#!/usr/bin/env python3
"""Shared offline vLLM chat backend for unified ideation.

This module keeps one in-process vLLM engine alive and batches chat-template
requests across concurrent callers. It is intentionally reusable from
`unified_ideation.py` so the shell runner only needs to launch a single Python
process for an entire experiment grid.

Environment:
    VLLM_OFFLINE_TQDM — if unset or 1/true, vLLM ``LLM.generate`` uses tqdm
    progress bars. Set to 0/false/off to disable (e.g. when logs must stay
    plain-line).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)

# Errors that affect a single request only (e.g. one prompt is too long).
# When the batched generate() raises one of these, the worker falls back to
# per-request generation so that a single bad prompt does not poison the whole
# batch and tear down the process.
_PER_REQUEST_ERROR_MARKERS = (
    "maximum context length",
    "VLLMValidationError",
    "input_tokens",
    "prompt is too long",
    "prompt_token_ids",
)


def _is_per_request_error(exc: BaseException) -> bool:
    """Heuristic for errors caused by one bad prompt rather than engine-wide failure."""
    name = type(exc).__name__
    if name in {"VLLMValidationError", "ValidationError"}:
        return True
    msg = str(exc)
    return any(marker in msg for marker in _PER_REQUEST_ERROR_MARKERS)


def strip_reasoning_prefix(text: str) -> str:
    """Drop any leading chain-of-thought produced by Qwen3 / other reasoning models.

    Rules:
    - If the response contains one or more ``</think>`` tags, keep only the
      substring after the LAST such tag. This also handles nested / malformed
      ``<think>`` blocks emitted by some fine-tunes.
    - If it contains a ``<think>`` opener but no closing ``</think>`` (the
      thinking hit max_tokens), return an empty string: the model never
      finished reasoning, so there is no post-think answer to keep. The empty
      response lets agent-level status detection mark the run as incomplete.
    - If it contains neither tag, return the text unchanged.

    The env var ``KEEP_REASONING=1`` disables stripping entirely (useful for
      inspecting a model's thinking trace).
    """
    if not isinstance(text, str) or not text:
        return text
    if os.getenv("KEEP_REASONING", "").strip().lower() in {"1", "true", "yes", "on"}:
        return text
    last_close = None
    for match in _THINK_CLOSE.finditer(text):
        last_close = match
    if last_close is not None:
        return text[last_close.end():].lstrip()
    if _THINK_OPEN.search(text) is not None:
        return ""
    return text


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "".join(chunks)
    return str(content)


def _pop_offline_vllm_env() -> Dict[str, str]:
    """Read and remove AIS-specific ``VLLM_*`` keys before importing ``vllm``.

    vLLM 0.19+ warns on unknown ``VLLM_*`` environment variables. This repo uses
    several for ``LLMGenerator`` (model path, batching, thinking toggle, etc.);
    they are not official engine flags, so we snapshot them here and ``pop``
    them from ``os.environ`` before ``from vllm import LLM`` runs.
    """
    keys = (
        "VLLM_MODEL",
        "VLLM_OFFLINE_BATCH_SIZE",
        "VLLM_OFFLINE_BATCH_WAIT_MS",
        "VLLM_OFFLINE_TENSOR_PARALLEL_SIZE",
        "VLLM_OFFLINE_MAX_MODEL_LEN",
        "VLLM_OFFLINE_GPU_MEMORY_UTILIZATION",
        "VLLM_OFFLINE_MAX_NUM_SEQS",
        "VLLM_OFFLINE_TQDM",
        "VLLM_DISABLE_THINKING",
        "VLLM_LANGUAGE_MODEL_ONLY",
    )
    out: Dict[str, str] = {}
    for k in keys:
        v = os.environ.pop(k, None)
        if v is not None:
            out[k] = v
    return out


class _QueuedRequest:
    def __init__(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> None:
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.event = threading.Event()
        self.result: Optional[str] = None
        self.error: Optional[BaseException] = None

    @property
    def generation_key(self) -> Tuple[str, float, int, Optional[int]]:
        return (self.model, round(self.temperature, 6), self.max_tokens, self.seed)


class LLMGenerator:
    def __init__(
        self,
        model_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_num_seqs: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        batch_wait_ms: Optional[int] = None,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "Offline vLLM inference requires the `vllm` and `transformers` packages "
                "in the active Python environment."
            ) from exc

        self._sampling_params_cls = SamplingParams
        self.model_name = model_name or os.getenv("VLLM_MODEL") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen3.6-35B-A3B"
        self.max_batch_size = max_batch_size or int(os.getenv("VLLM_OFFLINE_BATCH_SIZE", os.getenv("MAX_NUM_SEQS", "64")))
        self.batch_wait_ms = batch_wait_ms or int(os.getenv("VLLM_OFFLINE_BATCH_WAIT_MS", "10"))
        tensor_parallel_size = tensor_parallel_size or int(
            os.getenv("VLLM_OFFLINE_TENSOR_PARALLEL_SIZE", os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        )
        max_model_len = max_model_len or int(os.getenv("VLLM_OFFLINE_MAX_MODEL_LEN", os.getenv("MAX_MODEL_LEN", "32768")))
        gpu_memory_utilization = gpu_memory_utilization or float(
            os.getenv("VLLM_OFFLINE_GPU_MEMORY_UTILIZATION", os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
        )
        max_num_seqs = max_num_seqs or int(os.getenv("VLLM_OFFLINE_MAX_NUM_SEQS", str(self.max_batch_size)))

        # Qwen3-specific knobs
        self._language_model_only = (
            os.getenv("VLLM_LANGUAGE_MODEL_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
        )
        self._disable_thinking = (
            os.getenv("VLLM_DISABLE_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}
        )

        self.max_model_len = int(max_model_len)
        llm_kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )
        if self._language_model_only:
            llm_kwargs["language_model_only"] = True
        self._llm = LLM(**llm_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = self._tokenizer
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: deque[_QueuedRequest] = deque()
        self._worker = threading.Thread(target=self._worker_loop, name="vllm-offline-batcher", daemon=True)
        self._worker.start()

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_messages = [
            {
                "role": message.get("role", "user"),
                "content": normalize_message_content(message.get("content", "")),
            }
            for message in messages
        ]
        tmpl_kwargs: Dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        if self._disable_thinking:
            # Qwen3 models use enable_thinking in the chat template.
            # Non-Qwen3 tokenizers may not accept this kwarg, so we try
            # gracefully and fall back to the default template on TypeError.
            try:
                return self._tokenizer.apply_chat_template(
                    prompt_messages, enable_thinking=False, **tmpl_kwargs
                )
            except TypeError:
                pass
        return self._tokenizer.apply_chat_template(prompt_messages, **tmpl_kwargs)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Return the chat-template tokenized length for ``messages``.

        Uses the same chat template apply_chat_template applies inside vLLM, so
        this is the budget the vLLM tokenizer will see at generation time.
        """
        prompt = self._render_prompt(messages)
        try:
            ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        except Exception:
            # Fall back to a coarse estimate if the tokenizer rejects the input.
            return max(1, len(prompt) // 3)
        return len(ids)

    def truncate_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        safety_margin: int = 128,
        min_user_tokens: int = 64,
    ) -> Tuple[List[Dict[str, str]], bool]:
        """Drop or shorten messages so the prompt fits the model context.

        Returns ``(new_messages, was_truncated)``.

        Strategy:
        1. Compute ``budget = max_model_len - max_tokens - safety_margin``.
        2. Drop the earliest non-{system, last-user} messages one by one until
           the rendered prompt fits the budget.
        3. If still too long, truncate the content of the last user message,
           keeping a head and tail slice with a clearly marked elision in the
           middle so the model can recover something useful.

        System messages and the last user message are always preserved (the
        last user message is the actual instruction the agent issued; dropping
        it would silently change semantics).
        """
        if not messages:
            return messages, False

        budget = self.max_model_len - int(max_tokens) - int(safety_margin)
        if budget <= min_user_tokens:
            # max_tokens is so big that no prompt fits even after truncation.
            # Nothing safe we can do without changing the caller's max_tokens
            # contract; let the caller / vLLM raise a clear error.
            return messages, False

        if self.count_message_tokens(messages) <= budget:
            return messages, False

        n = len(messages)
        last_user_idx = next(
            (i for i in range(n - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        protected = {i for i, m in enumerate(messages) if m.get("role") == "system"}
        if last_user_idx is not None:
            protected.add(last_user_idx)

        kept = list(range(n))
        # Drop oldest non-protected messages first.
        for i in range(n):
            if i in protected:
                continue
            kept = [j for j in kept if j != i]
            candidate = [messages[j] for j in kept]
            if self.count_message_tokens(candidate) <= budget:
                return candidate, True

        candidate = [messages[j] for j in kept]
        if last_user_idx is None:
            return candidate, True

        # Stage 2: truncate the last user message body itself.
        try:
            new_last_idx = kept.index(last_user_idx)
        except ValueError:
            return candidate, True

        user_msg = candidate[new_last_idx]
        user_content = normalize_message_content(user_msg.get("content", ""))
        try:
            user_tokens = self._tokenizer.encode(user_content, add_special_tokens=False)
        except Exception:
            return candidate, True

        overhead = self.count_message_tokens(candidate) - len(user_tokens)
        keep_user_tokens = max(min_user_tokens, budget - max(0, overhead))
        if keep_user_tokens >= len(user_tokens):
            return candidate, True

        head_keep = keep_user_tokens // 2
        tail_keep = keep_user_tokens - head_keep
        try:
            head_text = self._tokenizer.decode(user_tokens[:head_keep], skip_special_tokens=True)
            tail_text = self._tokenizer.decode(user_tokens[-tail_keep:], skip_special_tokens=True)
        except Exception:
            return candidate, True

        dropped = len(user_tokens) - keep_user_tokens
        candidate[new_last_idx] = {
            **user_msg,
            "content": (
                f"{head_text}\n\n"
                f"[... TRUNCATED {dropped} TOKENS FROM MIDDLE TO FIT CONTEXT WINDOW ...]\n\n"
                f"{tail_text}"
            ),
        }
        return candidate, True

    def _take_batch(self) -> List[_QueuedRequest]:
        with self._condition:
            while not self._pending:
                self._condition.wait()

            deadline = time.monotonic() + (self.batch_wait_ms / 1000.0)
            while len(self._pending) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            first = self._pending.popleft()
            batch = [first]
            deferred: deque[_QueuedRequest] = deque()
            while self._pending and len(batch) < self.max_batch_size:
                candidate = self._pending.popleft()
                if candidate.generation_key == first.generation_key:
                    batch.append(candidate)
                else:
                    deferred.append(candidate)
            while deferred:
                self._pending.appendleft(deferred.pop())
            return batch

    def _fulfill(self, batch: List[_QueuedRequest], outputs: List[str]) -> None:
        for request, text in zip(batch, outputs):
            request.result = text.strip()
            request.event.set()

    def _fail(self, batch: List[_QueuedRequest], exc: BaseException) -> None:
        for request in batch:
            if request.event.is_set():
                continue
            request.error = exc
            request.event.set()

    def _decode_outputs(self, outputs: Any) -> List[str]:
        texts: List[str] = []
        for output in outputs:
            candidate = ""
            if getattr(output, "outputs", None):
                candidate = output.outputs[0].text
            texts.append(strip_reasoning_prefix(str(candidate)))
        return texts

    def _run_single(
        self,
        request: _QueuedRequest,
        prompt: str,
        sampling_params: Any,
    ) -> None:
        """Run one prompt in isolation, marking its result or error.

        Used as a fallback when the batched generate() fails on what looks like
        a per-request issue (e.g. a single prompt exceeding max_model_len).
        """
        try:
            outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
            texts = self._decode_outputs(outputs)
            request.result = (texts[0] if texts else "").strip()
        except BaseException as exc:
            print(
                f"[vllm] per-request generation failed and was isolated: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            request.error = exc
        finally:
            request.event.set()

    def _worker_loop(self) -> None:
        while True:
            batch = self._take_batch()
            sampling_params = self._sampling_params_cls(
                temperature=batch[0].temperature,
                max_tokens=batch[0].max_tokens,
                seed=batch[0].seed,
            )
            use_tqdm = os.getenv("VLLM_OFFLINE_TQDM", "1").strip().lower() not in {
                "0", "false", "no", "off",
            }

            # Render prompts. A render error is per-request (bad messages /
            # tokenizer rejection); mark just that request and keep the rest.
            prompts: List[Optional[str]] = []
            survivors: List[_QueuedRequest] = []
            survivor_prompts: List[str] = []
            for request in batch:
                try:
                    prompt = self._render_prompt(request.messages)
                except BaseException as exc:
                    print(
                        f"[vllm] prompt render failed and was isolated: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    request.error = exc
                    request.event.set()
                    prompts.append(None)
                    continue
                prompts.append(prompt)
                survivors.append(request)
                survivor_prompts.append(prompt)

            if not survivors:
                continue

            try:
                outputs = self._llm.generate(survivor_prompts, sampling_params, use_tqdm=use_tqdm)
                self._fulfill(survivors, self._decode_outputs(outputs))
            except BaseException as exc:
                if _is_per_request_error(exc) and len(survivors) > 1:
                    # Likely one bad prompt poisoned the batch. Re-issue every
                    # prompt one at a time so the good ones still complete.
                    print(
                        f"[vllm] batch generate failed with per-request error "
                        f"({type(exc).__name__}); falling back to single-prompt "
                        f"isolation for {len(survivors)} requests.",
                        flush=True,
                    )
                    for request, prompt in zip(survivors, survivor_prompts):
                        self._run_single(request, prompt, sampling_params)
                else:
                    # Engine-level failure (OOM, CUDA crash, ...): propagate to
                    # all surviving requests. The caller decides whether to
                    # raise or swallow via the on_error parameter.
                    self._fail(survivors, exc)

    def submit(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        *,
        on_error: str = "raise",
    ) -> str:
        requested_model = model or self.model_name
        if requested_model != self.model_name:
            raise RuntimeError(
                f"Offline vLLM engine loaded model {self.model_name!r}, but request asked for {requested_model!r}. "
                "Run one model per process when using offline vLLM."
            )
        request = _QueuedRequest(messages, requested_model, temperature, max_tokens, seed)
        with self._condition:
            self._pending.append(request)
            self._condition.notify()
        request.event.wait()
        if request.error is not None:
            if on_error == "empty":
                print(
                    f"[vllm] submit() suppressed per-request error "
                    f"({type(request.error).__name__}: {request.error}); returning empty string.",
                    flush=True,
                )
                return ""
            raise RuntimeError(str(request.error)) from request.error
        return request.result or ""

    def generate(
        self,
        convs: List[List[Dict[str, str]]],
        temperature: float = 0.3,
        max_tokens: int = 16384,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
        *,
        on_error: str = "raise",
    ) -> List[str]:
        model_name = model_name or self.model_name
        requests = [
            _QueuedRequest(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            for messages in convs
        ]
        with self._condition:
            for request in requests:
                self._pending.append(request)
            self._condition.notify_all()

        outputs: List[str] = []
        for request in requests:
            request.event.wait()
            if request.error is not None:
                if on_error == "empty":
                    print(
                        f"[vllm] generate() suppressed per-request error "
                        f"({type(request.error).__name__}: {request.error}); using empty string.",
                        flush=True,
                    )
                    outputs.append("")
                    continue
                raise RuntimeError(str(request.error)) from request.error
            outputs.append(request.result or "")
        return outputs


_GENERATOR: Optional[LLMGenerator] = None
_GENERATOR_LOCK = threading.Lock()
_GENERATOR_INIT_ERROR: Optional[BaseException] = None


def get_generator() -> LLMGenerator:
    global _GENERATOR, _GENERATOR_INIT_ERROR
    with _GENERATOR_LOCK:
        if _GENERATOR_INIT_ERROR is not None:
            raise RuntimeError(
                "Offline vLLM backend initialization previously failed. "
                "Fix the GPU memory / vLLM config issue and restart the process."
            ) from _GENERATOR_INIT_ERROR
        if _GENERATOR is None:
            try:
                _GENERATOR = LLMGenerator()
            except BaseException as exc:
                _GENERATOR_INIT_ERROR = exc
                raise
        return _GENERATOR


def warmup_generator() -> None:
    """Initialize the shared generator once up front.

    This prevents a burst of concurrent task threads from repeatedly retrying
    engine startup after the first failure.
    """
    get_generator()


def offline_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    seed: Optional[int] = None,
) -> str:
    generator = get_generator()
    return generator.submit(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
