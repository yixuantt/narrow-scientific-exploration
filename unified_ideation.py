import concurrent.futures
import copy
import importlib
import json
import os
import random
import re
import ssl
import sys
import threading
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

_PIPELINE_DIR = Path(__file__).resolve().parent / "scripts" / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from build_canonical_states import (
    build_memory_paper,
    choose_neighbors_stochastic,
    filter_corpus_records,
    precompute_all_terms,
    read_jsonl,
    tag_corpus_with_research_areas,
)
from run_inference_vllm import (
    get_generator,
    offline_chat_completion as run_vllm_offline_chat_completion,
    warmup_generator as warmup_vllm_offline_generator,
)


ROOT = Path(__file__).resolve().parent
EXTERNAL_DIR = ROOT / "external"
DEFAULT_MEMORY_CORPUS = (
    ROOT
    / "data"
    / "DBLP-Citation-network-V18"
    / "matched_all_master_g22_25.paper_corpus.jsonl"
)
_MEMORY_CORPUS_CACHE: Dict[Tuple[str, int, str, Optional[Tuple[int, ...]]], List[Dict[str, Any]]] = {}
_MEMORY_CORPUS_CACHE_LOCK = threading.Lock()
_CANONICAL_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
_CANONICAL_STATE_CACHE_LOCK = threading.Lock()
_IMPORTED_MODULE_CACHE: Dict[Tuple[str, str], Any] = {}
_IMPORTED_MODULE_CACHE_LOCK = threading.Lock()
_ENV_LOADED = False
_ENV_LOADED_LOCK = threading.Lock()


def load_env_file(path: Path) -> None:
    global _ENV_LOADED
    with _ENV_LOADED_LOCK:
        if _ENV_LOADED:
            return
        if not path.exists():
            _ENV_LOADED = True
            return

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
        _ENV_LOADED = True


def read_cached_json(path: Path) -> Dict[str, Any]:
    cache_key = str(path.resolve())
    with _CANONICAL_STATE_CACHE_LOCK:
        cached = _CANONICAL_STATE_CACHE.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        payload = read_json(path)
        _CANONICAL_STATE_CACHE[cache_key] = payload
        return copy.deepcopy(payload)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def to_relative_path(path: Path) -> str:
    resolved = path.resolve()
    return os.path.relpath(resolved, ROOT)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "run"


def model_dirname(model: Optional[str]) -> str:
    model_name = model or os.getenv("OPENAI_MODEL") or "unknown_model"
    return slugify(model_name)


def extract_balanced_json_object(text: str, start_index: int = 0) -> Optional[str]:
    start = text.find("{", start_index)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    code_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    return stripped


_THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_LEADING_OPEN_THINK = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)
_TRAILING_CLOSE_THINK = re.compile(r"</think\s*>\s*$", re.IGNORECASE)


def strip_think_block(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks (and orphan openers) from a response."""
    if not isinstance(text, str) or not text:
        return text
    cleaned = _THINK_BLOCK_PATTERN.sub("", text)
    cleaned = _LEADING_OPEN_THINK.sub("", cleaned, count=1)
    cleaned = _TRAILING_CLOSE_THINK.sub("", cleaned, count=1)
    return cleaned.strip()


def strip_surrounding_markdown_emphasis(text: str) -> str:
    stripped = text.strip()
    while True:
        updated = stripped
        if updated.startswith("**") and updated.endswith("**") and len(updated) >= 4:
            updated = updated[2:-2].strip()
        elif updated.startswith("*") and updated.endswith("*") and len(updated) >= 2:
            updated = updated[1:-1].strip()
        if updated == stripped:
            break
        stripped = updated
    return stripped


def quote_unquoted_json_keys(text: str) -> str:
    result: List[str] = []
    in_string = False
    escape = False
    expecting_key = False
    index = 0

    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "{":
            expecting_key = True
            result.append(char)
            index += 1
            continue

        if char == ",":
            expecting_key = True
            result.append(char)
            index += 1
            continue

        if expecting_key:
            if char.isspace():
                result.append(char)
                index += 1
                continue
            if char in "}]":
                expecting_key = False
                result.append(char)
                index += 1
                continue
            if char == '"':
                in_string = True
                expecting_key = False
                result.append(char)
                index += 1
                continue

            key_start = index
            while index < len(text) and text[index] not in ":\n\r{}[],":
                index += 1
            if index < len(text) and text[index] == ":":
                raw_key = text[key_start:index]
                key = raw_key.strip()
                if key and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ \-./()]*", key):
                    leading_ws_len = len(raw_key) - len(raw_key.lstrip())
                    trailing_ws_len = len(raw_key) - len(raw_key.rstrip())
                    if leading_ws_len:
                        result.append(raw_key[:leading_ws_len])
                    result.append(json.dumps(key))
                    if trailing_ws_len:
                        result.append(raw_key[len(raw_key) - trailing_ws_len :])
                    result.append(":")
                    index += 1
                    expecting_key = False
                    continue
                result.append(raw_key)
                result.append(":")
                index += 1
                expecting_key = False
                continue

            result.append(text[key_start:index])
            expecting_key = False
            continue

        result.append(char)
        index += 1

    repaired = "".join(result)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def escape_invalid_json_backslashes(text: str) -> str:
    result: List[str] = []
    in_string = False
    escape = False
    index = 0

    while index < len(text):
        char = text[index]

        if in_string:
            if escape:
                result.append(char)
                escape = False
                index += 1
                continue

            if char == "\\":
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                    result.append(char)
                    escape = True
                else:
                    result.append("\\\\")
                index += 1
                continue

            result.append(char)
            if char == '"':
                in_string = False
            index += 1
            continue

        result.append(char)
        if char == '"':
            in_string = True
        index += 1

    return "".join(result)


def parse_json_like_payload(payload_text: str) -> Tuple[Optional[Any], Optional[str]]:
    candidate = strip_code_fence(payload_text).strip().rstrip(";")
    if not candidate:
        return None, None

    candidates: List[str] = [candidate]
    repaired = quote_unquoted_json_keys(candidate)
    if repaired != candidate:
        candidates.append(repaired)
    escaped = escape_invalid_json_backslashes(candidate)
    if escaped != candidate:
        candidates.append(escaped)
    escaped_repaired = escape_invalid_json_backslashes(repaired)
    if escaped_repaired != repaired:
        candidates.append(escaped_repaired)

    seen: set[str] = set()
    for candidate_text in candidates:
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        return parsed, json.dumps(parsed, ensure_ascii=False)

    return None, None


def unwrap_tool_wrapper(action: str, payload: Any) -> Tuple[str, Any]:
    if action.lower() != "tool" or not isinstance(payload, dict):
        return action, payload

    tool_name = None
    for key in ("tool", "tool_name", "toolName", "name", "action"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            tool_name = value.strip()
            break

    arguments = None
    for key in ("arguments", "args", "parameters", "input"):
        if key in payload:
            arguments = payload[key]
            break

    if tool_name is None:
        return action, payload
    if arguments is None:
        arguments = {key: value for key, value in payload.items() if key not in {"tool", "tool_name", "toolName", "name", "action"}}
    return tool_name, arguments


def parse_markdown_labeled_fields(text: str) -> Optional[Dict[str, str]]:
    label_pattern = re.compile(
        r"(?:^\s*(?:[-*]\s+)?\*\*(?P<bold_label>[^*]+?)\*\*\s*$|^\s*(?:[-*]\s+)?\*\*(?P<bold_label_inline>[^*]+?)\*\*\s*|^\s*(?:[-*]\s+)?(?P<plain_label>(?:\d+\.\s*)?[A-Za-z][A-Za-z0-9 \-/()]+):\s*)",
        re.MULTILINE,
    )
    matches = list(label_pattern.finditer(text))
    if not matches:
        return None

    alias_map = {
        "hypothesis": "Short Hypothesis",
        "core hypothesis": "Short Hypothesis",
        "related work positioning": "Related Work",
        "related work": "Related Work",
        "risk factors and limitations": "Risk Factors and Limitations",
        "robustness test": "Experiments",
        "short hypothesis": "Short Hypothesis",
        "technical approach": "Abstract",
    }
    known_labels = {
        "Name",
        "Title",
        "Short Hypothesis",
        "Related Work",
        "Abstract",
        "Experiments",
        "Risk Factors and Limitations",
        "Feasibility",
    }

    filtered_matches: List[Tuple[str, int, int]] = []
    for match in matches:
        raw_label = match.group("bold_label") or match.group("bold_label_inline") or match.group("plain_label") or ""
        label = re.sub(r"^\d+\.\s*", "", raw_label.strip())
        label = label.rstrip(":").strip()
        label = alias_map.get(label.lower(), label)
        if label in known_labels:
            filtered_matches.append((label, match.start(), match.end()))

    if not filtered_matches:
        return None

    fields: Dict[str, str] = {}
    for idx, (label, _start, end) in enumerate(filtered_matches):
        value_start = end
        value_end = filtered_matches[idx + 1][1] if idx + 1 < len(filtered_matches) else len(text)
        value = text[value_start:value_end].strip()
        value = strip_surrounding_markdown_emphasis(value)
        if value:
            fields[label] = value

    heading_match = re.search(
        r"^\s*#{1,6}\s*Final(?: Research)? Proposal:?\s*(.+?)\s*$",
        text,
        re.MULTILINE,
    )
    if heading_match and "Name" not in fields:
        fields["Name"] = strip_surrounding_markdown_emphasis(heading_match.group(1).strip())

    if "Title" not in fields and "Name" in fields and ("Abstract" in fields or "Short Hypothesis" in fields):
        fields["Title"] = fields["Name"]

    required = {"Name", "Title", "Short Hypothesis", "Abstract"}
    if required.issubset(fields):
        return fields
    partial_required = {"Name", "Title", "Short Hypothesis"}
    if partial_required.issubset(fields):
        return fields
    return None


_FLAT_LLM_PRIMARY_FIELDS = ("Title", "Name", "Short Hypothesis", "Abstract")


def _flat_llm_dict_is_substantive(parsed: Any) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return False
    for key in _FLAT_LLM_PRIMARY_FIELDS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def parse_flat_llm_response(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Best-effort parse of a flat_llm response.

    Returns (parsed_dict, parse_error, parse_method). On success parse_error is
    None; on failure parsed_dict is None and parse_error explains why. Tries
    JSON first (with key/escape repairs), then falls back to extracting markdown
    labelled fields as used by the agent_laboratory parser.
    """
    if not isinstance(raw, str):
        return None, "non-string response", None
    if not raw.strip():
        return None, "empty response", None

    cleaned = strip_think_block(raw)
    cleaned = strip_code_fence(cleaned)
    if not cleaned.strip():
        return None, "response was only thinking/code fence wrappers", None

    json_blob = extract_balanced_json_object(cleaned)
    if json_blob:
        parsed, _normalized = parse_json_like_payload(json_blob)
        if isinstance(parsed, dict) and parsed:
            return parsed, None, "json_balanced"

    markdown_fields = parse_markdown_labeled_fields(cleaned)
    if isinstance(markdown_fields, dict) and markdown_fields:
        return markdown_fields, None, "markdown_labeled"

    if json_blob:
        return None, "found JSON-like block but could not parse it", None
    return None, "no JSON object or labelled markdown fields found", None


def parse_ai_scientist_response(response_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    normalized_response_text = response_text
    for label in ("ACTION", "ARGUMENTS", "THOUGHT"):
        normalized_response_text = re.sub(
            rf"\*\*\s*{label}\s*:\s*\*\*",
            f"{label}:",
            normalized_response_text,
            flags=re.IGNORECASE,
        )
        normalized_response_text = re.sub(
            rf"\*\*\s*{label}\s*\*\*\s*:",
            f"{label}:",
            normalized_response_text,
            flags=re.IGNORECASE,
        )

    action_label = r"(?:\*\*)?\s*ACTION(?::)?\s*(?:\*\*)?\s*:"
    arguments_label = r"(?:\*\*)?\s*ARGUMENTS(?::)?\s*(?:\*\*)?\s*:"
    action_match = re.search(
        rf"{action_label}\s*(.*?)\s*{arguments_label}",
        normalized_response_text,
        re.DOTALL | re.IGNORECASE,
    )
    arguments_match = re.search(
        rf"{arguments_label}\s*(.*?)(?:$|\n(?:\*\*)?\s*THOUGHT(?::)?\s*(?:\*\*)?:|\n$)",
        normalized_response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if action_match and arguments_match:
        action = action_match.group(1).strip()
        arguments_text = arguments_match.group(1).strip()
        parsed_payload, normalized_payload = parse_json_like_payload(arguments_text)
        if parsed_payload is not None and normalized_payload is not None:
            action, parsed_payload = unwrap_tool_wrapper(action, parsed_payload)
            return action, json.dumps(parsed_payload, ensure_ascii=False), "action_arguments"
        return action, strip_code_fence(arguments_text), "action_arguments"

    call_match = re.search(r"call:\s*([A-Za-z_][A-Za-z0-9_]*)", response_text, re.IGNORECASE)
    if call_match:
        action = call_match.group(1).strip()
        json_blob = extract_balanced_json_object(response_text, call_match.end())
        if json_blob:
            parsed_payload, normalized_payload = parse_json_like_payload(json_blob)
            if parsed_payload is not None and normalized_payload is not None:
                action, parsed_payload = unwrap_tool_wrapper(action, parsed_payload)
                return action, json.dumps(parsed_payload, ensure_ascii=False), "call_syntax"
            return action, json_blob, "call_syntax"

    fenced_json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if fenced_json_match:
        parsed_payload, normalized_payload = parse_json_like_payload(fenced_json_match.group(1))
        if parsed_payload is not None and normalized_payload is not None:
            if isinstance(parsed_payload, dict):
                if "idea" in parsed_payload:
                    return "FinalizeIdea", json.dumps(parsed_payload, ensure_ascii=False), "fenced_json"
                if "query" in parsed_payload:
                    return "SearchSemanticScholar", json.dumps(parsed_payload, ensure_ascii=False), "fenced_json"

    markdown_fields = parse_markdown_labeled_fields(response_text)
    if markdown_fields:
        arguments_text = json.dumps({"idea": markdown_fields}, ensure_ascii=False)
        return "FinalizeIdea", arguments_text, "markdown_fallback"

    return None, None, None


def normalize_openai_model(model: Optional[str], fallback: str = "gpt-4o-mini") -> str:
    if not model:
        return fallback

    lowered = model.lower()
    if "4o-mini" in lowered:
        return "gpt-4o-mini"
    if "gpt-4o" in lowered:
        return "gpt-4o"
    if lowered.startswith("o3"):
        return "o3-mini"
    if lowered.startswith("o1"):
        return "o1"
    return model


def normalize_base_url(base_url: Optional[str]) -> str:
    if not base_url:
        return "https://api.openai.com/v1"
    return base_url.rstrip("/")


def get_inference_backend() -> str:
    backend = os.getenv("UNIFIED_INFERENCE_BACKEND", "openai_api").strip().lower()
    alias_map = {
        "api": "openai_api",
        "http": "openai_api",
        "openai": "openai_api",
        "vllm": "vllm_offline",
        "offline": "vllm_offline",
        "offline_vllm": "vllm_offline",
    }
    return alias_map.get(backend, backend)


def is_local_openai_compatible_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def offline_vllm_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    seed: Optional[int] = None,
) -> str:
    return run_vllm_offline_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )


def build_ssl_context() -> Optional[ssl.SSLContext]:
    verify_flag = os.getenv("OPENAI_TLS_VERIFY", "true").strip().lower()
    ca_bundle = os.getenv("OPENAI_CA_BUNDLE")

    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    if verify_flag in {"0", "false", "no", "off"}:
        return ssl._create_unverified_context()
    return None


def openai_compatible_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    seed: Optional[int] = None,
) -> str:
    """Send a single chat completion request."""
    if get_inference_backend() == "vllm_offline":
        return offline_vllm_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    base_url = normalize_base_url(os.getenv("OPENAI_BASE_URL"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and is_local_openai_compatible_endpoint(base_url):
        api_key = "EMPTY"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not base_url.endswith("/chat/completions"):
        if base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
    else:
        url = base_url

    payload_dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload_dict["seed"] = seed
    payload = json.dumps(payload_dict).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ssl_context = build_ssl_context()
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        hint = ""
        reason_text = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in reason_text:
            hint = " Set OPENAI_TLS_VERIFY=false for self-signed endpoints or OPENAI_CA_BUNDLE=/path/to/ca.pem."
        raise RuntimeError(f"URLError: {exc.reason}.{hint}") from exc

    parsed = json.loads(raw)
    try:
        message = parsed["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected completion payload: {raw}") from exc

    if isinstance(message, list):
        text_chunks = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
        return "".join(text_chunks).strip()
    return str(message).strip()


def batch_chat_completions(
    requests: List[Dict[str, Any]],
    batch_size: int = 8,
) -> List[str]:
    """
    Send multiple chat completion requests concurrently (up to batch_size in flight at once).
    Each element of `requests` is a kwargs dict for openai_compatible_chat_completion.
    Returns responses in the same order as `requests`.
    """
    results: List[str] = [""] * len(requests)

    def _call(idx: int, kwargs: Dict[str, Any]) -> Tuple[int, str]:
        return idx, openai_compatible_chat_completion(**kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_call, i, req): i for i, req in enumerate(requests)}
        for future in concurrent.futures.as_completed(futures):
            idx, text = future.result()
            results[idx] = text

    return results


def validate_canonical_state(state: Dict[str, Any]) -> None:
    required = ["task_id", "subarea", "challenge", "goal", "memory_papers"]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Canonical state missing required fields: {', '.join(missing)}")
    if not isinstance(state.get("memory_papers"), list):
        raise ValueError("canonical_state.memory_papers must be a list")


def get_cached_memory_corpus(
    corpus_path: Path,
    *,
    min_context_size: int,
    paper_type: str,
    corpus_years: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    years_key = tuple(sorted(int(year) for year in corpus_years)) if corpus_years is not None else None
    cache_key = (str(corpus_path.resolve()), int(min_context_size), str(paper_type), years_key)

    with _MEMORY_CORPUS_CACHE_LOCK:
        cached = _MEMORY_CORPUS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        corpus = read_jsonl(corpus_path)
        corpus = filter_corpus_records(corpus, corpus_years, paper_type)
        precompute_all_terms(corpus)
        tag_corpus_with_research_areas(corpus, min_context_size)
        _MEMORY_CORPUS_CACHE[cache_key] = corpus
        return corpus


def apply_rq1a_memory_resample(
    state: Dict[str, Any],
    corpus_path: Path,
    memory_resample_seed: int,
    *,
    min_context_size: int = 5,
    paper_type: str = "all",
    corpus_years: Optional[Sequence[int]] = None,
    pool_multiplier: int = 8,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    RQ1a: fix anchor and research-area metadata; redraw seed papers from the same corpus
    slice used for research-area tagging.
    """
    source = state.get("source") or {}
    anchor_id = source.get("anchor_paper_id")
    if not anchor_id:
        raise ValueError("canonical_state.source.anchor_paper_id is required for memory resampling.")

    memory_size = len(state["memory_papers"])
    if memory_size < 1:
        raise ValueError("memory_papers must be non-empty.")

    corpus = get_cached_memory_corpus(
        corpus_path,
        min_context_size=min_context_size,
        paper_type=paper_type,
        corpus_years=corpus_years,
    )

    anchor = next((record for record in corpus if record["paper_id"] == anchor_id), None)
    if anchor is None:
        raise ValueError(f"Anchor paper_id {anchor_id!r} not found in corpus {corpus_path}.")

    anchor_typed = copy.deepcopy(anchor)
    anchor_typed["_context_id"] = source["context_id"]
    anchor_typed["_context_label"] = source["context_label"]
    kw = source.get("context_keywords")
    if kw is not None:
        anchor_typed["_context_phrases"] = list(kw)
    elif "_context_phrases" not in anchor_typed:
        anchor_typed["_context_phrases"] = []

    rng = random.Random(memory_resample_seed)
    memory_records = choose_neighbors_stochastic(
        anchor_typed,
        corpus,
        memory_size,
        rng,
        pool_multiplier=pool_multiplier,
    )

    out = copy.deepcopy(state)
    out["memory_papers"] = [build_memory_paper(record) for record in memory_records]
    provenance = {
        "memory_resample_seed": memory_resample_seed,
        "corpus_path": to_relative_path(corpus_path),
        "min_context_size": min_context_size,
        "paper_type": paper_type,
        "corpus_years": list(corpus_years) if corpus_years is not None else None,
        "pool_multiplier": pool_multiplier,
        "memory_paper_ids": [record.get("paper_id") for record in memory_records],
    }
    return out, provenance


def build_workshop_markdown(state: Dict[str, Any]) -> str:
    # Prompt policy: expose ONLY title + abstract for each literature paper.
    # No task_id / subarea / challenge / goal / keywords / constraints / year / venue.
    # No output-requirement injection either -- the output format is already
    # defined by each agent's own system prompt (AI-Scientist-v2's idea_generation
    # template and _FLAT_LLM_SYSTEM both specify it).
    memory_papers = state.get("memory_papers", [])

    lines: List[str] = ["## Literature Context"]

    if memory_papers:
        for index, paper in enumerate(memory_papers, start=1):
            title = safe_text(paper.get("title"), f"Paper {index}")
            abstract = safe_text(paper.get("abstract"), "").strip() or "No abstract provided."
            lines.extend(
                [
                    f"### Paper {index}",
                    f"- Title: {title}",
                    f"- Abstract: {abstract}",
                    "",
                ]
            )
    else:
        lines.append("No literature context provided.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def summarize_memory_papers(state: Dict[str, Any], limit: int = 6) -> str:
    # Prompt policy: title + abstract only (no goal / challenge / year / venue).
    papers = state.get("memory_papers", [])[:limit]
    if not papers:
        return "No literature review available."

    lines = ["Condensed literature review:"]
    for index, paper in enumerate(papers, start=1):
        # Keep full abstract; only collapse whitespace so the list stays one-line-per-paper.
        abstract = safe_text(paper.get("abstract"), "").strip()
        abstract = re.sub(r"\s+", " ", abstract)
        title = safe_text(paper.get("title"), f"Paper {index}")
        lines.append(f"{index}. {title}: {abstract or 'No abstract provided.'}")
    return "\n".join(lines)


def build_researchagent_context(state: Dict[str, Any]) -> Dict[str, Any]:
    # Prompt policy: treat memory_papers as a flat set. The upstream ResearchAgent
    # API requires `paper` + `references`, so we still fill those for library
    # compatibility, but `papers` preserves the full flat list for logging.
    memory_papers = state.get("memory_papers", [])[:8]
    papers = [
        {"title": p.get("title"), "abstract": p.get("abstract")}
        for p in memory_papers
    ]
    head = papers[0] if papers else {"title": "", "abstract": ""}
    references = papers[1:] if len(papers) > 1 else []

    return {
        "papers": papers,
        "paper": head,
        "references": references,
        "entities": [],
    }


def extract_fenced_block(text: str, label: str) -> str:
    pattern = rf"```{re.escape(label)}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def lexical_overlap_score(query: str, text: str) -> int:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms)


class LocalSemanticScholarTool:
    name = "SearchSemanticScholar"

    def __init__(self, papers: List[Dict[str, Any]], max_results: int = 5):
        self.description = "Search local memory papers instead of remote Semantic Scholar."
        self.papers = papers
        self.max_results = max_results

    def use_tool(self, query: str) -> str:
        ranked = sorted(
            self.papers,
            key=lambda paper: (
                lexical_overlap_score(query, f"{paper.get('title', '')} {paper.get('abstract', '')}"),
                paper.get("year") or 0,
            ),
            reverse=True,
        )[: self.max_results]

        if not ranked:
            return "No locally matched papers found."

        chunks = []
        for index, paper in enumerate(ranked, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"{index}: {paper.get('title', 'Unknown Title')}.",
                        f"Abstract: {paper.get('abstract', 'No abstract provided.')}",
                    ]
                )
            )
        return "\n\n".join(chunks)


def import_module_from_repo(repo_root: Path, module_name: str) -> Any:
    repo_str = str(repo_root)
    cache_key = (repo_str, module_name)
    with _IMPORTED_MODULE_CACHE_LOCK:
        cached = _IMPORTED_MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        module = importlib.import_module(module_name)
        _IMPORTED_MODULE_CACHE[cache_key] = module
        return module


def require_external_module(repo_root: Path, module_name: str, framework_name: str) -> Any:
    if not repo_root.exists():
        raise RuntimeError(
            f"{framework_name} source tree is required but was not found at {repo_root}. "
            "Place the original external repository under availability_repo/external/."
        )
    try:
        return import_module_from_repo(repo_root, module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import {framework_name} from {repo_root}. "
            "Install its requirements and keep the original external source tree in place."
        ) from exc


def clear_external_module_cache(module_roots: Sequence[str]) -> None:
    roots = tuple(module_roots)
    for name in list(sys.modules):
        if name in roots or any(name.startswith(f"{root}.") for root in roots):
            del sys.modules[name]


def run_ai_scientist_v2(
    state: Dict[str, Any],
    model: str,
    num_reflections: int = 5,
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    workshop_markdown = build_workshop_markdown(state)
    local_tool = LocalSemanticScholarTool(state.get("memory_papers", []))
    stages: List[Dict[str, Any]] = []
    msg_history: List[Dict[str, str]] = []
    last_tool_results = ""
    final_idea = None
    client_model = model

    repo_root = EXTERNAL_DIR / "AI-Scientist-v2"
    module = require_external_module(
        repo_root,
        "ai_scientist.perform_ideation_temp_free",
        "AI-Scientist-v2",
    )
    client, client_model = module.create_client(model)
    system_prompt = module.system_prompt
    generation_prompt_template = module.idea_generation_prompt
    reflection_prompt_template = module.idea_reflection_prompt

    for reflection_round in range(num_reflections):
        stage_name = "initial_generation" if reflection_round == 0 else "reflection"
        if reflection_round == 0:
            prompt_text = generation_prompt_template.format(
                workshop_description=workshop_markdown,
                prev_ideas_string="",
            )
        else:
            prompt_text = reflection_prompt_template.format(
                current_round=reflection_round + 1,
                num_reflections=num_reflections,
                last_tool_results=last_tool_results or "No new results.",
            )

        response_text, msg_history = module.get_response_from_llm(
            prompt=prompt_text,
            client=client,
            model=client_model,
            system_message=system_prompt,
            msg_history=msg_history,
        )

        stage_record: Dict[str, Any] = {
            "stage": stage_name,
            "round_index": reflection_round,
            "timestamp_utc": utc_now_iso(),
            "prompt": prompt_text,
            "raw_response": response_text,
        }

        action, arguments_text, parse_mode = parse_ai_scientist_response(response_text)
        if not action or not arguments_text:
            stage_record["parse_error"] = "Failed to parse ACTION/ARGUMENTS."
            stages.append(stage_record)
            continue

        stage_record["action"] = action
        stage_record["parse_mode"] = parse_mode
        stage_record["arguments_raw"] = arguments_text

        try:
            arguments_json = json.loads(arguments_text)
        except json.JSONDecodeError:
            stage_record["parse_error"] = "Arguments are not valid JSON."
            stages.append(stage_record)
            continue

        stage_record["arguments"] = arguments_json

        if action == local_tool.name:
            tool_query = arguments_json.get("query", "")
            last_tool_results = local_tool.use_tool(tool_query)
            stage_record["tool_result"] = last_tool_results
        elif action == "FinalizeIdea":
            final_idea = arguments_json.get("idea")
            stage_record["finalized_idea"] = final_idea
            stages.append(stage_record)
            break
        else:
            stage_record["tool_error"] = f"Unsupported action '{action}' in local harness."

        stages.append(stage_record)

    return {
        "agent": "ai_scientist_v2",
        "adapter_input": {
            "workshop_markdown": workshop_markdown,
        },
        "model_requested": model,
        "model_used": client_model,
        "run_seed": run_seed,
        "stages": stages,
        "final_output": final_idea,
        "status": "ok" if final_idea else "incomplete",
    }


def run_research_agent(
    state: Dict[str, Any],
    model: str,
    iterations: int = 2,
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    context = build_researchagent_context(state)
    adapter_input = json.loads(json.dumps(context))
    history: Dict[str, List[Dict[str, Any]]] = {"problems": [], "methods": [], "experiments": []}
    stages: List[Dict[str, Any]] = []
    try:
        repo_root = EXTERNAL_DIR / "ResearchAgent" / "code"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        clear_external_module_cache(["knowledge", "models", "pipelines", "utils"])

        from models.openai import OpenAIClient
        from pipelines.agents import (
            ExperimentDesigner,
            ExperimentValidator,
            MethodDeveloper,
            MethodValidator,
            ProblemIdentifier,
            ProblemValidator,
        )
        from utils.evaluation import get_avg_feedbacks_score, get_num_feedbacks_scores

        api_client = OpenAIClient(model=model)

        problem_identifier = ProblemIdentifier(api_client)
        problem_validator = ProblemValidator(api_client)
        method_developer = MethodDeveloper(api_client)
        method_validator = MethodValidator(api_client)
        experiment_designer = ExperimentDesigner(api_client)
        experiment_validator = ExperimentValidator(api_client)

        use_upstream = True
    except Exception as exc:
        raise RuntimeError(
            f"ResearchAgent source tree is required but could not be imported from {EXTERNAL_DIR / 'ResearchAgent' / 'code'}. "
            "Install its requirements and keep the original external source tree in availability_repo/external/."
        ) from exc

    if use_upstream:
        for index in range(iterations):
            generated = problem_identifier.run(context)
            context.update(generated)
            stages.append(
                {
                    "stage": "problem_identifier",
                    "iteration": index,
                    "timestamp_utc": utc_now_iso(),
                    "output": generated,
                }
            )
            feedback = problem_validator.run(context)
            context.update(feedback)
            stages.append(
                {
                    "stage": "problem_validator",
                    "iteration": index,
                    "timestamp_utc": utc_now_iso(),
                    "output": feedback,
                }
            )
            history["problems"].append(
                {
                    "problem": context.get("problem"),
                    "rationale": context.get("problem_rationale"),
                    "feedbacks": context.get("problem_feedbacks"),
                }
            )

        best_problem = max(
            history["problems"],
            key=lambda item: get_avg_feedbacks_score(item.get("feedbacks") or {})
            if get_num_feedbacks_scores(item.get("feedbacks") or {}) > 0
            else -1,
        )
        context.update(
            problem=best_problem.get("problem"),
            problem_rationale=best_problem.get("rationale"),
            problem_feedbacks=best_problem.get("feedbacks"),
        )
        stages.append(
            {
                "stage": "problem_selection",
                "timestamp_utc": utc_now_iso(),
                "output": best_problem,
            }
        )

        for index in range(iterations):
            generated = method_developer.run(context)
            context.update(generated)
            stages.append(
                {
                    "stage": "method_developer",
                    "iteration": index,
                    "timestamp_utc": utc_now_iso(),
                    "output": generated,
                }
            )
            feedback = method_validator.run(context)
            context.update(feedback)
            stages.append(
                {
                    "stage": "method_validator",
                    "iteration": index,
                    "timestamp_utc": utc_now_iso(),
                    "output": feedback,
                }
            )
            history["methods"].append(
                {
                    "method": context.get("method"),
                    "rationale": context.get("method_rationale"),
                    "feedbacks": context.get("method_feedbacks"),
                }
            )

        best_method = max(
            history["methods"],
            key=lambda item: get_avg_feedbacks_score(item.get("feedbacks") or {})
            if get_num_feedbacks_scores(item.get("feedbacks") or {}) > 0
            else -1,
        )
        context.update(
            method=best_method.get("method"),
            method_rationale=best_method.get("rationale"),
            method_feedbacks=best_method.get("feedbacks"),
        )
        stages.append(
            {
                "stage": "method_selection",
                "timestamp_utc": utc_now_iso(),
                "output": best_method,
            }
        )

        for index in range(iterations):
            generated = experiment_designer.run(context)
            context.update(generated)
            stages.append(
                {
                    "stage": "experiment_designer",
                    "iteration": index,
                    "timestamp_utc": utc_now_iso(),
                    "output": generated,
                }
            )
            feedback = experiment_validator.run(context)
            context.update(feedback)
            stages.append(
                {
                    "stage": "experiment_validator",
                    "iteration": index,
                    "timestamp_utc": utc_now_iso(),
                    "output": feedback,
                }
            )
            history["experiments"].append(
                {
                    "experiment": context.get("experiment"),
                    "rationale": context.get("experiment_rationale"),
                    "feedbacks": context.get("experiment_feedbacks"),
                }
            )

        best_experiment = max(
            history["experiments"],
            key=lambda item: get_avg_feedbacks_score(item.get("feedbacks") or {})
            if get_num_feedbacks_scores(item.get("feedbacks") or {}) > 0
            else -1,
        )
        context.update(
            experiment=best_experiment.get("experiment"),
            experiment_rationale=best_experiment.get("rationale"),
            experiment_feedbacks=best_experiment.get("feedbacks"),
        )
        stages.append(
            {
                "stage": "experiment_selection",
                "timestamp_utc": utc_now_iso(),
                "output": best_experiment,
            }
        )
    return {
        "agent": "research_agent",
        "adapter_input": adapter_input,
        "model_requested": model,
        "model_used": model,
        "run_seed": run_seed,
        "stages": stages,
        "final_output": {
            "problem": context.get("problem"),
            "problem_rationale": context.get("problem_rationale"),
            "method": context.get("method"),
            "method_rationale": context.get("method_rationale"),
            "experiment": context.get("experiment"),
            "experiment_rationale": context.get("experiment_rationale"),
            "history": history,
        },
        "status": "ok",
    }


def run_agent_laboratory(
    state: Dict[str, Any],
    model: str,
    max_steps: int = 8,
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    literature_summary = summarize_memory_papers(state)
    # Prompt policy: treat memory_papers as a flat set, no anchor/target paper.
    # research_topic is a generic instruction; the papers themselves are already
    # listed in literature_summary (title + abstract only, flat numbered list).
    research_topic = "Propose a novel, feasible research direction grounded in the literature review above."
    stages: List[Dict[str, Any]] = []
    dialogue = ""
    final_plan = None
    try:
        repo_root = EXTERNAL_DIR / "AgentLaboratory"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        clear_external_module_cache(["agents", "common_imports", "inference", "tools", "utils"])
        from agents import PhDStudentAgent, PostdocAgent

        openai_api_key = os.getenv("OPENAI_API_KEY")
        phd = PhDStudentAgent(model=model, notes=[], max_steps=max_steps, openai_api_key=openai_api_key)
        postdoc = PostdocAgent(model=model, notes=[], max_steps=max_steps, openai_api_key=openai_api_key)
        phd.lit_review_sum = literature_summary
        postdoc.lit_review_sum = literature_summary
        use_upstream = True
    except Exception as exc:
        raise RuntimeError(
            f"AgentLaboratory source tree is required but could not be imported from {EXTERNAL_DIR / 'AgentLaboratory'}. "
            "Install its requirements and keep the original external source tree in availability_repo/external/."
        ) from exc

    if use_upstream:
        for step in range(max_steps):
            postdoc_response = postdoc.inference(
                research_topic,
                "plan formulation",
                feedback=dialogue,
                step=step,
            )
            stage_record = {
                "stage": "postdoc_turn",
                "iteration": step,
                "timestamp_utc": utc_now_iso(),
                "raw_response": postdoc_response,
            }
            if "```DIALOGUE" in postdoc_response:
                dialogue_text = extract_fenced_block(postdoc_response, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue_text}"
                stage_record["dialogue"] = dialogue_text
            else:
                dialogue = ""

            if "```PLAN" in postdoc_response:
                final_plan = extract_fenced_block(postdoc_response, "PLAN")
                stage_record["plan"] = final_plan
                stages.append(stage_record)
                break

            stages.append(stage_record)

            phd_response = phd.inference(
                research_topic,
                "plan formulation",
                feedback=dialogue,
                step=step,
            )
            stage_record = {
                "stage": "phd_turn",
                "iteration": step,
                "timestamp_utc": utc_now_iso(),
                "raw_response": phd_response,
            }
            if "```DIALOGUE" in phd_response:
                dialogue_text = extract_fenced_block(phd_response, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue_text}"
                stage_record["dialogue"] = dialogue_text
            else:
                dialogue = ""
            stages.append(stage_record)
    adapter_input = {
        "research_topic": research_topic,
        "literature_review_summary": literature_summary,
    }
    return {
        "agent": "agent_laboratory",
        "adapter_input": adapter_input,
        "model_requested": model,
        "model_used": model,
        "run_seed": run_seed,
        "stages": stages,
        "final_output": {
            "plan": final_plan,
        },
        "status": "ok" if final_plan else "incomplete",
    }


_FLAT_LLM_SYSTEM = (
    "You are an experienced ML researcher. "
    "Given a research topic and a set of relevant papers, propose exactly one novel, "
    "feasible research idea. "
    "Output requirements: emit ONLY a single JSON object that matches the schema "
    "given by the user. No prose before or after, no markdown code fences, no "
    "thinking out loud, no <think> blocks. Begin your answer with '{' and end with '}'."
)

_FLAT_LLM_SYSTEM_STRICT = (
    "You are an experienced ML researcher. "
    "Output ONLY a single JSON object matching the schema. "
    "Do not include <think> blocks, planning, preamble, markdown fences, or "
    "commentary. Begin with '{' and end with '}'."
)

_FLAT_LLM_IDEA_SCHEMA = """{
  "Name": "<short snake_case identifier>",
  "Title": "<full paper title>",
  "Short Hypothesis": "<one sentence core claim>",
  "Related Work": "<how this differs from the provided papers>",
  "Abstract": "<~150 word abstract>",
  "Experiments": "<key experiments needed to validate the idea>",
  "Risk Factors and Limitations": "<main risks or limitations>"
}"""


_UNIFIED_MAX_TOKENS_DEFAULT = 16384


def _gen_max_tokens(default: int = _UNIFIED_MAX_TOKENS_DEFAULT) -> int:
    """Unified generation max_tokens for every agent path.

    Reads the env var UNIFIED_MAX_TOKENS (or FLAT_LLM_MAX_TOKENS as a
    fallback) so callers can shrink the budget without editing code.
    """
    raw = os.getenv("UNIFIED_MAX_TOKENS") or os.getenv("FLAT_LLM_MAX_TOKENS")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(256, value)


def _flat_llm_max_tokens(default: int = _UNIFIED_MAX_TOKENS_DEFAULT) -> int:
    """Backwards-compatible alias kept in case external callers import it."""
    return _gen_max_tokens(default)


def _safe_generate(
    convs: List[List[Dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
    seed: Optional[int],
    safety_margin: int = 128,
) -> List[str]:
    """Truncate over-long prompts client-side and isolate per-request errors.

    Wraps ``get_generator().generate(...)`` so a single oversized or
    tokenizer-rejected prompt no longer kills the whole Python process.

    - Each conversation is run through ``generator.truncate_messages`` first
      so that ``prompt_tokens + max_tokens + safety_margin <= max_model_len``.
      If a conversation needed to be shortened, a one-line warning is printed
      identifying which slot was affected (the original list order is
      preserved so the caller can still zip results back).
    - The underlying call uses ``on_error="empty"``: any per-request failure
      that survives truncation (or any engine-level failure that the worker
      could not isolate) yields an empty string for that slot rather than
      raising. The caller's existing parse-error / empty-response handling
      then records the failure in the run log without crashing the sweep.
    """
    generator = get_generator()
    safe_convs: List[List[Dict[str, str]]] = []
    for idx, messages in enumerate(convs):
        try:
            new_messages, was_truncated = generator.truncate_messages(
                messages,
                max_tokens=max_tokens,
                safety_margin=safety_margin,
            )
        except BaseException as exc:
            print(
                f"[unified_ideation] _safe_generate: truncation check failed for conv #{idx} "
                f"({type(exc).__name__}: {exc}); passing through unchanged.",
                flush=True,
            )
            safe_convs.append(messages)
            continue
        if was_truncated:
            print(
                f"[unified_ideation] _safe_generate: truncated conv #{idx} to fit "
                f"max_model_len={generator.max_model_len} (max_tokens={max_tokens}).",
                flush=True,
            )
        safe_convs.append(new_messages)
    return generator.generate(
        safe_convs,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        on_error="empty",
    )


def _flat_llm_retry_user_prompt(original_user_prompt: str) -> str:
    return (
        f"{original_user_prompt}\n\n"
        "---\n"
        "/no_think\n"
        "Your previous reply could not be parsed as JSON. "
        "Reply now with ONLY the JSON object using the schema above. "
        "Do not include any thinking, preamble, markdown fences, or commentary. "
        "Start your reply with '{' and end with '}'."
    )


def run_flat_llm(
    state: Dict[str, Any],
    model: str,
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Zero-shot baseline: single LLM call with topic + papers, no agent framework.

    Uses the same shared vLLM batcher as other agents so all flat_llm tasks
    submitted in parallel via run_grid_experiments are processed in one batch.
    """
    context = build_workshop_markdown(state)
    user_prompt = (
        f"{context}\n"
        "---\n"
        "Propose ONE novel research idea grounded in the literature above.\n"
        f"Reply with a single JSON object matching this schema:\n{_FLAT_LLM_IDEA_SCHEMA}"
    )
    messages = [
        {"role": "system", "content": _FLAT_LLM_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    max_tokens = _gen_max_tokens()

    raw = run_vllm_offline_chat_completion(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        seed=run_seed,
    )
    parsed, parse_error, parse_method = parse_flat_llm_response(raw)
    stages: List[Dict[str, Any]] = [
        {"stage": "flat_generation", "messages": messages, "response": raw}
    ]

    retry_messages: Optional[List[Dict[str, str]]] = None
    retry_raw: Optional[str] = None
    if parsed is None:
        retry_messages = [
            {"role": "system", "content": _FLAT_LLM_SYSTEM_STRICT},
            {"role": "user", "content": _flat_llm_retry_user_prompt(user_prompt)},
        ]
        retry_raw = run_vllm_offline_chat_completion(
            messages=retry_messages,
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            seed=run_seed,
        )
        retry_parsed, retry_err, retry_method = parse_flat_llm_response(retry_raw)
        stages.append(
            {"stage": "flat_generation_retry", "messages": retry_messages, "response": retry_raw}
        )
        if retry_parsed is not None:
            parsed = retry_parsed
            parse_error = None
            parse_method = (retry_method or "json_balanced") + "_retry"
        else:
            parse_error = retry_err or parse_error

    final_output: Dict[str, Any]
    if parsed is not None:
        final_output = dict(parsed)
        final_output["_parse_method"] = parse_method
    else:
        final_output = {"raw": raw, "raw_retry": retry_raw} if retry_raw is not None else {"raw": raw}

    return {
        "model_used": model,
        "run_seed": run_seed,
        "adapter_input": context,
        "stages": stages,
        "final_output": final_output,
        "status": "ok" if parsed is not None else "parse_error",
        "parse_error": parse_error,
        "parse_method": parse_method,
    }


def run_single_agent(
    agent: str,
    state: Dict[str, Any],
    model: str,
    reflections: int,
    iterations: int,
    max_steps: int,
    run_seed: Optional[int],
) -> Dict[str, Any]:
    normalized = agent.strip().lower()
    if normalized in {"ai_scientist_v2", "ai-scientist-v2", "aiscientistv2"}:
        return run_ai_scientist_v2(state=state, model=model, num_reflections=reflections, run_seed=run_seed)
    if normalized in {"research_agent", "researchagent"}:
        return run_research_agent(state=state, model=model, iterations=iterations, run_seed=run_seed)
    if normalized in {"agent_laboratory", "agentlaboratory"}:
        return run_agent_laboratory(state=state, model=model, max_steps=max_steps, run_seed=run_seed)
    if normalized in {"flat_llm", "flat-llm", "flatllm"}:
        return run_flat_llm(state=state, model=model, run_seed=run_seed)
    raise ValueError(f"Unsupported agent '{agent}'")


def build_log_record(
    agent: str,
    canonical_state_path: Path,
    state: Dict[str, Any],
    result: Dict[str, Any],
    model: str,
    run_seed: Optional[int],
    rq1a_memory_resample: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "schema_version": "0.1",
        "timestamp_utc": utc_now_iso(),
        "agent": agent,
        "inference_backend": get_inference_backend(),
        "canonical_state_path": to_relative_path(canonical_state_path),
        "task_id": state.get("task_id"),
        "model_requested": model,
        "model_used": result.get("model_used", model),
        "run_seed": run_seed,
        "status": result.get("status"),
        "canonical_state": state,
        "adapter_input": result.get("adapter_input"),
        "stages": result.get("stages", []),
        "final_output": result.get("final_output"),
    }
    if rq1a_memory_resample is not None:
        record["rq1a_memory_resample"] = rq1a_memory_resample
    if result.get("parse_error") is not None:
        record["parse_error"] = result["parse_error"]
    return record


def build_output_path(
    task_id: str,
    agent: str,
    output_dir: Path,
    run_seed: Optional[int] = None,
    memory_resample_seed: Optional[int] = None,
) -> Path:
    run_slug = slugify(task_id or "task")
    agent_slug = slugify(agent)
    mem_suffix = f"__mem_{memory_resample_seed}" if memory_resample_seed is not None else ""
    seed_suffix = f"__seed_{run_seed}" if run_seed is not None else ""
    return output_dir / f"{run_slug}__{agent_slug}{mem_suffix}{seed_suffix}.json"


def _prepare_task(
    task: Dict[str, Any],
    model: Optional[str],
    memory_corpus_path: Optional[Path],
    memory_min_context_size: int,
    memory_paper_type: str,
    memory_corpus_years: Optional[Sequence[int]],
    memory_pool_multiplier: int,
) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]], Path]:
    """Load and validate state for one task; apply memory resampling if needed.

    Returns (state, resolved_model, rq1a_meta, output_path).
    """
    load_env_file(ROOT / ".env")
    requested_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    state = read_cached_json(task["canonical_state_path"])
    validate_canonical_state(state)
    rq1a_meta: Optional[Dict[str, Any]] = None
    if task.get("memory_resample_seed") is not None:
        corpus_p = memory_corpus_path or DEFAULT_MEMORY_CORPUS
        if not corpus_p.exists():
            raise RuntimeError(f"Memory resample corpus not found: {corpus_p}")
        state, rq1a_meta = apply_rq1a_memory_resample(
            state,
            corpus_p,
            task["memory_resample_seed"],
            min_context_size=memory_min_context_size,
            paper_type=memory_paper_type,
            corpus_years=memory_corpus_years,
            pool_multiplier=memory_pool_multiplier,
        )
    ensure_directory(task["output_dir"])
    output_path = build_output_path(
        task_id=state.get("task_id", "task"),
        agent=task["agent"],
        output_dir=task["output_dir"],
        run_seed=task.get("run_seed"),
        memory_resample_seed=task.get("memory_resample_seed"),
    )
    return state, normalize_openai_model(requested_model), rq1a_meta, output_path


# ---------------------------------------------------------------------------
# Batch runners — one per agent type.  Each collects all tasks of its type,
# builds prompts upfront, and calls LLMGenerator.generate() once per stage so
# the full vLLM batch is maximally dense.
# ---------------------------------------------------------------------------

_BATCH_AGENTS = frozenset({"flat_llm", "ai_scientist_v2", "research_agent", "agent_laboratory"})


def _batch_run_flat_llm(
    tasks: List[Dict[str, Any]],
    model: Optional[str],
    memory_corpus_path: Optional[Path],
    memory_min_context_size: int,
    memory_paper_type: str,
    memory_corpus_years: Optional[Sequence[int]],
    memory_pool_multiplier: int,
    skip_existing: bool,
) -> List[Tuple[Dict[str, Any], Path]]:
    """Process all flat_llm tasks in a single GPU forward pass."""
    prepared: List[Tuple[Dict[str, Any], Dict[str, Any], str, Optional[Dict[str, Any]], Path]] = []
    skipped: List[Tuple[Dict[str, Any], Path]] = []
    for task in tasks:
        try:
            state, resolved_model, rq1a_meta, output_path = _prepare_task(
                task, model, memory_corpus_path, memory_min_context_size,
                memory_paper_type, memory_corpus_years, memory_pool_multiplier,
            )
        except Exception:
            print(
                f"[error] flat_llm _prepare_task failed for {task.get('task_id')}\n"
                f"{traceback.format_exc()}",
                file=sys.stderr, flush=True,
            )
            continue
        if skip_existing and output_path.exists():
            skipped.append((dict(task, skipped=True), output_path))
            continue
        prepared.append((task, state, resolved_model, rq1a_meta, output_path))

    results: List[Tuple[Dict[str, Any], Path]] = list(skipped)
    if not prepared:
        return results

    convs: List[List[Dict[str, str]]] = []
    user_prompts: List[str] = []
    contexts: List[str] = []
    for _task, state, _rm, _meta, _op in prepared:
        context = build_workshop_markdown(state)
        user_prompt = (
            f"{context}"
            "---\n"
            "Propose ONE novel research idea grounded in the literature above.\n"
            f"Reply with a single JSON object matching this schema:\n{_FLAT_LLM_IDEA_SCHEMA}"
        )
        contexts.append(context)
        user_prompts.append(user_prompt)
        convs.append([
            {"role": "system", "content": _FLAT_LLM_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])

    max_tokens = _gen_max_tokens()

    # Group by run_seed so each seed gets its own vLLM call (reproducibility).
    from collections import defaultdict as _defaultdict
    seed_to_idxs: Dict[Any, List[int]] = _defaultdict(list)
    for i, (task, *_) in enumerate(prepared):
        seed_to_idxs[task.get("run_seed")].append(i)
    raw_responses: List[str] = [""] * len(prepared)
    for seed_val, idxs in seed_to_idxs.items():
        batch_convs = [convs[i] for i in idxs]
        batch_resp = _safe_generate(
            batch_convs, temperature=0.7, max_tokens=max_tokens, seed=seed_val,
        )
        for i, resp in zip(idxs, batch_resp):
            raw_responses[i] = resp

    parsed_list: List[Optional[Dict[str, Any]]] = [None] * len(prepared)
    parse_methods: List[Optional[str]] = [None] * len(prepared)
    parse_errors: List[Optional[str]] = [None] * len(prepared)
    for i, raw in enumerate(raw_responses):
        parsed, perr, pmethod = parse_flat_llm_response(raw)
        parsed_list[i] = parsed
        parse_methods[i] = pmethod
        parse_errors[i] = perr

    retry_indices = [i for i, p in enumerate(parsed_list) if p is None]
    retry_responses: Dict[int, str] = {}
    retry_convs_by_idx: Dict[int, List[Dict[str, str]]] = {}
    if retry_indices:
        retry_seed_to_idxs: Dict[Any, List[int]] = _defaultdict(list)
        for i in retry_indices:
            retry_seed_to_idxs[prepared[i][0].get("run_seed")].append(i)
            retry_convs_by_idx[i] = [
                {"role": "system", "content": _FLAT_LLM_SYSTEM_STRICT},
                {"role": "user", "content": _flat_llm_retry_user_prompt(user_prompts[i])},
            ]
        for seed_val, idxs in retry_seed_to_idxs.items():
            batch_convs = [retry_convs_by_idx[i] for i in idxs]
            batch_resp = _safe_generate(
                batch_convs, temperature=0.3, max_tokens=max_tokens, seed=seed_val,
            )
            for i, resp in zip(idxs, batch_resp):
                retry_responses[i] = resp
                parsed, perr, pmethod = parse_flat_llm_response(resp)
                if parsed is not None:
                    parsed_list[i] = parsed
                    parse_errors[i] = None
                    parse_methods[i] = (pmethod or "json_balanced") + "_retry"
                else:
                    parse_errors[i] = perr or parse_errors[i]

    total = len(prepared)
    for idx, (task, state, resolved_model, rq1a_meta, output_path) in enumerate(prepared, start=1):
        i = idx - 1
        raw = raw_responses[i]
        parsed = parsed_list[i]
        parse_error = parse_errors[i]
        parse_method = parse_methods[i]

        stages = [{"stage": "flat_generation", "messages": convs[i], "response": raw}]
        if i in retry_responses:
            stages.append(
                {
                    "stage": "flat_generation_retry",
                    "messages": retry_convs_by_idx[i],
                    "response": retry_responses[i],
                }
            )

        if parsed is not None:
            final_output: Dict[str, Any] = dict(parsed)
            final_output["_parse_method"] = parse_method
        elif i in retry_responses:
            final_output = {"raw": raw, "raw_retry": retry_responses[i]}
        else:
            final_output = {"raw": raw}

        result = {
            "agent": "flat_llm",
            "model_requested": resolved_model,
            "model_used": resolved_model,
            "run_seed": task.get("run_seed"),
            "adapter_input": {"workshop_markdown": contexts[i]},
            "stages": stages,
            "final_output": final_output,
            "status": "ok" if parsed is not None else "parse_error",
            "parse_error": parse_error,
            "parse_method": parse_method,
        }
        log_record = build_log_record(
            agent="flat_llm",
            canonical_state_path=task["canonical_state_path"],
            state=state,
            result=result,
            model=resolved_model,
            run_seed=task.get("run_seed"),
            rq1a_memory_resample=rq1a_meta,
        )
        log_record["output_path"] = to_relative_path(output_path)
        log_record["memory_resample_seed"] = task.get("memory_resample_seed")
        if parse_method is not None:
            log_record["parse_method"] = parse_method
        write_json(output_path, log_record)
        retry_tag = " (retried)" if i in retry_responses and parsed is not None else (
            " (retry-failed)" if i in retry_responses else ""
        )
        print(
            f"[progress {idx}/{total}] flat_llm task={task.get('task_id')} "
            f"seed={task.get('run_seed')} status={result['status']}{retry_tag} -> {output_path}",
            flush=True,
        )
        results.append((task, output_path))

    return results


def _batch_run_ai_scientist_v2(
    tasks: List[Dict[str, Any]],
    model: Optional[str],
    reflections: int,
    memory_corpus_path: Optional[Path],
    memory_min_context_size: int,
    memory_paper_type: str,
    memory_corpus_years: Optional[Sequence[int]],
    memory_pool_multiplier: int,
    skip_existing: bool,
) -> List[Tuple[Dict[str, Any], Path]]:
    """Run AI-Scientist-v2 tasks through the bundled upstream implementation."""
    require_external_module(
        EXTERNAL_DIR / "AI-Scientist-v2",
        "ai_scientist.perform_ideation_temp_free",
        "AI-Scientist-v2",
    )
    results: List[Tuple[Dict[str, Any], Path]] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        path = run_and_log(
            agent="ai_scientist_v2",
            canonical_state_path=task["canonical_state_path"],
            output_dir=task["output_dir"],
            model=model,
            reflections=reflections,
            run_seed=task.get("run_seed"),
            memory_resample_seed=task.get("memory_resample_seed"),
            memory_corpus_path=memory_corpus_path,
            memory_min_context_size=memory_min_context_size,
            memory_paper_type=memory_paper_type,
            memory_corpus_years=memory_corpus_years,
            memory_pool_multiplier=memory_pool_multiplier,
            skip_existing=skip_existing,
        )
        if path:
            print(
                f"[progress {idx}/{total}] ai_scientist_v2 task={task.get('task_id')} "
                f"seed={task.get('run_seed')} -> {path}",
                flush=True,
            )
            results.append((task, path))
    return results


def _batch_run_research_agent(
    tasks: List[Dict[str, Any]],
    model: Optional[str],
    iterations: int,
    memory_corpus_path: Optional[Path],
    memory_min_context_size: int,
    memory_paper_type: str,
    memory_corpus_years: Optional[Sequence[int]],
    memory_pool_multiplier: int,
    skip_existing: bool,
) -> List[Tuple[Dict[str, Any], Path]]:
    """Run ResearchAgent tasks through the bundled upstream implementation."""
    repo_root = EXTERNAL_DIR / "ResearchAgent" / "code"
    if not repo_root.exists():
        raise RuntimeError(
            f"ResearchAgent source tree is required but was not found at {repo_root}."
        )
    results: List[Tuple[Dict[str, Any], Path]] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        path = run_and_log(
            agent="research_agent",
            canonical_state_path=task["canonical_state_path"],
            output_dir=task["output_dir"],
            model=model,
            iterations=iterations,
            run_seed=task.get("run_seed"),
            memory_resample_seed=task.get("memory_resample_seed"),
            memory_corpus_path=memory_corpus_path,
            memory_min_context_size=memory_min_context_size,
            memory_paper_type=memory_paper_type,
            memory_corpus_years=memory_corpus_years,
            memory_pool_multiplier=memory_pool_multiplier,
            skip_existing=skip_existing,
        )
        if path:
            print(
                f"[progress {idx}/{total}] research_agent task={task.get('task_id')} "
                f"seed={task.get('run_seed')} -> {path}",
                flush=True,
            )
            results.append((task, path))
    return results


def _batch_run_agent_laboratory(
    tasks: List[Dict[str, Any]],
    model: Optional[str],
    max_steps: int,
    memory_corpus_path: Optional[Path],
    memory_min_context_size: int,
    memory_paper_type: str,
    memory_corpus_years: Optional[Sequence[int]],
    memory_pool_multiplier: int,
    skip_existing: bool,
) -> List[Tuple[Dict[str, Any], Path]]:
    """Run AgentLaboratory tasks through the bundled upstream implementation."""
    repo_root = EXTERNAL_DIR / "AgentLaboratory"
    if not repo_root.exists():
        raise RuntimeError(
            f"AgentLaboratory source tree is required but was not found at {repo_root}."
        )
    results: List[Tuple[Dict[str, Any], Path]] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        path = run_and_log(
            agent="agent_laboratory",
            canonical_state_path=task["canonical_state_path"],
            output_dir=task["output_dir"],
            model=model,
            max_steps=max_steps,
            run_seed=task.get("run_seed"),
            memory_resample_seed=task.get("memory_resample_seed"),
            memory_corpus_path=memory_corpus_path,
            memory_min_context_size=memory_min_context_size,
            memory_paper_type=memory_paper_type,
            memory_corpus_years=memory_corpus_years,
            memory_pool_multiplier=memory_pool_multiplier,
            skip_existing=skip_existing,
        )
        if path:
            print(
                f"[progress {idx}/{total}] agent_laboratory task={task.get('task_id')} "
                f"seed={task.get('run_seed')} -> {path}",
                flush=True,
            )
            results.append((task, path))
    return results


def run_and_log(
    agent: str,
    canonical_state_path: Path,
    output_dir: Path,
    model: Optional[str] = None,
    reflections: int = 5,
    iterations: int = 2,
    max_steps: int = 8,
    run_seed: Optional[int] = None,
    memory_resample_seed: Optional[int] = None,
    memory_corpus_path: Optional[Path] = None,
    memory_min_context_size: int = 5,
    memory_paper_type: str = "all",
    memory_corpus_years: Optional[Sequence[int]] = None,
    memory_pool_multiplier: int = 8,
    skip_existing: bool = False,
) -> Optional[Path]:
    load_env_file(ROOT / ".env")
    requested_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    rq1a_meta: Optional[Dict[str, Any]] = None
    state = read_cached_json(canonical_state_path)
    validate_canonical_state(state)
    if memory_resample_seed is not None:
        corpus_p = memory_corpus_path or DEFAULT_MEMORY_CORPUS
        if not corpus_p.exists():
            raise RuntimeError(f"Memory resample corpus not found: {corpus_p}")
        state, rq1a_meta = apply_rq1a_memory_resample(
            state,
            corpus_p,
            memory_resample_seed,
            min_context_size=memory_min_context_size,
            paper_type=memory_paper_type,
            corpus_years=memory_corpus_years,
            pool_multiplier=memory_pool_multiplier,
        )

    ensure_directory(output_dir)
    output_path = build_output_path(
        task_id=state.get("task_id", "task"),
        agent=agent,
        output_dir=output_dir,
        run_seed=run_seed,
        memory_resample_seed=memory_resample_seed,
    )
    if skip_existing and output_path.exists():
        return output_path

    try:
        result = run_single_agent(
            agent=agent,
            state=state,
            model=normalize_openai_model(requested_model),
            reflections=reflections,
            iterations=iterations,
            max_steps=max_steps,
            run_seed=run_seed,
        )
        log_record = build_log_record(
            agent=agent,
            canonical_state_path=canonical_state_path,
            state=state,
            result=result,
            model=requested_model,
            run_seed=run_seed,
            rq1a_memory_resample=rq1a_meta,
        )
        log_record["output_path"] = to_relative_path(output_path)
        log_record["memory_resample_seed"] = memory_resample_seed
    except Exception as exc:
        task_id = state.get("task_id") if "state" in locals() else None
        print(
            f"[error] agent={agent} task_id={task_id} run_seed={run_seed} "
            f"memory_resample_seed={memory_resample_seed} "
            f"canonical_state_path={to_relative_path(canonical_state_path)}\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        return None

    write_json(output_path, log_record)
    return output_path


def read_experiment_grid(grid_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with grid_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_grid_experiments(
    grid_paths: Sequence[Path],
    output_root: Path,
    agents: Sequence[str],
    model: Optional[str] = None,
    reflections: int = 5,
    iterations: int = 2,
    max_steps: int = 8,
    task_parallelism: int = 64,
    memory_resample_seeds: Optional[Sequence[int]] = None,
    rq1a_baseline_memory: bool = True,
    memory_corpus_path: Optional[Path] = None,
    memory_min_context_size: int = 5,
    memory_paper_type: str = "all",
    memory_corpus_years: Optional[Sequence[int]] = None,
    memory_pool_multiplier: int = 8,
    skip_existing: bool = False,
) -> List[Tuple[Dict[str, Any], Path]]:
    tasks: List[Dict[str, Any]] = []
    resolved_output_root = output_root.resolve()
    model_output_root = resolved_output_root / model_dirname(model)
    resolved_memory_corpus = memory_corpus_path.resolve() if memory_corpus_path is not None else None
    resample_seeds = list(memory_resample_seeds or [])

    for grid_path in grid_paths:
        for row in read_experiment_grid(grid_path):
            canonical_state_path = Path(row["canonical_state_path"])
            if not canonical_state_path.is_absolute():
                canonical_state_path = (ROOT / canonical_state_path).resolve()
            year = str(row["year"])
            for agent in agents:
                agent_output_dir = model_output_root / year / agent
                if not resample_seeds:
                    tasks.append(
                        {
                            "agent": agent,
                            "task_id": row["task_id"],
                            "canonical_state_path": canonical_state_path,
                            "output_dir": agent_output_dir,
                            "run_seed": row["run_seed"],
                            "memory_resample_seed": None,
                        }
                    )
                    continue

                if rq1a_baseline_memory:
                    tasks.append(
                        {
                            "agent": agent,
                            "task_id": row["task_id"],
                            "canonical_state_path": canonical_state_path,
                            "output_dir": agent_output_dir,
                            "run_seed": row["run_seed"],
                            "memory_resample_seed": None,
                        }
                    )
                for mem_seed in resample_seeds:
                    tasks.append(
                        {
                            "agent": agent,
                            "task_id": row["task_id"],
                            "canonical_state_path": canonical_state_path,
                            "output_dir": agent_output_dir,
                            "run_seed": row["run_seed"],
                            "memory_resample_seed": mem_seed,
                        }
                    )

    results: List[Tuple[Dict[str, Any], Path]] = []
    skipped_results: List[Tuple[Dict[str, Any], Path]] = []
    runnable_tasks: List[Dict[str, Any]] = []
    for task in tasks:
        output_path = build_output_path(
            task_id=task["task_id"],
            agent=task["agent"],
            output_dir=task["output_dir"],
            run_seed=task["run_seed"],
            memory_resample_seed=task["memory_resample_seed"],
        )
        if skip_existing and output_path.exists():
            skipped_task = dict(task)
            skipped_task["skipped"] = True
            skipped_results.append((skipped_task, output_path))
        else:
            runnable_task = dict(task)
            runnable_task["skipped"] = False
            runnable_tasks.append(runnable_task)

    results.extend(skipped_results)
    if not runnable_tasks:
        print(
            f"[summary] succeeded=0 failed=0 skipped={len(skipped_results)} total={len(tasks)}",
            flush=True,
        )
        return results

    # Warm up the vLLM engine once before dispatching any tasks.
    if get_inference_backend() == "vllm_offline":
        warmup_vllm_offline_generator()

    # Shared kwargs passed to every batch runner.
    _batch_kwargs: Dict[str, Any] = dict(
        model=model,
        memory_corpus_path=resolved_memory_corpus,
        memory_min_context_size=memory_min_context_size,
        memory_paper_type=memory_paper_type,
        memory_corpus_years=memory_corpus_years,
        memory_pool_multiplier=memory_pool_multiplier,
        skip_existing=skip_existing,
    )

    # Group tasks by agent so each agent's batch runner receives its full set.
    tasks_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for task in runnable_tasks:
        tasks_by_agent.setdefault(task["agent"], []).append(task)

    batch_agent_results: List[Tuple[Dict[str, Any], Path]] = []
    unknown_tasks: List[Dict[str, Any]] = []

    for agent_name, agent_tasks in tasks_by_agent.items():
        if agent_name == "flat_llm":
            batch_agent_results.extend(
                _batch_run_flat_llm(agent_tasks, **_batch_kwargs)
            )
        elif agent_name in {"ai_scientist_v2", "ai-scientist-v2", "aiscientistv2"}:
            batch_agent_results.extend(
                _batch_run_ai_scientist_v2(agent_tasks, reflections=reflections, **_batch_kwargs)
            )
        elif agent_name in {"research_agent", "researchagent"}:
            batch_agent_results.extend(
                _batch_run_research_agent(agent_tasks, iterations=iterations, **_batch_kwargs)
            )
        elif agent_name in {"agent_laboratory", "agentlaboratory"}:
            batch_agent_results.extend(
                _batch_run_agent_laboratory(agent_tasks, max_steps=max_steps, **_batch_kwargs)
            )
        else:
            unknown_tasks.extend(agent_tasks)

    results.extend(batch_agent_results)
    completed = len(batch_agent_results)
    failed = 0

    # Unknown agents fall back to the original thread-pool path.
    if unknown_tasks:
        total_unknown = len(unknown_tasks)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=task_parallelism)
        try:
            future_to_task = {
                executor.submit(
                    run_and_log,
                    agent=task["agent"],
                    canonical_state_path=task["canonical_state_path"],
                    output_dir=task["output_dir"],
                    model=model,
                    reflections=reflections,
                    iterations=iterations,
                    max_steps=max_steps,
                    run_seed=task["run_seed"],
                    memory_resample_seed=task["memory_resample_seed"],
                    memory_corpus_path=resolved_memory_corpus,
                    memory_min_context_size=memory_min_context_size,
                    memory_paper_type=memory_paper_type,
                    memory_corpus_years=memory_corpus_years,
                    memory_pool_multiplier=memory_pool_multiplier,
                    skip_existing=skip_existing,
                ): task
                for task in unknown_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                mem_seed = task["memory_resample_seed"]
                variant = "canonical" if mem_seed is None else f"mem_{mem_seed}"
                try:
                    output_path = future.result()
                except Exception:
                    failed += 1
                    print(
                        f"[progress {completed + failed}/{total_unknown}] "
                        f"{task['agent']} task={task['task_id']} seed={task['run_seed']} "
                        f"variant={variant} -> FAILED",
                        flush=True,
                    )
                    continue
                if output_path is None:
                    failed += 1
                    print(
                        f"[progress {completed + failed}/{total_unknown}] "
                        f"{task['agent']} task={task['task_id']} seed={task['run_seed']} "
                        f"variant={variant} -> FAILED",
                        flush=True,
                    )
                    continue
                results.append((task, output_path))
                completed += 1
                print(
                    f"[progress {completed}/{total_unknown}] "
                    f"{task['agent']} task={task['task_id']} seed={task['run_seed']} "
                    f"variant={variant} -> {output_path}",
                    flush=True,
                )
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    total_runnable = len(runnable_tasks)
    print(
        f"[summary] succeeded={completed} failed={failed} skipped={len(skipped_results)} total={len(tasks)}",
        flush=True,
    )
    return results
