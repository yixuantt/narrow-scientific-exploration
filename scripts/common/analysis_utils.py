#!/usr/bin/env python3
"""Shared helpers for the analysis scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def join_nonempty(parts: Sequence[Any]) -> Optional[str]:
    values = [safe_str(part) for part in parts if safe_str(part)]
    if not values:
        return None
    return "\n\n".join(values)


def extract_plan_prefix(plan: str) -> str:
    split_match = re.split(
        r"^\s*##\s*3\.?\s*Experimental\b.*$",
        plan,
        maxsplit=1,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return split_match[0].strip() if split_match else plan.strip()


def extract_markdown_section(plan: str, heading_number: int) -> Optional[str]:
    pattern = re.compile(
        rf"^\s*##\s*{heading_number}\.?\s+.*?(?=^\s*##\s*{heading_number + 1}\.?\s+|\Z)",
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(plan)
    return match.group(0).strip() if match else None


def extract_agent_text(agent: str, final_output: Any) -> Optional[str]:
    """Extract comparable idea text from each framework's output schema."""
    if agent in ("flat_llm", "ai_scientist_v2"):
        if not isinstance(final_output, dict):
            return None
        title = final_output.get("Title") or final_output.get("Name")
        return join_nonempty(
            [title, final_output.get("Short Hypothesis"), final_output.get("Abstract")]
        )

    if agent == "research_agent":
        if not isinstance(final_output, dict):
            return None
        return join_nonempty([final_output.get("problem"), final_output.get("method")])

    if agent == "agent_laboratory":
        if not isinstance(final_output, dict):
            return None
        plan = final_output.get("plan")
        if not isinstance(plan, str) or not plan.strip():
            return None
        prefix = extract_plan_prefix(plan)
        sec1 = extract_markdown_section(prefix, 1)
        sec2 = extract_markdown_section(prefix, 2)
        return join_nonempty([sec1, sec2]) or prefix or None

    return None


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denom


def last_token_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, seq_lengths]


def embed_texts(texts: Sequence[str], model_name: str, batch_size: int) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Embedding analysis requires `transformers` and `torch`.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress(f"loading embedding model {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress(f"encoding {len(texts)} texts in {total_batches} batches")
    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=2048, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            if "qwen" in model_name.lower():
                pooled = last_token_pool(hidden, encoded["attention_mask"])
            else:
                pooled = mean_pool(hidden, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.detach().cpu().numpy())
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            progress(f"encoded batch {batch_idx}/{total_batches}")
    return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 1), dtype=np.float32)


def load_corpus_index(corpus_path: Path) -> Dict[str, Dict[str, Any]]:
    by_paper_id: Dict[str, Dict[str, Any]] = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            paper_id = safe_str(obj.get("paper_id"))
            if paper_id:
                by_paper_id[paper_id] = obj
    return by_paper_id


def _anchor_context_map(canonical_states_root: Path, context_prefix: str = "") -> Dict[str, str]:
    anchor_map: Dict[str, str] = {}
    for json_file in canonical_states_root.glob("**/*.json"):
        if json_file.name in ("context_summary.jsonl", "experiment_grid.jsonl"):
            continue
        try:
            obj = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        src = obj.get("source", {}) or {}
        context_id = safe_str(src.get("context_id"))
        if context_prefix and not context_id.startswith(context_prefix):
            continue
        anchor_id = safe_str(src.get("anchor_paper_id") or src.get("paper_id"))
        if anchor_id and context_id:
            anchor_map[anchor_id] = context_id
    return anchor_map


def _load_neighbors_from_edges(path: Path) -> Dict[str, List[str]]:
    neighbors: Dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                edge = json.loads(line)
            except Exception:
                continue
            src = safe_str(edge.get("source") or edge.get("source_paper_id"))
            dst = safe_str(edge.get("target") or edge.get("target_paper_id"))
            if not src or not dst or src == dst:
                continue
            neighbors.setdefault(src, set()).add(dst)
            neighbors.setdefault(dst, set()).add(src)
    return {k: sorted(v) for k, v in neighbors.items()}


def build_future_real_rows(
    corpus_by_id: Dict[str, Dict[str, Any]],
    canonical_states_root: Path,
    analysis_year: str,
    max_text_chars: int,
    *,
    graph_edges_path: Optional[Path] = None,
    context_prefix: str = "",
) -> List[Dict[str, Any]]:
    """Return graph-neighbor papers after an anchor year for each canonical anchor."""
    anchor_map = _anchor_context_map(canonical_states_root, context_prefix=context_prefix)
    if graph_edges_path is not None and graph_edges_path.exists():
        neighbor_map = _load_neighbors_from_edges(graph_edges_path)
    else:
        raise FileNotFoundError("Pass --graph-edges-path.")

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for anchor_id, context_id in anchor_map.items():
        for neighbor_id in neighbor_map.get(anchor_id, []):
            key = (anchor_id, neighbor_id)
            if key in seen:
                continue
            seen.add(key)
            rec = corpus_by_id.get(neighbor_id)
            if not rec:
                continue
            try:
                paper_year = int(rec.get("year"))
                if paper_year <= int(analysis_year):
                    continue
            except (TypeError, ValueError):
                continue
            text = join_nonempty([rec.get("title"), rec.get("abstract")])
            if not text:
                continue
            rows.append(
                {
                    "paper_id": neighbor_id,
                    "anchor_paper_id": anchor_id,
                    "context_id": context_id,
                    "year": paper_year,
                    "text": normalize_space(text)[:max_text_chars],
                }
            )
    return rows
