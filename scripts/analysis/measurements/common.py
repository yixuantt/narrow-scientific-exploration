#!/usr/bin/env python3
"""Shared I/O, vector, grouping, and uncertainty helpers."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON list of objects in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def year_from_identifier(value: Any) -> int | None:
    match = re.search(r"(?:^|_)(20\d{2})(?:_|$)", safe_text(value))
    return int(match.group(1)) if match else None


def l2_normalize(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Expected a two-dimensional embedding array, got {out.shape}")
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def load_aligned_embeddings(
    embeddings_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    embeddings = l2_normalize(np.load(embeddings_path, mmap_mode="r"))
    rows = load_rows(metadata_path)
    if len(embeddings) != len(rows):
        raise ValueError(
            f"Embedding/metadata length mismatch: {len(embeddings)} != {len(rows)}"
        )
    return embeddings, rows


def deduplicate_aligned(
    embeddings: np.ndarray,
    rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    seen: set[tuple[str, ...]] = set()
    keep: list[int] = []
    output_rows: list[dict[str, Any]] = []
    dropped = 0
    for index, row in enumerate(rows):
        key = tuple(safe_text(row.get(field)) for field in key_fields)
        if not any(key):
            key = (f"__row_{index}",)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        keep.append(index)
        output_rows.append(dict(row))
    return embeddings[np.asarray(keep, dtype=np.int64)], output_rows, dropped


def group_indices(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[tuple[str, ...], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = tuple(safe_text(row.get(field)) for field in fields)
        groups[key].append(index)
    return dict(groups)


def mean_pairwise_cosine(vectors: np.ndarray) -> tuple[float, int]:
    """Return the exact mean cosine similarity over all unordered row pairs."""
    n = len(vectors)
    if n < 2:
        return math.nan, 0
    summed = np.asarray(vectors, dtype=np.float64).sum(axis=0)
    mean = float((summed @ summed - n) / (n * (n - 1)))
    return mean, n * (n - 1) // 2


def mean_cross_cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    if len(left) == 0 or len(right) == 0:
        return math.nan, 0
    mean = float(np.asarray(left, dtype=np.float64).mean(axis=0) @ np.asarray(right, dtype=np.float64).mean(axis=0))
    return mean, len(left) * len(right)


def cosine_distance_to_centroid(vectors: np.ndarray, centroid_rows: np.ndarray) -> np.ndarray:
    if len(centroid_rows) == 0:
        return np.asarray([], dtype=float)
    centroid = np.asarray(centroid_rows, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0:
        return np.full(len(vectors), np.nan, dtype=float)
    centroid /= norm
    return 1.0 - np.asarray(vectors, dtype=np.float64) @ centroid


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def normal_ci(values: Iterable[float], confidence: float = 0.95) -> list[float | None]:
    array = finite(values)
    if len(array) < 2:
        return [None, None]
    if confidence != 0.95:
        raise ValueError("Only 95% normal-approximation intervals are implemented")
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    mean = float(array.mean())
    return [mean - 1.96 * se, mean + 1.96 * se]


def bootstrap_ci(
    rows: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    repetitions: int,
    seed: int,
) -> list[float | None]:
    if len(rows) < 2 or repetitions <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    n = len(rows)
    for index in range(repetitions):
        sample = [rows[i] for i in rng.integers(0, n, size=n)]
        estimates[index] = statistic(sample)
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) == 0:
        return [None, None]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def weighted_mean(rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key)
        if value is None or weight is None or not np.isfinite(float(value)) or float(weight) <= 0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return numerator / denominator if denominator else math.nan


def summarize_values(values: Iterable[float]) -> dict[str, Any]:
    array = finite(values)
    if len(array) == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None]}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "ci95": normal_ci(array),
    }


def keyword_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        return set()
    output = set()
    for item in items:
        text = " ".join(safe_text(item).lower().split())
        if text:
            output.add(text)
    return output


def require_fields(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], source: str) -> None:
    if not rows:
        raise ValueError(f"No rows found in {source}")
    missing = [field for field in fields if not any(safe_text(row.get(field)) for row in rows)]
    if missing:
        raise ValueError(f"{source} does not contain required fields: {', '.join(missing)}")
