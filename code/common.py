"""Shared helpers: fetch parquet data from the submission_data Hugging Face dataset, and the small vector/statistics routines used across the per-section analysis scripts."""

from __future__ import annotations

import contextlib
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import huggingface_hub  # noqa: F401  (registers the "hf://" fsspec protocol)
import numpy as np
import pandas as pd

HF_REPO = "yixuantt/submission_data"


def load_parquet(relative_path: str) -> pd.DataFrame:
    """Read one parquet file directly from the public HF dataset repo."""
    return pd.read_parquet(f"hf://datasets/{HF_REPO}/{relative_path}")


def results_dir() -> Path:
    """Where Code Ocean expects reproducible run outputs: /results if mounted (the
    platform's convention), else RESULTS_DIR if set, else ../results next to code/
    for local runs."""
    if Path("/results").is_dir():
        path = Path("/results")
    elif os.environ.get("RESULTS_DIR"):
        path = Path(os.environ["RESULTS_DIR"])
    else:
        path = Path(__file__).resolve().parent.parent / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def save_report(name: str):
    """Mirror everything printed inside the block to results/{name}.txt as well as stdout."""
    out_path = results_dir() / f"{name}.txt"
    with out_path.open("w", encoding="utf-8") as fh:
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, fh)
        try:
            yield
        finally:
            sys.stdout = original_stdout
    print(f"[wrote {out_path}]")


def stack_embeddings(df: pd.DataFrame, col: str = "embedding") -> np.ndarray:
    return np.vstack(df[col].to_numpy()).astype(np.float32)


def group_indices(rows: list[dict], field: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get(field) or ""].append(index)
    return dict(groups)


def mean_pairwise_cosine(vectors: np.ndarray) -> tuple[float, int]:
    """Exact mean cosine similarity over all unordered row pairs (embeddings are L2-normalized)."""
    n = len(vectors)
    if n < 2:
        return math.nan, 0
    summed = vectors.astype(np.float64).sum(axis=0)
    mean = float((summed @ summed - n) / (n * (n - 1)))
    return mean, n * (n - 1) // 2


def mean_cross_cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    if len(left) == 0 or len(right) == 0:
        return math.nan, 0
    mean = float(left.astype(np.float64).mean(axis=0) @ right.astype(np.float64).mean(axis=0))
    return mean, len(left) * len(right)


def cosine_distance_to_centroid(vectors: np.ndarray, centroid_rows: np.ndarray) -> np.ndarray:
    if len(centroid_rows) == 0:
        return np.asarray([], dtype=float)
    centroid = centroid_rows.astype(np.float64).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0:
        return np.full(len(vectors), np.nan, dtype=float)
    centroid = centroid / norm
    return 1.0 - vectors.astype(np.float64) @ centroid


def weighted_mean(values: list[float], weights: list[float]) -> float:
    values_a = np.asarray(values, dtype=float)
    weights_a = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_a) & (weights_a > 0)
    if not mask.any():
        return math.nan
    return float((values_a[mask] * weights_a[mask]).sum() / weights_a[mask].sum())


def group_balanced_mean(per_group_values: dict[str, float]) -> float:
    """Equal weight across groups (e.g. across agents or LLMs), matching Table 2-5's 'Pooled/All data' row."""
    values = [v for v in per_group_values.values() if np.isfinite(v)]
    return float(np.mean(values)) if values else math.nan
