"""Shared helpers: fetch parquet data from the submission_data Hugging Face dataset, and the small vector/statistics routines used across the per-section analysis scripts."""

from __future__ import annotations

import math
from collections import defaultdict

import huggingface_hub  # noqa: F401  (registers the "hf://" fsspec protocol)
import numpy as np
import pandas as pd

HF_REPO = "yixuantt/submission_data"


def load_parquet(relative_path: str) -> pd.DataFrame:
    """Read one parquet file directly from the public HF dataset repo."""
    return pd.read_parquet(f"hf://datasets/{HF_REPO}/{relative_path}")


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
