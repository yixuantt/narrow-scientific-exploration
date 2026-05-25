#!/usr/bin/env python3
"""Render balanced task-level embedding projection panels for RQ2 follow-on work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import hashlib
import random


ROOT = Path(__file__).resolve().parents[3]

TASKS = [
    "iclr_2022_325_matched_all_master_g22_25_bcsvd__61a596635244ab9dcbdfe4d4",
    "internationa_2022_323_matched_all_master_g22_25_bcsvd__62a7fc635aee126c0ff5e307",
    "neurips_2022_358_matched_all_master_g22_25_bcsvd__62a013765aee126c0ff68cc0",
    "internationa_2022_367_matched_all_master_g22_25_bcsvd__6201df495aee126c0f64db98",
]

TASK_LABELS = {
    "iclr_2022_325_matched_all_master_g22_25_bcsvd__61a596635244ab9dcbdfe4d4": "Example A",
    "internationa_2022_323_matched_all_master_g22_25_bcsvd__62a7fc635aee126c0ff5e307": "Example B",
    "neurips_2022_358_matched_all_master_g22_25_bcsvd__62a013765aee126c0ff68cc0": "Example C",
    "internationa_2022_367_matched_all_master_g22_25_bcsvd__6201df495aee126c0f64db98": "Example D",
}

N_INPUT = 5
MIN_GENERATED_FOLLOW = 7
MAX_GENERATED_FOLLOW = 14
SAMPLE_OVERRIDES = {
    "neurips_2022_358_matched_all_master_g22_25_bcsvd__62a013765aee126c0ff68cc0": {
        "n_compare": 7,
        "ai_salt": "neurips_2022_358_matched_all_master_g22_25_bcsvd__62a013765aee126c0ff68cc0:ai:candidate7:304",
        "follow_salt": "neurips_2022_358_matched_all_master_g22_25_bcsvd__62a013765aee126c0ff68cc0:follow:candidate7:304",
    },
}
INPUT_COLOR = "#D6D6D6"
INPUT_MARKER_COLOR = "#005A9C"
AI_COLOR = "#A5BAD5"
FOLLOW_ON_COLOR = "#A89BB5"
EDGE = "#4D4D4D"
TEXT = "#2F2F2F"
SPINE = "#7A7A7A"
INPUT_MARKER = "x"
AI_MARKER = "s"
FOLLOW_ON_MARKER = "^"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return x / denom


def add_cov_ellipse(ax, points: np.ndarray, color: str, scale: float = 1.0) -> None:
    if len(points) < 3:
        return
    cov = np.cov(points, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * scale * np.sqrt(vals)
    ax.add_patch(
        Ellipse(
            xy=points.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=0.9,
            alpha=0.13,
            zorder=1,
        )
    )


def set_square_limits(ax, points: np.ndarray, pad_frac: float = 0.08) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    if span <= 0:
        span = 1.0
    half = span * (0.5 + pad_frac)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)


def stable_sample(items, n: int, salt: str):
    items = sorted(items)
    if len(items) <= n:
        return items
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16], 16)
    return sorted(random.Random(seed).sample(items, n))


def select_points(
    task_id: str,
    task_data: dict,
    follow_ids_by_task: dict[str, set[str]],
    idea_embs: np.ndarray,
    literature_follow_embs: np.ndarray,
    literature_id_to_idx: dict[str, int],
    follow_id_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ai_indices = {row["emb_idx"] for row in task_data["tasks"] if row["task_id"] == task_id}
    memory_ids = []
    for row in task_data["tasks"]:
        if row["task_id"] == task_id:
            memory_ids = [pid for pid in row.get("memory_papers", []) if pid in literature_id_to_idx]
            break
    follow_ids = [pid for pid in follow_ids_by_task[task_id] if pid in follow_id_to_idx]
    n_compare = min(len(ai_indices), len(follow_ids), MAX_GENERATED_FOLLOW)
    override = SAMPLE_OVERRIDES.get(task_id)
    if override is not None:
        n_compare = min(n_compare, override["n_compare"])

    if len(memory_ids) < N_INPUT or n_compare < MIN_GENERATED_FOLLOW:
        raise ValueError(f"{task_id} has too few points for balanced projection")

    ai_salt = override["ai_salt"] if override is not None else f"{task_id}:ai:size_matched:{n_compare}"
    follow_salt = override["follow_salt"] if override is not None else f"{task_id}:follow:size_matched:{n_compare}"
    ai_sample = stable_sample(ai_indices, n_compare, ai_salt)
    follow_sample = stable_sample(follow_ids, n_compare, follow_salt)
    memory_sample = memory_ids[:N_INPUT]

    return (
        literature_follow_embs[[literature_id_to_idx[pid] for pid in memory_sample]],
        idea_embs[ai_sample],
        literature_follow_embs[[follow_id_to_idx[pid] for pid in follow_sample]],
        n_compare,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output PDF path")
    parser.add_argument("--png-out", type=Path, default=None, help="Optional PNG output path")
    args = parser.parse_args()
    data_dir = args.data_dir

    idea_embs = normalize(np.load(data_dir / "idea_embeddings.npy").astype("float32"))
    literature_follow_embs = normalize(np.load(data_dir / "rq2_memory_future_kw_embeddings.npy").astype("float32"))
    literature_follow_meta = load_json(data_dir / "rq2_memory_future_kw_embeddings_meta.json")
    task_data = load_json(data_dir / "rq2_task_data.json")
    three_way = load_json(data_dir / "rq2_three_way_sims.json")

    literature_id_to_idx = {
        meta["paper_id"]: i for i, meta in enumerate(literature_follow_meta) if meta["type"] == "memory"
    }
    follow_id_to_idx = {
        meta["paper_id"]: i for i, meta in enumerate(literature_follow_meta) if meta["type"] == "future"
    }
    follow_ids_by_task: dict[str, set[str]] = {}
    for row in three_way:
        follow_ids_by_task.setdefault(row["task_id"], set()).add(row["future_paper_id"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(7.45, 1.9), constrained_layout=False)

    for panel_idx, (ax, task_id) in enumerate(zip(axes, TASKS)):
        literature, ai, follow_on, n_compare = select_points(
            task_id,
            task_data,
            follow_ids_by_task,
            idea_embs,
            literature_follow_embs,
            literature_id_to_idx,
            follow_id_to_idx,
        )
        coords = PCA(n_components=2, random_state=42).fit_transform(np.vstack([literature, ai, follow_on]))
        literature_coords = coords[: len(literature)]
        ai_coords = coords[len(literature) : len(literature) + len(ai)]
        follow_coords = coords[len(literature) + len(ai) :]

        add_cov_ellipse(ax, follow_coords, FOLLOW_ON_COLOR, scale=1.0)
        add_cov_ellipse(ax, ai_coords, AI_COLOR, scale=1.0)
        ax.scatter(
            ai_coords[:, 0],
            ai_coords[:, 1],
            s=24,
            color=AI_COLOR,
            edgecolor=EDGE,
            linewidth=0.38,
            alpha=0.92,
            marker=AI_MARKER,
            label="Generated ideas",
            zorder=4,
        )
        ax.scatter(
            follow_coords[:, 0],
            follow_coords[:, 1],
            s=28,
            color=FOLLOW_ON_COLOR,
            edgecolor=EDGE,
            linewidth=0.42,
            alpha=0.84,
            marker=FOLLOW_ON_MARKER,
            label="Follow-on papers",
            zorder=4,
        )
        ax.scatter(
            literature_coords[:, 0],
            literature_coords[:, 1],
            s=42,
            color=INPUT_MARKER_COLOR,
            linewidth=1.05,
            alpha=0.99,
            marker=INPUT_MARKER,
            label="Input literature",
            zorder=6,
        )
        ax.set_title(
            f"{chr(ord('a') + panel_idx)}  {TASK_LABELS[task_id]}",
            loc="left",
            color=TEXT,
            fontsize=8.5,
            fontweight="bold",
            pad=2.0,
        )
        set_square_limits(ax, np.vstack([literature_coords, ai_coords, follow_coords]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(SPINE)
            spine.set_linewidth(0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = ["Input literature", "Generated ideas", "Follow-on papers"]
    by_label = dict(zip(labels, handles))
    fig.legend(
        [by_label[label] for label in legend_order],
        legend_order,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        handletextpad=0.35,
        columnspacing=0.7,
    )
    fig.subplots_adjust(top=0.77, left=0.018, right=0.995, bottom=0.04, wspace=0.08)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.015, dpi=300)
    print(args.out)
    if args.png_out:
        args.png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png_out, bbox_inches="tight", pad_inches=0.015, dpi=300)
        print(args.png_out)


if __name__ == "__main__":
    main()
