#!/usr/bin/env python3
"""Rebuild the four-panel idea concentration figure used in the paper.

The figure combines:
  a) same-area vs different-area similarity by agent framework
  b) same-area vs different-area similarity by LLM
  c) same-area agent-framework similarity matrix
  d) same-area LLM similarity matrix

Inputs are precomputed, L2-normalized embedding arrays and aligned metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from matplotlib.patches import Patch, Rectangle


RANDOM_SEED = 42
AGENT_ORDER = ["flat_llm", "ai_scientist_v2", "research_agent", "agent_laboratory"]
MODEL_SIZE_MAP = {
    "qwen_qwen3_5_0_8b": 0.8,
    "meta_llama_llama_3_2_1b_instruct": 1,
    "qwen_qwen3_5_4b": 4,
    "meta_llama_llama_3_1_8b_instruct": 8,
    "google_gemma_4_31b_it": 31,
    "qwen_qwen3_6_35b_a3b": 35,
}

BLUE = "#86A6C8"
BLUE_LIGHT = "#E8F0F7"
HUMAN = "#B9AEC2"
HUMAN_LIGHT = "#EEEAF2"
HATCH_EDGE = "#6B6B6B"
GRID = "#E5E5E5"
STRIPE = "#8A8A8A"


def short_agent(name: str) -> str:
    return {
        "flat_llm": "Zero-shot",
        "ai_scientist_v2": "AI Scientist",
        "research_agent": "ResearchAgent",
        "agent_laboratory": "Agent Lab.",
    }.get(name, name)


def short_model(name: str) -> str:
    return {
        "qwen_qwen3_5_0_8b": "Qwen 0.8B",
        "meta_llama_llama_3_2_1b_instruct": "Llama 1B",
        "qwen_qwen3_5_4b": "Qwen 4B",
        "meta_llama_llama_3_1_8b_instruct": "Llama 8B",
        "google_gemma_4_31b_it": "Gemma 31B",
        "qwen_qwen3_6_35b_a3b": "Qwen 35B",
    }.get(name, name)


def model_size(name: str) -> float:
    return MODEL_SIZE_MAP.get(name, 0.0)


def load_embeddings(emb_path: Path, meta_path: Path):
    embs = np.load(emb_path)
    with meta_path.open() as f:
        meta = json.load(f)
    if len(embs) != len(meta):
        raise ValueError(f"Embedding/meta length mismatch: {emb_path} and {meta_path}")
    return embs, meta


def build_indices(meta):
    ctx_idx = defaultdict(list)
    agent_idx = defaultdict(list)
    model_idx = defaultdict(list)
    for i, row in enumerate(meta):
        ctx_idx[row["context_id"]].append(i)
        if "agent" in row:
            agent_idx[row["agent"]].append(i)
        if "model" in row:
            model_idx[row["model"]].append(i)
    return ctx_idx, agent_idx, model_idx


def pairwise_sims(embs, idxs):
    if len(idxs) < 2:
        return np.array([], dtype=float)
    sub = embs[idxs]
    sim = sub @ sub.T
    tri = np.triu_indices(len(idxs), k=1)
    return sim[tri]


def within_for_groups(embs, ctx_idx, group_idx):
    sims = []
    group_set = set(group_idx)
    for ctx_rows in ctx_idx.values():
        idxs = [i for i in ctx_rows if i in group_set]
        if len(idxs) >= 2:
            sims.append(pairwise_sims(embs, idxs))
    return np.concatenate(sims) if sims else np.array([], dtype=float)


def sampled_out_of_context(embs, meta, idxs, n_sample):
    idxs = list(idxs)
    if len(idxs) < 2:
        return np.array([], dtype=float)
    n_possible = len(idxs) * (len(idxs) - 1) // 2
    target = min(n_sample, n_possible)
    ctx_ids = [meta[i]["context_id"] for i in idxs]
    sims = []
    attempts = 0
    max_attempts = max(target * 20, 10_000)
    while len(sims) < target and attempts < max_attempts:
        a, b = random.sample(range(len(idxs)), 2)
        attempts += 1
        if ctx_ids[a] == ctx_ids[b]:
            continue
        sims.append(float(embs[idxs[a]] @ embs[idxs[b]]))
    return np.array(sims, dtype=float)


def human_stats(human_embs, human_meta, n_out_sample):
    h_ctx, _, _ = build_indices(human_meta)
    within = []
    for idxs in h_ctx.values():
        sims = pairwise_sims(human_embs, idxs)
        if len(sims):
            within.append(sims)
    all_idxs = list(range(len(human_meta)))
    return {
        "within": np.concatenate(within) if within else np.array([], dtype=float),
        "out": sampled_out_of_context(human_embs, human_meta, all_idxs, n_out_sample),
        "ctx_idx": h_ctx,
    }


def grouped_cross_matrix(embs, meta, ctx_idx, groups, key, human_embs, human_ctx):
    labels = [short_agent(g) if key == "agent" else short_model(g) for g in groups] + ["Human"]
    n = len(labels)
    sums = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=float)

    for ctx, ctx_rows in ctx_idx.items():
        by_group = {g: [i for i in ctx_rows if meta[i][key] == g] for g in groups}
        human_rows = human_ctx.get(ctx, [])
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                if j < i:
                    continue
                if i == j:
                    val = mean_pairwise_from_rows(embs, by_group[gi])
                else:
                    val = mean_pairwise_from_rows(embs, by_group[gi], embs, by_group[gj])
                if np.isfinite(val):
                    weight = len(by_group[gi]) * (len(by_group[gj]) if i != j else max(len(by_group[gi]) - 1, 0))
                    sums[i, j] += val * weight
                    counts[i, j] += weight
                    if i != j:
                        sums[j, i] += val * weight
                        counts[j, i] += weight
            if human_rows:
                val = mean_pairwise_from_rows(embs, by_group[gi], human_embs, human_rows)
                if np.isfinite(val):
                    weight = len(by_group[gi]) * len(human_rows)
                    sums[i, -1] += val * weight
                    sums[-1, i] += val * weight
                    counts[i, -1] += weight
                    counts[-1, i] += weight

        val = mean_pairwise_from_rows(human_embs, human_rows)
        if np.isfinite(val):
            weight = len(human_rows) * max(len(human_rows) - 1, 0)
            sums[-1, -1] += val * weight
            counts[-1, -1] += weight

    with np.errstate(invalid="ignore", divide="ignore"):
        mat = sums / counts
    return labels, mat


def mean_pairwise_from_rows(embs_a, idx_a, embs_b=None, idx_b=None):
    if embs_b is None:
        if len(idx_a) < 2:
            return np.nan
        sub = embs_a[idx_a]
        summed = sub.sum(axis=0)
        n = len(idx_a)
        return float((summed @ summed - n) / (n * (n - 1)))
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan
    return float(embs_a[idx_a].mean(axis=0) @ embs_b[idx_b].mean(axis=0))


def bar_values(embs, meta, ctx_idx, groups, group_idx, h_stats, n_out_sample):
    rows = []
    for group in groups:
        idxs = group_idx[group]
        within = within_for_groups(embs, ctx_idx, idxs)
        out = sampled_out_of_context(embs, meta, idxs, n_out_sample)
        rows.append((within.mean(), out.mean(), within.std(), out.std()))
    rows.append((
        h_stats["within"].mean(),
        h_stats["out"].mean(),
        h_stats["within"].std(),
        h_stats["out"].std(),
    ))
    return np.array(rows, dtype=float)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.0, width=0.6, length=2.5)


def draw_bars(ax, labels, values, panel_label, title):
    x = np.arange(len(labels))
    width = 0.33
    is_human = np.array([label == "Human" for label in labels])
    colors = np.where(is_human, HUMAN, BLUE)
    light_colors = np.where(is_human, HUMAN_LIGHT, BLUE_LIGHT)

    ax.bar(
        x - width / 2,
        values[:, 0],
        width,
        yerr=values[:, 2],
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=0.45,
        error_kw={"elinewidth": 0.55, "ecolor": "#6A6A6A", "capsize": 1.8},
    )
    diff_bars = ax.bar(
        x + width / 2,
        values[:, 1],
        width,
        yerr=values[:, 3],
        color=light_colors,
        edgecolor=STRIPE,
        linewidth=0.45,
        hatch="//",
        error_kw={"elinewidth": 0.55, "ecolor": "#6A6A6A", "capsize": 1.8},
    )
    # Hatch color follows the patch edge color in Matplotlib. Overlay a clean
    # dark border so the hatch can stay light without losing print definition.
    for rect in diff_bars:
        ax.add_patch(Rectangle(
            (rect.get_x(), 0),
            rect.get_width(),
            rect.get_height(),
            fill=False,
            edgecolor="#4A4A4A",
            linewidth=0.45,
            zorder=rect.get_zorder() + 0.3,
        ))
    ax.axhline(values[-1, 0], color="#6B8294", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_ylim(0.58, 0.94)
    ax.set_ylabel("Cosine similarity", fontsize=8.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.0)
    ax.set_title(f"{panel_label}  {title}", loc="left", fontsize=8.5, fontweight="bold")
    style_axis(ax)


def draw_heatmap(ax, labels, mat, panel_label, title, rotate=40):
    cmap = plt.get_cmap("Blues")
    norm = colors.Normalize(vmin=0.73, vmax=0.84)
    n = len(labels)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(val)), edgecolor="none"))
            color = "white" if val >= 0.80 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color, fontweight="bold")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=7.0)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_title(f"{panel_label}  {title}", loc="left", fontsize=8.5, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", type=Path, default=Path("analysis_out/keyword_extraction/idea_embeddings.npy"))
    p.add_argument("--meta", type=Path, default=Path("analysis_out/keyword_extraction/idea_embeddings_meta.json"))
    p.add_argument("--human-emb", type=Path, default=Path("analysis_out/keyword_extraction/human_paper_non_exp_no0000_embeddings.npy"))
    p.add_argument("--human-meta", type=Path, default=Path("analysis_out/keyword_extraction/human_paper_non_exp_no0000_embeddings_meta.json"))
    p.add_argument("--human-cross-emb", type=Path, default=Path("analysis_out/keyword_extraction/human_paper_all_non_noise_no0000_embeddings.npy"))
    p.add_argument("--human-cross-meta", type=Path, default=Path("analysis_out/keyword_extraction/human_paper_all_non_noise_no0000_embeddings_meta.json"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_idea_concentration.pdf"))
    p.add_argument("--png-out", type=Path, default=None)
    p.add_argument("--out-sample", type=int, default=100_000)
    p.add_argument("--layout", choices=("two-row", "one-row"), default="two-row")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "hatch.linewidth": 0.35,
    })

    embs, meta = load_embeddings(args.emb, args.meta)
    ctx_idx, agent_idx, model_idx = build_indices(meta)
    human_embs, human_meta = load_embeddings(args.human_emb, args.human_meta)
    h_stats = human_stats(human_embs, human_meta, args.out_sample)
    human_cross_embs, human_cross_meta = load_embeddings(args.human_cross_emb, args.human_cross_meta)
    human_cross_ctx, _, _ = build_indices(human_cross_meta)

    agents = [a for a in AGENT_ORDER if a in agent_idx]
    models = sorted(model_idx, key=model_size)
    agent_labels = [short_agent(a) for a in agents] + ["Human"]
    model_labels = [short_model(m) for m in models] + ["Human"]
    agent_bar = bar_values(embs, meta, ctx_idx, agents, agent_idx, h_stats, args.out_sample)
    model_bar = bar_values(embs, meta, ctx_idx, models, model_idx, h_stats, args.out_sample)
    hm_agent_labels, agent_mat = grouped_cross_matrix(
        embs, meta, ctx_idx, agents, "agent", human_cross_embs, human_cross_ctx
    )
    hm_model_labels, model_mat = grouped_cross_matrix(
        embs, meta, ctx_idx, models, "model", human_cross_embs, human_cross_ctx
    )

    if args.layout == "one-row":
        fig = plt.figure(figsize=(7.87, 2.28))
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1.0, 1.0, 1.28, 1.28], wspace=0.46)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[0, 3])
        legend_y = 1.06
        legend_cols = 4
    else:
        fig = plt.figure(figsize=(7.05, 4.95))
        gs = GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[1.0, 1.08],
            height_ratios=[0.9, 1.08],
            wspace=0.34,
            hspace=0.52,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
        legend_y = 1.02
        legend_cols = 2

    draw_bars(ax_a, agent_labels, agent_bar, "a", "By agent framework")
    draw_bars(ax_b, model_labels, model_bar, "b", "By LLM")
    draw_heatmap(ax_c, hm_agent_labels, agent_mat, "c", "By agent framework", rotate=38)
    draw_heatmap(ax_d, hm_model_labels, model_mat, "d", "By LLM", rotate=42)

    legend_handles = [
        Patch(facecolor=BLUE, edgecolor="#4A4A4A", label="AI ideas, same research area"),
        Patch(facecolor=BLUE_LIGHT, edgecolor=STRIPE, hatch="//", label="AI ideas, different areas"),
        Patch(facecolor=HUMAN, edgecolor="#4A4A4A", label="Human papers, same area"),
        Patch(facecolor=HUMAN_LIGHT, edgecolor=STRIPE, hatch="//", label="Human papers, different areas"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=legend_cols,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        columnspacing=1.2,
        handlelength=0.8,
        handleheight=0.6,
        fontsize=9.0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.png_out:
        args.png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png_out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
