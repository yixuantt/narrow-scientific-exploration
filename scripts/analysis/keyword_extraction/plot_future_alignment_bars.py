#!/usr/bin/env python3
"""Rebuild the paper figure for AI/follow-on/literature alignment.

The input file contains one task-level row per generated idea task. Each row
stores three mean cosine similarities:
  - generated AI idea to starting literature
  - follow-on human work to starting literature
  - generated AI idea to follow-on human work
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


METRICS = [
    ("mean_sim_ai_memory", "AI--literature", "#86A6C8", ""),
    ("mean_sim_future_memory", "Follow-on--literature", "#D7D0DC", "//"),
    ("mean_sim_ai_future", "AI--follow-on", "#EEEEEE", "xx"),
]

AGENT_ORDER = ["agent_laboratory", "ai_scientist_v2", "flat_llm", "research_agent"]
MODEL_ORDER = [
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3.5-4B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
]


def short_agent(name: str) -> str:
    return {
        "agent_laboratory": "Agent Lab.",
        "ai_scientist_v2": "AI Scientist",
        "flat_llm": "Zero-shot",
        "research_agent": "ResearchAgent",
    }.get(name, name)


def short_model(name: str) -> str:
    return {
        "Qwen/Qwen3.5-0.8B": "Qwen 0.8B",
        "meta-llama/Llama-3.2-1B-Instruct": "Llama 1B",
        "Qwen/Qwen3.5-4B": "Qwen 4B",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama 8B",
        "google/gemma-4-31B-it": "Gemma 31B",
        "Qwen/Qwen3.6-35B-A3B": "Qwen 35B",
    }.get(name, name)


def ordered_values(values, preferred_order):
    seen = set(values)
    ordered = [v for v in preferred_order if v in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def aggregate(rows, group_key, groups):
    means = np.zeros((len(groups), len(METRICS)), dtype=float)
    stds = np.zeros_like(means)
    for i, group in enumerate(groups):
        group_rows = [r for r in rows if r[group_key] == group]
        if not group_rows:
            means[i, :] = np.nan
            stds[i, :] = np.nan
            continue
        for j, (metric, _, _, _) in enumerate(METRICS):
            vals = np.array([r[metric] for r in group_rows], dtype=float)
            means[i, j] = vals.mean()
            stds[i, j] = vals.std()
    return means, stds


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, width=0.6, length=2.5)
    ax.set_ylim(0.70, 1.03)
    ax.set_yticks(np.arange(0.70, 1.001, 0.05))
    ax.set_ylabel("Cosine similarity", fontsize=9)


def draw_grouped_bars(ax, labels, means, stds, panel_label, title):
    x = np.arange(len(labels), dtype=float)
    width = 0.23
    offsets = np.array([-width, 0.0, width])
    for j, (_, metric_label, color, hatch) in enumerate(METRICS):
        ax.bar(
            x + offsets[j],
            means[:, j],
            width,
            yerr=stds[:, j],
            label=metric_label,
            color=color,
            edgecolor="#4A4A4A",
            linewidth=0.55,
            hatch=hatch,
            error_kw={"elinewidth": 0.65, "ecolor": "#6A6A6A", "capsize": 2.0},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{panel_label}  {title}", loc="left", fontsize=9, fontweight="bold")
    style_axis(ax)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-agg",
        type=Path,
        default=Path("analysis_out/keyword_extraction/rq2_three_way_task_agg.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures_tex/fig_future_alignment_bars.pdf"),
    )
    parser.add_argument("--png-out", type=Path, default=Path("figures_tex/fig_future_alignment_bars.png"))
    parser.add_argument("--titles", choices=("paper", "modern"), default="modern")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.task_agg.open() as f:
        rows = json.load(f)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "hatch.linewidth": 0.35,
    })

    agent_groups = ordered_values([r["agent"] for r in rows], AGENT_ORDER)
    model_groups = ordered_values([r["model"] for r in rows], MODEL_ORDER)
    agent_means, agent_stds = aggregate(rows, "agent", agent_groups)
    model_means, model_stds = aggregate(rows, "model", model_groups)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    if args.titles == "modern":
        left_title, right_title = "By agent framework", "By LLM"
    else:
        left_title, right_title = "By agent framework", "By LLM"
    draw_grouped_bars(
        axes[0],
        [short_agent(g) for g in agent_groups],
        agent_means,
        agent_stds,
        "a",
        left_title,
    )
    draw_grouped_bars(
        axes[1],
        [short_model(g) for g in model_groups],
        model_means,
        model_stds,
        "b",
        right_title,
    )
    axes[1].set_ylabel("")

    handles = [Patch(facecolor=color, edgecolor="#4A4A4A", hatch=hatch, label=label) for _, label, color, hatch in METRICS]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.2,
        handlelength=0.8,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.74, bottom=0.25, left=0.08, right=0.99, wspace=0.30)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.png_out:
        args.png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png_out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
