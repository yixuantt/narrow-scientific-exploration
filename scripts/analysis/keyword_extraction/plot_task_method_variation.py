#!/usr/bin/env python3
"""Plot research-question vs technical-method variation by framework."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.common import plot_style as S  # noqa: E402


AGENT_LABELS = {
    "flat_llm": "Zero-shot",
    "ai_scientist_v2": "AI Scientist",
    "research_agent": "ResearchAgent",
    "agent_laboratory": "Agent Lab.",
}


def load_proportions(path: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    question_props: dict[str, list[float]] = defaultdict(list)
    method_props: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            agent = row.get("agent", "unknown")
            question_terms = row.get("task_keywords", [])
            method_terms = row.get("method_keywords", [])
            question_new = row.get("task_new", [])
            method_new = row.get("method_new", [])
            if question_terms:
                question_props[agent].append(len(question_new) / len(question_terms))
            if method_terms:
                method_props[agent].append(len(method_new) / len(method_terms))
    return question_props, method_props


def draw_panel(ax: plt.Axes, data: dict[str, list[float]], title: str, color: str, fill: str) -> None:
    values = [data.get(agent, [0.0]) for agent in S.AGENTS]
    positions = np.arange(len(S.AGENTS))

    violins = ax.violinplot(
        values,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor(fill)
        body.set_edgecolor("#4A4A4A")
        body.set_linewidth(0.6)
        body.set_alpha(1.0)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.7},
        capprops={"color": "#4A4A4A", "linewidth": 0.7},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("#4A4A4A")
        patch.set_linewidth(0.7)

    ax.set_title(title, loc="left")
    ax.set_xticks(positions)
    ax.set_xticklabels([AGENT_LABELS.get(agent, S.AGENT_LABELS[agent]) for agent in S.AGENTS], rotation=25, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="idea_keywords JSONL from extract_keywords.py")
    parser.add_argument("--out", type=Path, required=True, help="Output PDF path")
    parser.add_argument("--png-out", type=Path, default=None, help="Optional PNG output path")
    args = parser.parse_args()

    S.apply()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.pad_inches": 0.02,
        }
    )

    question_props, method_props = load_proportions(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    draw_panel(axes[0], question_props, "a  Research questions", "#86A6C8", "#E8F0F7")
    draw_panel(axes[1], method_props, "b  Technical methods", "#B9AEC2", "#EEEAF2")
    axes[0].set_ylabel("Proportion absent\nfrom seed literature")
    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.31, wspace=0.18)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    if args.png_out:
        args.png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
