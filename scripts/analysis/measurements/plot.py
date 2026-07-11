#!/usr/bin/env python3
"""Plot summaries emitted by the measurement modules."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AI_COLOR = "#A9C2DA"
AI_EDGE = "#607D98"
HUMAN_COLOR = "#D8CFDD"
HUMAN_EDGE = "#847789"
QUESTION_COLOR = "#DCE8F1"
METHOD_COLOR = "#D8CFDD"
GRID = "#E8E8E8"


def label(value: str) -> str:
    mapping = {
        "flat_llm": "Zero-shot",
        "ai_scientist_v2": "AI Scientist",
        "research_agent": "ResearchAgent",
        "agent_laboratory": "Agent Lab.",
        "co_scientist": "Co-Scientist",
        "gpt_5_4": "GPT-5.4",
    }
    return mapping.get(value, value.replace("_", " "))


def errors(mean: float, interval: Sequence[float | None] | None) -> tuple[float, float]:
    if not interval or interval[0] is None or interval[1] is None:
        return 0.0, 0.0
    return max(0.0, mean - float(interval[0])), max(0.0, float(interval[1]) - mean)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)


def paired_groups(data: dict[str, Any], measure: str, group: str) -> dict[str, Any]:
    if measure == "frontier":
        return {"overall": data["summaries"]["overall"]} if group == "overall" else data["summaries"][f"by_{group}"]
    return data["summaries"][group]


def paired_keys(measure: str) -> tuple[str, str, str, str]:
    if measure == "breadth":
        return "ai_mean", "human_mean", "ai_ci95", "human_ci95"
    if measure == "distance":
        return "ai_mean", "human_mean", "ai_ci95", "human_ci95"
    if measure == "frontier":
        return "idea_mean", "human_mean", "idea_ci95", "human_ci95"
    if measure == "impact":
        return "ai_mean", "human_mean", "ai_ci95", "human_ci95"
    raise ValueError(measure)


def plot_paired(axis: plt.Axes, groups: dict[str, Any], measure: str, scope: str) -> None:
    if measure == "breadth":
        groups = {name: value[scope] for name, value in groups.items()}
    names = [
        name
        for name, value in groups.items()
        if value.get(
            "n_tasks",
            value.get("n_groups", value.get("n_units", value.get("n_ai", 0))),
        )
        > 0
    ]
    ai_key, human_key, ai_ci_key, human_ci_key = paired_keys(measure)
    ai = np.asarray([groups[name][ai_key] for name in names], dtype=float)
    human = np.asarray([groups[name][human_key] for name in names], dtype=float)
    ai_err = np.asarray([errors(groups[name][ai_key], groups[name].get(ai_ci_key)) for name in names]).T
    human_err = np.asarray([errors(groups[name][human_key], groups[name].get(human_ci_key)) for name in names]).T
    x = np.arange(len(names))
    width = 0.36
    axis.bar(
        x - width / 2,
        ai,
        width,
        yerr=ai_err,
        color=AI_COLOR,
        edgecolor=AI_EDGE,
        linewidth=0.8,
        capsize=2,
        label="AI ideas",
    )
    axis.bar(
        x + width / 2,
        human,
        width,
        yerr=human_err,
        color=HUMAN_COLOR,
        edgecolor=HUMAN_EDGE,
        linewidth=0.8,
        hatch="////",
        capsize=2,
        label="Human papers",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([label(name) for name in names], rotation=32, ha="right")
    ylabels = {
        "breadth": "Exploration breadth",
        "distance": "Exploration distance",
        "frontier": "Frontier alignment",
        "impact": "Potential impact",
    }
    axis.set_ylabel(ylabels[measure])
    style_axis(axis)


def plot_matrix(axis: plt.Axes, matrix: dict[str, Any]) -> None:
    values = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in matrix["breadth"]],
        dtype=float,
    )
    image = axis.imshow(values, cmap="Blues", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    labels = [label(value) for value in matrix["labels"]]
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=40, ha="right")
    axis.set_yticklabels(labels)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_novelty(axis: plt.Axes, data: dict[str, Any], group: str) -> None:
    groups = data["summaries"]["overall"] if group == "overall" else data["summaries"][f"by_{group}"]
    if group == "overall":
        groups = {"overall": groups}
    names = list(groups)
    question = np.asarray([groups[name]["new_research_question"]["share"] for name in names]) * 100
    method = np.asarray([groups[name]["new_method"]["share"] for name in names]) * 100
    question_err = np.asarray(
        [errors(groups[name]["new_research_question"]["share"], groups[name]["new_research_question"]["ci95"]) for name in names]
    ).T * 100
    method_err = np.asarray(
        [errors(groups[name]["new_method"]["share"], groups[name]["new_method"]["ci95"]) for name in names]
    ).T * 100
    x = np.arange(len(names))
    width = 0.36
    axis.bar(
        x - width / 2,
        question,
        width,
        yerr=question_err,
        color=QUESTION_COLOR,
        edgecolor="#607D98",
        linewidth=0.8,
        capsize=2,
        label="New research question",
    )
    axis.bar(
        x + width / 2,
        method,
        width,
        yerr=method_err,
        color=METHOD_COLOR,
        edgecolor="#847789",
        linewidth=0.8,
        hatch="////",
        capsize=2,
        label="New method",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([label(name) for name in names], rotation=32, ha="right")
    axis.set_ylabel("Share of ideas (%)")
    axis.set_ylim(0, 105)
    style_axis(axis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("breadth", "distance", "frontier", "impact", "novelty"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--group", default="agent")
    parser.add_argument("--scope", choices=("same_area", "different_area_same_field"), default="same_area")
    parser.add_argument("--include-matrix", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "hatch.linewidth": 0.45,
        }
    )
    use_matrix = args.measure == "breadth" and args.include_matrix and args.group in data.get("matrices", {})
    figure, axes = plt.subplots(1, 2 if use_matrix else 1, figsize=(10, 4) if use_matrix else (7.2, 3.8))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    if args.measure == "novelty":
        plot_novelty(axes[0], data, args.group)
    else:
        groups = paired_groups(data, args.measure, args.group)
        plot_paired(axes[0], groups, args.measure, args.scope)
    axes[0].legend(frameon=False)
    if use_matrix:
        plot_matrix(axes[1], data["matrices"][args.group])
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    plt.close(figure)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
