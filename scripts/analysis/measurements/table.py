#!/usr/bin/env python3
"""Export measurement summaries as CSV or standalone LaTeX tabular code."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(source, target)
    return text


def format_number(value: Any, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def display_label(value: Any) -> str:
    text = str(value)
    return {
        "all": "All",
        "flat_llm": "Zero-shot",
        "ai_scientist_v2": "AI Scientist",
        "research_agent": "ResearchAgent",
        "agent_laboratory": "Agent Laboratory",
        "co_scientist": "Co-Scientist",
        "google_gemma_4_31b_it": "Gemma-4-31B-IT",
        "meta_llama_llama_3_1_8b_instruct": "Llama-3.1-8B",
        "nousresearch_hermes_4_14b": "Hermes-4-14B",
        "qwen_qwen3_6_35b_a3b": "Qwen3-35B-A3B",
        "gpt_5_4": "GPT-5.4",
    }.get(text, text.replace("_", " "))


def interval(value: Any, digits: int) -> str:
    if not isinstance(value, Sequence) or len(value) != 2 or value[0] is None or value[1] is None:
        return ""
    return f"[{float(value[0]):.{digits}f}, {float(value[1]):.{digits}f}]"


def group_data(data: dict[str, Any], measure: str, group: str) -> dict[str, Any]:
    summaries = data["summaries"]
    if measure == "frontier":
        return {"all": summaries["overall"]} if group == "pooled" else summaries[f"by_{group}"]
    if measure == "distance" and group in {"year", "seed_year"}:
        return summaries["by_year"]
    if group == "pooled":
        return summaries["pooled"]
    return summaries[group]


def comparison_rows(
    data: dict[str, Any], measure: str, group: str, scope: str, digits: int
) -> tuple[list[str], list[list[str]]]:
    groups = group_data(data, measure, group)
    output = []
    for name, values in groups.items():
        if measure == "breadth":
            values = values[scope]
        ai_key = "idea_mean" if measure == "frontier" else "ai_mean"
        ai_ci_key = "idea_ci95" if measure == "frontier" else "ai_ci95"
        n = values.get("n_tasks", values.get("n_groups", values.get("n_units", values.get("n_ai", 0))))
        output.append(
            [
                display_label(name),
                str(n),
                format_number(values.get(ai_key), digits),
                format_number(values.get("human_mean"), digits),
                format_number(values.get("difference"), digits),
                interval(values.get("difference_ci95"), digits),
                interval(values.get(ai_ci_key), digits),
                interval(values.get("human_ci95"), digits),
            ]
        )
    return (
        ["Group", "N", "AI", "Human", "AI-Human", "Difference 95% CI", "AI 95% CI", "Human 95% CI"],
        output,
    )


def novelty_rows(
    data: dict[str, Any], group: str, digits: int
) -> tuple[list[str], list[list[str]]]:
    summaries = data["summaries"]
    groups = {"all": summaries["overall"]} if group == "pooled" else summaries[f"by_{group}"]
    output = []
    for name, values in groups.items():
        question = values["new_research_question"]
        method = values["new_method"]
        both = values["both_new"]
        output.append(
            [
                display_label(name),
                str(values["n"]),
                format_number(question["share"], digits),
                interval(question["ci95"], digits),
                format_number(method["share"], digits),
                interval(method["ci95"], digits),
                format_number(both["share"], digits),
                interval(both["ci95"], digits),
            ]
        )
    return (
        [
            "Group",
            "N",
            "New question",
            "Question 95% CI",
            "New method",
            "Method 95% CI",
            "Both new",
            "Both 95% CI",
        ],
        output,
    )


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_latex(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = "l" + "r" * (len(header) - 1)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(value) for value in header) + " \\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("breadth", "distance", "frontier", "impact", "novelty"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--group", default="agent")
    parser.add_argument("--scope", choices=("same_area", "different_area_same_field"), default="same_area")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--tex-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.csv_out is None and args.tex_out is None:
        raise ValueError("Pass --csv-out, --tex-out, or both")
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if args.measure == "novelty":
        header, rows = novelty_rows(data, args.group, args.digits)
    else:
        header, rows = comparison_rows(data, args.measure, args.group, args.scope, args.digits)
    if args.csv_out is not None:
        write_csv(args.csv_out, header, rows)
        print(f"saved {args.csv_out}")
    if args.tex_out is not None:
        write_latex(args.tex_out, header, rows)
        print(f"saved {args.tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
