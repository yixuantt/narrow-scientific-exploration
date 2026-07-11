#!/usr/bin/env python3
"""Aggregate independent novelty annotations with majority voting.

The two labels are marginal and may overlap: an idea may introduce both a new
research question and a new method. The output therefore reports both marginal
shares and the four joint categories.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import read_jsonl, safe_text, write_json, write_jsonl


LABELS = ("new_research_question", "new_method")
CATEGORIES = (
    "reused_question_reused_method",
    "reused_question_new_method",
    "new_question_reused_method",
    "new_question_new_method",
)


def binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    text = safe_text(value).lower()
    if text in {"0", "false", "no", "reused", "old"}:
        return 0
    if text in {"1", "true", "yes", "new"}:
        return 1
    return None


def vote_for(row: dict[str, Any], label: str) -> int | None:
    direct = binary(row.get(label))
    if direct is not None:
        return direct
    fallback_key = "task_new" if label == "new_research_question" else "method_new"
    fallback = row.get(fallback_key)
    if isinstance(fallback, list):
        return int(bool(fallback))
    return None


def load_annotator(path: Path, id_key: str) -> tuple[dict[str, dict[str, Any]], int]:
    output: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in read_jsonl(path):
        identifier = safe_text(row.get(id_key))
        if not identifier:
            continue
        if identifier in output:
            duplicates += 1
        # Restarted annotation jobs append repaired rows. Keep the latest row.
        output[identifier] = row
    return output, duplicates


def majority_vote(votes: Sequence[int | None]) -> int | None:
    clean = [int(vote) for vote in votes if vote in (0, 1)]
    if len(clean) < 2:
        return None
    ones = sum(clean)
    zeros = len(clean) - ones
    if ones > zeros:
        return 1
    if zeros > ones:
        return 0
    return None


def category(question: int, method: int) -> str:
    if question == 0 and method == 0:
        return CATEGORIES[0]
    if question == 0 and method == 1:
        return CATEGORIES[1]
    if question == 1 and method == 0:
        return CATEGORIES[2]
    return CATEGORIES[3]


def wilson_interval(successes: int, n: int) -> list[float | None]:
    if n == 0:
        return [None, None]
    z = 1.96
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def pack(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = Counter(row["category"] for row in rows)
    question_count = sum(int(row["new_research_question"]) for row in rows)
    method_count = sum(int(row["new_method"]) for row in rows)
    both_count = counts["new_question_new_method"]
    return {
        "n": n,
        "new_research_question": {
            "count": question_count,
            "share": question_count / n if n else None,
            "ci95": wilson_interval(question_count, n),
        },
        "new_method": {
            "count": method_count,
            "share": method_count / n if n else None,
            "ci95": wilson_interval(method_count, n),
        },
        "both_new": {
            "count": both_count,
            "share": both_count / n if n else None,
            "ci95": wilson_interval(both_count, n),
        },
        "categories": {
            name: {
                "count": int(counts[name]),
                "share": counts[name] / n if n else None,
            }
            for name in CATEGORIES
        },
    }


def fleiss_kappa(votes: np.ndarray) -> float | None:
    if votes.ndim != 2 or votes.shape[0] == 0 or votes.shape[1] < 2:
        return None
    n_items, n_raters = votes.shape
    positive = votes.sum(axis=1)
    negative = n_raters - positive
    observed = np.mean((positive * (positive - 1) + negative * (negative - 1)) / (n_raters * (n_raters - 1)))
    prevalence = float(votes.mean())
    expected = prevalence * prevalence + (1 - prevalence) * (1 - prevalence)
    if math.isclose(1.0, expected):
        return None
    return float((observed - expected) / (1 - expected))


def gwet_ac1(votes: np.ndarray) -> float | None:
    if votes.ndim != 2 or votes.shape[0] == 0 or votes.shape[1] < 2:
        return None
    n_raters = votes.shape[1]
    positive = votes.sum(axis=1)
    negative = n_raters - positive
    observed = np.mean((positive * (positive - 1) + negative * (negative - 1)) / (n_raters * (n_raters - 1)))
    prevalence = float(votes.mean())
    chance = 2 * prevalence * (1 - prevalence)
    if math.isclose(1.0, chance):
        return None
    return float((observed - chance) / (1 - chance))


def agreement(
    common_ids: Sequence[str],
    annotators: Sequence[dict[str, dict[str, Any]]],
    names: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {"annotators": list(names), "n_common_ids": len(common_ids)}
    for label in LABELS:
        complete = []
        for identifier in common_ids:
            row_votes = [vote_for(index[identifier], label) for index in annotators]
            if all(vote is not None for vote in row_votes):
                complete.append([int(vote) for vote in row_votes])
        matrix = np.asarray(complete, dtype=int)
        output[label] = {
            "n_complete": len(matrix),
            "all_agree_share": float(np.mean(np.all(matrix == matrix[:, :1], axis=1))) if len(matrix) else None,
            "fleiss_kappa": fleiss_kappa(matrix),
            "gwet_ac1": gwet_ac1(matrix),
        }
        pairwise = {}
        for left in range(len(annotators)):
            for right in range(left + 1, len(annotators)):
                valid = []
                for identifier in common_ids:
                    a = vote_for(annotators[left][identifier], label)
                    b = vote_for(annotators[right][identifier], label)
                    if a is not None and b is not None:
                        valid.append(int(a == b))
                pairwise[f"{names[left]}__{names[right]}"] = {
                    "n": len(valid),
                    "agreement": float(np.mean(valid)) if valid else None,
                }
        output[label]["pairwise"] = pairwise
    return output


def grouped(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    output = {"overall": pack(rows)}
    for field in fields:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = safe_text(row.get(field)) or "unknown"
            buckets[value].append(row)
        output[f"by_{field}"] = {name: pack(group_rows) for name, group_rows in sorted(buckets.items())}
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-files", type=Path, nargs="+", required=True)
    parser.add_argument("--annotator-names", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--id-key", default="run_id")
    parser.add_argument("--group-fields", nargs="+", default=["agent", "model", "primary_field", "year"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.annotator_files) < 3:
        raise ValueError("Majority voting requires at least three annotator files")
    if len(args.annotator_files) % 2 == 0:
        raise ValueError("Use an odd number of annotators to avoid tied votes")
    names = args.annotator_names or [path.stem for path in args.annotator_files]
    if len(names) != len(args.annotator_files):
        raise ValueError("--annotator-names must match --annotator-files")
    annotators = []
    duplicate_counts = {}
    for name, path in zip(names, args.annotator_files):
        index, duplicates = load_annotator(path, args.id_key)
        annotators.append(index)
        duplicate_counts[name] = duplicates
    common_ids = sorted(set.intersection(*(set(index) for index in annotators)))
    majority_rows: list[dict[str, Any]] = []
    invalid = 0
    for identifier in common_ids:
        labels: dict[str, int] = {}
        valid = True
        for label in LABELS:
            votes = [vote_for(index[identifier], label) for index in annotators]
            majority = majority_vote(votes)
            if majority is None:
                valid = False
                break
            labels[label] = majority
        if not valid:
            invalid += 1
            continue
        metadata = dict(annotators[0][identifier])
        for label in LABELS:
            metadata.pop(label, None)
        majority_rows.append(
            {
                **metadata,
                args.id_key: identifier,
                **labels,
                "category": category(labels["new_research_question"], labels["new_method"]),
            }
        )
    output = {
        "measure": "question_and_method_novelty",
        "definition": "Independent majority votes for idea-level new research-question and new-method labels.",
        "overlap_note": "The two labels are not mutually exclusive; joint categories are reported separately.",
        "annotators": names,
        "counts": {
            "common_ids": len(common_ids),
            "majority_valid": len(majority_rows),
            "majority_invalid": invalid,
            "duplicates_dropped": duplicate_counts,
        },
        "agreement": agreement(common_ids, annotators, names),
        "summaries": grouped(majority_rows, args.group_fields),
    }
    write_jsonl(args.out_dir / "novelty_majority_labels.jsonl", majority_rows)
    write_json(args.out_dir / "novelty_majority_summary.json", output)
    print(json.dumps(output["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
