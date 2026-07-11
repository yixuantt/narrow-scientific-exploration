#!/usr/bin/env python3
"""Compute exploration breadth from aligned idea and human-paper embeddings.

Breadth is one minus cosine similarity. AI and human records are downsampled to
equal counts inside each research area before pairwise distances are computed.
The primary summary is pair-count weighted; context-level records are retained
so equal-context summaries or clustered resampling can also be computed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import (
    bootstrap_ci,
    deduplicate_aligned,
    group_indices,
    load_aligned_embeddings,
    mean_cross_cosine,
    mean_pairwise_cosine,
    require_fields,
    safe_text,
    weighted_mean,
    write_json,
    write_jsonl,
)


def _dedup(
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    preferred: Sequence[Sequence[str]],
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    for fields in preferred:
        if all(any(safe_text(row.get(field)) for row in rows) for field in fields):
            return deduplicate_aligned(embeddings, rows, fields)
    return embeddings, rows, 0


def _field(row: Mapping[str, Any], field_key: str, context_key: str) -> str:
    explicit = safe_text(row.get(field_key))
    if explicit:
        return explicit
    context = safe_text(row.get(context_key))
    prefix = context.rsplit("_", 1)[0] if "_" in context else context
    return prefix.replace("_", " ").title() if prefix else "Unknown"


def matched_context_records(
    idea_embeddings: np.ndarray,
    idea_rows: Sequence[dict[str, Any]],
    human_embeddings: np.ndarray,
    human_rows: Sequence[dict[str, Any]],
    *,
    dimension: str,
    group: str,
    context_key: str,
    field_key: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[int]]]]:
    rng = np.random.default_rng(seed)
    idea_contexts = group_indices(idea_rows, [context_key])
    human_contexts = group_indices(human_rows, [context_key])
    matched: dict[str, dict[str, list[int]]] = {}
    records: list[dict[str, Any]] = []

    for (context_id,), context_idea_indices in sorted(idea_contexts.items()):
        if dimension != "pooled":
            context_idea_indices = [
                index for index in context_idea_indices if safe_text(idea_rows[index].get(dimension)) == group
            ]
        context_human_indices = list(human_contexts.get((context_id,), []))
        size = min(len(context_idea_indices), len(context_human_indices))
        if size < 2:
            continue
        ai_idx = sorted(rng.choice(context_idea_indices, size=size, replace=False).tolist())
        human_idx = sorted(rng.choice(context_human_indices, size=size, replace=False).tolist())
        ai_similarity, ai_pairs = mean_pairwise_cosine(idea_embeddings[ai_idx])
        human_similarity, human_pairs = mean_pairwise_cosine(human_embeddings[human_idx])
        ai_breadth = 1.0 - ai_similarity
        human_breadth = 1.0 - human_similarity
        source_row = idea_rows[ai_idx[0]]
        records.append(
            {
                "scope": "same_area",
                "dimension": dimension,
                "group": group,
                "context_id": context_id,
                "field": _field(source_row, field_key, context_key),
                "n_ai": size,
                "n_human": size,
                "n_ai_pairs": ai_pairs,
                "n_human_pairs": human_pairs,
                "ai_breadth": ai_breadth,
                "human_breadth": human_breadth,
                "difference": ai_breadth - human_breadth,
            }
        )
        matched[context_id] = {"ai": ai_idx, "human": human_idx}
    return records, matched


def different_area_records(
    idea_embeddings: np.ndarray,
    idea_rows: Sequence[dict[str, Any]],
    human_embeddings: np.ndarray,
    human_rows: Sequence[dict[str, Any]],
    matched: Mapping[str, Mapping[str, list[int]]],
    *,
    dimension: str,
    group: str,
    field_key: str,
    context_key: str,
) -> list[dict[str, Any]]:
    by_field: dict[str, list[str]] = defaultdict(list)
    for context_id, indices in matched.items():
        if indices["ai"]:
            by_field[_field(idea_rows[indices["ai"][0]], field_key, context_key)].append(context_id)

    output: list[dict[str, Any]] = []
    for field, contexts in sorted(by_field.items()):
        contexts = sorted(contexts)
        for left_pos, left_context in enumerate(contexts):
            for right_context in contexts[left_pos + 1 :]:
                left = matched[left_context]
                right = matched[right_context]
                ai_similarity, ai_pairs = mean_cross_cosine(
                    idea_embeddings[left["ai"]], idea_embeddings[right["ai"]]
                )
                human_similarity, human_pairs = mean_cross_cosine(
                    human_embeddings[left["human"]], human_embeddings[right["human"]]
                )
                output.append(
                    {
                        "scope": "different_area_same_field",
                        "dimension": dimension,
                        "group": group,
                        "field": field,
                        "context_left": left_context,
                        "context_right": right_context,
                        "n_ai_pairs": ai_pairs,
                        "n_human_pairs": human_pairs,
                        "ai_breadth": 1.0 - ai_similarity,
                        "human_breadth": 1.0 - human_similarity,
                        "difference": human_similarity - ai_similarity,
                    }
                )
    return output


def summarize_records(rows: Sequence[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_units": 0}

    def estimate(sample: Sequence[dict[str, Any]], key: str, weight: str) -> float:
        return weighted_mean(sample, key, weight)

    ai_mean = estimate(rows, "ai_breadth", "n_ai_pairs")
    human_mean = estimate(rows, "human_breadth", "n_human_pairs")

    def difference(sample: Sequence[dict[str, Any]]) -> float:
        return estimate(sample, "ai_breadth", "n_ai_pairs") - estimate(
            sample, "human_breadth", "n_human_pairs"
        )

    return {
        "n_units": len(rows),
        "n_ai_pairs": int(sum(int(row["n_ai_pairs"]) for row in rows)),
        "n_human_pairs": int(sum(int(row["n_human_pairs"]) for row in rows)),
        "ai_mean": ai_mean,
        "human_mean": human_mean,
        "difference": ai_mean - human_mean,
        "ai_ci95": bootstrap_ci(
            rows,
            lambda sample: estimate(sample, "ai_breadth", "n_ai_pairs"),
            repetitions=repetitions,
            seed=seed,
        ),
        "human_ci95": bootstrap_ci(
            rows,
            lambda sample: estimate(sample, "human_breadth", "n_human_pairs"),
            repetitions=repetitions,
            seed=seed + 1,
        ),
        "difference_ci95": bootstrap_ci(
            rows, difference, repetitions=repetitions, seed=seed + 2
        ),
        "context_mean_ai": float(np.mean([row["ai_breadth"] for row in rows])),
        "context_mean_human": float(np.mean([row["human_breadth"] for row in rows])),
    }


def summarize_group_balanced(
    grouped_rows: Mapping[str, Sequence[dict[str, Any]]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    usable = {group: list(rows) for group, rows in grouped_rows.items() if rows}
    if not usable:
        return {"n_units": 0, "n_groups": 0}

    def point(rows_by_group: Mapping[str, Sequence[dict[str, Any]]], key: str, weight: str) -> float:
        estimates = [weighted_mean(rows, key, weight) for rows in rows_by_group.values()]
        estimates = [value for value in estimates if np.isfinite(value)]
        return float(np.mean(estimates)) if estimates else math.nan

    ai_mean = point(usable, "ai_breadth", "n_ai_pairs")
    human_mean = point(usable, "human_breadth", "n_human_pairs")

    def intervals() -> tuple[list[float | None], list[float | None], list[float | None]]:
        if repetitions <= 0:
            return [None, None], [None, None], [None, None]
        rng = np.random.default_rng(seed)
        ai_values = np.empty(repetitions, dtype=float)
        human_values = np.empty(repetitions, dtype=float)
        for repetition in range(repetitions):
            sampled = {}
            for group, rows in usable.items():
                sampled[group] = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
            ai_values[repetition] = point(sampled, "ai_breadth", "n_ai_pairs")
            human_values[repetition] = point(sampled, "human_breadth", "n_human_pairs")
        diff_values = ai_values - human_values
        return (
            [float(np.quantile(ai_values, 0.025)), float(np.quantile(ai_values, 0.975))],
            [float(np.quantile(human_values, 0.025)), float(np.quantile(human_values, 0.975))],
            [float(np.quantile(diff_values, 0.025)), float(np.quantile(diff_values, 0.975))],
        )

    ai_ci, human_ci, diff_ci = intervals()
    return {
        "n_units": len({row.get("context_id", (row.get("context_left"), row.get("context_right"))) for rows in usable.values() for row in rows}),
        "n_groups": len(usable),
        "n_ai_pairs": int(sum(int(row["n_ai_pairs"]) for rows in usable.values() for row in rows)),
        "n_human_pairs": int(sum(int(row["n_human_pairs"]) for rows in usable.values() for row in rows)),
        "ai_mean": ai_mean,
        "human_mean": human_mean,
        "difference": ai_mean - human_mean,
        "ai_ci95": ai_ci,
        "human_ci95": human_ci,
        "difference_ci95": diff_ci,
        "group_balancing": "Equal weight across groups after pair-weighted aggregation within each group.",
    }


def breadth_matrix(
    idea_embeddings: np.ndarray,
    idea_rows: Sequence[dict[str, Any]],
    human_embeddings: np.ndarray,
    human_rows: Sequence[dict[str, Any]],
    *,
    dimension: str,
    context_key: str,
) -> dict[str, Any]:
    groups = sorted({safe_text(row.get(dimension)) for row in idea_rows if safe_text(row.get(dimension))})
    labels = groups + ["human"]
    sums = np.zeros((len(labels), len(labels)), dtype=float)
    counts = np.zeros_like(sums)
    idea_contexts = group_indices(idea_rows, [context_key])
    human_contexts = group_indices(human_rows, [context_key])

    for context_tuple in sorted(set(idea_contexts) | set(human_contexts)):
        idea_context = idea_contexts.get(context_tuple, [])
        human_context = human_contexts.get(context_tuple, [])
        members = {
            group: [index for index in idea_context if safe_text(idea_rows[index].get(dimension)) == group]
            for group in groups
        }
        for left_pos, left_group in enumerate(groups):
            left_idx = members[left_group]
            similarity, count = mean_pairwise_cosine(idea_embeddings[left_idx])
            if count:
                sums[left_pos, left_pos] += (1.0 - similarity) * count
                counts[left_pos, left_pos] += count
            for right_pos in range(left_pos + 1, len(groups)):
                similarity, count = mean_cross_cosine(
                    idea_embeddings[left_idx], idea_embeddings[members[groups[right_pos]]]
                )
                if count:
                    distance = 1.0 - similarity
                    sums[left_pos, right_pos] += distance * count
                    sums[right_pos, left_pos] += distance * count
                    counts[left_pos, right_pos] += count
                    counts[right_pos, left_pos] += count
            similarity, count = mean_cross_cosine(
                idea_embeddings[left_idx], human_embeddings[human_context]
            )
            if count:
                distance = 1.0 - similarity
                sums[left_pos, -1] += distance * count
                sums[-1, left_pos] += distance * count
                counts[left_pos, -1] += count
                counts[-1, left_pos] += count
        similarity, count = mean_pairwise_cosine(human_embeddings[human_context])
        if count:
            sums[-1, -1] += (1.0 - similarity) * count
            counts[-1, -1] += count

    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = sums / counts
    return {
        "labels": labels,
        "breadth": [[None if not np.isfinite(value) else float(value) for value in row] for row in matrix],
        "pair_counts": counts.astype(np.int64).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea-embeddings", type=Path, required=True)
    parser.add_argument("--idea-meta", type=Path, required=True)
    parser.add_argument("--human-embeddings", type=Path, required=True)
    parser.add_argument("--human-meta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-key", default="context_id")
    parser.add_argument("--field-key", default="primary_field")
    parser.add_argument("--group-fields", nargs="+", default=["agent", "model"])
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    idea_embeddings, idea_rows = load_aligned_embeddings(args.idea_embeddings, args.idea_meta)
    human_embeddings, human_rows = load_aligned_embeddings(args.human_embeddings, args.human_meta)
    require_fields(idea_rows, [args.context_key], str(args.idea_meta))
    require_fields(human_rows, [args.context_key], str(args.human_meta))
    idea_embeddings, idea_rows, idea_duplicates = _dedup(
        idea_embeddings, idea_rows, [("run_id",), ("task_id", "agent", "model")]
    )
    human_embeddings, human_rows, human_duplicates = _dedup(
        human_embeddings, human_rows, [("paper_id",), (args.context_key, "title")]
    )

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    dimensions = []
    for field in args.group_fields:
        values = sorted({safe_text(row.get(field)) for row in idea_rows if safe_text(row.get(field))})
        dimensions.append((field, values))

    for dim_index, (dimension, groups) in enumerate(dimensions):
        summaries[dimension] = {}
        for group_index, group in enumerate(groups):
            records, matched = matched_context_records(
                idea_embeddings,
                idea_rows,
                human_embeddings,
                human_rows,
                dimension=dimension,
                group=group,
                context_key=args.context_key,
                field_key=args.field_key,
                seed=args.seed + 1000 * dim_index + group_index,
            )
            cross_records = different_area_records(
                idea_embeddings,
                idea_rows,
                human_embeddings,
                human_rows,
                matched,
                dimension=dimension,
                group=group,
                field_key=args.field_key,
                context_key=args.context_key,
            )
            all_records.extend(records)
            all_records.extend(cross_records)
            summaries[dimension][group] = {
                "same_area": summarize_records(
                    records, args.bootstrap_repetitions, args.seed + 10_000 * dim_index + group_index
                ),
                "different_area_same_field": summarize_records(
                    cross_records,
                    args.bootstrap_repetitions,
                    args.seed + 20_000 * dim_index + group_index,
                ),
            }

    pooling_dimension = "agent" if "agent" in summaries else args.group_fields[0]
    same_by_group = {
        group: [row for row in all_records if row["dimension"] == pooling_dimension and row["group"] == group and row["scope"] == "same_area"]
        for group in summaries[pooling_dimension]
    }
    different_by_group = {
        group: [row for row in all_records if row["dimension"] == pooling_dimension and row["group"] == group and row["scope"] == "different_area_same_field"]
        for group in summaries[pooling_dimension]
    }
    summaries["pooled"] = {
        "all": {
            "same_area": summarize_group_balanced(
                same_by_group, args.bootstrap_repetitions, args.seed + 90_000
            ),
            "different_area_same_field": summarize_group_balanced(
                different_by_group, args.bootstrap_repetitions, args.seed + 100_000
            ),
        }
    }

    matrices = {
        field: breadth_matrix(
            idea_embeddings,
            idea_rows,
            human_embeddings,
            human_rows,
            dimension=field,
            context_key=args.context_key,
        )
        for field in args.group_fields
    }
    output = {
        "measure": "exploration_breadth",
        "definition": "One minus pairwise cosine similarity within a research area.",
        "matching": "AI and human records are downsampled to equal counts within each area and comparison group.",
        "aggregation": "Pair-count-weighted means; context-level means are also reported.",
        "counts": {
            "idea_rows": len(idea_rows),
            "human_rows": len(human_rows),
            "idea_duplicates_dropped": idea_duplicates,
            "human_duplicates_dropped": human_duplicates,
        },
        "summaries": summaries,
        "matrices": matrices,
    }
    write_jsonl(args.out_dir / "exploration_breadth_records.jsonl", all_records)
    write_json(args.out_dir / "exploration_breadth_summary.json", output)
    print(json.dumps(output["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
