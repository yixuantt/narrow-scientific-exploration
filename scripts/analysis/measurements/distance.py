#!/usr/bin/env python3
"""Compute exploration distance from each seed-paper centroid.

The script first averages idea-level distances within each seed task and
agent/model subgroup, then compares those task-level means with follow-on human
papers linked to the same task. Uncertainty is obtained by resampling tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import (
    bootstrap_ci,
    cosine_distance_to_centroid,
    deduplicate_aligned,
    group_indices,
    load_aligned_embeddings,
    require_fields,
    safe_text,
    year_from_identifier,
    write_json,
    write_jsonl,
)


def _dedup(
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    candidates: Sequence[Sequence[str]],
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    for fields in candidates:
        if all(any(safe_text(row.get(field)) for row in rows) for field in fields):
            return deduplicate_aligned(embeddings, rows, fields)
    return embeddings, rows, 0


def _year(row: dict[str, Any], year_key: str) -> int | None:
    for key in (year_key, "seed_year", "year"):
        try:
            value = row.get(key)
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            pass
    return year_from_identifier(row.get("task_id"))


def _paper_id(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    return text if text.startswith("CorpusId:") else f"CorpusId:{text}"


def expand_canonical_seeds(
    embeddings: np.ndarray,
    rows: Sequence[dict[str, Any]],
    canonical_root: Path,
    task_key: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lookup = {
        _paper_id(row.get("paper_id")): index
        for index, row in enumerate(rows)
        if _paper_id(row.get("paper_id"))
    }
    indices: list[int] = []
    output_rows: list[dict[str, Any]] = []
    for path in sorted(canonical_root.glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = state.get("source") or {}
        task_id = safe_text(state.get(task_key) or state.get("task_id") or source.get("seed_id"))
        if not task_id:
            continue
        for paper in state.get("memory_papers") or []:
            paper_id = _paper_id(paper.get("paper_id") or paper.get("corpusid"))
            index = lookup.get(paper_id)
            if index is None:
                continue
            indices.append(index)
            output_rows.append(
                {
                    **rows[index],
                    task_key: task_id,
                    "paper_id": paper_id,
                    "context_id": source.get("context_id") or paper.get("context_id"),
                    "seed_year": source.get("anchor_year"),
                }
            )
    return embeddings[np.asarray(indices, dtype=np.int64)], output_rows


def expand_followon_links(
    embeddings: np.ndarray,
    rows: Sequence[dict[str, Any]],
    links_path: Path,
    task_key: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lookup = {
        _paper_id(row.get("paper_id") or row.get("future_paper_id")): index
        for index, row in enumerate(rows)
        if _paper_id(row.get("paper_id") or row.get("future_paper_id"))
    }
    indices: list[int] = []
    output_rows: list[dict[str, Any]] = []
    with links_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            link = json.loads(line)
            paper_id = _paper_id(link.get("future_paper_id") or link.get("paper_id"))
            index = lookup.get(paper_id)
            task_id = safe_text(link.get(task_key) or link.get("seed_id"))
            if index is None or not task_id:
                continue
            indices.append(index)
            output_rows.append(
                {
                    **rows[index],
                    **link,
                    task_key: task_id,
                    "paper_id": paper_id,
                }
            )
    return embeddings[np.asarray(indices, dtype=np.int64)], output_rows


def build_task_records(
    idea_embeddings: np.ndarray,
    idea_rows: Sequence[dict[str, Any]],
    seed_embeddings: np.ndarray,
    seed_rows: Sequence[dict[str, Any]],
    followon_embeddings: np.ndarray,
    followon_rows: Sequence[dict[str, Any]],
    *,
    task_key: str,
    year_key: str,
    group_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_groups = group_indices(seed_rows, [task_key])
    idea_groups = group_indices(idea_rows, [task_key])
    followon_groups = group_indices(followon_rows, [task_key])
    observations: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []

    for (task_id,), seed_indices in sorted(seed_groups.items()):
        idea_indices = idea_groups.get((task_id,), [])
        followon_indices = followon_groups.get((task_id,), [])
        if not idea_indices or not followon_indices:
            continue
        centroid_rows = seed_embeddings[seed_indices]
        human_distances = cosine_distance_to_centroid(followon_embeddings[followon_indices], centroid_rows)
        human_distances = human_distances[np.isfinite(human_distances)]
        if len(human_distances) == 0:
            continue
        source_row = idea_rows[idea_indices[0]]
        seed_year = _year(source_row, year_key)
        for index, distance in zip(followon_indices, cosine_distance_to_centroid(followon_embeddings[followon_indices], centroid_rows)):
            if np.isfinite(distance):
                observations.append(
                    {
                        "task_id": task_id,
                        "source": "human_followon",
                        "distance": float(distance),
                        "seed_year": seed_year,
                        "paper_id": followon_rows[index].get("paper_id"),
                    }
                )

        dimensions: list[tuple[str, str, list[int]]] = [("pooled", "all", list(idea_indices))]
        for field in group_fields:
            values = sorted({safe_text(idea_rows[index].get(field)) for index in idea_indices if safe_text(idea_rows[index].get(field))})
            for value in values:
                dimensions.append(
                    (field, value, [index for index in idea_indices if safe_text(idea_rows[index].get(field)) == value])
                )

        for dimension, group, subgroup_indices in dimensions:
            ai_distances = cosine_distance_to_centroid(idea_embeddings[subgroup_indices], centroid_rows)
            ai_distances = ai_distances[np.isfinite(ai_distances)]
            if len(ai_distances) == 0:
                continue
            task_records.append(
                {
                    "task_id": task_id,
                    "seed_year": seed_year,
                    "dimension": dimension,
                    "group": group,
                    "n_seed_papers": len(seed_indices),
                    "n_ai": len(ai_distances),
                    "n_human": len(human_distances),
                    "ai_distance": float(ai_distances.mean()),
                    "human_distance": float(human_distances.mean()),
                    "difference": float(ai_distances.mean() - human_distances.mean()),
                }
            )
            for index, distance in zip(subgroup_indices, cosine_distance_to_centroid(idea_embeddings[subgroup_indices], centroid_rows)):
                if np.isfinite(distance):
                    observations.append(
                        {
                            "task_id": task_id,
                            "source": "ai",
                            "distance": float(distance),
                            "seed_year": seed_year,
                            "dimension": dimension,
                            "group": group,
                            "run_id": idea_rows[index].get("run_id"),
                        }
                    )
    return task_records, observations


def summarize(rows: Sequence[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_tasks": 0}

    def mean(sample: Sequence[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in sample]))

    def diff(sample: Sequence[dict[str, Any]]) -> float:
        return mean(sample, "difference")

    return {
        "n_tasks": len(rows),
        "n_ai_observations": int(sum(int(row["n_ai"]) for row in rows)),
        "n_human_observations": int(sum(int(row["n_human"]) for row in rows)),
        "ai_mean": mean(rows, "ai_distance"),
        "human_mean": mean(rows, "human_distance"),
        "difference": mean(rows, "difference"),
        "ai_ci95": bootstrap_ci(
            rows,
            lambda sample: mean(sample, "ai_distance"),
            repetitions=repetitions,
            seed=seed,
        ),
        "human_ci95": bootstrap_ci(
            rows,
            lambda sample: mean(sample, "human_distance"),
            repetitions=repetitions,
            seed=seed + 1,
        ),
        "difference_ci95": bootstrap_ci(
            rows, diff, repetitions=repetitions, seed=seed + 2
        ),
    }


def summarize_all(records: Sequence[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    dimensions = sorted({safe_text(row.get("dimension")) for row in records})
    for dim_pos, dimension in enumerate(dimensions):
        output[dimension] = {}
        groups = sorted({safe_text(row.get("group")) for row in records if row.get("dimension") == dimension})
        for group_pos, group in enumerate(groups):
            rows = [row for row in records if row.get("dimension") == dimension and row.get("group") == group]
            output[dimension][group] = summarize(
                rows, repetitions, seed + 10_000 * dim_pos + group_pos
            )
    by_year: dict[str, Any] = {}
    pooled = [row for row in records if row.get("dimension") == "pooled"]
    for year_pos, year in enumerate(sorted({row.get("seed_year") for row in pooled if row.get("seed_year") is not None})):
        rows = [row for row in pooled if row.get("seed_year") == year]
        by_year[str(year)] = summarize(rows, repetitions, seed + 50_000 + year_pos)
    output["by_year"] = by_year
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea-embeddings", type=Path, required=True)
    parser.add_argument("--idea-meta", type=Path, required=True)
    parser.add_argument("--seed-embeddings", type=Path, required=True)
    parser.add_argument("--seed-meta", type=Path, required=True)
    parser.add_argument("--followon-embeddings", type=Path, required=True)
    parser.add_argument("--followon-meta", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, default=None)
    parser.add_argument("--followon-links", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-key", default="task_id")
    parser.add_argument("--year-key", default="seed_year")
    parser.add_argument("--group-fields", nargs="+", default=["agent", "model"])
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    idea_embeddings, idea_rows = load_aligned_embeddings(args.idea_embeddings, args.idea_meta)
    seed_embeddings, seed_rows = load_aligned_embeddings(args.seed_embeddings, args.seed_meta)
    followon_embeddings, followon_rows = load_aligned_embeddings(
        args.followon_embeddings, args.followon_meta
    )
    if args.canonical_root is not None:
        seed_embeddings, seed_rows = expand_canonical_seeds(
            seed_embeddings, seed_rows, args.canonical_root, args.task_key
        )
    if args.followon_links is not None:
        followon_embeddings, followon_rows = expand_followon_links(
            followon_embeddings, followon_rows, args.followon_links, args.task_key
        )
    require_fields(idea_rows, [args.task_key], str(args.idea_meta))
    require_fields(seed_rows, [args.task_key], str(args.seed_meta))
    require_fields(followon_rows, [args.task_key], str(args.followon_meta))
    idea_embeddings, idea_rows, idea_duplicates = _dedup(
        idea_embeddings, idea_rows, [("run_id",), (args.task_key, "agent", "model")]
    )
    seed_embeddings, seed_rows, seed_duplicates = _dedup(
        seed_embeddings, seed_rows, [(args.task_key, "paper_id")]
    )
    followon_embeddings, followon_rows, followon_duplicates = _dedup(
        followon_embeddings, followon_rows, [(args.task_key, "paper_id")]
    )
    records, observations = build_task_records(
        idea_embeddings,
        idea_rows,
        seed_embeddings,
        seed_rows,
        followon_embeddings,
        followon_rows,
        task_key=args.task_key,
        year_key=args.year_key,
        group_fields=args.group_fields,
    )
    output = {
        "measure": "exploration_distance",
        "definition": "Cosine distance from a record embedding to the centroid of its seed-paper set.",
        "aggregation": "Idea-level distances are averaged within seed task and subgroup; summaries average task-level means.",
        "counts": {
            "task_records": len(records),
            "observations": len(observations),
            "idea_duplicates_dropped": idea_duplicates,
            "seed_duplicates_dropped": seed_duplicates,
            "followon_duplicates_dropped": followon_duplicates,
        },
        "summaries": summarize_all(
            records, args.bootstrap_repetitions, args.seed
        ),
    }
    write_jsonl(args.out_dir / "exploration_distance_records.jsonl", records)
    write_jsonl(args.out_dir / "exploration_distance_observations.jsonl", observations)
    write_json(args.out_dir / "exploration_distance_summary.json", output)
    print(json.dumps(output["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
