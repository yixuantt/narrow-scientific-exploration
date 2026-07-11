#!/usr/bin/env python3
"""Compute next-year research-frontier alignment from scholarly keywords.

For each field and year, the frontier is the most frequent fraction of unique
keywords in next-year human papers. Idea keywords are unioned within each
field/year/agent/model group. Follow-on keywords are unioned once per task in
the same group. Evaluation follow-on papers are excluded from frontier
construction by default.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common import (
    bootstrap_ci,
    keyword_set,
    read_jsonl,
    require_fields,
    safe_text,
    write_json,
    write_jsonl,
)


def normalized_paper_id(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    return text if text.startswith("CorpusId:") else f"CorpusId:{text}"


def canonical_paper_metadata(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = state.get("source") or {}
        default_field = safe_text(source.get("primary_field"))
        for paper in state.get("memory_papers") or []:
            paper_id = normalized_paper_id(paper.get("paper_id") or paper.get("corpusid"))
            year = integer(paper, ["year"])
            field = safe_text(paper.get("primary_field")) or default_field
            if not paper_id or year is None or not field:
                continue
            old = output.get(paper_id)
            if old is None or year < int(old["year"]):
                output[paper_id] = {"paper_id": paper_id, "primary_field": field, "year": year}
    return output


def merge_paper_metadata(
    annotations: Sequence[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    paper_id_key: str,
) -> list[dict[str, Any]]:
    output = []
    for row in annotations:
        paper_id = normalized_paper_id(row.get(paper_id_key))
        meta = metadata.get(paper_id)
        if meta:
            output.append({**row, **meta, paper_id_key: paper_id})
        else:
            output.append(dict(row))
    return output


def join_followon_annotations(
    annotations: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]] | None,
    *,
    task_key: str,
    field_key: str,
    seed_year_key: str,
    paper_id_key: str,
) -> list[dict[str, Any]]:
    annotation_index = {
        normalized_paper_id(row.get(paper_id_key) or row.get("future_paper_id")): row
        for row in annotations
        if normalized_paper_id(row.get(paper_id_key) or row.get("future_paper_id"))
    }
    output: list[dict[str, Any]] = []
    if links is not None:
        for link in links:
            paper_id = normalized_paper_id(link.get("future_paper_id") or link.get(paper_id_key))
            annotation = annotation_index.get(paper_id)
            if annotation is None:
                continue
            output.append(
                {
                    **annotation,
                    **link,
                    paper_id_key: paper_id,
                    task_key: link.get(task_key) or link.get("seed_id"),
                }
            )
        return output

    for annotation in annotations:
        task_ids = annotation.get("source_seed_ids")
        if not isinstance(task_ids, list):
            task_ids = [annotation.get(task_key)]
        fields = annotation.get("primary_fields")
        fallback_field = fields[0] if isinstance(fields, list) and fields else annotation.get(field_key)
        for task_id in task_ids:
            if not safe_text(task_id):
                continue
            output.append(
                {
                    **annotation,
                    task_key: task_id,
                    field_key: fallback_field,
                    seed_year_key: annotation.get(seed_year_key),
                    paper_id_key: normalized_paper_id(
                        annotation.get(paper_id_key) or annotation.get("future_paper_id")
                    ),
                }
            )
    return output


def integer(row: dict[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        try:
            value = row.get(key)
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def build_frontiers(
    paper_rows: Iterable[dict[str, Any]],
    *,
    field_key: str,
    year_key: str,
    keyword_key: str,
    top_fraction: float,
    min_size: int,
    excluded_paper_ids: set[str],
    paper_id_key: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    counters: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    paper_counts: Counter[tuple[str, int]] = Counter()
    for row in paper_rows:
        paper_id = safe_text(row.get(paper_id_key))
        if paper_id and paper_id in excluded_paper_ids:
            continue
        field = safe_text(row.get(field_key))
        year = integer(row, [year_key, "year"])
        terms = keyword_set(row.get(keyword_key))
        if not field or year is None or not terms:
            continue
        key = (field, year)
        paper_counts[key] += 1
        counters[key].update(terms)

    output: dict[tuple[str, int], dict[str, Any]] = {}
    for key, counter in counters.items():
        size = min(len(counter), max(min_size, math.ceil(len(counter) * top_fraction)))
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        terms = {term for term, _ in ranked[:size]}
        output[key] = {
            "terms": terms,
            "n_terms": len(terms),
            "n_candidate_terms": len(counter),
            "n_source_papers": int(paper_counts[key]),
        }
    return output


def build_records(
    ideas: Sequence[dict[str, Any]],
    followons: Sequence[dict[str, Any]],
    frontiers: dict[tuple[str, int], dict[str, Any]],
    *,
    field_key: str,
    seed_year_key: str,
    task_key: str,
    agent_key: str,
    model_key: str,
    keyword_key: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    followon_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in followons:
        task_id = safe_text(row.get(task_key))
        if task_id:
            followon_by_task[task_id].append(row)

    groups: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    seen_runs: set[str] = set()
    for row_number, idea in enumerate(ideas):
        run_id = safe_text(idea.get("run_id")) or f"__row_{row_number}"
        if run_id in seen_runs:
            skipped["duplicate_idea"] += 1
            continue
        seen_runs.add(run_id)
        task_id = safe_text(idea.get(task_key))
        linked = followon_by_task.get(task_id, [])
        if not linked:
            skipped["no_followon"] += 1
            continue
        field = safe_text(idea.get(field_key)) or safe_text(linked[0].get(field_key))
        seed_year = integer(idea, [seed_year_key, "seed_year", "year"])
        if seed_year is None:
            seed_year = integer(linked[0], [seed_year_key, "seed_year"])
        if not field or seed_year is None:
            skipped["missing_field_or_year"] += 1
            continue
        frontier = frontiers.get((field, seed_year + 1))
        if not frontier:
            skipped["missing_frontier"] += 1
            continue
        idea_terms = keyword_set(idea.get(keyword_key))
        if not idea_terms:
            skipped["missing_idea_keywords"] += 1
            continue
        agent = safe_text(idea.get(agent_key)) or "unknown"
        model = safe_text(idea.get(model_key)) or "unknown"
        key = (field, seed_year, agent, model)
        group = groups.setdefault(
            key,
            {
                "field": field,
                "seed_year": seed_year,
                "frontier_year": seed_year + 1,
                "agent": agent,
                "model": model,
                "idea_keywords": set(),
                "followon_keywords": set(),
                "tasks": set(),
                "runs": set(),
                "followon_papers": set(),
            },
        )
        group["idea_keywords"].update(idea_terms)
        group["runs"].add(run_id)
        if task_id not in group["tasks"]:
            group["tasks"].add(task_id)
            for followon in linked:
                group["followon_keywords"].update(keyword_set(followon.get(keyword_key)))
                paper_id = safe_text(followon.get("paper_id"))
                if paper_id:
                    group["followon_papers"].add(paper_id)

    records: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        frontier = frontiers[(group["field"], group["frontier_year"])]
        terms = frontier["terms"]
        idea_terms = group["idea_keywords"]
        followon_terms = group["followon_keywords"]
        if not followon_terms:
            skipped["missing_followon_keywords"] += 1
            continue
        idea_hits = idea_terms & terms
        followon_hits = followon_terms & terms
        records.append(
            {
                "field": group["field"],
                "seed_year": group["seed_year"],
                "frontier_year": group["frontier_year"],
                "agent": group["agent"],
                "model": group["model"],
                "n_tasks": len(group["tasks"]),
                "n_ideas": len(group["runs"]),
                "n_followon_papers": len(group["followon_papers"]),
                "n_frontier_keywords": frontier["n_terms"],
                "n_frontier_source_papers": frontier["n_source_papers"],
                "n_idea_keywords": len(idea_terms),
                "n_followon_keywords": len(followon_terms),
                "idea_frontier_hits": len(idea_hits),
                "followon_frontier_hits": len(followon_hits),
                "idea_frontier_coverage": len(idea_hits) / len(terms),
                "followon_frontier_coverage": len(followon_hits) / len(terms),
                "difference": (len(idea_hits) - len(followon_hits)) / len(terms),
                "idea_keyword_share": len(idea_hits) / len(idea_terms),
                "followon_keyword_share": len(followon_hits) / len(followon_terms),
            }
        )
    return records, dict(skipped)


def summarize(rows: Sequence[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_groups": 0}

    def mean(sample: Sequence[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in sample]))

    return {
        "n_groups": len(rows),
        "n_tasks": int(sum(int(row["n_tasks"]) for row in rows)),
        "n_ideas": int(sum(int(row["n_ideas"]) for row in rows)),
        "idea_mean": mean(rows, "idea_frontier_coverage"),
        "human_mean": mean(rows, "followon_frontier_coverage"),
        "difference": mean(rows, "difference"),
        "idea_ci95": bootstrap_ci(
            rows,
            lambda sample: mean(sample, "idea_frontier_coverage"),
            repetitions=repetitions,
            seed=seed,
        ),
        "human_ci95": bootstrap_ci(
            rows,
            lambda sample: mean(sample, "followon_frontier_coverage"),
            repetitions=repetitions,
            seed=seed + 1,
        ),
        "difference_ci95": bootstrap_ci(
            rows,
            lambda sample: mean(sample, "difference"),
            repetitions=repetitions,
            seed=seed + 2,
        ),
    }


def grouped_summaries(records: Sequence[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    output = {"overall": summarize(records, repetitions, seed)}
    for dim_pos, field in enumerate(("agent", "model", "field", "seed_year")):
        groups: dict[str, Any] = {}
        values = sorted({safe_text(row.get(field)) for row in records})
        for group_pos, value in enumerate(values):
            rows = [row for row in records if safe_text(row.get(field)) == value]
            groups[value] = summarize(
                rows, repetitions, seed + 10_000 * (dim_pos + 1) + group_pos
            )
        output[f"by_{field}"] = groups
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea-annotations", type=Path, required=True)
    parser.add_argument("--human-paper-annotations", type=Path, required=True)
    parser.add_argument("--followon-annotations", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, default=None)
    parser.add_argument("--followon-links", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--field-key", default="primary_field")
    parser.add_argument("--paper-year-key", default="year")
    parser.add_argument("--seed-year-key", default="seed_year")
    parser.add_argument("--task-key", default="task_id")
    parser.add_argument("--agent-key", default="agent")
    parser.add_argument("--model-key", default="model")
    parser.add_argument("--keyword-key", default="keywords")
    parser.add_argument("--paper-id-key", default="paper_id")
    parser.add_argument("--frontier-top-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-frontier-size", type=int, default=1)
    parser.add_argument("--include-followons-in-frontier", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ideas = list(read_jsonl(args.idea_annotations))
    papers = list(read_jsonl(args.human_paper_annotations))
    followon_annotations = list(read_jsonl(args.followon_annotations))
    if args.canonical_root is not None:
        papers = merge_paper_metadata(
            papers, canonical_paper_metadata(args.canonical_root), args.paper_id_key
        )
    followons = join_followon_annotations(
        followon_annotations,
        list(read_jsonl(args.followon_links)) if args.followon_links else None,
        task_key=args.task_key,
        field_key=args.field_key,
        seed_year_key=args.seed_year_key,
        paper_id_key=args.paper_id_key,
    )
    require_fields(ideas, [args.task_key, args.keyword_key], str(args.idea_annotations))
    require_fields(papers, [args.field_key, args.paper_year_key, args.keyword_key], str(args.human_paper_annotations))
    require_fields(followons, [args.task_key, args.keyword_key], str(args.followon_annotations))
    excluded = set()
    if not args.include_followons_in_frontier:
        excluded = {safe_text(row.get(args.paper_id_key)) for row in followons if safe_text(row.get(args.paper_id_key))}
    frontiers = build_frontiers(
        papers,
        field_key=args.field_key,
        year_key=args.paper_year_key,
        keyword_key=args.keyword_key,
        top_fraction=args.frontier_top_fraction,
        min_size=args.minimum_frontier_size,
        excluded_paper_ids=excluded,
        paper_id_key=args.paper_id_key,
    )
    records, skipped = build_records(
        ideas,
        followons,
        frontiers,
        field_key=args.field_key,
        seed_year_key=args.seed_year_key,
        task_key=args.task_key,
        agent_key=args.agent_key,
        model_key=args.model_key,
        keyword_key=args.keyword_key,
    )
    output = {
        "measure": "frontier_alignment",
        "definition": "Share of next-year field-frontier keywords covered by a grouped keyword union.",
        "frontier": {
            "top_fraction": args.frontier_top_fraction,
            "minimum_size": args.minimum_frontier_size,
            "followons_excluded": not args.include_followons_in_frontier,
            "n_field_year_frontiers": len(frontiers),
        },
        "counts": {
            "idea_rows": len(ideas),
            "human_paper_rows": len(papers),
            "followon_rows": len(followons),
            "group_records": len(records),
            "skipped": skipped,
        },
        "summaries": grouped_summaries(
            records, args.bootstrap_repetitions, args.seed
        ),
    }
    write_jsonl(args.out_dir / "frontier_alignment_records.jsonl", records)
    write_json(args.out_dir / "frontier_alignment_summary.json", output)
    print(json.dumps(output["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
