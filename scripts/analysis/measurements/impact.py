#!/usr/bin/env python3
"""Estimate potential impact from local historical citation neighborhoods.

Human citation impact is log1p(citations) centered within research area and
publication year. Each generated idea receives the mean centered score of its
k nearest historical human papers in the same area. Follow-on papers are scored
directly against the same area/year citation baseline. Comparisons are paired
at the seed-task level.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote

import numpy as np

from .common import (
    bootstrap_ci,
    deduplicate_aligned,
    load_aligned_embeddings,
    load_rows,
    normal_ci,
    require_fields,
    safe_text,
    year_from_identifier,
    write_json,
    write_jsonl,
)


def integer(row: dict[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        try:
            value = row.get(key)
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def number(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value >= 0 else None


def paper_id(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    return text if text.startswith("CorpusId:") else f"CorpusId:{text}"


def canonical_human_metadata(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = state.get("source") or {}
        for paper in state.get("memory_papers") or []:
            identifier = paper_id(paper.get("paper_id") or paper.get("corpusid"))
            year = integer(paper, ["year"])
            if not identifier or year is None:
                continue
            candidate = {
                "paper_id": identifier,
                "context_id": paper.get("context_id") or source.get("context_id"),
                "primary_field": paper.get("primary_field") or source.get("primary_field"),
                "year": year,
                "citation_count": paper.get("citation_count"),
            }
            old = output.get(identifier)
            if old is None or year < int(old["year"]):
                output[identifier] = candidate
    return output


def merge_human_metadata(
    rows: Sequence[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        identifier = paper_id(row.get("paper_id"))
        merged = dict(metadata.get(identifier, {}))
        merged.update(row)
        merged["paper_id"] = identifier
        output.append(merged)
    return output


def infer_field(context_id: str) -> str:
    prefix = context_id.rsplit("_", 1)[0]
    return {
        "biology": "Biology",
        "business": "Business",
        "chemistry": "Chemistry",
        "computer_science": "Computer Science",
        "economics": "Economics",
        "engineering": "Engineering",
        "environmental_science": "Environmental Science",
        "materials_science": "Materials Science",
        "mathematics": "Mathematics",
        "medicine": "Medicine",
        "physics": "Physics",
        "sociology": "Sociology",
    }.get(prefix, prefix.replace("_", " ").title())


def discover_partition_files(root: Path, directory: str, pattern: str) -> dict[str, Path]:
    output = {}
    for path in sorted((root / directory).glob(pattern)):
        part = path.parent.name
        if part.startswith("primary_field="):
            output[unquote(part.split("=", 1)[1])] = path
    return output


def full_area_year_baselines(
    data_root: Path,
    candidate_rows: Sequence[dict[str, Any]],
    *,
    context_key: str,
    year_key: str,
    batch_size: int,
) -> dict[tuple[str, int], dict[str, float]]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("--data-root requires pyarrow") from exc

    wanted: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in candidate_rows:
        context = safe_text(row.get(context_key))
        year = integer(row, [year_key, "year"])
        if context and year is not None:
            field = safe_text(row.get("primary_field")) or infer_field(context)
            wanted[field][context].add(year)

    context_files = discover_partition_files(data_root, "paper_context", "primary_field=*/*.parquet")
    index_files = discover_partition_files(data_root, "papers_index", "primary_field=*/data_*.parquet")
    output: dict[tuple[str, int], dict[str, float]] = {}
    for field, contexts in sorted(wanted.items()):
        context_path = context_files.get(field)
        index_path = index_files.get(field)
        if context_path is None or index_path is None:
            continue
        context_values = pa.array(sorted(contexts), type=pa.string())
        years = sorted({year for values in contexts.values() for year in values})
        year_values = pa.array(years, type=pa.int32())
        members: dict[tuple[str, int], list[int]] = defaultdict(list)
        candidate_ids: set[int] = set()
        context_file = pq.ParquetFile(context_path)
        for batch in context_file.iter_batches(columns=["corpusid", "context_id", "year"], batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            mask = pc.and_(
                pc.is_in(table["context_id"], value_set=context_values),
                pc.is_in(table["year"], value_set=year_values),
            )
            if not pc.any(mask).as_py():
                continue
            data = table.filter(mask).to_pydict()
            for corpusid, context, year in zip(data["corpusid"], data["context_id"], data["year"]):
                key = (str(context), int(year))
                members[key].append(int(corpusid))
                candidate_ids.add(int(corpusid))

        citations: dict[int, float] = {}
        if candidate_ids:
            value_set = pa.array(sorted(candidate_ids), type=pa.int64())
            index_file = pq.ParquetFile(index_path)
            for batch in index_file.iter_batches(columns=["corpusid", "citationcount"], batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                mask = pc.is_in(table["corpusid"], value_set=value_set)
                if not pc.any(mask).as_py():
                    continue
                data = table.filter(mask).to_pydict()
                for corpusid, citation in zip(data["corpusid"], data["citationcount"]):
                    if citation is not None:
                        citations[int(corpusid)] = float(citation)
        for key, identifiers in members.items():
            values = [citations[identifier] for identifier in identifiers if identifier in citations]
            if values:
                logs = np.log1p(np.asarray(values, dtype=float))
                output[key] = {
                    "count": float(len(values)),
                    "log_sum": float(logs.sum()),
                    "log_mean": float(logs.mean()),
                }
    return output


def citation_baselines(
    rows: Sequence[dict[str, Any]],
    *,
    context_key: str,
    year_key: str,
    citation_key: str,
    external: dict[tuple[str, int], dict[str, float]] | None = None,
) -> tuple[dict[tuple[str, int], float], np.ndarray]:
    values = np.full(len(rows), np.nan, dtype=float)
    groups: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for index, row in enumerate(rows):
        context = safe_text(row.get(context_key))
        year = integer(row, [year_key, "year"])
        citations = number(row, citation_key)
        if context and year is not None and citations is not None:
            groups[(context, year)].append((index, float(np.log1p(citations))))
    baselines: dict[tuple[str, int], float] = {}
    for key, entries in groups.items():
        log_values = [value for _, value in entries]
        external_group = external.get(key) if external else None
        baseline = float(external_group["log_mean"]) if external_group else float(np.mean(log_values))
        baselines[key] = baseline
        total = float(external_group["log_sum"]) if external_group else float(sum(log_values))
        count = int(external_group["count"]) if external_group else len(entries)
        for index, value in entries:
            comparison = (total - value) / (count - 1) if count > 1 else baseline
            values[index] = value - comparison
    return baselines, values


def score_queries(
    query_embeddings: np.ndarray,
    query_rows: Sequence[dict[str, Any]],
    human_embeddings: np.ndarray,
    human_rows: Sequence[dict[str, Any]],
    human_scores: np.ndarray,
    *,
    context_key: str,
    seed_year_key: str,
    human_year_key: str,
    k: int,
    batch_size: int,
    row_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_context_year: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(human_rows):
        context = safe_text(row.get(context_key))
        year = integer(row, [human_year_key, "year"])
        if context and year is not None and np.isfinite(human_scores[index]):
            by_context_year[(context, year)].append(index)

    eligible_cache: dict[tuple[str, int], np.ndarray] = {}
    query_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for index, row in enumerate(query_rows):
        context = safe_text(row.get(context_key))
        year = integer(row, [seed_year_key, "seed_year", "year"])
        if year is None:
            year = year_from_identifier(row.get("task_id"))
        if not context or year is None:
            skipped["missing_context_or_year"] += 1
            continue
        query_groups[(context, year)].append(index)

    output: list[dict[str, Any]] = []
    for (context, seed_year), query_indices in sorted(query_groups.items()):
        cache_key = (context, seed_year)
        eligible = eligible_cache.get(cache_key)
        if eligible is None:
            indices = []
            for (candidate_context, candidate_year), candidate_indices in by_context_year.items():
                if candidate_context == context and candidate_year <= seed_year:
                    indices.extend(candidate_indices)
            eligible = np.asarray(sorted(set(indices)), dtype=np.int64)
            eligible_cache[cache_key] = eligible
        if len(eligible) == 0:
            skipped["no_historical_neighbors"] += len(query_indices)
            continue
        take = min(k, len(eligible))
        candidate_matrix = human_embeddings[eligible]
        candidate_scores = human_scores[eligible]
        for start in range(0, len(query_indices), batch_size):
            batch_indices = query_indices[start : start + batch_size]
            similarities = query_embeddings[batch_indices] @ candidate_matrix.T
            if take == len(eligible):
                top_positions = np.tile(np.arange(len(eligible)), (len(batch_indices), 1))
            else:
                top_positions = np.argpartition(similarities, -take, axis=1)[:, -take:]
            for row_position, idea_index in enumerate(batch_indices):
                positions = top_positions[row_position]
                row = query_rows[idea_index]
                output.append(
                    {
                        "row_kind": row_kind,
                        "run_id": row.get("run_id"),
                        "task_id": row.get("task_id"),
                        "context_id": context,
                        "seed_year": seed_year,
                        "agent": row.get("agent"),
                        "model": row.get("model"),
                        "n_neighbors": take,
                        "mean_neighbor_similarity": float(similarities[row_position, positions].mean()),
                        "impact_score": float(candidate_scores[positions].mean()),
                    }
                )
    return output, dict(skipped)


def expand_followon_embeddings(
    embeddings: np.ndarray,
    rows: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]],
    task_key: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lookup = {
        paper_id(row.get("paper_id") or row.get("future_paper_id")): index
        for index, row in enumerate(rows)
        if paper_id(row.get("paper_id") or row.get("future_paper_id"))
    }
    indices: list[int] = []
    output_rows: list[dict[str, Any]] = []
    for link in links:
        identifier = paper_id(link.get("future_paper_id") or link.get("paper_id"))
        index = lookup.get(identifier)
        task_id = safe_text(link.get(task_key) or link.get("seed_id"))
        if index is None or not task_id:
            continue
        indices.append(index)
        output_rows.append(
            {
                **rows[index],
                **link,
                task_key: task_id,
                "paper_id": identifier,
            }
        )
    return embeddings[np.asarray(indices, dtype=np.int64)], output_rows


def score_followons(
    rows: Sequence[dict[str, Any]],
    baselines: dict[tuple[str, int], float],
    *,
    context_key: str,
    year_key: str,
    citation_key: str,
    task_key: str,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    skipped: dict[str, int] = defaultdict(int)
    for row_number, row in enumerate(rows):
        task_id = safe_text(row.get(task_key))
        paper_id = safe_text(row.get("paper_id")) or f"__row_{row_number}"
        if not task_id or (task_id, paper_id) in seen:
            skipped["missing_or_duplicate"] += 1
            continue
        seen.add((task_id, paper_id))
        context = safe_text(row.get(context_key))
        year = integer(row, [year_key, "year", "future_year"])
        citations = number(row, citation_key)
        if citations is None:
            citations = number(row, "future_citationcount")
        baseline = baselines.get((context, year)) if year is not None else None
        if not context or year is None or citations is None or baseline is None:
            skipped["missing_baseline_or_citations"] += 1
            continue
        by_task[task_id].append(float(np.log1p(citations) - baseline))
    return dict(by_task), dict(skipped)


def task_records(
    idea_scores: Sequence[dict[str, Any]],
    human_by_task: dict[str, list[float]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    ideas_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in idea_scores:
        task_id = safe_text(row.get("task_id"))
        if task_id in human_by_task:
            ideas_by_task[task_id].append(row)

    output: list[dict[str, Any]] = []
    for task_id, rows in sorted(ideas_by_task.items()):
        human_scores = human_by_task[task_id]
        dimensions: list[tuple[str, str, list[dict[str, Any]]]] = [("pooled", "all", rows)]
        for field in group_fields:
            values = sorted({safe_text(row.get(field)) for row in rows if safe_text(row.get(field))})
            for value in values:
                dimensions.append((field, value, [row for row in rows if safe_text(row.get(field)) == value]))
        for dimension, group, subgroup in dimensions:
            ai_mean = float(np.mean([row["impact_score"] for row in subgroup]))
            human_mean = float(np.mean(human_scores))
            output.append(
                {
                    "task_id": task_id,
                    "seed_year": subgroup[0].get("seed_year"),
                    "dimension": dimension,
                    "group": group,
                    "n_ai": len(subgroup),
                    "n_human": len(human_scores),
                    "ai_impact": ai_mean,
                    "human_impact": human_mean,
                    "difference": ai_mean - human_mean,
                }
            )
    return output


def summarize(rows: Sequence[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_tasks": 0}

    def mean(sample: Sequence[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in sample]))

    return {
        "n_tasks": len(rows),
        "n_ai": int(sum(int(row["n_ai"]) for row in rows)),
        "n_human": int(sum(int(row["n_human"]) for row in rows)),
        "ai_mean": mean(rows, "ai_impact"),
        "human_mean": mean(rows, "human_impact"),
        "difference": mean(rows, "difference"),
        "ai_ci95": bootstrap_ci(
            rows, lambda sample: mean(sample, "ai_impact"), repetitions=repetitions, seed=seed
        ),
        "human_ci95": bootstrap_ci(
            rows,
            lambda sample: mean(sample, "human_impact"),
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
    return output


def unpaired_summary(
    ai_rows: Sequence[dict[str, Any]], human_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    ai = np.asarray([float(row["impact_score"]) for row in ai_rows], dtype=float)
    human = np.asarray([float(row["impact_score"]) for row in human_rows], dtype=float)
    if len(ai) == 0 or len(human) == 0:
        return {"n_ai": len(ai), "n_human": len(human)}
    difference = float(ai.mean() - human.mean())
    if len(ai) > 1 and len(human) > 1:
        se = float(np.sqrt(ai.var(ddof=1) / len(ai) + human.var(ddof=1) / len(human)))
        difference_ci = [difference - 1.96 * se, difference + 1.96 * se]
    else:
        difference_ci = [None, None]
    return {
        "n_ai": len(ai),
        "n_human": len(human),
        "ai_mean": float(ai.mean()),
        "human_mean": float(human.mean()),
        "difference": difference,
        "ai_ci95": normal_ci(ai),
        "human_ci95": normal_ci(human),
        "difference_ci95": difference_ci,
    }


def idea_level_summaries(
    idea_scores: Sequence[dict[str, Any]], followon_scores: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "pooled": {"all": unpaired_summary(idea_scores, followon_scores)}
    }
    for field in ("agent", "model"):
        output[field] = {}
        values = sorted({safe_text(row.get(field)) for row in idea_scores if safe_text(row.get(field))})
        for value in values:
            ai_group = [row for row in idea_scores if safe_text(row.get(field)) == value]
            task_ids = {safe_text(row.get("task_id")) for row in ai_group}
            human_group = [row for row in followon_scores if safe_text(row.get("task_id")) in task_ids]
            output[field][value] = unpaired_summary(ai_group, human_group)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea-embeddings", type=Path, required=True)
    parser.add_argument("--idea-meta", type=Path, required=True)
    parser.add_argument("--human-embeddings", type=Path, required=True)
    parser.add_argument("--human-meta", type=Path, required=True)
    parser.add_argument("--followon-embeddings", type=Path, required=True)
    parser.add_argument("--followon-meta", type=Path, required=True)
    parser.add_argument("--followon-links", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--parquet-batch-size", type=int, default=1_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-key", default="context_id")
    parser.add_argument("--seed-year-key", default="seed_year")
    parser.add_argument("--human-year-key", default="year")
    parser.add_argument("--followon-year-key", default="year")
    parser.add_argument("--citation-key", default="citation_count")
    parser.add_argument("--task-key", default="task_id")
    parser.add_argument("--group-fields", nargs="+", default=["agent", "model"])
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")
    idea_embeddings, idea_rows = load_aligned_embeddings(args.idea_embeddings, args.idea_meta)
    human_embeddings, human_rows = load_aligned_embeddings(args.human_embeddings, args.human_meta)
    followon_embeddings, followon_embedding_rows = load_aligned_embeddings(
        args.followon_embeddings, args.followon_meta
    )
    followon_links = load_rows(args.followon_links)
    followon_embeddings, followon_rows = expand_followon_embeddings(
        followon_embeddings, followon_embedding_rows, followon_links, args.task_key
    )
    if args.canonical_root is not None:
        human_rows = merge_human_metadata(
            human_rows, canonical_human_metadata(args.canonical_root)
        )
    require_fields(idea_rows, [args.context_key, args.task_key], str(args.idea_meta))
    require_fields(human_rows, [args.context_key, args.human_year_key, args.citation_key], str(args.human_meta))
    require_fields(followon_rows, [args.context_key, args.task_key], str(args.followon_links))
    idea_embeddings, idea_rows, idea_duplicates = deduplicate_aligned(
        idea_embeddings,
        idea_rows,
        ("run_id",) if any(safe_text(row.get("run_id")) for row in idea_rows) else (args.task_key, "agent", "model"),
    )
    external_baselines = None
    if args.data_root is not None:
        external_baselines = full_area_year_baselines(
            args.data_root,
            human_rows,
            context_key=args.context_key,
            year_key=args.human_year_key,
            batch_size=args.parquet_batch_size,
        )
    baselines, human_scores = citation_baselines(
        human_rows,
        context_key=args.context_key,
        year_key=args.human_year_key,
        citation_key=args.citation_key,
        external=external_baselines,
    )
    idea_scores, idea_skipped = score_queries(
        idea_embeddings,
        idea_rows,
        human_embeddings,
        human_rows,
        human_scores,
        context_key=args.context_key,
        seed_year_key=args.seed_year_key,
        human_year_key=args.human_year_key,
        k=args.neighbors,
        batch_size=args.batch_size,
        row_kind="ai_idea",
    )
    followon_scores, followon_skipped = score_queries(
        followon_embeddings,
        followon_rows,
        human_embeddings,
        human_rows,
        human_scores,
        context_key=args.context_key,
        seed_year_key=args.seed_year_key,
        human_year_key=args.human_year_key,
        k=args.neighbors,
        batch_size=args.batch_size,
        row_kind="human_followon",
    )
    human_by_task: dict[str, list[float]] = defaultdict(list)
    for row in followon_scores:
        task_id = safe_text(row.get("task_id"))
        if task_id:
            human_by_task[task_id].append(float(row["impact_score"]))
    records = task_records(idea_scores, human_by_task, args.group_fields)
    output = {
        "measure": "potential_impact",
        "definition": "Mean within-area historical citation residual among the k nearest semantic neighbors, applied to both ideas and follow-on papers.",
        "citation_normalization": "log1p(citations) minus the mean log1p(citations) in the same research area and publication year.",
        "aggregation": "The primary summaries compare idea-level scores with follow-on scores on the corresponding task subset. Paired task-level summaries are reported separately.",
        "parameters": {"neighbors": args.neighbors, "historical_year_rule": "paper_year <= seed_year"},
        "counts": {
            "idea_rows": len(idea_rows),
            "human_landscape_rows": len(human_rows),
            "followon_rows": len(followon_rows),
            "idea_scores": len(idea_scores),
            "followon_scores": len(followon_scores),
            "task_records": len(records),
            "idea_duplicates_dropped": idea_duplicates,
            "idea_skipped": idea_skipped,
            "followon_skipped": followon_skipped,
        },
        "summaries": idea_level_summaries(idea_scores, followon_scores),
        "task_level_summaries": grouped_summaries(
            records, args.bootstrap_repetitions, args.seed
        ),
    }
    write_jsonl(args.out_dir / "potential_impact_idea_scores.jsonl", idea_scores)
    write_jsonl(args.out_dir / "potential_impact_followon_scores.jsonl", followon_scores)
    write_jsonl(args.out_dir / "potential_impact_records.jsonl", records)
    write_json(args.out_dir / "potential_impact_summary.json", output)
    print(json.dumps(output["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
