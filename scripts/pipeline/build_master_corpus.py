#!/usr/bin/env python3
"""
Build a unified master corpus JSONL from per-paper JSON files.

The input is a venue/year directory tree containing one JSON file per paper,
with title, abstract, venue, year, and identifier fields.

The output JSONL schema is understood by scripts/pipeline/build_canonical_states.py (via its --input argument).

Usage examples:
  # ICLR only (single venue)
  python scripts/pipeline/build_master_corpus.py --input-root corpus/iclr \
      --output-jsonl corpus/iclr_master.jsonl

  # NeurIPS (all years merged into one file)
  python scripts/pipeline/build_master_corpus.py --input-root corpus/neurips \
      --output-jsonl corpus/neurips_master.jsonl

  # ICML
  python scripts/pipeline/build_master_corpus.py --input-root corpus/icml \
      --output-jsonl corpus/icml_master.jsonl

  # Combined across venues (e.g. NeurIPS + ICML + ICLR)
  python scripts/pipeline/build_master_corpus.py \
      --input-root corpus/iclr corpus/neurips corpus/icml \
      --output-jsonl corpus/all_master.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-paper JSON files from one or more corpus directories into "
            "a single master JSONL and a stats JSON summary."
        )
    )
    parser.add_argument(
        "--input-root",
        nargs="+",
        default=["corpus/iclr"],
        help=(
            "One or more input root directories containing per-year subdirectories "
            "with individual paper JSON files. Default: corpus/iclr"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL path (one record per line).",
    )
    parser.add_argument(
        "--output-stats",
        default=None,
        help=(
            "Output stats JSON path. Defaults to <output-jsonl>.stats.json "
            "in the same directory."
        ),
    )
    parser.add_argument(
        "--dedupe-by",
        choices=("forum_id", "title"),
        default="forum_id",
        help=(
            "Primary deduplication key. Use 'title' when mixing venues that "
            "may share the same paper via different IDs. Default: forum_id"
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="If set, only include papers from these years.",
    )
    return parser.parse_args()


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_master_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    forum_id = normalize_text(data.get("forum_id"))
    title = normalize_text(data.get("title"))
    abstract = normalize_text(data.get("abstract"))
    venue = normalize_text(data.get("venue"))
    paper_type = normalize_text(data.get("paper_type"))
    date = normalize_text(data.get("date"))
    year = data.get("year")
    keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []

    paper_id = forum_id or path.stem
    return {
        "paper_id": paper_id,
        "forum_id": forum_id,
        "title": title,
        "abstract": abstract,
        "venue": venue,
        "paper_type": paper_type,
        "year": year,
        "date": date,
        "keywords": keywords,
        "tl_dr": normalize_text(data.get("tl_dr")),
        "decision": normalize_text(data.get("decision")),
        "openreview_url": normalize_text(data.get("openreview_url")),
        # extra fields from S2 (None when using OpenReview source)
        "s2_paper_id": normalize_text(data.get("s2_paper_id")),
        "arxiv_id": normalize_text(data.get("arxiv_id")),
        "source_path": str(path.as_posix()),
    }


def dedupe_key_value(record: dict[str, Any], key_name: str) -> str | None:
    value = normalize_text(record.get(key_name))
    if key_name == "title" and value:
        return value.lower()
    return value


def main() -> int:
    args = parse_args()

    output_jsonl = Path(args.output_jsonl)
    output_stats = Path(args.output_stats) if args.output_stats else (
        output_jsonl.parent / (output_jsonl.stem + ".stats.json")
    )

    allowed_years: set[int] | None = set(args.years) if args.years else None

    # Collect all paper JSON files across all input roots
    all_paths: list[Path] = []
    for root_str in args.input_root:
        root = Path(root_str)
        if not root.exists():
            print(f"WARNING: input-root does not exist: {root}", flush=True)
            continue
        found = sorted(root.glob("*/*.json"))
        print(f"  {root}: found {len(found)} paper files", flush=True)
        all_paths.extend(found)

    if not all_paths:
        raise SystemExit("No paper JSON files found. Check --input-root paths.")

    records: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    venue_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    seen_keys: dict[str, Path] = {}
    duplicate_count = 0
    skipped_year = 0

    for path in all_paths:
        raw = load_raw(path)
        record = build_master_record(path, raw)

        # Year filter
        if allowed_years is not None:
            year = record.get("year")
            if not isinstance(year, int) or year not in allowed_years:
                skipped_year += 1
                continue

        # Deduplication
        key = dedupe_key_value(record, args.dedupe_by)
        if key and key in seen_keys:
            duplicate_count += 1
            continue
        if key:
            seen_keys[key] = path

        for field in ("title", "abstract", "venue", "year", "date", "paper_type"):
            if record.get(field) in (None, "", []):
                missing_counts[field] += 1

        if isinstance(record.get("year"), int):
            year_counts[record["year"]] += 1
        venue_str = normalize_text(record.get("venue")) or "unknown"
        venue_counts[venue_str[:40]] += 1
        if record.get("paper_type"):
            type_counts[str(record["paper_type"])] += 1

        records.append(record)

    records.sort(
        key=lambda item: (
            item.get("year") if isinstance(item.get("year"), int) else 0,
            item.get("paper_type") or "",
            item.get("title") or "",
        )
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "input_roots": args.input_root,
        "output_jsonl": str(output_jsonl),
        "total_input_files": len(all_paths),
        "total_records_written": len(records),
        "dedupe_by": args.dedupe_by,
        "duplicate_count": duplicate_count,
        "skipped_year_filter": skipped_year,
        "year_counts": dict(sorted(year_counts.items())),
        "paper_type_counts": dict(sorted(type_counts.items())),
        "missing_required_field_counts": dict(sorted(missing_counts.items())),
    }
    output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
