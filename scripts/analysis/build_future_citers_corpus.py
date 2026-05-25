#!/usr/bin/env python3
"""
build_future_citers_corpus.py
=============================

Materialize the "real future citers" corpus for each anchor paper as a JSONL
cache, so downstream module-extraction / embedding does not have to touch the
citation graph again.

For each canonical-state anchor, walk the citation graph (REVERSE edges:
target -> [source...] = papers that cite target), keep only citers whose year
> --analysis-year, and emit one JSONL row per (anchor, citer):

  {"paper_id": "...",
   "anchor_paper_id": "...",
   "context_id": "...",
   "year": int,
   "text": "<title>. <abstract>"}

Run with `--analysis-year 2022` to materialize all 2023+ citers, etc.; passing
`--analysis-year all` runs once per year in {2022,2023,2024,2025} and writes
year-tagged outputs:

  results/v1/corpora/future_citers.year2022.jsonl
  results/v1/corpora/future_citers.year2023.jsonl
  ...

For the analysis we typically want the *union* across all anchor years so
the module extractor sees every future citer once. Pass --merged to also write
`future_citers.jsonl` deduped by (anchor, citer).

Usage
-----

  python -m scripts.analysis.build_future_citers_corpus \
      --canonical-states-root data/canonical_states/clean_main_batch \
      --corpus-path data/DBLP-Citation-network-V18/matched_all_master_g22_25.paper_corpus.jsonl \
      --graph-edges-path data/DBLP-Citation-network-V18/matched_all_master_g22_25.paper_graph_edges.jsonl \
      --analysis-year all --merged \
      --out-dir results/v1/corpora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common import analysis_utils as A


def write_rows(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canonical-states-root", type=Path,
                   default=ROOT / "data" / "canonical_states" / "clean_main_batch")
    p.add_argument("--corpus-path", type=Path,
                   default=ROOT / "data" / "DBLP-Citation-network-V18"
                   / "matched_all_master_g22_25.paper_corpus.jsonl")
    p.add_argument("--graph-edges-path", type=Path,
                   default=ROOT / "data" / "DBLP-Citation-network-V18"
                   / "matched_all_master_g22_25.paper_graph_edges.jsonl")
    p.add_argument("--context-prefix", type=str, default="")
    p.add_argument("--max-text-chars", type=int, default=6000)
    p.add_argument("--analysis-year", type=str, default="2022",
                   help='One of {2022, 2023, 2024, 2025} or "all" to scan every '
                        'anchor year and emit one JSONL per year.')
    p.add_argument("--years", type=str, nargs="*",
                   default=["2022", "2023", "2024", "2025"],
                   help='Years to scan when --analysis-year all.')
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "v1" / "corpora")
    p.add_argument("--merged", action="store_true",
                   help="Also write a deduped union across all years -> "
                        "future_citers.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] loading {args.corpus_path}", flush=True)
    corpus_by_id = A.load_corpus_index(args.corpus_path)
    print(f"[corpus] indexed {len(corpus_by_id)} papers", flush=True)

    if args.analysis_year != "all":
        years = [args.analysis_year]
    else:
        years = list(args.years)

    union_seen: set = set()
    union_rows: List[dict] = []

    for year in years:
        print(f"\n=== analysis_year = {year} ===", flush=True)
        rows = A.build_future_real_rows(
            corpus_by_id,
            args.canonical_states_root,
            year,
            args.max_text_chars,
            graph_edges_path=args.graph_edges_path,
            context_prefix=args.context_prefix,
        )
        out_path = args.out_dir / f"future_citers.year{year}.jsonl"
        n = write_rows(rows, out_path)
        print(f"[write] {n} rows -> {out_path}", flush=True)

        if args.merged:
            for r in rows:
                key = (r.get("anchor_paper_id"), r.get("paper_id"))
                if key in union_seen:
                    continue
                union_seen.add(key)
                union_rows.append(r)

    if args.merged:
        out_path = args.out_dir / "future_citers.jsonl"
        n = write_rows(union_rows, out_path)
        print(f"\n[write merged] {n} unique (anchor, citer) rows -> {out_path}",
              flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
