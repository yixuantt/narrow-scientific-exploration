#!/usr/bin/env python3
"""Stream DBLP JSONL and keep rows whose (title, year) match corpus/all_master.jsonl.

Matching uses normalized title plus integer year. Optional DOI join is used when
master rows include ``doi``.

Large DBLP files: use ``--workers`` > 1 to scan disjoint byte ranges in parallel (line-aligned).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]  # repository root

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--master",
        type=Path,
        default=ROOT / "corpus" / "all_master.jsonl",
        help="Master corpus JSONL (default: corpus/all_master.jsonl).",
    )
    p.add_argument(
        "--dblp",
        type=Path,
        default=ROOT
        / "data"
        / "DBLP-Citation-network-V18"
        / "DBLP-Citation-network-V18.jsonl",
        help="Full DBLP citation-network JSONL.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data"
        / "DBLP-Citation-network-V18"
        / "DBLP-Citation-network-V18.matched_all_master.jsonl",
        help="Output JSONL: DBLP records that match master.",
    )
    p.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON stats path (default: <output>.stats.json).",
    )
    p.add_argument(
        "--year-window",
        type=int,
        default=0,
        help="Allow DBLP year within ± this window vs master year (default: 0 = exact).",
    )
    p.add_argument(
        "--max-dblp-lines",
        type=int,
        default=None,
        help="Debug: only first process, stop after this many DBLP lines (forces --workers 1).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
        help="Parallel workers over DBLP byte ranges (default: min(8, CPUs); use 1 for single-process).",
    )
    return p.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_title_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text).lower())


def master_title_year_key(title: Any, year: Any, year_window: int) -> Set[Tuple[str, int]]:
    tk = normalize_title_key(title)
    if not tk:
        return set()
    try:
        y0 = int(year)
    except (TypeError, ValueError):
        return set()
    return {(tk, y0 + d) for d in range(-year_window, year_window + 1)}


def normalize_doi(raw: Any) -> str:
    s = normalize_text(raw).lower()
    if not s:
        return ""
    if s.startswith("http://doi.org/"):
        s = "https://" + s[7:]
    if s.startswith("https://doi.org/"):
        s = s[len("https://doi.org/") :]
    elif s.startswith("doi:"):
        s = s[4:].strip()
    return s.strip()


def load_master_keys(path: Path, year_window: int) -> tuple[Set[Tuple[str, int]], Set[str]]:
    keys: Set[Tuple[str, int]] = set()
    dois: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.update(master_title_year_key(row.get("title"), row.get("year"), year_window))
            d = normalize_doi(row.get("doi"))
            if d:
                dois.add(d)
    return keys, dois


def dblp_row_keys(row: Dict[str, Any], year_window: int) -> Set[Tuple[str, int]]:
    tk = normalize_title_key(row.get("title"))
    if not tk:
        return set()
    try:
        y0 = int(row.get("year"))
    except (TypeError, ValueError):
        return set()
    return {(tk, y0 + d) for d in range(-year_window, year_window + 1)}


def _line_iter_binary(path: Path, start_byte: int, end_byte: int, file_size: int) -> Iterable[bytes]:
    """Read complete lines from [start_byte, end_byte); last chunk reads until EOF."""
    is_last = end_byte >= file_size
    with path.open("rb") as fh:
        fh.seek(start_byte)
        if start_byte > 0:
            fh.readline()
        while True:
            if not is_last and fh.tell() >= end_byte:
                break
            line_b = fh.readline()
            if not line_b:
                break
            yield line_b


def _iter_lines_text(it: Iterable[bytes]) -> Iterable[str]:
    for b in it:
        yield b.decode("utf-8", errors="replace")


# Globals set in worker initializer (fork-friendly).
_TY_KEYS: Set[Tuple[str, int]] = set()
_MASTER_DOIS: Set[str] = set()
_YEAR_W: int = 0


def _worker_init(ty_keys: Set[Tuple[str, int]], master_dois: Set[str], year_window: int) -> None:
    global _TY_KEYS, _MASTER_DOIS, _YEAR_W
    _TY_KEYS = ty_keys
    _MASTER_DOIS = master_dois
    _YEAR_W = year_window


def _worker_run(
    task: Tuple[str, int, int, int, str, int],
) -> Dict[str, Any]:
    dblp_s, start_b, end_b, file_size, out_part_s, wid = task
    dblp_path = Path(dblp_s)
    out_part = Path(out_part_s)

    raw_it = _line_iter_binary(dblp_path, start_b, end_b, file_size)
    lines = _iter_lines_text(raw_it)

    chunk_span = max(1, (end_b - start_b) if end_b > start_b else 1)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=chunk_span,
            unit="B",
            desc=f"w{wid}",
            position=wid,
            leave=True,
            mininterval=0.3,
        )

    scanned = 0
    matched = 0
    by_doi = 0
    by_title = 0
    emitted_ids: Set[str] = set()

    out_part.parent.mkdir(parents=True, exist_ok=True)
    with out_part.open("w", encoding="utf-8") as fout:
        for line in lines:
            enc = line.encode("utf-8", errors="replace")
            if pbar is not None:
                pbar.update(len(enc))
            scanned += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = normalize_text(row.get("id"))
            d = normalize_doi(row.get("doi"))
            kset = dblp_row_keys(row, _YEAR_W)

            hit = False
            if d and d in _MASTER_DOIS:
                hit = True
                by_doi += 1
            elif kset & _TY_KEYS:
                hit = True
                by_title += 1

            if not hit:
                continue
            if rid and rid in emitted_ids:
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if rid:
                emitted_ids.add(rid)
            matched += 1

    if pbar is not None:
        pbar.close()

    return {
        "worker_id": wid,
        "scanned": scanned,
        "matched": matched,
        "by_doi": by_doi,
        "by_title": by_title,
        "part_path": str(out_part),
    }


def byte_ranges(file_size: int, n_chunks: int) -> List[Tuple[int, int]]:
    if n_chunks <= 1 or file_size <= 0:
        return [(0, file_size)]
    out: List[Tuple[int, int]] = []
    for i in range(n_chunks):
        start = i * file_size // n_chunks
        end = (i + 1) * file_size // n_chunks if i < n_chunks - 1 else file_size
        out.append((start, end))
    return out


def run_sequential(
    args: argparse.Namespace,
    ty_keys: Set[Tuple[str, int]],
    master_dois: Set[str],
    mk_count: int,
) -> None:
    dblp_path = args.dblp.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fin = dblp_path.open("r", encoding="utf-8")
    inner = fin
    pbar = None
    if tqdm is not None:
        inner = tqdm(fin, unit=" lines", desc="DBLP", mininterval=0.5)
        pbar = inner

    scanned = 0
    matched = 0
    by_doi = 0
    by_title = 0
    emitted_ids: Set[str] = set()

    with out_path.open("w", encoding="utf-8") as fout:
        for line in inner:
            if args.max_dblp_lines is not None and scanned >= args.max_dblp_lines:
                break
            scanned += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = normalize_text(row.get("id"))
            d = normalize_doi(row.get("doi"))
            kset = dblp_row_keys(row, args.year_window)

            hit = False
            if d and d in master_dois:
                hit = True
                by_doi += 1
            elif kset & ty_keys:
                hit = True
                by_title += 1

            if not hit:
                continue
            if rid and rid in emitted_ids:
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if rid:
                emitted_ids.add(rid)
            matched += 1

    fin.close()
    if pbar is not None:
        pbar.close()

    stats_path = args.stats_output or out_path.with_suffix(out_path.suffix + ".stats.json")
    stats: Dict[str, Any] = {
        "master_path": str(args.master.expanduser().resolve()),
        "dblp_path": str(args.dblp.expanduser().resolve()),
        "output": str(out_path),
        "year_window": args.year_window,
        "workers": 1,
        "master_title_year_key_count": mk_count,
        "master_doi_count": len(master_dois),
        "dblp_lines_scanned": scanned,
        "dblp_rows_emitted": matched,
        "match_component_doi_rows": by_doi,
        "match_component_title_year_rows": by_title,
        "note": "Per row, DOI match is preferred over title+year when master has doi.",
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    master_path = args.master.expanduser().resolve()
    dblp_path = args.dblp.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    if not master_path.is_file():
        print(f"master not found: {master_path}", file=sys.stderr)
        return 1
    if not dblp_path.is_file():
        print(f"dblp not found: {dblp_path}", file=sys.stderr)
        return 1

    print(f"[intersect] loading keys from {master_path} ...", flush=True)
    ty_keys, master_dois = load_master_keys(master_path, args.year_window)
    mk_count = len(ty_keys)
    print(
        f"[intersect] master (title,year) keys: {mk_count}, master doi keys: {len(master_dois)}",
        flush=True,
    )

    workers = max(1, args.workers)
    if args.max_dblp_lines is not None:
        workers = 1
    file_size = dblp_path.stat().st_size

    if workers == 1 or file_size < 8 * 1024 * 1024:
        if workers > 1 and file_size < 8 * 1024 * 1024:
            print("[intersect] file < 8 MiB, using single worker", flush=True)
        run_sequential(args, ty_keys, master_dois, mk_count)
        return 0

    ranges = byte_ranges(file_size, workers)
    parts: List[Path] = []
    tasks: List[Tuple[str, int, int, int, str, int]] = []
    for wid, (start_b, end_b) in enumerate(ranges):
        part = out_path.with_suffix(out_path.suffix + f".part{wid}.jsonl")
        parts.append(part)
        tasks.append((str(dblp_path), start_b, end_b, file_size, str(part), wid))

    print(
        f"[intersect] DBLP size={file_size} bytes, workers={workers}, chunks={len(tasks)}",
        flush=True,
    )

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(
        workers,
        initializer=_worker_init,
        initargs=(ty_keys, master_dois, args.year_window),
    ) as pool:
        results = pool.map(_worker_run, tasks)

    scanned = sum(r["scanned"] for r in results)
    matched = sum(r["matched"] for r in results)
    by_doi = sum(r["by_doi"] for r in results)
    by_title = sum(r["by_title"] for r in results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fout:
        for part in parts:
            if not part.is_file():
                continue
            with part.open("rb") as fin:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            part.unlink(missing_ok=True)

    stats_path = args.stats_output or out_path.with_suffix(out_path.suffix + ".stats.json")
    stats = {
        "master_path": str(master_path),
        "dblp_path": str(dblp_path),
        "output": str(out_path),
        "year_window": args.year_window,
        "workers": workers,
        "master_title_year_key_count": mk_count,
        "master_doi_count": len(master_dois),
        "dblp_lines_scanned": scanned,
        "dblp_rows_emitted": matched,
        "match_component_doi_rows": by_doi,
        "match_component_title_year_rows": by_title,
        "worker_chunks": results,
        "note": "Per row, DOI match is preferred over title+year when master has doi.",
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
