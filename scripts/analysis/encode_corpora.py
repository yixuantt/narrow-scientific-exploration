#!/usr/bin/env python3
"""
encode_corpora.py
=================

Encode every text in a canonical-corpus JSONL (output of
`extract_modules.py --build-from ...` or directly from `extract_modules.py`'s
input format) using the Qwen3-Embedding-4B (or any HF) model and save:

    <out-dir>/<corpus_name>.embeddings.npy        float32 (N, D), L2-normalized
    <out-dir>/<corpus_name>.meta.jsonl            N lines aligned to .npy

The embeddings are L2-normalized so cosine == dot product downstream. A
sidecar manifest `<out-dir>/encode_manifest.json` records (model, dim, sha1)
per corpus to make incremental re-runs safe.

Usage
-----

  python -m scripts.analysis.encode_corpora \
      --inputs results/v1/corpora/generated.jsonl \
               results/v1/corpora/inputs.jsonl \
               results/v1/corpora/future_citers.jsonl \
               results/v1/corpora/corpus.jsonl \
      --out-dir results/v1/embeddings \
      --embedding-model Qwen/Qwen3-Embedding-4B \
      --batch-size 32

If `<out-dir>/<name>.embeddings.npy` already exists with matching row count and
the same model name in the manifest, that corpus is skipped (use --overwrite
to force).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common import analysis_utils as A


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype("float32")


def encode_one(input_path: Path, out_dir: Path, *,
               embedding_model: str, batch_size: int,
               max_text_chars: int, overwrite: bool,
               manifest: Dict[str, Any]) -> Dict[str, Any]:
    name = input_path.stem  # e.g. "generated"
    out_emb = out_dir / f"{name}.embeddings.npy"
    out_meta = out_dir / f"{name}.meta.jsonl"

    rows = list(read_jsonl(input_path))
    n = len(rows)
    if n == 0:
        print(f"[encode] {input_path}: empty -- skipped", flush=True)
        return {"name": name, "n": 0, "skipped": True}

    if (not overwrite
        and out_emb.exists()
        and manifest.get(name, {}).get("embedding_model") == embedding_model
        and manifest.get(name, {}).get("n") == n):
        print(f"[encode] {name}: cached ({n} rows, model match) -- skipped", flush=True)
        return {"name": name, "n": n, "skipped": True}

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for r in rows:
        text = r.get("text") or ""
        text = text[:max_text_chars]
        if not text.strip():
            continue
        texts.append(text)
        metas.append({
            "doc_id": r.get("doc_id"),
            "doc_kind": r.get("doc_kind"),
            **(r.get("meta") or {}),
        })
    print(f"[encode] {name}: {len(texts)} texts -> {embedding_model}", flush=True)
    t0 = time.time()
    emb = A.embed_texts(texts, embedding_model, batch_size).astype("float32")
    emb = l2_normalize(emb)
    dt = time.time() - t0
    print(f"[encode] {name}: shape={emb.shape}, took {dt/60:.1f} min "
          f"({len(texts)/max(dt,1e-6):.1f} texts/sec)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, emb)
    with out_meta.open("w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    return {"name": name, "n": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "embedding_model": embedding_model,
            "out_emb": str(out_emb),
            "out_meta": str(out_meta),
            "skipped": False}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="Canonical-corpus JSONL files (each row has 'text').")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "v1" / "embeddings")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-text-chars", type=int, default=6000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "encode_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    summary: List[Dict[str, Any]] = []
    for ip in args.inputs:
        res = encode_one(ip, args.out_dir,
                          embedding_model=args.embedding_model,
                          batch_size=args.batch_size,
                          max_text_chars=args.max_text_chars,
                          overwrite=args.overwrite,
                          manifest=manifest)
        summary.append(res)
        if not res.get("skipped"):
            manifest[res["name"]] = {
                "embedding_model": res["embedding_model"],
                "n": res["n"],
                "dim": res.get("dim"),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
