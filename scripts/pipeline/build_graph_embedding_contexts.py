#!/usr/bin/env python3
"""
Build bcsvd research contexts from the DBLP matched corpus.

Input
    data/DBLP-Citation-network-V18/DBLP-Citation-network-V18.matched_all_master.jsonl
    Each line carries a DBLP paper with fields id, year, title, references (list of DBLP ids).

Method
    bcsvd   Bibliographic coupling TF-IDF matrix reduced with TruncatedSVD.

Clustering
    hdbscan   sklearn.cluster.HDBSCAN (density, -1 = noise).

Default research-area filters
    Remove the largest non-noise cluster and keep only clusters containing
    papers from every year in the target analysis window.

Outputs
    <out>/<prefix>_bcsvd_embed.embeddings.npy
    <out>/<prefix>_bcsvd_embed.paper_ids.json
    <out>/<prefix>_bcsvd_hdbscan.paper_contexts.jsonl
    <out>/<prefix>_bcsvd_hdbscan.context_index.json
    <out>/<prefix>_bcsvd_hdbscan.meta.json

All paper_id strings are DBLP ids.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import networkx as nx

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_matched_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Load {path.name}", unit="line"):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_citation_digraph(rows: List[dict]) -> Tuple[nx.DiGraph, Set[str]]:
    """Directed graph (citing -> cited) over ALL matched DBLP ids.

    We keep every matched id as a node so references from younger papers can
    reach older (< year_min) papers too. The context embedding is computed only
    over the selected year window, but the companion graph keeps the full
    matched DBLP citation structure for downstream neighbor lookup.
    """
    ids: Set[str] = {r["id"] for r in rows if r.get("id")}
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    n_edges = 0
    for r in rows:
        src = r.get("id")
        if not src:
            continue
        for ref in (r.get("references") or []):
            if not ref or ref == src:
                continue
            if ref in ids:
                G.add_edge(src, ref)
                n_edges += 1
    print(f"[graph] directed citation graph: nodes={G.number_of_nodes()}, edges={n_edges}")
    return G, ids


def edges_within(G: nx.DiGraph, node_subset: Set[str]) -> int:
    c = 0
    for u, v in G.edges():
        if u in node_subset and v in node_subset:
            c += 1
    return c


# ---------------------------------------------------------------------------
# Bibliographic coupling + SVD
# ---------------------------------------------------------------------------

def bibliographic_coupling_matrix(subset: List[dict]):
    from scipy.sparse import csr_matrix
    ref_to_col: Dict[str, int] = {}
    rows_i: List[int] = []
    cols_j: List[int] = []
    data: List[int] = []
    for i, r in enumerate(subset):
        seen: Set[str] = set()
        for ref in (r.get("references") or []):
            if not ref or ref in seen:
                continue
            seen.add(ref)
            j = ref_to_col.get(ref)
            if j is None:
                j = len(ref_to_col)
                ref_to_col[ref] = j
            rows_i.append(i)
            cols_j.append(j)
            data.append(1)
    A = csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows_i, cols_j)),
        shape=(len(subset), len(ref_to_col)),
        dtype=np.float32,
    )
    return A, ref_to_col


def bcsvd_embeddings(subset: List[dict], svd_dim: int, seed: int) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    A, ref_to_col = bibliographic_coupling_matrix(subset)
    print(f"[bc] reference-incidence matrix: shape={A.shape}, nnz={A.nnz}, "
          f"mean_refs_per_paper={A.nnz / max(A.shape[0], 1):.2f}")
    if A.nnz == 0:
        raise RuntimeError("Bibliographic coupling matrix is empty (no references).")

    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    At = tfidf.fit_transform(A)

    n_comp = min(svd_dim, A.shape[1] - 1, A.shape[0] - 1)
    if n_comp < 2:
        raise RuntimeError(f"svd_dim too small for matrix {A.shape}")
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    X = svd.fit_transform(At).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    print(f"[bcsvd] SVD explained variance sum={float(svd.explained_variance_ratio_.sum()):.4f}")
    return X


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    h = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return h.fit_predict(X)


def filter_research_area_labels(
    subset_rows: List[dict],
    labels: np.ndarray,
    *,
    year_min: int,
    year_max: int,
    drop_largest_cluster: bool,
    require_all_years: bool,
) -> Tuple[np.ndarray, dict]:
    """Apply the manuscript's main-analysis research-area filters."""
    filtered = np.asarray(labels, dtype=int).copy()
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    cluster_years: Dict[int, Set[int]] = defaultdict(set)
    for idx, lab_raw in enumerate(filtered):
        lab = int(lab_raw)
        if lab == -1:
            continue
        cluster_members[lab].append(idx)
        year = subset_rows[idx].get("year")
        if year is not None:
            cluster_years[lab].add(int(year))

    raw_labels = sorted(cluster_members)
    drop_labels: Set[int] = set()
    largest_label = None
    if drop_largest_cluster and cluster_members:
        largest_label = max(cluster_members, key=lambda lab: len(cluster_members[lab]))
        drop_labels.add(largest_label)

    required_years = set(range(year_min, year_max + 1))
    inactive_labels: List[int] = []
    if require_all_years:
        for lab in raw_labels:
            if lab in drop_labels:
                continue
            if not required_years.issubset(cluster_years.get(lab, set())):
                inactive_labels.append(lab)
                drop_labels.add(lab)

    if drop_labels:
        for idx, lab_raw in enumerate(filtered):
            if int(lab_raw) in drop_labels:
                filtered[idx] = -1

    kept_labels = [lab for lab in raw_labels if lab not in drop_labels]
    summary = {
        "raw_num_real_clusters": len(raw_labels),
        "raw_assigned_papers": int(sum(len(cluster_members[lab]) for lab in raw_labels)),
        "drop_largest_cluster": bool(drop_largest_cluster),
        "dropped_largest_label": int(largest_label) if largest_label is not None else None,
        "dropped_largest_size": int(len(cluster_members[largest_label])) if largest_label is not None else 0,
        "require_all_years": bool(require_all_years),
        "required_years": sorted(required_years) if require_all_years else [],
        "dropped_inactive_clusters": len(inactive_labels),
        "kept_num_real_clusters": len(kept_labels),
        "kept_assigned_papers": int(sum(len(cluster_members[lab]) for lab in kept_labels)),
    }
    print(
        "[filter] raw_clusters={raw_num_real_clusters}, raw_assigned={raw_assigned_papers}, "
        "dropped_largest={dropped_largest_size}, dropped_inactive={dropped_inactive_clusters}, "
        "kept_clusters={kept_num_real_clusters}, kept_assigned={kept_assigned_papers}".format(**summary),
        flush=True,
    )
    return filtered, summary


# ---------------------------------------------------------------------------
# Context writers
# ---------------------------------------------------------------------------

def _cluster_compactness(X: np.ndarray, idxs: List[int], cap: int = 200) -> Tuple[float, float]:
    """Return (mean_intra_cosine, mean_intra_euclidean) over up to `cap` members.

    mean_intra_cosine is computed on L2-normalised copies of the rows;
    mean_intra_euclidean is computed on the raw rows. Both metrics ignore the
    diagonal.
    """
    if len(idxs) < 2:
        return 1.0, 0.0
    take = idxs[:cap]
    M = X[take].astype(np.float32, copy=True)
    n = M.shape[0]
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sim = Mn @ Mn.T
    mean_cos = float((sim.sum() - n) / (n * (n - 1)))
    # Euclidean: mean off-diag distance
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms = (M * M).sum(axis=1)
    D2 = norms[:, None] + norms[None, :] - 2.0 * (M @ M.T)
    D2 = np.clip(D2, 0.0, None)
    D = np.sqrt(D2)
    mean_euc = float((D.sum() - 0.0) / (n * (n - 1)))
    return mean_cos, mean_euc


def write_context_bundle(
    subset_rows: List[dict],
    paper_ids: List[str],
    X: np.ndarray,
    labels: np.ndarray,
    *,
    prefix: str,
    out_dir: Path,
    method: str,
    clustering: str,
    config: dict,
) -> Tuple[Path, Path, Path]:
    titles = {r["id"]: (r.get("title") or "") for r in subset_rows}
    pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    members: Dict[int, List[str]] = defaultdict(list)
    for pid, lab in zip(paper_ids, labels):
        members[int(lab)].append(pid)

    # Sort clusters by size desc (noise first if any, named separately).
    real_clusters = sorted(
        [c for c in members if c != -1],
        key=lambda c: (-len(members[c]), c),
    )
    has_noise = -1 in members
    context_prefix = f"{prefix}_{method}_{clustering}"

    contexts: Dict[str, dict] = {}
    paper_ctx_rows: List[dict] = []

    # IMPORTANT: do NOT put the internal method ("bcsvd"), clustering ("hdbscan"),
    # raw context_id ("matched_all_master_g22_25_bcsvd_hdbscan_0108"), cluster
    # index ("context 108"), or cluster-stats bookkeeping (paper counts, mean
    # intra-cosine/euclidean) into context_label. Those strings get surfaced in
    # the ideation agent's prompt (subarea / challenge / goal / keywords) and
    # the LLM echoes them back into the idea text (e.g. "we propose
    # amortized-bcsvd" or "evaluated on the bcsvd-hdbscan benchmark"), which
    # pollutes the module-provenance metrics. The context_label shown to the
    # agent is just a list of sample paper titles; the internal identifiers and
    # compactness stats live only in the context-index JSON (for provenance
    # and downstream filtering).
    def _titles_label(mem_ids: List[str]) -> str:
        sample_titles = [titles.get(p, "").strip() for p in mem_ids[:3]]
        joined = "; ".join(t for t in sample_titles if t)
        return joined or "research context"

    if has_noise:
        noise_cid = f"{context_prefix}_noise"
        mem = members[-1]
        label = _titles_label(mem)
        contexts[noise_cid] = {
            "context_id": noise_cid,
            "context_label": label,
            "member_count": len(mem),
            "member_paper_ids": mem,
            "mean_intra_cosine": float("nan"),
            "mean_intra_euclidean": float("nan"),
            # No internal identifiers; downstream (build_canonical_states.py) falls
            # back to splitting context_label when this is empty, which is fine.
            "context_seed_keywords": [],
            "small_context": True,
            "noise_cluster": True,
        }
        for p in mem:
            paper_ctx_rows.append({
                "paper_id": p,
                "title": titles.get(p, ""),
                "context_id": noise_cid,
                "context_label": label,
                "seed_eligible": False,
            })

    for order_idx, c in enumerate(real_clusters):
        cid = f"{context_prefix}_{order_idx:04d}"
        mem = members[c]
        mean_cos, mean_euc = _cluster_compactness(X, [pid_to_idx[p] for p in mem])
        # Agent-facing label = sample paper titles only. All pipeline / stats
        # metadata stays in the structured fields below, never in the label.
        label = _titles_label(mem)
        contexts[cid] = {
            "context_id": cid,
            "context_label": label,
            "member_count": len(mem),
            "member_paper_ids": mem,
            "mean_intra_cosine": mean_cos,
            "mean_intra_euclidean": mean_euc,
            # Seed keywords are populated later from DBLP/title-based keyword
            # extraction in earlier research-area construction scripts (if enabled).
            # Never leak cid/method/clustering here: see comment above.
            "context_seed_keywords": [],
            "small_context": len(mem) < 5,
            "noise_cluster": False,
        }
        for p in mem:
            paper_ctx_rows.append({
                "paper_id": p,
                "title": titles.get(p, ""),
                "context_id": cid,
                "context_label": label,
                "seed_eligible": True,
            })

    ctx_path = out_dir / f"{prefix}_{method}_{clustering}.paper_contexts.jsonl"
    idx_path = out_dir / f"{prefix}_{method}_{clustering}.context_index.json"
    meta_path = out_dir / f"{prefix}_{method}_{clustering}.meta.json"

    write_jsonl(ctx_path, paper_ctx_rows)

    size_hist = Counter(len(v) for k, v in members.items() if k != -1)
    sizes_sorted = sorted((len(v) for k, v in members.items() if k != -1), reverse=True)

    num_noise = int((labels == -1).sum())
    index_payload = {
        "contexts": contexts,
        "min_context_size": 5,
        "context_mode": "graph_embedding",
        "method": method,
        "clustering": clustering,
        "num_papers": len(paper_ids),
        "num_contexts": len(contexts),
        "num_real_clusters": len(real_clusters),
        "num_noise_papers": num_noise,
        "cluster_size_top10": sizes_sorted[:10],
        "cluster_size_min_max": [sizes_sorted[-1] if sizes_sorted else 0,
                                  sizes_sorted[0] if sizes_sorted else 0],
        "config": config,
    }
    write_json(idx_path, index_payload)
    write_json(meta_path, {
        "method": method,
        "clustering": clustering,
        "num_papers": len(paper_ids),
        "num_contexts": len(contexts),
        "num_real_clusters": len(real_clusters),
        "num_noise_papers": num_noise,
        "cluster_size_histogram": dict(sorted(size_hist.items())),
        "config": config,
        "outputs": {
            "paper_contexts": str(ctx_path),
            "context_index": str(idx_path),
        },
    })
    print(f"[write] {ctx_path.name}, {idx_path.name}  "
          f"({len(real_clusters)} real clusters, {num_noise} noise)")
    return ctx_path, idx_path, meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    default_matched = (
        ROOT
        / "data"
        / "DBLP-Citation-network-V18"
        / "DBLP-Citation-network-V18.matched_all_master.jsonl"
    )
    default_out = ROOT / "data" / "DBLP-Citation-network-V18"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-jsonl", type=Path, default=default_matched)
    ap.add_argument("--output-dir", type=Path, default=default_out)
    ap.add_argument("--output-prefix", type=str,
                    default="matched_all_master_g22_25",
                    help="Filename prefix for all outputs.")
    ap.add_argument("--year-min", type=int, default=2022)
    ap.add_argument("--year-max", type=int, default=2025)

    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=15)
    ap.add_argument("--hdbscan-min-samples", type=int, default=5)

    ap.add_argument("--bc-svd-dim", type=int, default=128)
    ap.add_argument(
        "--keep-largest-cluster",
        action="store_true",
        help="Do not remove the largest non-noise HDBSCAN cluster. The manuscript pipeline removes it.",
    )
    ap.add_argument(
        "--no-longitudinal-filter",
        action="store_true",
        help="Do not require research areas to contain papers from every year in the target window.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print graph stats and exit without running embeddings.")
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] output_dir={out_dir}")
    print(f"[cfg] output_prefix={args.output_prefix}")
    print("[cfg] method=bcsvd  clustering=hdbscan")

    t_all = time.time()
    all_rows = read_matched_jsonl(args.matched_jsonl)
    by_year = Counter(int(r["year"]) for r in all_rows if r.get("year") is not None)
    print(f"[load] {len(all_rows)} total rows. Year histogram: {dict(sorted(by_year.items()))}")

    subset = [
        r for r in all_rows
        if r.get("id")
        and r.get("year") is not None
        and args.year_min <= int(r["year"]) <= args.year_max
    ]
    subset.sort(key=lambda r: r["id"])
    paper_ids = [r["id"] for r in subset]
    subset_set = set(paper_ids)
    print(f"[subset] year in [{args.year_min},{args.year_max}]: {len(paper_ids)} papers")

    G, all_ids = build_citation_digraph(all_rows)
    n_within = edges_within(G, subset_set)
    print(f"[graph] edges within {args.year_min}-{args.year_max}: {n_within}")

    # -------------------------------------------------------------------
    # Companion corpus + edges in DBLP-id namespace so build_canonical_states.py
    # can link anchors, memory, and context assignments without any re-mapping.
    # corpus uses paper_id = DBLP id (renamed from matched JSONL's "id");
    # edges are directed citing -> cited restricted to matched nodes.
    # -------------------------------------------------------------------
    corpus_out = out_dir / f"{args.output_prefix}.paper_corpus.jsonl"
    edges_out = out_dir / f"{args.output_prefix}.paper_graph_edges.jsonl"
    with corpus_out.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            pid = r.get("id")
            if not pid:
                continue
            row = {
                "paper_id": pid,
                "title": r.get("title"),
                "abstract": r.get("abstract"),
                "venue": r.get("venue"),
                "year": r.get("year"),
                "keywords": r.get("keywords"),
                "authors": r.get("authors"),
                "doi": r.get("doi"),
                "url": r.get("url"),
                "doc_type": r.get("doc_type"),
                "n_citation": r.get("n_citation"),
                "dblp_id": pid,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[companion] wrote {corpus_out.name}  (paper_id = DBLP id)")

    with edges_out.open("w", encoding="utf-8") as fh:
        written = 0
        for u, v in G.edges():
            fh.write(json.dumps({"source": u, "target": v, "weight": 1}) + "\n")
            written += 1
    print(f"[companion] wrote {edges_out.name}  ({written} directed citing->cited edges)")

    graph_meta = {
        "year_min": args.year_min,
        "year_max": args.year_max,
        "num_rows_total": len(all_rows),
        "num_subset_papers": len(paper_ids),
        "graph_nodes_total": G.number_of_nodes(),
        "graph_edges_total": G.number_of_edges(),
        "graph_edges_within_subset": n_within,
        "subset_year_histogram": {int(y): c for y, c in sorted(
            Counter(int(r["year"]) for r in subset).items()
        )},
    }
    write_json(out_dir / f"{args.output_prefix}.graph_build.meta.json", graph_meta)

    if args.dry_run:
        print("[dry-run] exiting before embedding.")
        return

    base_config = {
        "seed": args.seed,
        "year_range": [args.year_min, args.year_max],
        "graph_direction": "directed",
        "graph_meta": graph_meta,
    }

    print("[method] Bibliographic coupling + SVD")
    t0 = time.time()
    X = bcsvd_embeddings(subset, svd_dim=args.bc_svd_dim, seed=args.seed)
    print(f"[bcsvd] done in {time.time()-t0:.1f}s, shape={X.shape}")
    np.save(out_dir / f"{args.output_prefix}_bcsvd_embed.embeddings.npy", X)
    write_json(out_dir / f"{args.output_prefix}_bcsvd_embed.paper_ids.json", paper_ids)

    print("[cluster] method=bcsvd, clustering=hdbscan")
    t0 = time.time()
    labels = cluster_hdbscan(
        X,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
    )
    uniq = np.unique(labels)
    raw_n_real = int(len(uniq) - (1 if -1 in uniq else 0))
    raw_n_noise = int((labels == -1).sum())
    print(f"[cluster] method=bcsvd, hdbscan: {raw_n_real} clusters, "
          f"{raw_n_noise} noise, {time.time()-t0:.1f}s")

    labels, filter_summary = filter_research_area_labels(
        subset,
        labels,
        year_min=args.year_min,
        year_max=args.year_max,
        drop_largest_cluster=not args.keep_largest_cluster,
        require_all_years=not args.no_longitudinal_filter,
    )
    uniq = np.unique(labels)
    n_real = int(len(uniq) - (1 if -1 in uniq else 0))
    n_noise = int((labels == -1).sum())
    print(f"[cluster] after research-area filters: {n_real} clusters, "
          f"{n_noise} noise, {time.time()-t0:.1f}s")

    config = dict(base_config)
    config.update({
        "method": "bcsvd",
        "clustering": "hdbscan",
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "bc_svd_dim": args.bc_svd_dim,
        "research_area_filter": filter_summary,
    })

    write_context_bundle(
        subset_rows=subset,
        paper_ids=paper_ids,
        X=X,
        labels=labels,
        prefix=args.output_prefix,
        out_dir=out_dir,
        method="bcsvd",
        clustering="hdbscan",
        config=config,
    )

    print(f"[done] total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
