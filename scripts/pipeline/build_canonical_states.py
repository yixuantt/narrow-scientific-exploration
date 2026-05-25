#!/usr/bin/env python3
"""Build canonical seed-paper states from DBLP citation-graph research areas."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_DBLP_DIR = ROOT / "data" / "DBLP-Citation-network-V18"
DEFAULT_DBLP_PREFIX = "matched_all_master_g22_25"
DEFAULT_DBLP_CONTEXT_BUNDLE = f"{DEFAULT_DBLP_PREFIX}_bcsvd_hdbscan"
DEFAULT_INPUT = DEFAULT_DBLP_DIR / f"{DEFAULT_DBLP_PREFIX}.paper_corpus.jsonl"
DEFAULT_CONTEXTS = DEFAULT_DBLP_DIR / f"{DEFAULT_DBLP_CONTEXT_BUNDLE}.paper_contexts.jsonl"
DEFAULT_CONTEXT_INDEX = DEFAULT_DBLP_DIR / f"{DEFAULT_DBLP_CONTEXT_BUNDLE}.context_index.json"
DEFAULT_GRAPH_EDGES = DEFAULT_DBLP_DIR / f"{DEFAULT_DBLP_PREFIX}.paper_graph_edges.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "canonical_states" / "dblp_g22_25_bcsvd_hdbscan"
DEFAULT_ANCHOR_YEARS = (2022, 2023, 2024, 2025)

DEFAULT_CONSTRAINTS = {
    "budget": "academic lab",
    "compute": "up to 8 A100-equivalent GPUs",
    "data": "publicly available datasets only",
    "evaluation": "must include at least one robustness or stress-test setting beyond headline accuracy",
}

STOPWORDS = {
    "and",
    "for",
    "from",
    "into",
    "over",
    "with",
    "without",
    "using",
    "use",
    "via",
    "through",
    "towards",
    "toward",
    "based",
    "learning",
    "model",
    "models",
    "network",
    "networks",
    "task",
    "tasks",
    "method",
    "methods",
    "approach",
    "approaches",
    "framework",
    "frameworks",
    "study",
    "studies",
    "new",
    "improved",
    "improving",
    "robust",
    "efficient",
    "deep",
    "representation",
    "optimization",
    "generalization",
    "theory",
    "inference",
    "prediction",
    "generation",
    "classification",
    "regression",
    "training",
    "testing",
    "vision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample seed-paper sets from DBLP citation-graph research areas.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Master corpus JSONL.")
    parser.add_argument("--contexts-path", type=Path, default=DEFAULT_CONTEXTS, help="Per-paper research-area assignments.")
    parser.add_argument("--context-index", type=Path, default=DEFAULT_CONTEXT_INDEX, help="Research-area metadata JSON.")
    parser.add_argument("--graph-edges", type=Path, default=DEFAULT_GRAPH_EDGES, help="Paper graph edge JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Canonical state output directory.")
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=list(DEFAULT_ANCHOR_YEARS),
        help="Target anchor year(s). Default: 2022 2023 2024 2025.",
    )
    parser.add_argument("--per-context", type=int, default=25, help="Seed-paper sets sampled inside each year x research-area cell.")
    parser.add_argument("--max-contexts-per-year", type=int, default=20, help="Maximum research areas kept per year.")
    parser.add_argument("--min-context-size", type=int, default=5, help="Discard research areas with fewer visible papers than this.")
    parser.add_argument("--seed-set-size", dest="memory_size", type=int, metavar="N", default=5, help="Number of seed papers per generation run.")
    parser.add_argument("--memory-size", dest="memory_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--run-seeds", type=int, nargs="*", default=[0, 1, 2], help="Repeated-run seeds.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible sampling.")
    parser.add_argument("--paper-type", type=str, default="all", choices=["all", "oral", "poster"], help="Optional anchor paper-type filter.")
    parser.add_argument(
        "--include-outside-dblp-graph",
        action="store_true",
        help="Also sample anchors from _outside_dblp_graph. By default that area is skipped.",
    )
    parser.add_argument(
        "--include-graph-isolates",
        action="store_true",
        help="Also sample anchors from graph_isolates (degree-0 and/or merged periphery). "
        "By default only graph_mod_* citation communities are used as research areas.",
    )
    parser.add_argument(
        "--sort-contexts-by",
        choices=["anchors", "tightness", "size"],
        default="anchors",
        help=(
            "Order research areas within each year BEFORE truncation to --max-contexts-per-year. "
            "'anchors' (default): most in-year anchor candidates first. "
            "'tightness': tightest research areas first, ranked by mean_intra_cosine (falls back to "
            "mean_intra_euclidean when no cosine metric is available). "
            "'size': largest visible research areas first."
        ),
    )
    parser.add_argument(
        "--tightness-min-anchors",
        type=int,
        default=1,
        help="When --sort-contexts-by=tightness, drop research areas whose anchor count for the "
             "target year is below this threshold BEFORE sorting (default 1 = keep any area "
             "that still has a usable anchor that year).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    return token


def normalize_phrase_tokens(text: str) -> List[str]:
    normalized = []
    for token in tokenize(text):
        token = normalize_token(token)
        if token and token not in STOPWORDS:
            normalized.append(token)
    return normalized


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", normalize_text(text).lower()).strip("_")
    return text or "item"


def to_relative_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT)


def paper_terms(record: Dict[str, Any]) -> frozenset[str]:
    cached = record.get("_terms")
    if cached is not None:
        return cached
    terms: set[str] = set()
    for token in normalize_phrase_tokens(normalize_text(record.get("title"))):
        terms.add(token)
    for token in normalize_phrase_tokens(normalize_text(record.get("abstract"))):
        terms.add(token)
    record["_terms"] = frozenset(terms)
    return record["_terms"]


def precompute_all_terms(corpus: Sequence[Dict[str, Any]]) -> None:
    try:
        from tqdm import tqdm

        iterator = tqdm(corpus, desc="Precomputing term sets", unit="rec", dynamic_ncols=True)
    except ImportError:
        iterator = corpus  # type: ignore[assignment]
    for record in iterator:
        if "_terms" not in record:
            record["_terms"] = paper_terms(record)


def load_context_assignments(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {
        normalize_text(row.get("paper_id")): row
        for row in rows
        if normalize_text(row.get("paper_id"))
    }


def load_context_index(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("contexts", {})


def load_graph_adjacency(path: Path) -> Dict[str, Dict[str, float]]:
    adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in read_jsonl(path):
        src = normalize_text(row.get("source"))
        dst = normalize_text(row.get("target"))
        if not src or not dst or src == dst:
            continue
        weight = float(row.get("weight") or 1.0)
        adjacency[src][dst] = max(weight, adjacency[src].get(dst, 0.0))
        adjacency[dst][src] = max(weight, adjacency[dst].get(src, 0.0))
    return adjacency


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# Patterns that identify internal pipeline artifacts (embedding method,
# clustering algorithm, raw context-id, and label templates) which must
# not reach the agent prompt.
#
# ORDER MATTERS. Patterns are applied top-to-bottom; composite templates must
# fire before their sub-tokens are stripped, otherwise the template stops
# matching (e.g. once "bcsvd" is gone we can no longer recognise the
# "bcsvd-hdbscan context 108" label prefix). For the same reason:
#   (a) label templates first,
#   (b) multi-token composites next,
#   (c) single tokens last.
# Kept in sync with _LEAK_PATTERNS in scripts/analysis/build_module_vocabulary.py.
_INTERNAL_TOKEN_PATTERNS = (
    # (a) Label templates.
    # "bcsvd-hdbscan context 108", "research context 7", "<method>-<clust> noise cluster".
    re.compile(
        r"^\s*(?:[a-z0-9]+[-_])?[a-z]+\s+(?:context|noise\s+cluster)(?:\s+\d+)?\s*[-:|]?\s*",
        re.IGNORECASE,
    ),
    # Legacy stats-blob parenthetical:
    #   "(23 papers, mean intra-cos=0.915, euc=0.393)"
    #   "(23 papers; unclustered)"
    # plus the surrounding pipe/colon separator that used to follow it.
    re.compile(r"\(\s*\d+\s+papers?[^)]*\)\s*[|:\-]?\s*", re.IGNORECASE),
    # (b) Multi-token composites that must be stripped whole.
    re.compile(r"matched[_ ]?all[_ ]?master[\w-]*", re.IGNORECASE),
    # "bcsvd-hdbscan", "bsvd-hdbscan", "bcsv-hdbscan", "bc-svd hdbscan", ...
    re.compile(r"\bb[cs]{1,2}v?d?[-_ ]+hdb?s?can\b", re.IGNORECASE),
    re.compile(r"\bg\d{2}[_-]\d{2}[_-]bc\w*", re.IGNORECASE),
    # (c) Single pipeline tokens.
    re.compile(r"\bbc[-_ ]?svd\b", re.IGNORECASE),
    re.compile(r"\bbc[-_ ]?umap\b", re.IGNORECASE),
    re.compile(r"\bbsvd\b", re.IGNORECASE),
    re.compile(r"\bbcsv\b(?!d)", re.IGNORECASE),
    re.compile(r"\bhdb?scan\b|\bhdscan\b", re.IGNORECASE),
    re.compile(r"\bnode2vec\b|\bn2v\b", re.IGNORECASE),
    re.compile(r"\blouvain\b", re.IGNORECASE),
    re.compile(r"\bkmeans\b|\bk[-_ ]means\b", re.IGNORECASE),
)


def _scrub_internal_tokens(s: str) -> str:
    """Remove internal pipeline identifiers from a human-readable string.

    Collapses leftover whitespace and strips leading/trailing separators so the
    result reads cleanly when dropped into an agent prompt.
    """
    if not isinstance(s, str) or not s:
        return s
    out = s
    for pat in _INTERNAL_TOKEN_PATTERNS:
        out = pat.sub(" ", out)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"^[\s|,:;/\-]+", "", out)
    out = re.sub(r"[\s|,:;/\-]+$", "", out)
    return out.strip()


def _scrub_context_label(label: Optional[str]) -> str:
    """Return an agent-safe context_label. Falls back to 'research context'."""
    cleaned = _scrub_internal_tokens(normalize_space(label or ""))
    if not cleaned or len(cleaned) < 4:
        return "research context"
    return cleaned


def derive_context_keywords(meta: Dict[str, Any]) -> List[str]:
    raw_keywords = meta.get("context_seed_keywords") or []
    if isinstance(raw_keywords, list):
        cleaned: List[str] = []
        for kw in raw_keywords:
            kw_norm = _scrub_internal_tokens(normalize_space(kw))
            if kw_norm and len(kw_norm) >= 2:
                cleaned.append(kw_norm)
        if cleaned:
            return cleaned[:6]
    label = _scrub_context_label(meta.get("context_label"))
    parts = [p.strip() for p in label.split("/") if p.strip()]
    return parts or [label or "research context"]


def infer_subarea(context_meta: Dict[str, Any]) -> str:
    label = _scrub_context_label(context_meta.get("context_label"))
    keywords = derive_context_keywords(context_meta)
    if label and label != "research context":
        return label
    if len(keywords) >= 2:
        return f"{keywords[0]} / {keywords[1]}"
    return keywords[0] if keywords else "research context"


def infer_challenge(anchor: Dict[str, Any], context_meta: Dict[str, Any]) -> str:
    context_label = _scrub_context_label(context_meta.get("context_label"))
    tl_dr = normalize_space(anchor.get("tl_dr"))
    if tl_dr:
        return (
            f"Inside the research context '{context_label}', extend the anchor paper while resolving the remaining "
            f"weakness implied by this contribution. Anchor TL;DR: {tl_dr}"
        )
    abstract = normalize_space(anchor.get("abstract"))
    if len(abstract) > 260:
        abstract = abstract[:257] + "..."
    return (
        f"Inside the research context '{context_label}', extend the anchor paper into a sharper and more novel "
        f"research direction while preserving feasibility. Anchor abstract: {abstract}"
    )


def infer_goal(anchor: Dict[str, Any], context_meta: Dict[str, Any]) -> str:
    return (
        f"Within the research context '{_scrub_context_label(context_meta.get('context_label'))}', propose a novel and "
        f"feasible top-venue research direction that builds on '{anchor.get('title', 'the anchor paper')}' and "
        f"closely related literature."
    )


def build_memory_paper(
    record: Dict[str, Any],
    context_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context_meta = context_meta or {}
    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "abstract": record.get("abstract"),
        "venue": record.get("venue"),
        "year": record.get("year"),
        "paper_type": record.get("paper_type"),
        "openreview_url": record.get("openreview_url"),
        # NOTE: context_id is preserved verbatim for downstream joins, but it is
        # never meant to be read by the agent. Every place that surfaces this to
        # a prompt (unified_ideation.build_workshop_markdown et al.) uses
        # context_label, which we scrub here.
        "context_id": context_meta.get("context_id", record.get("_context_id")),
        "context_label": _scrub_context_label(
            context_meta.get("context_label", record.get("_context_label"))
        ),
    }


def filter_corpus_records(
    corpus: Sequence[Dict[str, Any]],
    years: Optional[Sequence[int]],
    paper_type: str,
) -> List[Dict[str, Any]]:
    years_set = {int(year) for year in years} if years is not None else None
    filtered: List[Dict[str, Any]] = []
    for row in corpus:
        year = row.get("year")
        if years_set is not None:
            if year is None or int(year) not in years_set:
                continue
        if paper_type != "all" and normalize_text(row.get("paper_type")).lower() != paper_type.lower():
            continue
        filtered.append(row)
    return filtered


def tag_corpus_with_research_areas(corpus: Sequence[Dict[str, Any]], min_context_size: int) -> None:
    counts: Dict[str, int] = defaultdict(int)
    for row in corpus:
        context_id = normalize_text(row.get("_context_id"))
        if context_id:
            counts[context_id] += 1

    for row in corpus:
        context_id = normalize_text(row.get("_context_id"))
        if not context_id or counts.get(context_id, 0) < min_context_size:
            row["_context_id"] = "graph_isolates"
            row["_context_label"] = "DBLP citation graph isolates"
            row["_context_phrases"] = []


def choose_neighbors_stochastic(
    anchor: Dict[str, Any],
    corpus: Sequence[Dict[str, Any]],
    memory_size: int,
    rng: random.Random,
    pool_multiplier: int = 8,
) -> List[Dict[str, Any]]:
    anchor_pid = normalize_text(anchor.get("paper_id"))
    anchor_year = int(anchor.get("year") or 0)
    anchor_context = normalize_text(anchor.get("_context_id"))
    anchor_terms = paper_terms(anchor)

    same_context: List[Dict[str, Any]] = []
    fallback_pool: List[Dict[str, Any]] = []
    for candidate in corpus:
        candidate_pid = normalize_text(candidate.get("paper_id"))
        if not candidate_pid or candidate_pid == anchor_pid:
            continue
        candidate_year = int(candidate.get("year") or 0)
        if candidate_year > anchor_year:
            continue
        candidate_context = normalize_text(candidate.get("_context_id"))
        if candidate_context == anchor_context and candidate_context:
            same_context.append(candidate)
        else:
            fallback_pool.append(candidate)

    def _score(candidate: Dict[str, Any]) -> Tuple[int, int]:
        overlap = len(anchor_terms & paper_terms(candidate))
        recency = -abs(anchor_year - int(candidate.get("year") or 0))
        return (overlap, recency)

    same_context.sort(key=_score, reverse=True)
    fallback_pool.sort(key=_score, reverse=True)

    pool_cap = max(memory_size * max(pool_multiplier, 1), memory_size)
    candidate_pool = same_context[:pool_cap]
    if len(candidate_pool) < memory_size:
        candidate_pool.extend(fallback_pool[: max(pool_cap - len(candidate_pool), memory_size)])

    selected: List[Dict[str, Any]] = [anchor]
    if candidate_pool:
        take = min(memory_size - 1, len(candidate_pool))
        selected.extend(rng.sample(candidate_pool, k=take))

    if len(selected) < memory_size:
        for candidate in same_context + fallback_pool:
            cid = normalize_text(candidate.get("paper_id"))
            if cid == anchor_pid:
                continue
            if any(normalize_text(item.get("paper_id")) == cid for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= memory_size:
                break
    return selected[:memory_size]


def rank_memory_candidates(
    anchor: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    adjacency: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    anchor_pid = normalize_text(anchor.get("paper_id"))
    anchor_year = int(anchor.get("year") or 0)
    anchor_terms = paper_terms(anchor)
    ranked: List[Tuple[Tuple[float, int, int, int], Dict[str, Any]]] = []
    for candidate in candidates:
        candidate_pid = normalize_text(candidate.get("paper_id"))
        if candidate_pid == anchor_pid:
            continue
        if int(candidate.get("year") or 0) > anchor_year:
            continue
        edge_weight = adjacency.get(anchor_pid, {}).get(candidate_pid, 0.0)
        overlap = len(anchor_terms & paper_terms(candidate))
        same_year = int(int(candidate.get("year") or 0) == anchor_year)
        recency = -abs(anchor_year - int(candidate.get("year") or 0))
        ranked.append(((edge_weight, overlap, same_year, recency), candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked]


def choose_memory_papers(
    anchor: Dict[str, Any],
    context_members: Sequence[Dict[str, Any]],
    adjacency: Dict[str, Dict[str, float]],
    memory_size: int,
) -> List[Dict[str, Any]]:
    ranked = rank_memory_candidates(anchor, context_members, adjacency)
    chosen = [anchor]
    for candidate in ranked:
        if len(chosen) >= memory_size:
            break
        chosen.append(candidate)
    return chosen[:memory_size]


def build_task_id(anchor: Dict[str, Any], context_meta: Dict[str, Any], index: int) -> str:
    venue_tag = slugify((anchor.get("venue") or "paper").split()[0])[:12]
    return (
        f"{venue_tag}_{anchor['year']}_{index:03d}_"
        f"{slugify(context_meta['context_id'])[:32]}_{slugify(anchor['paper_id'])[:24]}"
    )


def build_canonical_state(
    anchor: Dict[str, Any],
    context_meta: Dict[str, Any],
    context_members: Sequence[Dict[str, Any]],
    adjacency: Dict[str, Dict[str, float]],
    memory_size: int,
    index: int,
    visible_years: Sequence[int],
    anchor_candidates_in_year: int,
) -> Dict[str, Any]:
    memory_records = choose_memory_papers(anchor, context_members, adjacency, memory_size)
    return {
        "task_id": build_task_id(anchor, context_meta, index),
        "source": {
            "dataset": "dblp_citation_network_v18",
            "anchor_paper_id": anchor.get("paper_id"),
            "anchor_year": anchor.get("year"),
            "anchor_paper_type": anchor.get("paper_type"),
            "context_id": context_meta["context_id"],
            # Raw context_label (may contain internal tokens) is kept here for
            # provenance but is NOT read by any prompt builder; they all use
            # infer_subarea/challenge/goal or derive_context_keywords, which
            # apply _scrub_context_label / _scrub_internal_tokens.
            "context_label": _scrub_context_label(context_meta["context_label"]),
            "context_keywords": derive_context_keywords(context_meta),
            "context_size_visible": len(context_members),
            "anchor_candidates_in_year": anchor_candidates_in_year,
            "context_year_min": min(visible_years) if visible_years else None,
            "context_year_max": max(visible_years) if visible_years else None,
            "context_type": "global_paper_graph",
        },
        "subarea": infer_subarea(context_meta),
        "challenge": infer_challenge(anchor, context_meta),
        "goal": infer_goal(anchor, context_meta),
        "constraints": dict(DEFAULT_CONSTRAINTS),
        "keywords": derive_context_keywords(context_meta),
        "language": "en",
        "memory_papers": [build_memory_paper(record, context_meta) for record in memory_records],
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.contexts_path.exists():
        raise SystemExit(
            f"context assignments not found: {args.contexts_path}\n"
            "Run: python scripts/pipeline/build_graph_embedding_contexts.py"
        )
    if not args.context_index.exists():
        raise SystemExit(
            f"context index not found: {args.context_index}\n"
            "Run: python scripts/pipeline/build_graph_embedding_contexts.py"
        )
    if not args.graph_edges.exists():
        raise SystemExit(
            f"graph edges not found: {args.graph_edges}\n"
            "Run: python scripts/pipeline/build_graph_embedding_contexts.py"
        )

    corpus = read_jsonl(args.input)
    precompute_all_terms(corpus)
    corpus_by_id = {normalize_text(row.get("paper_id")): row for row in corpus if normalize_text(row.get("paper_id"))}
    context_assignments = load_context_assignments(args.contexts_path)
    context_index = load_context_index(args.context_index)
    adjacency = load_graph_adjacency(args.graph_edges)
    seed_eligibility_by_context: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for pid, assign in context_assignments.items():
        ctx = normalize_text(assign.get("context_id"))
        if not ctx:
            continue
        seed_eligibility_by_context[ctx][pid] = bool(assign.get("seed_eligible", True))

    all_years = sorted({int(row["year"]) for row in corpus})
    target_years = set(args.years or all_years)

    if not args.include_outside_dblp_graph:
        print(
            "Excluding _outside_dblp_graph from anchor sampling (use --include-outside-dblp-graph to allow).",
            flush=True,
        )
    if not args.include_graph_isolates:
        print(
            "Excluding graph_isolates from anchor sampling (use --include-graph-isolates to allow).",
            flush=True,
        )

    manifest_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    context_summary_rows: List[Dict[str, Any]] = []
    task_counter = 0

    for year in all_years:
        if year not in target_years:
            continue
        visible_corpus = [
            row for row in corpus if int(row.get("year") or -1) <= year and normalize_text(row.get("paper_id")) in context_assignments
        ]
        visible_by_context: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in visible_corpus:
            paper_id = normalize_text(row.get("paper_id"))
            context_id = normalize_text(context_assignments[paper_id].get("context_id"))
            if context_id:
                visible_by_context[context_id].append(row)

        eligible_contexts: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]] = []
        for context_id, members in visible_by_context.items():
            if (
                context_id == "_outside_dblp_graph"
                and not args.include_outside_dblp_graph
            ):
                continue
            if context_id == "graph_isolates" and not args.include_graph_isolates:
                continue
            meta = context_index.get(context_id)
            if not meta:
                continue
            if meta.get("small_context"):
                continue
            if len(members) < args.min_context_size:
                continue
            seed_eligible_map = seed_eligibility_by_context.get(context_id, {})
            anchors = [
                member
                for member in members
                if int(member.get("year") or -1) == year
                and (args.paper_type == "all" or member.get("paper_type") == args.paper_type)
                and seed_eligible_map.get(normalize_text(member.get("paper_id")), True)
            ]
            if not anchors:
                continue
            eligible_contexts.append((context_id, meta, members, anchors))

        if args.sort_contexts_by == "tightness" and args.tightness_min_anchors > 1:
            eligible_contexts = [
                item for item in eligible_contexts
                if len(item[3]) >= args.tightness_min_anchors
            ]

        def _context_tightness(meta: Dict[str, Any]) -> float:
            """Prefer mean_intra_cosine; fall back to -mean_intra_euclidean (smaller euc = tighter)."""
            cos = meta.get("mean_intra_cosine")
            if cos is not None and cos == cos:  # not NaN
                return float(cos)
            euc = meta.get("mean_intra_euclidean")
            if euc is not None and euc == euc:
                return -float(euc)
            return float("-inf")

        def _eligible_sort_key(item: Tuple[str, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]) -> Tuple:
            _context_id, meta, members, anchors = item
            label = normalize_space(meta.get("context_label"))
            if args.sort_contexts_by == "tightness":
                return (
                    -_context_tightness(meta),
                    -len(anchors),
                    -len(members),
                    label,
                )
            if args.sort_contexts_by == "size":
                return (
                    -len(members),
                    -len(anchors),
                    label,
                )
            # anchors (default)
            return (-len(anchors), -len(members), label)

        eligible_contexts.sort(key=_eligible_sort_key)
        eligible_contexts = eligible_contexts[: args.max_contexts_per_year]

        for context_id, meta, members, anchors in eligible_contexts:
            visible_years = sorted({int(member.get("year") or 0) for member in members if member.get("year") is not None})
            context_summary_rows.append(
                {
                    "year": year,
                    "context_id": context_id,
                    "context_label": meta.get("context_label"),
                    "context_size_visible": len(members),
                    "anchor_candidates_in_year": len(anchors),
                    "context_year_min": min(visible_years) if visible_years else None,
                    "context_year_max": max(visible_years) if visible_years else None,
                    "context_seed_keywords": meta.get("context_seed_keywords"),
                    "mean_intra_cosine": meta.get("mean_intra_cosine"),
                    "mean_intra_euclidean": meta.get("mean_intra_euclidean"),
                }
            )

            rng.shuffle(anchors)
            selected = anchors[: min(args.per_context, len(anchors))]
            for anchor in selected:
                task_counter += 1
                state = build_canonical_state(
                    anchor=anchor,
                    context_meta=meta,
                    context_members=members,
                    adjacency=adjacency,
                    memory_size=args.memory_size,
                    index=task_counter,
                    visible_years=visible_years,
                    anchor_candidates_in_year=len(anchors),
                )
                output_path = args.output_dir / f"{state['task_id']}.json"
                write_json(output_path, state)
                manifest_rows.append(
                    {
                        "task_id": state["task_id"],
                        "year": year,
                        "context_id": context_id,
                        "context_label": meta.get("context_label"),
                        "context_keywords": derive_context_keywords(meta),
                        "context_size_visible": len(members),
                        "anchor_candidates_in_year": len(anchors),
                        "anchor_paper_id": anchor.get("paper_id"),
                        "anchor_title": anchor.get("title"),
                        "anchor_paper_type": anchor.get("paper_type"),
                        "path": to_relative_path(output_path),
                    }
                )
                for run_seed in args.run_seeds:
                    grid_rows.append(
                        {
                            "unit_id": f"{state['task_id']}__seed_{run_seed}",
                            "task_id": state["task_id"],
                            "year": year,
                            "context_id": context_id,
                            "context_label": meta.get("context_label"),
                            "canonical_state_path": to_relative_path(output_path),
                            "run_seed": run_seed,
                        }
                    )

    write_jsonl(args.output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(args.output_dir / "experiment_grid.jsonl", grid_rows)
    write_jsonl(args.output_dir / "context_summary.jsonl", context_summary_rows)

    summary = {
        "input": to_relative_path(args.input),
        "contexts_path": to_relative_path(args.contexts_path),
        "context_index": to_relative_path(args.context_index),
        "graph_edges": to_relative_path(args.graph_edges),
        "output_dir": to_relative_path(args.output_dir),
        "years": sorted(target_years),
        "paper_type": args.paper_type,
        "per_context": args.per_context,
        "max_contexts_per_year": args.max_contexts_per_year,
        "sort_contexts_by": args.sort_contexts_by,
        "tightness_min_anchors": args.tightness_min_anchors,
        "min_context_size": args.min_context_size,
        "memory_size": args.memory_size,
        "run_seeds": args.run_seeds,
        "num_tasks": len(manifest_rows),
        "num_experiment_units": len(grid_rows),
        "num_contexts_used": len({row["context_id"] for row in manifest_rows}),
        "contexts_per_year": dict(sorted(Counter(str(row["year"]) for row in context_summary_rows).items())),
        "tasks_per_year": dict(sorted(Counter(str(row["year"]) for row in manifest_rows).items())),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
