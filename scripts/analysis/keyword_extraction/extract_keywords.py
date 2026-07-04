#!/usr/bin/env python3
"""Cumulative Task/Method keyword extraction over original papers + generated ideas.

Two global keyword libraries (task, method) start empty and grow cumulatively
across two stages:

    Stage 1 (papers):  unique memory papers from ``data/canonical_states/clean_main_batch``.
    Stage 2 (ideas) :  every valid idea under ``runs/ideation_main`` (status==ok and
                       passing CoT leak / empty filters, matching analyze_runs_rq1.py).

Each document is sent to the LLM together with the current vocabulary snapshot.
The LLM separates (1) Problem components, i.e. what the contribution is about,
from (2) Approach components, i.e. how the contribution addresses the problem.
The output fields remain ``task_keywords`` and ``method_keywords`` for backward
compatibility.
We use the offline vLLM backend (``run_inference_vllm.LLMGenerator``) so that
we can drive one GPU-resident engine in-process and batch across the whole
dataset.

Concurrency model:
    Each batch sees a frozen snapshot of the vocabularies. The LLM outputs from
    that batch are parsed, and the union of all *new* keywords in the batch is
    merged into the global vocab before the next batch starts. This makes the
    LLM call inside a batch independent, which vLLM can batch together, while
    still honouring the cumulative semantics at a coarse granularity.

Prompt-size guard:
    When the serialized vocab plus one document would not fit within the
    configured token budget, we fall back to a lexical top-K view of the vocab
    (terms sharing alphabetic tokens with the document, ranked by overlap,
    truncated to ``--vocab-topk``). This mirrors what the user requested:
    "full vocab normally; if it stops fitting, lexical match -> top 100".

Outputs (under ``--out-dir``, default ``analysis_out/keyword_extraction``):
    paper_keywords.jsonl   one row per unique paper  (paper_id, task, method, ...)
    idea_keywords.jsonl    one row per valid idea (includes task/method plus
                           task_from_memory, task_new, method_from_memory,
                           method_new vs memory papers).
    task_vocab.json        final Task keyword library (canonical list)
    method_vocab.json      final Method keyword library
    summary.json           run-level summary + timings
    state.json             resume metadata (done ids, last batch)

The script is restart-safe: rerunning skips documents whose ids are already
present in the corresponding JSONL and re-loads the existing vocab files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Make the project root importable so we can reuse shared analysis helpers.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.common.analysis_utils import extract_agent_text  # noqa: E402

DEFAULT_EXTRACTION_MODEL = "google/gemma-4-31B-it"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOG = logging.getLogger("keyword_extraction")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass
class PaperDoc:
    paper_id: str
    title: Optional[str]
    abstract: Optional[str]

    @property
    def text(self) -> str:
        parts = []
        if self.title:
            parts.append(str(self.title).strip())
        if self.abstract:
            parts.append(str(self.abstract).strip())
        return "\n\n".join(p for p in parts if p)


@dataclass
class IdeaDoc:
    run_path: str            # relative to repo root
    run_id: str              # sha1(run_path), stable doc id
    task_id: Optional[str]
    agent: str
    model: Optional[str]
    run_seed: Optional[int]
    anchor_paper_id: Optional[str]
    year: Optional[str]
    text: str
    # Ordered memory paper ids from canonical_state.memory_papers (same run file).
    memory_paper_ids: List[str] = field(default_factory=list)


def _collect_unique_papers(canonical_root: Path) -> List[PaperDoc]:
    """Walk every canonical_state JSON and dedupe memory_papers by paper_id."""
    pid2paper: "OrderedDict[str, PaperDoc]" = OrderedDict()
    files = sorted(canonical_root.rglob("*.json"))
    LOG.info("Scanning %d canonical_state files from %s", len(files), canonical_root)
    for path in files:
        try:
            obj = json.loads(path.read_text())
        except Exception as exc:
            LOG.warning("Skip malformed canonical_state %s: %s", path, exc)
            continue
        for mp in obj.get("memory_papers") or []:
            pid = mp.get("paper_id")
            if not pid or pid in pid2paper:
                continue
            title = mp.get("title")
            abstract = mp.get("abstract")
            if not (title or abstract):
                continue
            pid2paper[pid] = PaperDoc(
                paper_id=str(pid),
                title=title,
                abstract=abstract,
            )
    LOG.info("Collected %d unique memory papers.", len(pid2paper))
    return list(pid2paper.values())


def _parse_run_path(run_path: Path, repo_root: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Infer (model, year, agent) from the canonical runs/ideation_main layout."""
    try:
        rel = run_path.relative_to(repo_root)
    except ValueError:
        return None, None, None
    parts = rel.parts
    # runs/ideation_main/<model>/<year>/<agent>/<file>.json
    if len(parts) >= 5 and parts[0] == "runs" and parts[1] == "ideation_main":
        return parts[2], parts[3], parts[4]
    return None, None, None


def _collect_valid_ideas(runs_root: Path, repo_root: Path) -> List[IdeaDoc]:
    """Load every runs/ideation_main idea, dropping CoT-leak / empty / failed runs.

    Uses ``analysis_utils.extract_agent_text`` which already bakes in CoT leak
    and empty-text filtering per agent.
    """
    files = sorted(runs_root.rglob("*.json"))
    LOG.info("Scanning %d run files from %s", len(files), runs_root)
    ideas: List[IdeaDoc] = []
    dropped = {"bad_json": 0, "not_ok": 0, "no_text": 0}
    for path in files:
        try:
            obj = json.loads(path.read_text())
        except Exception:
            dropped["bad_json"] += 1
            continue
        if obj.get("status") and str(obj.get("status")).lower() != "ok":
            dropped["not_ok"] += 1
            continue
        agent = obj.get("agent") or ""
        text = extract_agent_text(agent, obj.get("final_output"))
        if not text or not text.strip():
            dropped["no_text"] += 1
            continue
        rel_str = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)
        model, year, agent_dir = _parse_run_path(path, repo_root)
        anchor = ((obj.get("canonical_state") or {}).get("source") or {}).get("anchor_paper_id")
        mem_ids: List[str] = []
        for mp in (obj.get("canonical_state") or {}).get("memory_papers") or []:
            pid = mp.get("paper_id")
            if pid:
                mem_ids.append(str(pid))
        ideas.append(
            IdeaDoc(
                run_path=rel_str,
                run_id=hashlib.sha1(rel_str.encode("utf-8")).hexdigest(),
                task_id=obj.get("task_id"),
                agent=agent or (agent_dir or "unknown"),
                model=model,
                run_seed=obj.get("run_seed"),
                anchor_paper_id=anchor,
                year=year,
                text=text,
                memory_paper_ids=mem_ids,
            )
        )
    LOG.info(
        "Collected %d valid ideas (dropped: bad_json=%d, not_ok=%d, no_text=%d).",
        len(ideas), dropped["bad_json"], dropped["not_ok"], dropped["no_text"],
    )
    return ideas


# --------------------------------------------------------------------------- #
# Keyword vocabulary
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _normalize_keyword(kw: Any) -> Optional[str]:
    if not isinstance(kw, str):
        return None
    kw = kw.strip().strip('"').strip("'").strip()
    if not kw:
        return None
    # Normalize internal whitespace and lowercase.
    kw = re.sub(r"\s+", " ", kw).lower()
    # Strip trailing punctuation like ".", "," that sometimes slips in.
    kw = kw.strip(" .,;:/\\")
    if len(kw) < 2 or len(kw) > 120:
        return None
    # Drop obvious non-terms.
    if kw in {"n/a", "none", "null", "unknown", "etc", "tbd"}:
        return None
    return kw


_BANNED_PHRASES_EXACT: set = {
    "neural network",
    "neural networks",
    "deep learning",
    "machine learning",
    "artificial intelligence",
    "end-to-end training",
    "end to end training",
    "self-supervised learning",
    "self supervised learning",
    "transformer-based",
    "transformer based",
    "deep neural network",
    "deep neural networks",
    "model",
    "approach",
    "framework",
    "method",
    "methods",
    "analysis",
    "study",
    "effect",
    "effects",
    "impact",
    "impacts",
    "application",
    "applications",
    "system",
    "systems",
    "algorithm",
    "learning",
    "training",
    "network",
    "networks",
}


def _is_banned_keyword(kw: str) -> bool:
    return kw.strip().lower() in _BANNED_PHRASES_EXACT


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _fuzzy_dedupe(items: List[str], threshold: float = 0.75) -> List[str]:
    """Remove near-duplicate items based on token Jaccard similarity."""
    out: List[str] = []
    seen_token_sets: List[set] = []
    for item in items:
        toks = set(_tokenize_for_lexical(item))
        if not toks:
            continue
        is_dup = False
        for existing in seen_token_sets:
            if _jaccard(toks, existing) >= threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(item)
            seen_token_sets.append(toks)
    return out


def _cross_axis_dedup(
    tasks: List[str], methods: List[str]
) -> Tuple[List[str], List[str]]:
    """Ensure methods do not repeat any task phrase for the same document."""
    task_set = set(tasks)
    methods_clean = [m for m in methods if m not in task_set]
    return tasks, methods_clean


def _force_reuse(
    items: List[str],
    vocab_toksets: List[Tuple[str, set]],
    threshold: float = 0.6,
) -> List[str]:
    """Replace each item with the most similar vocab entry if Jaccard >= threshold."""
    out: List[str] = []
    for kw in items:
        kw_toks = set(_tokenize_for_lexical(kw))
        if not kw_toks:
            continue
        best_score = 0.0
        best_match = kw
        for existing, existing_toks in vocab_toksets:
            if not existing_toks:
                continue
            inter = len(kw_toks & existing_toks)
            if inter == 0:
                continue
            union = len(kw_toks) + len(existing_toks) - inter
            score = inter / union
            if score > best_score:
                best_score = score
                best_match = existing
        out.append(best_match if best_score >= threshold else kw)
    return out


def _tokenize_for_lexical(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class Vocabulary:
    """Canonicalized keyword list with insertion order preserved."""

    def __init__(self, name: str, initial: Optional[Iterable[str]] = None) -> None:
        self.name = name
        self._items: "OrderedDict[str, None]" = OrderedDict()
        if initial:
            for kw in initial:
                self.add(kw)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, kw: str) -> bool:
        return kw in self._items

    def as_list(self) -> List[str]:
        return list(self._items.keys())

    def add(self, kw: str) -> bool:
        norm = _normalize_keyword(kw)
        if norm is None or norm in self._items:
            return False
        self._items[norm] = None
        return True

    def add_many(
        self, kws: Iterable[str], mutex_with: Optional["Vocabulary"] = None
    ) -> List[str]:
        added: List[str] = []
        for kw in kws:
            norm = _normalize_keyword(kw)
            if norm is None or norm in self._items:
                continue
            if mutex_with is not None and norm in mutex_with:
                continue
            self._items[norm] = None
            added.append(norm)
        return added


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a careful science assistant that extracts two facets from a paper "
    "or generated research idea. The output field names are kept as "
    "task_keywords and method_keywords for compatibility, but use these "
    "definitions:\n"
    "  * task_keywords are Problem components. A Problem component names what "
    "the contribution is about: the object of study, target system, population, "
    "phenomenon, relationship, outcome, property, or problem that the work tries "
    "to explain, predict, improve, or create.\n"
    "  * method_keywords are Approach components. An Approach component names "
    "how the contribution addresses the problem: a method, intervention, "
    "mechanism, material, measurement, data source, model, theory, experimental "
    "design, or analytic strategy.\n\n"
    "Reuse a vocabulary entry only when it names the same Problem component or "
    "the same Approach component in a compatible field context. Keep phrases "
    "short and stable, but do not merge concepts merely because they share broad "
    "words across different fields. Do not narrate the paper, copy motivational "
    "wording, or force everything into computer-science terminology.\n\n"
    "What counts as the contribution\n"
    "  Only tag what this work actually proposes, combines, extends, or relies on as "
    "a main part of its own claim. Skip background, related work, citations, and generic "
    "scene-setting unless that setting is itself what the paper studies.\n\n"
    "task_keywords — Problem components\n"
    "  Identify the ONE core Problem component. If the contribution genuinely "
    "centers on two distinct Problem components, list both.\n"
    "  Include the object of study or target system when it is central: a disease, "
    "material, organism, population, institution, molecule, ecosystem, market, legal "
    "regime, or technological system.\n"
    "  Include the focal problem, phenomenon, relation, outcome, or property when it "
    "is central: antibiotic resistance, soil carbon sequestration, housing price "
    "spillovers, phase transition, crop yield loss, adolescent depression risk, "
    "photocatalytic hydrogen evolution, or structure-property relation.\n"
    "  Do NOT enumerate every sub-challenge, evaluation scenario, organism, geography, "
    "or symptom mentioned in the text; include those only when they define the "
    "Problem component of the contribution.\n"
    "  Do not put tools, assays, algorithms, datasets, statistical models, or "
    "experimental designs here; those belong in method_keywords.\n\n"
    "method_keywords — Approach components\n"
    "  Name 2-4 specific Approach components a researcher could recognize or reproduce: "
    "methods, interventions, mechanisms, measurements, materials, instruments, "
    "assays, study designs, data sources, theories, statistical estimators, "
    "simulation models, optimization methods, architectures, or named analytic "
    "procedures.\n"
    "  Examples include `randomized controlled trial`, `policy intervention`, "
    "`inflammatory pathway`, `rna sequencing`, `satellite imagery`, `cohort study`, "
    "`difference in differences`, `cryo electron microscopy`, `density functional "
    "theory`, `graph neural network`, `polymer electrolyte membrane`, and "
    "`bayesian hierarchical model`.\n"
    "  Avoid vague umbrellas (`empirical analysis`, `experimental study`, "
    "`computational modeling`, `machine learning`) unless the text pins down a "
    "specific variant that is central to the contribution.\n"
    "  Do not repeat a task_keywords item here.\n"
    "  Do not list every minor measurement, control variable, reagent, dataset column, "
    "or implementation detail; focus on the mechanisms that are central to the claim.\n\n"
    "Using the vocabularies (two lists below in the user message)\n"
    "  Reuse an existing entry when this contribution genuinely builds on, combines, "
    "or depends on that concept — not when the word only shows up in passing.\n"
    "  When you reuse, copy the line from the vocabulary exactly.\n"
    "  Add a new phrase only if nothing in the vocabulary covers that same "
    "Problem component or Approach component.\n\n"
    "Consistency\n"
    "  If the vocabulary already contains a phrase that names the same component, "
    "reuse that exact wording. Do not create near-synonym variants (e.g. do not add "
    "\"small far away objects\" if \"far objects\" already exists).\n\n"
    "Mutual exclusion\n"
    "  task_keywords and method_keywords must be disjoint. The same phrase must never "
    "appear in both lists for a single document.\n\n"
    "Style rules\n"
    "  Lowercase, 2-5 words per phrase, no trailing punctuation, no pronouns.\n"
    "  Ban standalone filler: `method`, `approach`, `framework`, `analysis`, "
    "`study`, `model`, `system`, `effect`, `impact`, `application`, `machine learning`, "
    "`deep learning`, `neural network`, `end-to-end training`, `transformer-based`.\n"
    "  No author names, paper titles, venue names, grant/program names, or generic "
    "discipline names alone. Dataset or cohort names may appear only when the named "
    "resource itself is central to the contribution.\n"
    "  About 1-2 task_keywords/Problem components (hard max 3) and 2-3 "
    "method_keywords/Approach components (hard max 4). "
    "method_keywords may be empty if the text never names a concrete method, "
    "intervention, or mechanism.\n\n"
    "Return only valid JSON:\n"
    "  {\"task_keywords\": [\"...\", ...], \"method_keywords\": [\"...\", ...]}\n"
)


ANALYSIS_FIELDS = [
    "Aim",
    "Motivation",
    "Questions addressed",
    "Method",
    "Evaluation metrics",
    "Findings",
    "Contributions",
    "Limitations",
    "Future work",
]

ANNOTATION_SYSTEM_PROMPT = (
    "You are a careful scholarly annotator. Given a research manuscript, write a "
    "concise scholarly analysis covering Aim, Motivation, Questions addressed, "
    "Method, Evaluation metrics, Findings, Contributions, Limitations, and Future "
    "work. Finally extract 5-12 concise scholarly keywords grounded in that "
    "analysis. Return exactly one valid JSON object, with no markdown, no prose "
    "outside the JSON, and no missing JSON fields."
)

IDEA_SEED_SYSTEM_PROMPT = (
    ANNOTATION_SYSTEM_PROMPT
    + " For generated ideas, also compare the idea against the provided seed/memory "
      "paper annotations. For every keyword you extract, decide whether the concept "
      "comes from the seed papers or is new in the generated idea."
)


def _render_vocab_block(task_vocab: Sequence[str], method_vocab: Sequence[str]) -> str:
    def _fmt(v: Sequence[str]) -> str:
        if not v:
            return "  (empty — this is the first document that may introduce entries)"
        return "\n".join(f"  - {kw}" for kw in v)
    return (
        "Current task_keywords vocabulary:\n"
        f"{_fmt(task_vocab)}\n\n"
        "Current method_keywords vocabulary:\n"
        f"{_fmt(method_vocab)}\n"
    )


def _lexical_topk(vocab: Sequence[str], text_tokens: Sequence[str], k: int) -> List[str]:
    """Rank vocab entries by token-coverage relative to the vocab term.

    Uses coverage (overlap / len(term_tokens)) so long generic phrases do not
    dominate just because they contain many tokens. Ties broken by absolute
    overlap, then original order. This is a cheap fallback used only when the
    full vocab can't fit in the prompt.
    """
    if k <= 0 or not vocab:
        return []
    token_set = set(text_tokens)
    scored: List[Tuple[float, int, int, str]] = []
    for idx, kw in enumerate(vocab):
        toks = set(_tokenize_for_lexical(kw))
        if not toks:
            continue
        abs_overlap = len(toks & token_set)
        coverage = abs_overlap / len(toks)
        scored.append((-coverage, -abs_overlap, idx, kw))
    scored.sort()
    top = [kw for _, _, _, kw in scored[:k]]
    return top


def _build_messages(
    doc_text: str,
    task_vocab: Sequence[str],
    method_vocab: Sequence[str],
    doc_kind: str,
    doc_id: str,
    *,
    memory_papers_block: Optional[str] = None,
    require_idea_memory_partition: bool = False,
    skip_vocab_block: bool = False,
) -> List[Dict[str, str]]:
    chunks: List[str] = [
        f"Document kind: {doc_kind}\n",
        f"Document id: {doc_id}\n\n",
    ]
    if not skip_vocab_block:
        chunks.append(_render_vocab_block(task_vocab, method_vocab))
    if memory_papers_block:
        chunks.append(
            "\nMemory papers (keywords already extracted for these papers):\n"
            f"{memory_papers_block}"
        )
    chunks.append("\n---\nDocument text:\n")
    chunks.append(doc_text.strip())
    chunks.append("\n---\n")
    if skip_vocab_block:
        chunks.append(
            "Extract task_keywords as Problem components, and method_keywords "
            "as Approach components. Every phrase must reflect "
            "the contribution itself, not passing mentions. Be concise, specific, "
            "and field-appropriate."
        )
    else:
        chunks.append(
            "Extract task_keywords as Problem components, and method_keywords "
            "as Approach components. Every phrase must reflect "
            "the contribution itself, not passing mentions. Reuse vocabulary lines "
            "only when this work names the same component in a compatible field context."
        )
    if require_idea_memory_partition and doc_kind == "idea":
        chunks.append(
            "\n\nThis document is an IDEA. Return one JSON object with these fields:\n"
            "  \"task_keywords\", \"method_keywords\",\n"
            "  \"task_from_memory\", \"task_new\",\n"
            "  \"method_from_memory\", \"method_new\",\n"
            "  \"task_reasoning\", \"method_reasoning\".\n"
            "\n"
            "Partition rules (critical):\n"
            "  1. For EVERY keyword in task_keywords and method_keywords, search the "
            "     MEMORY PAPERS keywords above for the closest matching concept.\n"
            "  2. If you can find a matching concept (same core idea, even if wording differs), "
            "     put the keyword in *_from_memory.\n"
            "  3. ONLY if you genuinely cannot find any matching concept in the MEMORY PAPERS, "
            "     put the keyword in *_new.\n"
            "  4. When in doubt, prefer *_from_memory.\n"
            "  5. Each keyword must appear in exactly one of *_from_memory or *_new.\n"
            "\n"
            "Reasoning fields:\n"
            "  - task_reasoning: a list of strings, one per task_from_memory keyword, "
            "    in the format: \"idea_keyword → memory_keyword (brief reason)\".\n"
            "  - method_reasoning: same for method_from_memory keywords.\n"
            "  - If *_from_memory is empty, the reasoning list is empty.\n"
            "\n"
            "Examples:\n"
            "  - Memory task: \"antibiotic resistance\"\n"
            "    Idea task: \"drug resistant infections\"\n"
            "    → from_memory; reasoning: \"drug resistant infections → antibiotic resistance (same biological problem, wording broadened)\"\n"
            "  - Memory method: \"rna sequencing\"\n"
            "    Idea method: \"single cell transcriptomics\"\n"
            "    → from_memory; reasoning: \"single cell transcriptomics → rna sequencing (same measurement family, more specific variant)\"\n"
            "  - Memory task: \"urban heat islands\"\n"
            "    Idea task: \"heat exposure inequality\"\n"
            "    → from_memory; reasoning: \"heat exposure inequality → urban heat islands (same environmental hazard, social outcome added)\"\n"
            "  - Memory method: \"difference in differences\"\n"
            "    Idea method: \"synthetic control design\"\n"
            "    → new; no reasoning entry needed\n"
            "  - Memory task: \"photocatalytic water splitting\"\n"
            "    Idea task: \"electrochemical carbon dioxide reduction\"\n"
            "    → new; no reasoning entry needed"
        )
    chunks.append("\n\nOutput ONLY the JSON object.")
    user = "".join(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _build_annotation_messages(
    doc_text: str,
    doc_kind: str,
    doc_id: str,
    *,
    seed_block: Optional[str] = None,
) -> List[Dict[str, str]]:
    prompt = f"""Document kind: {doc_kind}
Document id: {doc_id}
"""
    if seed_block:
        prompt += f"""

Seed/memory paper annotations for comparison. Use these to judge whether generated idea keywords are derived from seed papers or new:
{seed_block}
"""
    prompt += f"""

Please proceed to conduct a scholarly analysis of the provided research manuscript. Your analysis should encapsulate the core components of the study as delineated in the enumeration below:
Aim: What is the aim of the study?
Motivation: What is the motivation of the study?
Questions addressed: What question does this study address?
Methods: What methods does the study use to solve the question?
Evaluation metrics: What evaluation metrics are used in this study?
Findings: What does the study find?
Contributions: What are the contributions of this study?
Limitations: What are the limitations of this study?
Future work: What is the future work of this study?

Subsequently, organize the distilled information into a structured JSON format, omitting any supplementary explanations.

Return exactly one JSON object with this structure:
{{
  "analysis": {{
    "Aim": "...",
    "Motivation": "...",
    "Questions addressed": "...",
    "Method": "...",
    "Evaluation metrics": "...",
    "Findings": "...",
    "Contributions": "...",
    "Limitations": "...",
    "Future work": "..."
  }},
  "keywords": ["...", "..."]
"""
    if seed_block:
        prompt += ',\n  "seed_keyword_judgments": [\n    {"keyword": "...", "from_seed": true, "matched_seed_paper_id": "... or null", "matched_seed_keyword": "... or null", "reason": "..."}\n  ]\n'
    prompt += (
        "}\n\n"
        "Rules:\n"
        "- Output JSON only; do not wrap it in markdown fences.\n"
        "- Include every analysis field exactly as shown above.\n"
        "- keywords must be a non-empty list of 5-12 short scholarly noun phrases.\n"
        "- If a field is uncertain, write a brief best-effort value rather than omitting it.\n\n"
        "Research manuscript:\n" + doc_text.strip()
    )
    return [
        {"role": "system", "content": IDEA_SEED_SYSTEM_PROMPT if seed_block else ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_messages_fit(
    generator,
    doc_text: str,
    task_vocab: Sequence[str],
    method_vocab: Sequence[str],
    doc_kind: str,
    doc_id: str,
    max_input_tokens: int,
    topk: int,
    doc_text_cap_chars: int = 4000,
    *,
    memory_papers_block: Optional[str] = None,
    require_idea_memory_partition: bool = False,
    batch_text_tokens: Optional[List[str]] = None,
    skip_vocab_block: bool = False,
) -> Tuple[List[Dict[str, str]], bool, int, int]:
    """Build messages that fit in ``max_input_tokens``.

    Returns (messages, used_lexical_fallback, n_task_in_prompt, n_method_in_prompt).
    """
    # Hard cap on absurdly long documents before tokenization to keep the
    # rendered prompt bounded even when vocab is empty.
    # Ideas (especially agent_laboratory plans) can be much longer than paper abstracts.
    cap = doc_text_cap_chars
    if doc_kind == "idea":
        cap = max(doc_text_cap_chars, 8000)
    doc_text_trim = doc_text.strip()
    if len(doc_text_trim) > cap:
        doc_text_trim = doc_text_trim[:cap].rstrip() + " ... [truncated]"

    # Attempt 1: full vocab (or no vocab if skip_vocab_block).
    messages = _build_messages(
        doc_text_trim, task_vocab, method_vocab, doc_kind, doc_id,
        memory_papers_block=memory_papers_block,
        require_idea_memory_partition=require_idea_memory_partition,
        skip_vocab_block=skip_vocab_block,
    )
    try:
        n_tok = generator.count_message_tokens(messages)
    except Exception:
        n_tok = max_input_tokens + 1  # force fallback path
    if n_tok <= max_input_tokens:
        return messages, False, len(task_vocab) if not skip_vocab_block else 0, len(method_vocab) if not skip_vocab_block else 0

    # Attempt 2: lexical top-K on vocab (only when vocab block is used).
    if skip_vocab_block:
        # Vocab block is already skipped; if still too long, truncate document text.
        while n_tok > max_input_tokens and len(doc_text_trim) > 2000:
            doc_text_trim = doc_text_trim[:len(doc_text_trim) - 500].rstrip() + " ... [truncated]"
            messages = _build_messages(
                doc_text_trim, task_vocab, method_vocab, doc_kind, doc_id,
                memory_papers_block=memory_papers_block,
                require_idea_memory_partition=require_idea_memory_partition,
                skip_vocab_block=skip_vocab_block,
            )
            try:
                n_tok = generator.count_message_tokens(messages)
            except Exception:
                break
        return messages, True, 0, 0

    doc_tokens = _tokenize_for_lexical(doc_text_trim)
    if memory_papers_block:
        doc_tokens = doc_tokens + _tokenize_for_lexical(memory_papers_block)
    # Prefer batch-level token union for richer coverage when available.
    lexical_tokens = batch_text_tokens if batch_text_tokens is not None else doc_tokens
    task_top = _lexical_topk(list(task_vocab), lexical_tokens, topk)
    method_top = _lexical_topk(list(method_vocab), lexical_tokens, topk)
    messages = _build_messages(
        doc_text_trim, task_top, method_top, doc_kind, doc_id,
        memory_papers_block=memory_papers_block,
        require_idea_memory_partition=require_idea_memory_partition,
        skip_vocab_block=skip_vocab_block,
    )
    try:
        n_tok = generator.count_message_tokens(messages)
    except Exception:
        n_tok = max_input_tokens + 1
    while n_tok > max_input_tokens and topk > 10:
        topk = max(10, topk // 2)
        task_top = _lexical_topk(list(task_vocab), doc_tokens, topk)
        method_top = _lexical_topk(list(method_vocab), doc_tokens, topk)
        messages = _build_messages(
            doc_text_trim, task_top, method_top, doc_kind, doc_id,
            memory_papers_block=memory_papers_block,
            require_idea_memory_partition=require_idea_memory_partition,
            skip_vocab_block=skip_vocab_block,
        )
        try:
            n_tok = generator.count_message_tokens(messages)
        except Exception:
            break
    return messages, True, len(task_top), len(method_top)


# --------------------------------------------------------------------------- #
# JSON parsing (shared regex for fenced / partial JSON)
# --------------------------------------------------------------------------- #

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# Persistence / resume
# --------------------------------------------------------------------------- #


def _load_vocab(path: Path, name: str) -> Vocabulary:
    if not path.exists():
        return Vocabulary(name)
    try:
        data = json.loads(path.read_text())
        items = data.get("keywords") if isinstance(data, dict) else data
        if isinstance(items, list):
            return Vocabulary(name, initial=items)
    except Exception as exc:
        LOG.warning("Failed to load vocab from %s: %s. Starting empty.", path, exc)
    return Vocabulary(name)


def _save_vocab(path: Path, vocab: Vocabulary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {"name": vocab.name, "size": len(vocab), "keywords": vocab.as_list()},
            ensure_ascii=False, indent=2,
        )
    )
    tmp.replace(path)


def _load_done_ids(path: Path, key: str) -> set:
    done: set = set()
    if not path.exists():
        return done
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get(key):
                done.add(str(row[key]))
    return done


def _load_paper_keywords_index(jsonl_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Map paper_id -> {"task_keywords": [...], "method_keywords": [...]} from paper_keywords.jsonl.

    Later lines win if a paper_id appears more than once (resume re-runs).
    """
    index: Dict[str, Dict[str, List[str]]] = {}
    if not jsonl_path.exists():
        return index
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = row.get("paper_id")
            if not pid:
                continue
            tasks = row.get("task_keywords") or []
            methods = row.get("method_keywords") or []
            if not isinstance(tasks, list):
                tasks = []
            if not isinstance(methods, list):
                methods = []
            index[str(pid)] = {
                "task_keywords": [k for k in (_normalize_keyword(x) for x in tasks) if k],
                "method_keywords": [k for k in (_normalize_keyword(x) for x in methods) if k],
            }
    return index


def _memory_keyword_pools(
    paper_ids: Sequence[str],
    paper_kw_index: Dict[str, Dict[str, List[str]]],
) -> Tuple[set, set]:
    task_pool: set = set()
    method_pool: set = set()
    for pid in paper_ids:
        row = paper_kw_index.get(str(pid)) or {}
        task_pool.update(row.get("task_keywords") or [])
        method_pool.update(row.get("method_keywords") or [])
    return task_pool, method_pool


def _format_memory_paper_keywords_block(
    paper_ids: Sequence[str],
    paper_kw_index: Dict[str, Dict[str, List[str]]],
    paper_abstract_index: Optional[Dict[str, "PaperDoc"]] = None,
    abstract_cap_chars: int = 600,
) -> str:
    """Human-readable block listing each memory paper's title and keywords."""
    if not paper_ids:
        return "(no memory_paper_ids on this run record)\n"
    lines: List[str] = []
    for pid in paper_ids:
        pid_s = str(pid)
        row = paper_kw_index.get(pid_s)
        if not row:
            lines.append(f"paper_id {pid_s}\n  (no row in paper_keywords.jsonl yet — run papers stage first)\n")
            continue
        tasks = row.get("task_keywords") or []
        meths = row.get("method_keywords") or []
        lines.append(f"paper_id {pid_s}")
        # Add title when available (abstract omitted to keep prompt focused)
        if paper_abstract_index:
            paper_doc = paper_abstract_index.get(pid_s)
            if paper_doc and paper_doc.title:
                lines.append(f"  title: {paper_doc.title}")
        lines.append("  task_keywords:")
        if tasks:
            lines.extend(f"    - {t}" for t in tasks)
        else:
            lines.append("    (empty)")
        lines.append("  method_keywords:")
        if meths:
            lines.extend(f"    - {m}" for m in meths)
        else:
            lines.append("    (empty)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class ParsedKeywordResponse:
    task_keywords: List[str]
    method_keywords: List[str]
    task_from_memory: List[str]
    task_new: List[str]
    method_from_memory: List[str]
    method_new: List[str]
    parse_error: bool
    partition_repaired: bool = False
    partition_omitted: bool = False
    task_reasoning: List[str] = field(default_factory=list)
    method_reasoning: List[str] = field(default_factory=list)


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _finalize_memory_partitions(
    task_keywords: List[str],
    method_keywords: List[str],
    llm_from_t: List[str],
    llm_new_t: List[str],
    llm_from_m: List[str],
    llm_new_m: List[str],
    task_pool: set,
    method_pool: set,
) -> Tuple[List[str], List[str], List[str], List[str], bool]:
    """Disjoint reused/new lists covering task_keywords and method_keywords.

    Trusts the LLM's semantic judgment, but enforces two hard post-hoc rules:
      1. Exact match: if a keyword (case-insensitive) is present in the
         memory-paper pool, it is forced into *_from_memory regardless of the
         LLM's label.
      2. Substring match: if a keyword is a non-trivial substring (or
         superstring) of a memory-paper keyword (min length 12 chars), it is
         also forced into *_from_memory.
    These corrections repair the LLM's position-bias errors where it "forgets"
    memory-paper keywords seen earlier in the prompt.
    """
    t_order = _dedupe_preserve_order([k for k in task_keywords if k])
    m_order = _dedupe_preserve_order([k for k in method_keywords if k])

    def _align(
        ordered: List[str],
        pool: set,
        llm_from: List[str],
        llm_new: List[str],
    ) -> Tuple[List[str], List[str], bool]:
        llm_from_set = {k for k in llm_from if k}
        llm_new_set = {k for k in llm_new if k}

        # Case-insensitive pool for robust matching.
        pool_lc: Dict[str, str] = {p.lower(): p for p in pool}

        def _exact_in_pool(kw: str) -> bool:
            return kw.lower() in pool_lc

        def _substring_in_pool(kw: str) -> bool:
            kw_lc = kw.lower()
            for p_lc in pool_lc:
                if len(p_lc) < 12 or len(kw_lc) < 12:
                    continue
                if kw_lc in p_lc or p_lc in kw_lc:
                    return True
            return False

        final_reused: List[str] = []
        final_new: List[str] = []
        for k in ordered:
            in_from = k in llm_from_set
            in_new = k in llm_new_set

            if _exact_in_pool(k) or _substring_in_pool(k):
                # Post-hoc correction: memory-paper match overrides LLM label.
                final_reused.append(k)
            elif in_from and not in_new:
                final_reused.append(k)
            elif in_new and not in_from:
                final_new.append(k)
            elif in_from and in_new:
                # Conflict: trust LLM's positive signal (reused)
                final_reused.append(k)
            else:
                # Missing from both: default to new
                final_new.append(k)

        repaired = set(final_reused) != llm_from_set or set(final_new) != llm_new_set
        return final_reused, final_new, repaired

    rt, nt, r1 = _align(t_order, task_pool, llm_from_t, llm_new_t)
    rm, nm, r2 = _align(m_order, method_pool, llm_from_m, llm_new_m)
    return rt, nt, rm, nm, r1 or r2


def _parse_keyword_json(raw: str, *, idea_partition: bool) -> ParsedKeywordResponse:
    """Parse model JSON into keyword lists (and optional idea-only memory partition)."""
    if not raw or not raw.strip():
        return ParsedKeywordResponse([], [], [], [], [], [], True)
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    obj: Any = None
    try:
        obj = json.loads(txt)
    except Exception:
        m = _JSON_OBJ_RE.search(txt)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return ParsedKeywordResponse([], [], [], [], [], [], True)

    tasks = obj.get("task_keywords") or obj.get("tasks") or []
    methods = obj.get("method_keywords") or obj.get("methods") or []
    if not isinstance(tasks, list):
        tasks = []
    if not isinstance(methods, list):
        methods = []
    tasks_norm = _dedupe_preserve_order(
        [k for k in (_normalize_keyword(x) for x in tasks) if k and not _is_banned_keyword(k)]
    )
    methods_norm = _dedupe_preserve_order(
        [k for k in (_normalize_keyword(x) for x in methods) if k and not _is_banned_keyword(k)]
    )
    tasks_norm = _fuzzy_dedupe(tasks_norm, threshold=0.6)
    methods_norm = _fuzzy_dedupe(methods_norm, threshold=0.6)
    tasks_norm, methods_norm = _cross_axis_dedup(tasks_norm, methods_norm)
    # Hard cap to keep keywords focused on the core contribution.
    tasks_norm = tasks_norm[:3]
    methods_norm = methods_norm[:4]

    if not idea_partition:
        return ParsedKeywordResponse(
            tasks_norm, methods_norm, [], [], [], [], False,
        )

    def _first_list(d: Dict[str, Any], *keys: str) -> List[str]:
        for key in keys:
            v = d.get(key)
            if isinstance(v, list) and v:
                return _dedupe_preserve_order([k for k in (_normalize_keyword(x) for x in v) if k])
        for key in keys:
            v = d.get(key)
            if isinstance(v, list):
                return _dedupe_preserve_order([k for k in (_normalize_keyword(x) for x in v) if k])
        return []

    rt = _first_list(
        obj,
        "task_from_memory",
        "task_reused_from_memory",
        "task_reused",
        "tasks_reused_from_memory",
    )
    tn = _first_list(obj, "task_new", "tasks_new")
    rm = _first_list(
        obj,
        "method_from_memory",
        "method_reused_from_memory",
        "method_reused",
        "methods_reused_from_memory",
    )
    mn = _first_list(obj, "method_new", "methods_new")

    # Parse optional reasoning fields that map idea keywords to memory-paper concepts.
    tr = obj.get("task_reasoning") or []
    mr = obj.get("method_reasoning") or []
    if not isinstance(tr, list):
        tr = []
    if not isinstance(mr, list):
        mr = []

    has_partition = bool(rt or tn or rm or mn)
    partition_omitted = (not has_partition) and bool(tasks_norm or methods_norm)
    return ParsedKeywordResponse(
        tasks_norm,
        methods_norm,
        rt,
        tn,
        rm,
        mn,
        False,
        partition_repaired=False,
        partition_omitted=partition_omitted,
        task_reasoning=tr,
        method_reasoning=mr,
    )



@dataclass
class StageStats:
    n_total: int = 0
    n_skipped_resume: int = 0
    n_processed: int = 0
    n_parse_errors: int = 0
    n_lexical_fallback: int = 0
    elapsed_s: float = 0.0


def _chunked(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _run_stage(
    *,
    generator,
    docs: Sequence[Any],
    doc_kind: str,
    task_vocab: Vocabulary,
    method_vocab: Vocabulary,
    output_jsonl: Path,
    resume_key: str,
    text_getter,
    id_getter,
    meta_getter,
    batch_size: int,
    max_input_tokens: int,
    max_output_tokens: int,
    temperature: float,
    vocab_topk: int,
    paper_kw_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
    idea_memory_partition: bool = False,
    paper_abstract_index: Optional[Dict[str, PaperDoc]] = None,
) -> StageStats:
    stats = StageStats(n_total=len(docs))
    done_ids = _load_done_ids(output_jsonl, resume_key)
    if done_ids:
        LOG.info(
            "[%s] resume: %d/%d already have keyword rows, skipping.",
            doc_kind, len(done_ids), len(docs),
        )

    pending = [d for d in docs if id_getter(d) not in done_ids]
    stats.n_skipped_resume = len(docs) - len(pending)

    t0 = time.time()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a") as fout:
        for batch_idx, batch in enumerate(_chunked(pending, batch_size)):
            task_snapshot = task_vocab.as_list()
            method_snapshot = method_vocab.as_list()

            # Pre-compute token sets for forced reuse matching.
            task_vocab_toksets = [
                (kw, set(_tokenize_for_lexical(kw))) for kw in task_snapshot
            ]
            method_vocab_toksets = [
                (kw, set(_tokenize_for_lexical(kw))) for kw in method_snapshot
            ]

            # Pre-compute batch-level token union for richer lexical fallback.
            batch_text_tokens: List[str] = []
            for doc in batch:
                batch_text_tokens.extend(_tokenize_for_lexical(text_getter(doc)))
                req_partition = bool(
                    idea_memory_partition and doc_kind == "idea"
                )
                if req_partition:
                    mp_ids = getattr(doc, "memory_paper_ids", None) or []
                    mem_block = _format_memory_paper_keywords_block(
                        mp_ids, paper_kw_index or {}, paper_abstract_index
                    )
                    batch_text_tokens.extend(_tokenize_for_lexical(mem_block))

            messages_list: List[List[Dict[str, str]]] = []
            fallback_flags: List[bool] = []
            prompt_task_sizes: List[int] = []
            prompt_method_sizes: List[int] = []
            for doc in batch:
                mem_block: Optional[str] = None
                req_partition = bool(
                    idea_memory_partition and doc_kind == "idea"
                )
                if req_partition:
                    mp_ids = getattr(doc, "memory_paper_ids", None) or []
                    mem_block = _format_memory_paper_keywords_block(
                        mp_ids, paper_kw_index or {}, paper_abstract_index
                    )
                # For ideas, skip the global vocab block to keep the prompt focused
                # on memory papers only; let the LLM express concepts freely.
                skip_vocab = (doc_kind == "idea")
                messages, used_fallback, n_t, n_m = _build_messages_fit(
                    generator=generator,
                    doc_text=text_getter(doc),
                    task_vocab=task_snapshot,
                    method_vocab=method_snapshot,
                    doc_kind=doc_kind,
                    doc_id=id_getter(doc),
                    max_input_tokens=max_input_tokens,
                    topk=vocab_topk,
                    memory_papers_block=mem_block,
                    require_idea_memory_partition=req_partition,
                    batch_text_tokens=batch_text_tokens,
                    skip_vocab_block=skip_vocab,
                )
                messages_list.append(messages)
                fallback_flags.append(used_fallback)
                prompt_task_sizes.append(n_t)
                prompt_method_sizes.append(n_m)

            outputs = generator.generate(
                messages_list,
                temperature=temperature,
                max_tokens=max_output_tokens,
                seed=0,
                on_error="empty",
            )

            batch_new_tasks: List[str] = []
            batch_new_methods: List[str] = []
            for doc, raw, used_fallback, n_t, n_m in zip(
                batch, outputs, fallback_flags, prompt_task_sizes, prompt_method_sizes
            ):
                req_partition = bool(
                    idea_memory_partition and doc_kind == "idea"
                )
                parsed = _parse_keyword_json(raw, idea_partition=req_partition)
                parse_err = parsed.parse_error
                tasks = parsed.task_keywords
                methods = parsed.method_keywords

                # Force reuse: map near-synonyms to existing vocab entries.
                # Skip for ideas because the prompt no longer shows the global vocab;
                # let the LLM express concepts freely based on memory-paper context.
                if doc_kind != "idea":
                    tasks = _force_reuse(tasks, task_vocab_toksets, threshold=0.6)
                    methods = _force_reuse(methods, method_vocab_toksets, threshold=0.6)
                # Re-apply dedup/cross-axis after replacement may have collapsed items.
                tasks = _dedupe_preserve_order(
                    [k for k in tasks if k and not _is_banned_keyword(k)]
                )
                methods = _dedupe_preserve_order(
                    [k for k in methods if k and not _is_banned_keyword(k)]
                )
                tasks, methods = _cross_axis_dedup(tasks, methods)

                if parse_err:
                    stats.n_parse_errors += 1
                if used_fallback:
                    stats.n_lexical_fallback += 1

                rt: List[str] = []
                nt: List[str] = []
                rm: List[str] = []
                nm: List[str] = []
                part_repaired = False
                if req_partition and not parse_err:
                    tp, mp = _memory_keyword_pools(
                        getattr(doc, "memory_paper_ids", None) or [],
                        paper_kw_index or {},
                    )
                    rt, nt, rm, nm, part_repaired = _finalize_memory_partitions(
                        tasks,
                        methods,
                        parsed.task_from_memory,
                        parsed.task_new,
                        parsed.method_from_memory,
                        parsed.method_new,
                        tp,
                        mp,
                    )

                row: Dict[str, Any] = {
                    resume_key: id_getter(doc),
                    "doc_kind": doc_kind,
                    "task_keywords": tasks,
                    "method_keywords": methods,
                    "llm_meta": {
                        "parse_error": parse_err,
                        "used_lexical_fallback": used_fallback,
                        "vocab_in_prompt": {"task": n_t, "method": n_m},
                        "raw_chars": len(raw or ""),
                        "partition_omitted": parsed.partition_omitted if req_partition else False,
                        "partition_repaired": part_repaired if req_partition else False,
                    },
                }
                if req_partition:
                    row["task_from_memory"] = rt
                    row["task_new"] = nt
                    row["method_from_memory"] = rm
                    row["method_new"] = nm
                    # Persist LLM's reasoning so humans can audit the mapping.
                    row["task_reasoning"] = parsed.task_reasoning
                    row["method_reasoning"] = parsed.method_reasoning
                row.update(meta_getter(doc) or {})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats.n_processed += 1

                for kw in tasks:
                    if kw not in task_vocab:
                        batch_new_tasks.append(kw)
                for kw in methods:
                    if kw not in method_vocab:
                        batch_new_methods.append(kw)

            added_t = task_vocab.add_many(batch_new_tasks, mutex_with=method_vocab)
            added_m = method_vocab.add_many(batch_new_methods, mutex_with=task_vocab)
            fout.flush()

            if (batch_idx % 20 == 0) or (stats.n_processed == len(pending)):
                LOG.info(
                    "[%s] batch %d/%d processed=%d vocab(task=%d, method=%d) "
                    "+new(t=%d,m=%d) parse_err=%d fallback=%d elapsed=%.1fs",
                    doc_kind,
                    batch_idx + 1,
                    (len(pending) + batch_size - 1) // batch_size,
                    stats.n_processed,
                    len(task_vocab), len(method_vocab),
                    len(added_t), len(added_m),
                    stats.n_parse_errors, stats.n_lexical_fallback,
                    time.time() - t0,
                )

    stats.elapsed_s = time.time() - t0
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _paper_meta(doc: PaperDoc) -> Dict[str, Any]:
    return {"title": doc.title, "has_abstract": bool(doc.abstract)}


def _idea_meta(doc: IdeaDoc) -> Dict[str, Any]:
    return {
        "run_path": doc.run_path,
        "task_id": doc.task_id,
        "agent": doc.agent,
        "model": doc.model,
        "run_seed": doc.run_seed,
        "anchor_paper_id": doc.anchor_paper_id,
        "year": doc.year,
        "memory_paper_ids": list(doc.memory_paper_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=str(_REPO_ROOT))
    ap.add_argument(
        "--canonical-root",
        default="data/canonical_states/clean_main_batch",
        help="Directory of canonical_state JSONs (memory_papers source).",
    )
    ap.add_argument(
        "--runs-root",
        default="runs/ideation_main",
        help="Directory of generated idea runs.",
    )
    ap.add_argument("--out-dir", default="analysis_out/keyword_extraction")
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", DEFAULT_EXTRACTION_MODEL))
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "2")))
    ap.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "16384")))
    ap.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")))
    ap.add_argument("--batch-size", type=int, default=128,
                    help="vLLM submission batch size per stage step.")
    ap.add_argument("--max-output-tokens", type=int, default=512)
    ap.add_argument(
        "--idea-max-output-tokens",
        type=int,
        default=1024,
        help="Ideas stage max new tokens (larger JSON: task/method + memory reuse split).",
    )
    ap.add_argument("--max-input-tokens", type=int, default=28000,
                    help="Max tokens allowed in the rendered chat prompt. "
                         "When the full vocab + document exceeds this, lexical top-K fallback kicks in.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--vocab-topk", type=int, default=300,
                    help="Lexical top-K cap when the full vocab does not fit.")
    ap.add_argument("--stage", choices=["papers", "ideas", "both"], default="both")
    ap.add_argument("--limit-papers", type=int, default=0, help="Debug cap on #papers.")
    ap.add_argument("--limit-ideas", type=int, default=0, help="Debug cap on #ideas.")
    ap.add_argument("--disable-thinking", action="store_true", default=True,
                    help="Enable Qwen3 chat template with reasoning OFF (default).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    repo_root = Path(args.repo_root).resolve()
    canonical_root = (repo_root / args.canonical_root).resolve()
    runs_root = (repo_root / args.runs_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_out = out_dir / "paper_keywords.jsonl"
    idea_out = out_dir / "idea_keywords.jsonl"
    task_vocab_path = out_dir / "task_vocab.json"
    method_vocab_path = out_dir / "method_vocab.json"
    summary_path = out_dir / "summary.json"

    # Resume: load existing vocabs (if any) so re-runs keep growing them.
    task_vocab = _load_vocab(task_vocab_path, "task")
    method_vocab = _load_vocab(method_vocab_path, "method")
    LOG.info(
        "Loaded vocabs (resume): task=%d, method=%d",
        len(task_vocab), len(method_vocab),
    )

    # vLLM configuration through env vars used by run_inference_vllm.LLMGenerator.
    os.environ.setdefault("VLLM_MODEL", args.model)
    os.environ.setdefault("TENSOR_PARALLEL_SIZE", str(args.tensor_parallel_size))
    os.environ.setdefault("MAX_MODEL_LEN", str(args.max_model_len))
    os.environ.setdefault("GPU_MEMORY_UTILIZATION", str(args.gpu_memory_utilization))
    os.environ.setdefault("VLLM_OFFLINE_BATCH_SIZE", str(args.batch_size))
    os.environ.setdefault("VLLM_OFFLINE_MAX_NUM_SEQS", str(args.batch_size))
    if args.disable_thinking:
        os.environ["VLLM_DISABLE_THINKING"] = "1"
    os.environ.setdefault("VLLM_OFFLINE_TQDM", "0")

    # Lazy import so argparse errors never pay the vLLM bring-up cost.
    from run_inference_vllm import get_generator

    LOG.info("Bringing up offline vLLM engine: model=%s TP=%d max_model_len=%d",
             args.model, args.tensor_parallel_size, args.max_model_len)
    generator = get_generator()
    LOG.info("Engine ready.")

    run_summary: Dict[str, Any] = {"model": args.model, "stages": {}}

    # Build abstract index early so it can be reused across stages.
    paper_abstract_index: Optional[Dict[str, PaperDoc]] = None

    if args.stage in ("papers", "both"):
        papers = _collect_unique_papers(canonical_root)
        if args.limit_papers:
            papers = papers[: args.limit_papers]
        paper_abstract_index = {p.paper_id: p for p in papers}
        LOG.info("Stage=papers: %d docs to consider.", len(papers))
        stats = _run_stage(
            generator=generator,
            docs=papers,
            doc_kind="paper",
            task_vocab=task_vocab,
            method_vocab=method_vocab,
            output_jsonl=paper_out,
            resume_key="paper_id",
            text_getter=lambda d: d.text,
            id_getter=lambda d: d.paper_id,
            meta_getter=_paper_meta,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            vocab_topk=args.vocab_topk,
        )
        _save_vocab(task_vocab_path, task_vocab)
        _save_vocab(method_vocab_path, method_vocab)
        run_summary["stages"]["papers"] = stats.__dict__ | {
            "final_task_vocab_size": len(task_vocab),
            "final_method_vocab_size": len(method_vocab),
        }

    if args.stage in ("ideas", "both"):
        ideas = _collect_valid_ideas(runs_root, repo_root)
        if args.limit_ideas:
            ideas = ideas[: args.limit_ideas]
        paper_kw_index = _load_paper_keywords_index(paper_out)
        if not paper_kw_index:
            LOG.warning(
                "%s missing or empty: memory paper keyword pools will be empty; "
                "partition repair will put all idea keywords in *_new.",
                paper_out,
            )
        if not paper_abstract_index:
            paper_abstract_index = {p.paper_id: p for p in _collect_unique_papers(canonical_root)}
        idea_max_out = max(int(args.max_output_tokens), int(args.idea_max_output_tokens))
        LOG.info("Stage=ideas: %d docs to consider (idea_max_output_tokens=%d).", len(ideas), idea_max_out)
        stats = _run_stage(
            generator=generator,
            docs=ideas,
            doc_kind="idea",
            task_vocab=task_vocab,
            method_vocab=method_vocab,
            output_jsonl=idea_out,
            resume_key="run_id",
            text_getter=lambda d: d.text,
            id_getter=lambda d: d.run_id,
            meta_getter=_idea_meta,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=idea_max_out,
            temperature=args.temperature,
            vocab_topk=args.vocab_topk,
            paper_kw_index=paper_kw_index,
            idea_memory_partition=True,
            paper_abstract_index=paper_abstract_index,
        )
        _save_vocab(task_vocab_path, task_vocab)
        _save_vocab(method_vocab_path, method_vocab)
        run_summary["stages"]["ideas"] = stats.__dict__ | {
            "final_task_vocab_size": len(task_vocab),
            "final_method_vocab_size": len(method_vocab),
        }

    run_summary["final_task_vocab_size"] = len(task_vocab)
    run_summary["final_method_vocab_size"] = len(method_vocab)
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2))
    LOG.info(
        "Done. Final task_vocab=%d, method_vocab=%d. Summary: %s",
        len(task_vocab), len(method_vocab), summary_path,
    )


if __name__ == "__main__":
    main()
