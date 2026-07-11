#!/usr/bin/env python3
"""Extract standardized scholarly annotations from papers and generated ideas.

The same prompt and output schema are used for both document types. Outputs
contain a structured ``analysis`` object and 5--12 scholarly ``keywords``.
Downstream users choose which annotation fields to embed.

The extraction model is selected at runtime with ``--model``. This module does
not perform novelty labeling; independent novelty annotator outputs are inputs
to ``scripts.analysis.measurements.novelty``.
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
from typing import Any, Callable, Iterator, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_EXTRACTION_MODEL = "google/gemma-4-31B-it"
LOG = logging.getLogger("keyword_extraction")

ANALYSIS_FIELDS = (
    "Aim",
    "Motivation",
    "Questions addressed",
    "Method",
    "Evaluation metrics",
    "Findings",
    "Contributions",
    "Limitations",
    "Future work",
)

ANNOTATION_SYSTEM_PROMPT = (
    "You are a careful scholarly annotator. Given a research manuscript, write a "
    "concise scholarly analysis covering Aim, Motivation, Questions addressed, "
    "Method, Evaluation metrics, Findings, Contributions, Limitations, and Future "
    "work. Finally extract 5-12 concise scholarly keywords grounded in that "
    "analysis. Return exactly one valid JSON object, with no markdown, no prose "
    "outside the JSON, and no missing JSON fields."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _join_nonempty(parts: Sequence[Any]) -> str | None:
    values = [str(part).strip() for part in parts if part is not None and str(part).strip()]
    return "\n\n".join(values) if values else None


def _extract_markdown_section(plan: str, heading_number: int) -> str | None:
    pattern = re.compile(
        rf"^\s*##\s*{heading_number}\.?\s+.*?(?=^\s*##\s*{heading_number + 1}\.?\s+|\Z)",
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(plan)
    return match.group(0).strip() if match else None


def extract_agent_text(agent: str, final_output: Any) -> str | None:
    """Normalize supported agent outputs to one comparable idea document."""
    if not isinstance(final_output, dict):
        return None
    if agent in ("flat_llm", "ai_scientist_v2"):
        return _join_nonempty(
            [
                final_output.get("Title") or final_output.get("Name"),
                final_output.get("Short Hypothesis"),
                final_output.get("Abstract"),
            ]
        )
    if agent == "research_agent":
        return _join_nonempty([final_output.get("problem"), final_output.get("method")])
    if agent == "agent_laboratory":
        plan = final_output.get("plan")
        if not isinstance(plan, str) or not plan.strip():
            return None
        prefix = re.split(
            r"^\s*##\s*3\.?\s*Experimental\b.*$",
            plan,
            maxsplit=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )[0].strip()
        return _join_nonempty(
            [_extract_markdown_section(prefix, 1), _extract_markdown_section(prefix, 2)]
        ) or prefix
    if agent == "co_scientist":
        ranked = final_output.get("ranked_hypotheses")
        if not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], dict):
            return None
        top = ranked[0]
        experiments = top.get("experiments")
        if isinstance(experiments, list):
            experiments = "\n".join(str(item) for item in experiments)
        return _join_nonempty(
            [top.get("title"), top.get("hypothesis"), top.get("rationale"), experiments]
        )
    return _join_nonempty(
        [
            final_output.get("title") or final_output.get("Title"),
            final_output.get("hypothesis") or final_output.get("Short Hypothesis"),
            final_output.get("abstract") or final_output.get("Abstract"),
            final_output.get("method"),
        ]
    )


@dataclass
class Document:
    identifier: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _collect_unique_papers(canonical_root: Path) -> list[Document]:
    papers: OrderedDict[str, Document] = OrderedDict()
    paths = sorted(canonical_root.rglob("*.json"))
    LOG.info("Scanning %d canonical-state files from %s", len(paths), canonical_root)
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Skipping malformed canonical state %s: %s", path, exc)
            continue
        source = state.get("source") or {}
        for paper in state.get("memory_papers") or []:
            identifier = paper.get("paper_id") or paper.get("corpusid")
            if identifier is None or str(identifier) in papers:
                continue
            title = paper.get("title")
            abstract = paper.get("abstract")
            text = _join_nonempty([title, abstract])
            if not text:
                continue
            identifier = str(identifier)
            papers[identifier] = Document(
                identifier=identifier,
                text=text,
                metadata={
                    "paper_id": identifier,
                    "title": title,
                    "has_abstract": bool(abstract),
                    "context_id": paper.get("context_id") or source.get("context_id"),
                    "primary_field": paper.get("primary_field") or source.get("primary_field"),
                    "year": _integer(paper.get("year")),
                },
            )
    LOG.info("Collected %d unique papers", len(papers))
    return list(papers.values())


def _parse_run_path(path: Path, repo_root: Path) -> tuple[str | None, str | None, str | None]:
    try:
        parts = path.relative_to(repo_root).parts
    except ValueError:
        return None, None, None
    if len(parts) >= 6 and parts[0] == "runs":
        return parts[2], parts[3], parts[4]
    return None, None, None


def _collect_valid_ideas(runs_root: Path, repo_root: Path) -> list[Document]:
    documents: list[Document] = []
    dropped = {"bad_json": 0, "not_ok": 0, "no_text": 0}
    paths = sorted(runs_root.rglob("*.json"))
    LOG.info("Scanning %d run files from %s", len(paths), runs_root)
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dropped["bad_json"] += 1
            continue
        if row.get("status") and str(row.get("status")).lower() != "ok":
            dropped["not_ok"] += 1
            continue
        model, path_year, path_agent = _parse_run_path(path, repo_root)
        agent = str(row.get("agent") or path_agent or "unknown")
        text = extract_agent_text(agent, row.get("final_output"))
        if not text:
            dropped["no_text"] += 1
            continue
        relative = str(path.relative_to(repo_root))
        canonical = row.get("canonical_state") or {}
        source = canonical.get("source") or {}
        memory_ids = [
            str(paper.get("paper_id"))
            for paper in canonical.get("memory_papers") or []
            if paper.get("paper_id") is not None
        ]
        year = _integer(row.get("seed_year") or path_year or source.get("year"))
        run_id = str(row.get("run_id") or hashlib.sha1(relative.encode("utf-8")).hexdigest())
        documents.append(
            Document(
                identifier=run_id,
                text=text,
                metadata={
                    "run_id": run_id,
                    "run_path": relative,
                    "task_id": row.get("task_id"),
                    "agent": agent,
                    "model": row.get("model") or model,
                    "run_seed": row.get("run_seed"),
                    "anchor_paper_id": source.get("anchor_paper_id"),
                    "context_id": source.get("context_id"),
                    "primary_field": source.get("primary_field"),
                    "year": year,
                    "seed_year": year,
                    "memory_paper_ids": memory_ids,
                },
            )
        )
    LOG.info("Collected %d valid ideas; dropped=%s", len(documents), dropped)
    return documents


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        output = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        output.append(value)
        return output
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("rows", "data", "papers", "ideas"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    raise ValueError(f"Expected a JSON array or JSONL records in {path}")


def _documents_from_input(path: Path, kind: str) -> list[Document]:
    documents: list[Document] = []
    for position, row in enumerate(_read_rows(path)):
        if kind == "paper":
            identifier = row.get("paper_id") or row.get("corpusid") or row.get("id")
            text = row.get("text") or _join_nonempty([row.get("title"), row.get("abstract")])
        else:
            identifier = row.get("run_id") or row.get("id")
            text = row.get("text")
            if not text:
                text = extract_agent_text(str(row.get("agent") or ""), row.get("final_output"))
        if not text:
            continue
        if identifier is None:
            digest = hashlib.sha1(
                f"{position}:{json.dumps(row, sort_keys=True, ensure_ascii=False)}".encode("utf-8")
            ).hexdigest()
            identifier = digest
        identifier = str(identifier)
        metadata = dict(row)
        metadata["paper_id" if kind == "paper" else "run_id"] = identifier
        documents.append(Document(identifier=identifier, text=str(text), metadata=metadata))
    return documents


def _build_annotation_messages(doc_text: str, doc_kind: str, doc_id: str) -> list[dict[str, str]]:
    prompt = f"""Document kind: {doc_kind}
Document id: {doc_id}

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
}}

Rules:
- Output JSON only; do not wrap it in markdown fences.
- Include every analysis field exactly as shown above.
- keywords must be a non-empty list of 5-12 short scholarly noun phrases.
- If a field is uncertain, write a brief best-effort value rather than omitting it.

Research manuscript:
{doc_text.strip()}"""
    return [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _fit_messages(
    generator: Any,
    text: str,
    kind: str,
    identifier: str,
    max_input_tokens: int,
) -> tuple[list[dict[str, str]], bool]:
    trimmed = text.strip()
    truncated = False
    while True:
        messages = _build_annotation_messages(trimmed, kind, identifier)
        try:
            tokens = generator.count_message_tokens(messages)
        except Exception:
            tokens = 0
        if tokens <= max_input_tokens or len(trimmed) <= 2000:
            return messages, truncated
        trimmed = trimmed[: max(2000, len(trimmed) - max(1000, len(trimmed) // 8))].rstrip()
        trimmed += " ... [truncated]"
        truncated = True


def _normalize_keyword(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value.strip().strip(".,;:"))
    if not text or len(text) > 160:
        return None
    return text.lower()


@dataclass
class ParsedAnnotation:
    analysis: dict[str, str]
    keywords: list[str]
    parse_error: bool
    schema_errors: list[str] = field(default_factory=list)


def _parse_annotation(raw: str) -> ParsedAnnotation:
    if not raw or not raw.strip():
        return ParsedAnnotation({}, [], True, ["empty_response"])
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        try:
            value = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            value = None
    if not isinstance(value, dict):
        return ParsedAnnotation({}, [], True, ["invalid_json_object"])

    source = value.get("analysis")
    if not isinstance(source, dict):
        source = {}
    aliases = {
        "Questions addressed": ("Questions addressed", "Research Question", "Research Questions"),
        "Method": ("Method", "Methods", "Technical Method"),
    }
    analysis: dict[str, str] = {}
    schema_errors: list[str] = []
    for field_name in ANALYSIS_FIELDS:
        keys = aliases.get(field_name, (field_name,))
        field_value = next((source.get(key) for key in keys if source.get(key) is not None), None)
        if field_value is None or not str(field_value).strip():
            schema_errors.append(f"missing_analysis_field:{field_name}")
            analysis[field_name] = ""
        else:
            analysis[field_name] = str(field_value).strip()

    raw_keywords = value.get("keywords")
    if not isinstance(raw_keywords, list):
        raw_keywords = []
        schema_errors.append("keywords_not_list")
    keywords: list[str] = []
    seen: set[str] = set()
    for item in raw_keywords:
        keyword = _normalize_keyword(item)
        if keyword and keyword not in seen:
            seen.add(keyword)
            keywords.append(keyword)
    if not 5 <= len(keywords) <= 12:
        schema_errors.append(f"keyword_count:{len(keywords)}")
    keywords = keywords[:12]
    return ParsedAnnotation(analysis, keywords, bool(schema_errors), schema_errors)


@dataclass
class StageStats:
    n_total: int = 0
    n_skipped_resume: int = 0
    n_processed: int = 0
    n_parse_errors: int = 0
    n_input_truncated: int = 0
    elapsed_s: float = 0.0


def _chunked(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _completed_ids(path: Path, id_key: str) -> set[str]:
    if not path.exists():
        return set()
    output: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            llm_meta = row.get("llm_meta") or {}
            if row.get(id_key) is not None and not llm_meta.get("parse_error"):
                output.add(str(row[id_key]))
    return output


def _run_stage(
    *,
    generator: Any,
    documents: Sequence[Document],
    kind: str,
    output: Path,
    id_key: str,
    batch_size: int,
    max_input_tokens: int,
    max_output_tokens: int,
    temperature: float,
) -> StageStats:
    stats = StageStats(n_total=len(documents))
    completed = _completed_ids(output, id_key)
    pending = [document for document in documents if document.identifier not in completed]
    stats.n_skipped_resume = len(documents) - len(pending)
    output.parent.mkdir(parents=True, exist_ok=True)
    error_output = output.with_name(f"{output.stem}.errors{output.suffix}")
    started = time.time()
    with (
        output.open("a", encoding="utf-8") as handle,
        error_output.open("a", encoding="utf-8") as error_handle,
    ):
        for batch in _chunked(pending, batch_size):
            fitted = [
                _fit_messages(
                    generator,
                    document.text,
                    kind,
                    document.identifier,
                    max_input_tokens,
                )
                for document in batch
            ]
            responses = generator.generate(
                [messages for messages, _ in fitted],
                temperature=temperature,
                max_tokens=max_output_tokens,
                seed=0,
                on_error="empty",
            )
            for document, response, (_, truncated) in zip(batch, responses, fitted):
                parsed = _parse_annotation(response)
                stats.n_processed += 1
                stats.n_parse_errors += int(parsed.parse_error)
                stats.n_input_truncated += int(truncated)
                row = {
                    **document.metadata,
                    id_key: document.identifier,
                    "doc_kind": kind,
                    "analysis": parsed.analysis,
                    "keywords": parsed.keywords,
                    "llm_meta": {
                        "parse_error": parsed.parse_error,
                        "schema_errors": parsed.schema_errors,
                        "input_truncated": truncated,
                        "raw_chars": len(response or ""),
                        "keyword_count": len(parsed.keywords),
                    },
                }
                target = error_handle if parsed.parse_error else handle
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            error_handle.flush()
    stats.elapsed_s = time.time() - started
    return stats


def _resolve_documents(
    input_path: Path | None,
    fallback: Callable[[], list[Document]],
    kind: str,
) -> list[Document]:
    return _documents_from_input(input_path, kind) if input_path is not None else fallback()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--canonical-root", type=Path, default=Path("data/canonical_states/clean_main_batch"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/ideation_main"))
    parser.add_argument("--paper-input", type=Path, default=None, help="Optional generic paper JSON/JSONL input")
    parser.add_argument("--idea-input", type=Path, default=None, help="Optional generic idea JSON/JSONL input")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_out/keyword_extraction"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", DEFAULT_EXTRACTION_MODEL))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "2")))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "16384")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=28000)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stage", choices=("papers", "ideas", "both"), default="both")
    parser.add_argument("--limit-papers", type=int, default=0)
    parser.add_argument("--limit-ideas", type=int, default=0)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    canonical_root = resolve(args.canonical_root)
    runs_root = resolve(args.runs_root)
    paper_input = resolve(args.paper_input) if args.paper_input else None
    idea_input = resolve(args.idea_input) if args.idea_input else None
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VLLM_MODEL", args.model)
    os.environ.setdefault("TENSOR_PARALLEL_SIZE", str(args.tensor_parallel_size))
    os.environ.setdefault("MAX_MODEL_LEN", str(args.max_model_len))
    os.environ.setdefault("GPU_MEMORY_UTILIZATION", str(args.gpu_memory_utilization))
    os.environ.setdefault("VLLM_OFFLINE_BATCH_SIZE", str(args.batch_size))
    os.environ.setdefault("VLLM_OFFLINE_MAX_NUM_SEQS", str(args.batch_size))
    if args.disable_thinking:
        os.environ["VLLM_DISABLE_THINKING"] = "1"

    from scripts.analysis.keyword_extraction.vllm_backend import get_generator

    generator = get_generator()
    summary: dict[str, Any] = {
        "model": args.model,
        "schema": "scholarly_annotation_v1",
        "stages": {},
    }

    if args.stage in ("papers", "both"):
        papers = _resolve_documents(
            paper_input,
            lambda: _collect_unique_papers(canonical_root),
            "paper",
        )
        if args.limit_papers:
            papers = papers[: args.limit_papers]
        summary["stages"]["papers"] = _run_stage(
            generator=generator,
            documents=papers,
            kind="paper",
            output=out_dir / "paper_annotations.jsonl",
            id_key="paper_id",
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        ).__dict__

    if args.stage in ("ideas", "both"):
        ideas = _resolve_documents(
            idea_input,
            lambda: _collect_valid_ideas(runs_root, repo_root),
            "idea",
        )
        if args.limit_ideas:
            ideas = ideas[: args.limit_ideas]
        summary["stages"]["ideas"] = _run_stage(
            generator=generator,
            documents=ideas,
            kind="idea",
            output=out_dir / "idea_annotations.jsonl",
            id_key="run_id",
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        ).__dict__

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
