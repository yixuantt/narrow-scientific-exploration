import argparse
from pathlib import Path
from typing import List, Optional

from unified_ideation import ROOT, build_output_path, read_json, run_and_log, run_grid_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI research-agent frameworks on canonical seed-paper states.")
    parser.add_argument(
        "--canonical-state",
        type=Path,
        default=ROOT / "examples" / "canonical_state.sample.json",
        help="Path to the canonical seed-paper state JSON file.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        help="One framework: all, flat_llm, ai_scientist_v2, research_agent, agent_laboratory. all runs all four.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit framework list. Overrides --agent and is mainly for --grid-path mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "ideation",
        help="Directory for unified JSON logs.",
    )
    parser.add_argument(
        "--llm",
        dest="model",
        type=str,
        default=None,
        help="LLM name or endpoint model identifier. Overrides OPENAI_MODEL from .env.",
    )
    parser.add_argument("--model", dest="model", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Reflection rounds for AI Scientist v2.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Problem/method/experiment iterations for ResearchAgent.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum dialogue turns for Agent Laboratory plan formulation.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Run seed used for repeated experimental units.",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        nargs="*",
        default=None,
        help="Optional experiment_grid.jsonl path(s). When set, run the whole grid in one Python process.",
    )
    parser.add_argument(
        "--task-parallelism",
        type=int,
        default=8,
        help="Thread-pool width for --grid-path mode. Keep this modest; the shared vLLM batcher handles prompt batching.",
    )
    parser.add_argument(
        "--include-original-seed-set",
        type=int,
        default=1,
        choices=[0, 1],
        help="In --grid-path mode, include the original seed-paper set before resampled runs.",
    )
    parser.add_argument("--rq1a-baseline-memory", dest="include_original_seed_set", type=int, choices=[0, 1], default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-set-resample-seed",
        dest="memory_resample_seed",
        type=int,
        nargs="*",
        metavar="SEED",
        default=None,
        help="Redraw seed-paper sets from the corpus. In single-run mode pass one seed; in --grid-path mode pass zero or more seeds.",
    )
    parser.add_argument("--memory-resample-seed", dest="memory_resample_seed", type=int, nargs="*", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-paper-corpus",
        dest="memory_corpus",
        type=Path,
        metavar="PATH",
        default=None,
        help=(
            "JSONL corpus for seed-paper resampling. Defaults to the DBLP companion "
            "paper corpus emitted by build_graph_embedding_contexts.py."
        ),
    )
    parser.add_argument("--memory-corpus", dest="memory_corpus", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-paper-min-research-area-size",
        dest="memory_min_context_size",
        type=int,
        metavar="N",
        default=5,
        help="Minimum research-area size when tagging the corpus for seed-paper resampling.",
    )
    parser.add_argument("--memory-min-research-area-size", dest="memory_min_context_size", type=int, metavar="N", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--memory-min-context-size", dest="memory_min_context_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-paper-type",
        dest="memory_paper_type",
        type=str,
        default="all",
        choices=["all", "oral", "poster"],
        help="Corpus filter for seed-paper resampling (match build_canonical_states if needed).",
    )
    parser.add_argument("--memory-paper-type", dest="memory_paper_type", type=str, choices=["all", "oral", "poster"], default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-paper-corpus-years",
        dest="memory_corpus_years",
        type=int,
        nargs="*",
        metavar="YEAR",
        default=None,
        help="Optional year filter on corpus before tagging (e.g. match --years used when building canonical states).",
    )
    parser.add_argument("--memory-corpus-years", dest="memory_corpus_years", type=int, nargs="*", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-paper-pool-multiplier",
        dest="memory_pool_multiplier",
        type=int,
        metavar="N",
        default=8,
        help="Sample non-anchor seed papers from top ranked neighbors.",
    )
    parser.add_argument("--memory-pool-multiplier", dest="memory_pool_multiplier", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-existing",
        type=int,
        default=0,
        choices=[0, 1],
        help="Skip tasks whose target output json already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.agents:
        agents = args.agents
    elif args.agent == "all":
        agents = ["flat_llm", "ai_scientist_v2", "research_agent", "agent_laboratory"]
    else:
        agents = [args.agent]

    corpus_years: Optional[List[int]] = args.memory_corpus_years if args.memory_corpus_years else None
    memory_resample_seeds = args.memory_resample_seed or []

    if args.grid_path:
        results = run_grid_experiments(
            grid_paths=args.grid_path,
            output_root=args.output_dir,
            agents=agents,
            model=args.model,
            reflections=args.reflections,
            iterations=args.iterations,
            max_steps=args.max_steps,
            task_parallelism=args.task_parallelism,
            memory_resample_seeds=memory_resample_seeds,
            rq1a_baseline_memory=bool(args.include_original_seed_set),
            memory_corpus_path=args.memory_corpus,
            memory_min_context_size=args.memory_min_context_size,
            memory_paper_type=args.memory_paper_type,
            memory_corpus_years=corpus_years,
            memory_pool_multiplier=args.memory_pool_multiplier,
            skip_existing=bool(args.skip_existing),
        )
        for task, output_path in results:
            mem_seed = task["memory_resample_seed"]
            variant = "canonical" if mem_seed is None else f"mem_{mem_seed}"
            action = "skipped" if task.get("skipped") else "wrote"
            print(
                f"[{task['agent']}] {action} {task['canonical_state_path'].name} "
                f"seed={task['run_seed']} variant={variant} -> {output_path}",
                flush=True,
            )
        return

    for agent in agents:
        if len(memory_resample_seeds) > 1:
            raise ValueError("Single-run mode accepts at most one --seed-set-resample-seed. Use --grid-path for multiple seeds.")
        memory_resample_seed = memory_resample_seeds[0] if memory_resample_seeds else None
        canonical_state = read_json(args.canonical_state)
        output_path = build_output_path(
            task_id=canonical_state.get("task_id", "task"),
            agent=agent,
            output_dir=args.output_dir,
            run_seed=args.run_seed,
            memory_resample_seed=memory_resample_seed,
        )
        if args.skip_existing and output_path.exists():
            print(f"[{agent}] skipped existing unified log at {output_path}", flush=True)
            continue
        output_path = run_and_log(
            agent=agent,
            canonical_state_path=args.canonical_state,
            output_dir=args.output_dir,
            model=args.model,
            reflections=args.reflections,
            iterations=args.iterations,
            max_steps=args.max_steps,
            run_seed=args.run_seed,
            memory_resample_seed=memory_resample_seed,
            memory_corpus_path=args.memory_corpus,
            memory_min_context_size=args.memory_min_context_size,
            memory_paper_type=args.memory_paper_type,
            memory_corpus_years=corpus_years,
            memory_pool_multiplier=args.memory_pool_multiplier,
            skip_existing=bool(args.skip_existing),
        )
        print(f"[{agent}] wrote unified log to {output_path}", flush=True)


if __name__ == "__main__":
    main()
