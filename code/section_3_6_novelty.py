"""Section 3.6 -- Novelty: AI Ideas Propose New Methods More Often Than New Research Questions.

Three LLM annotators independently label each AI-generated idea for whether it poses a new research question and/or a new method; a label counts only when at least two of the three votes agree (majority_vote). The headline share is the fraction of majority-labeled ideas with each novelty type, pooled over all ideas and broken down by agent/model/field for Section 3.5.

Data: novelty/novelty_labels.parquet
"""

from __future__ import annotations

from collections import defaultdict

from common import load_parquet


def pack(rows: list[dict]) -> dict:
    n = len(rows)
    question_count = sum(int(r["majority_new_research_question"]) for r in rows)
    method_count = sum(int(r["majority_new_method"]) for r in rows)
    return {
        "n": n,
        "new_research_question_share": question_count / n if n else None,
        "new_method_share": method_count / n if n else None,
    }


def grouped(rows: list[dict], field: str) -> dict[str, dict]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(field) or "unknown"].append(r)
    return {name: pack(group_rows) for name, group_rows in sorted(buckets.items())}


def main():
    df = load_parquet("novelty/novelty_labels.parquet")
    print(f"n_rows={len(df)}")

    valid = df.dropna(subset=["majority_new_research_question", "majority_new_method"])
    print(f"n_valid (>= 2 agreeing annotator votes)={len(valid)}")
    rows = valid.to_dict("records")

    overall = pack(rows)
    print("\n[Section 3.6] Novelty of AI-generated ideas (pooled):")
    print(f"  New research question = {overall['new_research_question_share']:.3f}")
    print(f"  New method             = {overall['new_method_share']:.3f}")

    # --- Section 3.5: consistency across agent frameworks and LLMs ---
    by_agent = grouped(rows, "agent")
    print("\n[Section 3.5] Novelty (new research question / new method) by agent framework:")
    for agent, stats in by_agent.items():
        print(f"  {agent:>25s}: question={stats['new_research_question_share']:.3f}  method={stats['new_method_share']:.3f}")

    by_model = grouped(rows, "model")
    print("\n[Section 3.5] Novelty (new research question / new method) by LLM:")
    for model, stats in by_model.items():
        print(f"  {model:>35s}: question={stats['new_research_question_share']:.3f}  method={stats['new_method_share']:.3f}")

    by_field = grouped(rows, "primary_field")
    print("\n[Supplementary] Novelty by research field:")
    for field, stats in by_field.items():
        print(f"  {field:>15s}: question={stats['new_research_question_share']:.3f}  method={stats['new_method_share']:.3f}")


if __name__ == "__main__":
    main()
