"""Section 3.4 -- Potential Scientific Impact: AI Ideas Score Lower on a Citation-Neighborhood Proxy. (paper Table 5)

Impact is proxied by `local_log_residual`: the log-citation residual of a record relative to same-context, same-vintage neighbors. The headline pooled number is the simple (unpaired) mean over all AI ideas vs. all follow-on human papers. Section 3.5's by-agent/by-LLM breakdown instead pairs each seed task's AI subgroup mean against that task's human mean, then averages across tasks. The proxy's validity is checked by predicting each later human paper's score from its strictly-earlier same-context neighbors and Spearman-correlating the prediction with the paper's actual score.

Data: impact/ai_scores.parquet, impact/followon_scores.parquet, impact/validation.parquet
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from common import load_parquet

SCORE_COL = "local_log_residual"


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 4 or np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        return None
    rho = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
    return rho if np.isfinite(rho) else None


def task_group_records(idea_rows, human_by_task, group_fields):
    """One row per (task, dimension, group): mean AI score for that subgroup vs.
    the task's overall mean human (follow-on) score.
    """
    ideas_by_task = defaultdict(list)
    for row in idea_rows:
        task_id = row.get("task_id")
        if task_id in human_by_task:
            ideas_by_task[task_id].append(row)

    records = []
    for task_id, rows in ideas_by_task.items():
        human_mean = float(np.mean(human_by_task[task_id]))
        dimensions = [("pooled", "all", rows)]
        for field in group_fields:
            values = sorted({row[field] for row in rows if row.get(field)})
            for value in values:
                dimensions.append((field, value, [row for row in rows if row.get(field) == value]))
        for dimension, group, subgroup in dimensions:
            records.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "ai_impact": float(np.mean([row[SCORE_COL] for row in subgroup])),
                    "human_impact": human_mean,
                }
            )
    return records


def mean_by(records, key, dimension):
    groups = defaultdict(list)
    for r in records:
        if r["dimension"] == dimension:
            groups[r["group"]].append(r[key])
    return {g: float(np.mean(v)) for g, v in groups.items()}


def main():
    ai_scores = load_parquet("impact/ai_scores.parquet")
    followon_scores = load_parquet("impact/followon_scores.parquet")
    validation = load_parquet("impact/validation.parquet")
    print(f"ai_rows={len(ai_scores)} followon_rows={len(followon_scores)} validation_rows={len(validation)}")

    # --- Headline pooled numbers (Table 5): simple unpaired means ---
    ai_mean = float(ai_scores[SCORE_COL].mean())
    human_mean = float(followon_scores[SCORE_COL].mean())
    print("\n[Section 3.4] Potential scientific impact proxy (Table 5):")
    print(f"  AI-generated ideas    = {ai_mean:.3f}")
    print(f"  Follow-on human papers = {human_mean:.3f}")
    print(f"  Difference             = {ai_mean - human_mean:+.3f}")

    # --- Proxy validation: predict later human papers from earlier same-context neighbors ---
    predicted = validation["predicted_local_log_residual"].to_numpy(dtype=float)
    target = validation["target_log_residual"].to_numpy(dtype=float)
    rho = spearman(predicted, target)
    print(f"\n[Section 3.4] Proxy validation on human landscape: n={len(validation)}, Spearman rho={rho:.3f}")

    # --- Section 3.5: consistency across agent frameworks and LLMs (task-paired means) ---
    idea_rows = ai_scores.to_dict("records")
    human_by_task = defaultdict(list)
    for row in followon_scores.to_dict("records"):
        human_by_task[row["task_id"]].append(row[SCORE_COL])

    records = task_group_records(idea_rows, human_by_task, group_fields=["agent", "model"])

    by_agent = mean_by(records, "ai_impact", "agent")
    print("\n[Section 3.5] Impact proxy (AI) by agent framework:")
    for agent, value in sorted(by_agent.items()):
        print(f"  {agent:>25s}: {value:.3f}")

    by_model = mean_by(records, "ai_impact", "model")
    print("\n[Section 3.5] Impact proxy (AI) by LLM:")
    for model, value in sorted(by_model.items()):
        print(f"  {model:>35s}: {value:.3f}")


if __name__ == "__main__":
    main()
