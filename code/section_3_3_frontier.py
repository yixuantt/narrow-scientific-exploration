"""Section 3.3 -- Frontier Alignment: AI Ideas Undershoot the Research Frontier. (paper Table 4)

Each (primary_field, seed_year) task group's set of frontier terms is the top-10% most-growing keywords of the following year. Coverage is the fraction of a group's (agent, model, field, seed_year) keyword vocabulary that lands inside that frontier. AI ideas are compared with the follow-on human papers of the same tasks. Groups are averaged unweighted -- the same average-of-group-ratios reported for Section 3.5's by-agent/by-model rows.

Data: frontier/field_year_frontiers.parquet, frontier/idea_keywords.parquet, frontier/followon_keywords.parquet
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from common import load_parquet, save_report


def build_groups(idea_rows, followon_by_task):
    groups = defaultdict(lambda: {"idea_terms": set(), "followon_terms": set(), "tasks": set()})
    for row in idea_rows:
        key = (row["primary_field"], row["seed_year"], row["agent"], row["model"])
        g = groups[key]
        g["idea_terms"].update(row["keywords"])
        task_id = row["task_id"]
        if task_id not in g["tasks"]:
            g["tasks"].add(task_id)
            g["followon_terms"].update(followon_by_task.get(task_id, set()))
    return groups


def coverages(groups, frontier_terms):
    rows = []
    for (field, seed_year, agent, model), g in groups.items():
        terms = frontier_terms.get((field, seed_year + 1))
        if not terms or not g["followon_terms"]:
            continue
        rows.append(
            {
                "field": field,
                "seed_year": seed_year,
                "agent": agent,
                "model": model,
                "idea_coverage": len(g["idea_terms"] & terms) / len(terms),
                "human_coverage": len(g["followon_terms"] & terms) / len(terms),
            }
        )
    return rows


def mean_by(rows, key, field):
    groups = defaultdict(list)
    for r in rows:
        groups[r[field]].append(r[key])
    return {g: float(np.mean(v)) for g, v in groups.items()}


def main():
    frontiers = load_parquet("frontier/field_year_frontiers.parquet").to_dict("records")
    idea_rows = load_parquet("frontier/idea_keywords.parquet").to_dict("records")
    followon_rows = load_parquet("frontier/followon_keywords.parquet").to_dict("records")
    print(f"n_frontiers={len(frontiers)} idea_rows={len(idea_rows)} followon_rows={len(followon_rows)}")

    frontier_terms = {(r["primary_field"], r["frontier_year"]): set(r["frontier_terms"]) for r in frontiers}
    followon_by_task = {r["task_id"]: set(r["keywords"]) for r in followon_rows}

    groups = build_groups(idea_rows, followon_by_task)
    rows = coverages(groups, frontier_terms)

    idea_mean = float(np.mean([r["idea_coverage"] for r in rows]))
    human_mean = float(np.mean([r["human_coverage"] for r in rows]))
    print(f"\n[Section 3.3] Frontier-keyword coverage (Table 4), n_groups={len(rows)}:")
    print(f"  AI-generated ideas    = {idea_mean:.3f}")
    print(f"  Follow-on human papers = {human_mean:.3f}")
    print(f"  Difference             = {idea_mean - human_mean:+.3f}")

    # --- Section 3.5: consistency across agent frameworks and LLMs ---
    by_agent = mean_by(rows, "idea_coverage", "agent")
    print("\n[Section 3.5] Frontier coverage (AI) by agent framework:")
    for agent, value in sorted(by_agent.items()):
        print(f"  {agent:>25s}: {value:.3f}")

    by_model = mean_by(rows, "idea_coverage", "model")
    print("\n[Section 3.5] Frontier coverage (AI) by LLM:")
    for model, value in sorted(by_model.items()):
        print(f"  {model:>35s}: {value:.3f}")


if __name__ == "__main__":
    with save_report("section_3_3_frontier"):
        main()
