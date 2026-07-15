"""Section 3.1 -- Exploration Breadth: AI Ideas Are More Concentrated Than Human Papers. (paper Table 2)

Within each citation-defined research area (`context_id`), AI-generated ideas and human-authored papers are downsampled to equal counts, and breadth is 1 - mean pairwise cosine similarity within that matched sample. The headline "Pooled/All data" number averages equally across agent frameworks (also reused by Section 3.5, "Consistency Across Agent Frameworks and LLMs", to report the per-agent and per-LLM breakdowns of this same measure).

Data: breadth/idea_records.parquet, breadth/human_records.parquet
"""

from __future__ import annotations

import math

import numpy as np

from common import (
    group_balanced_mean,
    group_indices,
    load_parquet,
    mean_cross_cosine,
    mean_pairwise_cosine,
    stack_embeddings,
    weighted_mean,
)

# Within each research area, AI ideas and human papers are downsampled to equal
# counts (a matched comparison) using a fixed RNG seed. The seed only fixes *which*
# equal-sized subset is drawn per area; the AI < Human breadth gap is a property of
# the matched design, not of this particular draw, and is stable across seeds.
SEED = 42


def matched_same_area(idea_emb, idea_rows, human_emb, human_rows, dimension, group, seed):
    """Downsample AI and human records to equal counts within each research area."""
    rng = np.random.default_rng(seed)
    idea_contexts = group_indices(idea_rows, "context_id")
    human_contexts = group_indices(human_rows, "context_id")
    records = []
    matched = {}
    for context_id, ctx_idea_idx in sorted(idea_contexts.items()):
        if dimension != "pooled":
            ctx_idea_idx = [i for i in ctx_idea_idx if idea_rows[i].get(dimension) == group]
        ctx_human_idx = human_contexts.get(context_id, [])
        size = min(len(ctx_idea_idx), len(ctx_human_idx))
        if size < 2:
            continue
        ai_idx = sorted(rng.choice(ctx_idea_idx, size=size, replace=False).tolist())
        human_idx = sorted(rng.choice(ctx_human_idx, size=size, replace=False).tolist())
        ai_sim, ai_pairs = mean_pairwise_cosine(idea_emb[ai_idx])
        human_sim, human_pairs = mean_pairwise_cosine(human_emb[human_idx])
        records.append(
            {
                "context_id": context_id,
                "n_ai_pairs": ai_pairs,
                "n_human_pairs": human_pairs,
                "ai_breadth": 1.0 - ai_sim,
                "human_breadth": 1.0 - human_sim,
            }
        )
        matched[context_id] = {"ai": ai_idx, "human": human_idx}
    return records, matched


def cross_group_breadth_within_context(idea_emb, idea_rows, dimension):
    """Within the SAME research area, breadth between ideas from different groups
    (different agent frameworks, or different LLMs) -- paper Fig. 2c-d.
    """
    groups = sorted({r.get(dimension) for r in idea_rows if r.get(dimension)})
    idea_contexts = group_indices(idea_rows, "context_id")
    cell_sum = {(a, b): 0.0 for i, a in enumerate(groups) for b in groups[i + 1 :]}
    cell_count = {key: 0 for key in cell_sum}

    for _, ctx_idx in idea_contexts.items():
        members = {g: [i for i in ctx_idx if idea_rows[i].get(dimension) == g] for g in groups}
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                sim, pairs = mean_cross_cosine(idea_emb[members[left]], idea_emb[members[right]])
                if pairs:
                    cell_sum[(left, right)] += (1.0 - sim) * pairs
                    cell_count[(left, right)] += pairs

    cell_means = [cell_sum[k] / cell_count[k] for k in cell_sum if cell_count[k] > 0]
    return float(np.mean(cell_means)) if cell_means else math.nan


def pooled_same_area(idea_emb, idea_rows, human_emb, human_rows, dimension, groups):
    """Equal-weight-across-group pooled summary -- the 'Pooled/All data' row of Table 2."""
    per_group_ai, per_group_human, all_records = {}, {}, []
    for gi, group in enumerate(groups):
        records, _ = matched_same_area(idea_emb, idea_rows, human_emb, human_rows, dimension, group, SEED + gi)
        all_records.extend(records)
        per_group_ai[group] = weighted_mean([r["ai_breadth"] for r in records], [r["n_ai_pairs"] for r in records])
        per_group_human[group] = weighted_mean(
            [r["human_breadth"] for r in records], [r["n_human_pairs"] for r in records]
        )
    return group_balanced_mean(per_group_ai), group_balanced_mean(per_group_human), per_group_ai, per_group_human


def main():
    idea_df = load_parquet("breadth/idea_records.parquet")
    human_df = load_parquet("breadth/human_records.parquet")
    idea_rows = idea_df.drop(columns=["embedding"]).to_dict("records")
    human_rows = human_df.drop(columns=["embedding"]).to_dict("records")
    idea_emb = stack_embeddings(idea_df)
    human_emb = stack_embeddings(human_df)

    print(f"idea_rows={len(idea_rows)} human_rows={len(human_rows)}")

    # --- Headline pooled numbers (Table 2, "Pooled/All data") ---
    agents = sorted({r["agent"] for r in idea_rows})
    ai_mean, human_mean, by_agent_ai, by_agent_human = pooled_same_area(
        idea_emb, idea_rows, human_emb, human_rows, "agent", agents
    )
    print("\n[Section 3.1] Exploration breadth, same research area (Table 2):")
    print(f"  AI-generated ideas   = {ai_mean:.3f}")
    print(f"  Human-authored papers = {human_mean:.3f}")
    print(f"  Difference            = {ai_mean - human_mean:+.3f}")

    # --- Section 3.5: consistency across agent frameworks and LLMs ---
    print("\n[Section 3.5] Same-area breadth by agent framework:")
    for agent in agents:
        print(f"  {agent:>25s}: AI={by_agent_ai[agent]:.3f}  Human={by_agent_human[agent]:.3f}")

    models = sorted({r["model"] for r in idea_rows})
    _, _, by_model_ai, by_model_human = pooled_same_area(idea_emb, idea_rows, human_emb, human_rows, "model", models)
    print("\n[Section 3.5] Same-area breadth by LLM:")
    for model in models:
        print(f"  {model:>35s}: AI={by_model_ai[model]:.3f}  Human={by_model_human[model]:.3f}")

    # --- Cross-agent / cross-model breadth within the SAME research area (Fig. 2c-d) ---
    cross_agent_mean = cross_group_breadth_within_context(idea_emb, idea_rows, "agent")
    cross_model_mean = cross_group_breadth_within_context(idea_emb, idea_rows, "model")
    print(f"\n[Section 3.1] Cross-agent-framework breadth (same area) = {cross_agent_mean:.3f}")
    print(f"[Section 3.1] Cross-LLM breadth (same area)             = {cross_model_mean:.3f}")


if __name__ == "__main__":
    main()
