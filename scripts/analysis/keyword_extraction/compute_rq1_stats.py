#!/usr/bin/env python3
"""Compute RQ1 statistics from precomputed embeddings (no GPU needed)."""

import json
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    args = parser.parse_args()

    embs = np.load(args.emb)
    with open(args.meta) as f:
        meta = json.load(f)

    print(f"Loaded embeddings: {embs.shape}, meta: {len(meta)}")

    # Group by context
    ctx_groups = defaultdict(list)
    for idx, m in enumerate(meta):
        ctx_groups[m["context_id"]].append((idx, m))

    categories = defaultdict(list)
    for ctx_id, members in tqdm(ctx_groups.items(), desc="contexts"):
        if len(members) < 10:
            continue
        for (i, mi), (j, mj) in combinations(members, 2):
            sim = (embs[i] @ embs[j]).item()
            same_model = mi["model"] == mj["model"]
            same_agent = mi["agent"] == mj["agent"]
            same_anchor = mi.get("anchor", "") == mj.get("anchor", "") and mi.get("anchor", "")

            if same_model and same_agent and not same_anchor:
                cat = "intra"
            elif same_model and not same_agent:
                cat = "same_model_diff_agent"
            elif not same_model and same_agent:
                cat = "diff_model_same_agent"
            elif not same_model and not same_agent:
                cat = "diff_model_diff_agent"
            else:
                continue
            categories[cat].append(sim)

    print("\n" + "=" * 70)
    print("RQ1: Semantic similarity within the same research context")
    print("=" * 70)
    for cat in ["intra", "same_model_diff_agent", "diff_model_same_agent", "diff_model_diff_agent"]:
        vals = categories[cat]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        print(f"\n{cat}:")
        print(f"  N pairs: {len(vals):,}")
        print(f"  Mean cosine similarity: {mean:.4f}")
        print(f"  Std: {std:.4f}")

    baseline = sum(categories["intra"]) / len(categories["intra"])
    print("\n--- Effect sizes (drop from intra-group baseline) ---")
    for cat in ["same_model_diff_agent", "diff_model_same_agent", "diff_model_diff_agent"]:
        vals = categories[cat]
        if vals:
            drop = baseline - (sum(vals) / len(vals))
            print(f"  {cat}: {drop:.4f}")

    print("\n--- Per-model effect (fixed agent, diff_model_same_agent) ---")
    model_sims = defaultdict(list)
    for ctx_id, members in ctx_groups.items():
        for (i, mi), (j, mj) in combinations(members, 2):
            if mi["model"] != mj["model"] and mi["agent"] == mj["agent"]:
                sim = (embs[i] @ embs[j]).item()
                model_sims[mi["agent"]].append(sim)
    for agent, vals in sorted(model_sims.items()):
        print(f"  {agent}: {sum(vals)/len(vals):.4f} (n={len(vals):,})")

    print("\n--- Per-agent effect (fixed model, same_model_diff_agent) ---")
    agent_sims = defaultdict(list)
    for ctx_id, members in ctx_groups.items():
        for (i, mi), (j, mj) in combinations(members, 2):
            if mi["model"] == mj["model"] and mi["agent"] != mj["agent"]:
                sim = (embs[i] @ embs[j]).item()
                agent_sims[mi["model"]].append(sim)
    for model, vals in sorted(agent_sims.items()):
        short = model.replace("meta_llama_", "").replace("google_", "").replace("qwen_", "").replace("_instruct", "")
        print(f"  {short}: {sum(vals)/len(vals):.4f} (n={len(vals):,})")


if __name__ == "__main__":
    main()
