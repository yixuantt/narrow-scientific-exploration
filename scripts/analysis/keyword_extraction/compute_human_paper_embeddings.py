#!/usr/bin/env python3
"""Compute embeddings for human papers and their RQ1 baseline stats."""

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

random.seed(42)
np.random.seed(42)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


@torch.no_grad()
def encode_texts(texts, model, tokenizer, batch_size=64, max_length=4096):
    all_embs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding", total=total):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        pooled = summed / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--input", type=str, default="human_papers_by_context.json",
                        help="Input JSON file with papers by context")
    parser.add_argument("--output-prefix", type=str, default="human_paper",
                        help="Output filename prefix")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_out/keyword_extraction"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load human papers
    with open(out_dir / args.input) as f:
        ctx_data = json.load(f)

    texts = []
    meta = []
    for ctx in ctx_data:
        ctx_id = ctx["context_id"]
        for p in ctx["papers"]:
            texts.append(p["text"])
            meta.append({"context_id": ctx_id, "paper_id": p["paper_id"]})

    print(f"Encoding {len(texts)} human papers from {len(ctx_data)} contexts")

    model, tokenizer = load_model(args.model)
    embs = encode_texts(texts, model, tokenizer, batch_size=args.batch_size, max_length=args.max_length)

    np.save(out_dir / f"{args.output_prefix}_embeddings.npy", embs.numpy())
    with open(out_dir / f"{args.output_prefix}_embeddings_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings: {embs.shape}")

    # Compute stats
    ctx_groups = defaultdict(list)
    for idx, m in enumerate(meta):
        ctx_groups[m["context_id"]].append(idx)

    within_sims = []
    out_sims = []

    # Within-context pairs
    for ctx_id, idxs in ctx_groups.items():
        if len(idxs) < 2:
            continue
        sub = embs[idxs]
        sim_mat = sub @ sub.T
        triu = np.triu_indices(len(idxs), k=1)
        within_sims.extend(sim_mat[triu].tolist())

    # Out-of-context pairs (sample)
    ctx_ids = list(ctx_groups.keys())
    all_idxs = list(range(len(meta)))
    n_sample = min(500_000, len(all_idxs) * (len(all_idxs) - 1) // 2)
    sampled = 0
    ctx_arr = [meta[i]["context_id"] for i in all_idxs]
    while sampled < n_sample:
        a, b = random.sample(all_idxs, 2)
        if ctx_arr[a] != ctx_arr[b]:
            out_sims.append((embs[a] @ embs[b]).item())
            sampled += 1

    print("\n" + "=" * 50)
    print("HUMAN PAPERS baseline")
    print("=" * 50)
    print(f"Within-context:  mean={np.mean(within_sims):.4f}, std={np.std(within_sims):.4f}, n={len(within_sims):,}")
    print(f"Out-of-context:  mean={np.mean(out_sims):.4f}, std={np.std(out_sims):.4f}, n={len(out_sims):,}")
    print(f"Gap:             {np.mean(within_sims) - np.mean(out_sims):.4f}")


if __name__ == "__main__":
    main()
