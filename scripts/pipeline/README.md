# DBLP Citation Pipeline

This directory contains the released corpus-to-task pipeline. It uses the
DBLP citation graph and constructs research areas with bibliographic coupling
plus SVD (`bcsvd`).

## Stages

```bash
# 1. Merge venue/year paper JSON files into the master corpus.
python scripts/pipeline/build_master_corpus.py --help

# 2. Keep DBLP citation-network rows that match the master corpus by title/year.
python scripts/pipeline/intersect_dblp_with_master.py --help

# 3. Build bcsvd research-area assignments from the DBLP citation graph.
python scripts/pipeline/build_graph_embedding_contexts.py --help

# 4. Sample canonical seed-paper states from the DBLP research areas.
python scripts/pipeline/build_canonical_states.py --help
```

`build_graph_embedding_contexts.py` writes the companion paper corpus, graph
edges, context assignments, and context index. Its default research-area filters
match the manuscript: after HDBSCAN, the largest non-noise cluster is treated as
heterogeneous and removed, and only clusters containing papers from every target
year are retained. By default, `build_canonical_states.py` reads the
`matched_all_master_g22_25_bcsvd_hdbscan` context bundle produced by that step.

The canonical-state step follows the manuscript pipeline directly: for each
target anchor year, it exposes only papers from that year or earlier, drops
outside-graph and graph-isolate bins by default, removes contexts that are too
small or have no eligible anchor papers in the target year, ranks the remaining
DBLP citation-defined research areas by available anchors, and samples seed
sets from the selected year-by-area cells. Each seed set contains one anchor
paper plus related in-context memory papers selected from citation and lexical
similarity signals.
