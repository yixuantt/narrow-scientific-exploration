# AI Scientist analysis code

This repository contains code needed to reproduce the manuscript pipeline and figures. Data, generated runs, intermediate outputs, and paper drafts are not included.

## Contents

- `scripts/pipeline/`: corpus construction, citation graph construction, research-area construction, and canonical seed-paper state generation.
- `run_unified_ideation.py`, `unified_ideation.py`, `run_inference_vllm.py`: unified runner for the zero-shot baseline, AIScientist, ResearchAgent, and AgentLaboratory frameworks across LLMs.
- `external/`: bundled source files from AIScientist, ResearchAgent, and AgentLaboratory that are imported by the unified runner. There are no internal fallback prompts for these frameworks.
- `scripts/analysis/`: follow-on-paper corpora, embeddings, keyword extraction, statistics, and figure scripts used in the paper.
- `scripts/common/`: shared style and analysis utilities.

## DBLP Citation Pipeline

```bash
python scripts/pipeline/build_master_corpus.py --help
python scripts/pipeline/intersect_dblp_with_master.py --help
python scripts/pipeline/build_graph_embedding_contexts.py --help
python scripts/pipeline/build_canonical_states.py --help

python run_unified_ideation.py --help

python scripts/analysis/keyword_extraction/extract_keywords.py --help
python scripts/analysis/keyword_extraction/plot_idea_concentration.py --help
python scripts/analysis/keyword_extraction/plot_future_alignment_bars.py --help
python scripts/analysis/keyword_extraction/plot_task_method_variation.py --help
```

The released pipeline builds citation-graph research areas from DBLP citation
records. The `build_graph_embedding_contexts.py` step uses bibliographic
coupling with SVD (`bcsvd`) and writes the DBLP-id paper corpus, graph edges,
research-area assignments, and context index consumed by `build_canonical_states.py`.
It removes the largest heterogeneous HDBSCAN cluster and retains only research
areas active in every target year, matching the manuscript's main-analysis
research-area construction.
Canonical seed-paper states are then sampled directly from those DBLP
citation-defined research areas: for each anchor year, only earlier-or-current
papers are visible, outside-graph and graph-isolate bins are excluded by
default, small or inactive areas are dropped, and each retained year-by-area
cell yields seed sets containing one anchor paper plus related memory papers
chosen by citation and lexical similarity.

The data paths are command-line arguments. The ignored directories (`data/`, `corpus/`, `runs/`, `analysis_out/`, `results/`) are local working directories.

All text-embedding analyses use `Qwen/Qwen3-Embedding-4B`, including generated
ideas, follow-on papers, seed papers, and human-paper baselines.
Research-question and method extraction uses `google/gemma-4-31B-it` by default,
matching the Gemma-4-31B-IT extractor described in the manuscript.

The unified runner requires the bundled `external/` source files when AIScientist, ResearchAgent, or AgentLaboratory is used. Install each framework's requirements before running those agents.
