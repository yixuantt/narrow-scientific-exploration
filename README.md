# AI Research Agents Narrow Scientific Exploration -- Empirical Analysis Code

Reproduces the headline statistics reported in the paper's Empirical Analysis section (Sections 3.1-3.6).

## Layout

```
code/
  common.py                  Shared HF-fetch and vector-statistics helpers
  section_3_1_breadth.py      Section 3.1 -- Exploration Breadth (Table 2)
  section_3_2_distance.py     Section 3.2 -- Exploration Distance (Table 3, Fig. 3)
  section_3_3_frontier.py     Section 3.3 -- Frontier Alignment (Table 4)
  section_3_4_impact.py       Section 3.4 -- Potential Scientific Impact (Table 5)
  section_3_6_novelty.py      Section 3.6 -- Novelty (new methods vs. new research questions)
  requirements.txt
  run                         Runs all five scripts in sequence
```

Section 3.5 ("Consistency Across Agent Frameworks and LLMs") is not a separate script: it reuses the same four measures (breadth, distance, frontier, impact) broken down by agent framework and by LLM, and each section script above prints that breakdown after its pooled headline number.

## Data

Each script reads small, analysis-ready parquet inputs from the public Hugging Face dataset. Those files are derived from the full raw run output published on Zenodo (`ideation.parquet`, https://zenodo.org/records/21201261).

## Running

```bash
pip install -r code/requirements.txt
cd code
./run
```

or run an individual section, e.g.:

```bash
python3 code/section_3_1_breadth.py
```
