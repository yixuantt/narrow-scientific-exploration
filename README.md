# Narrow Scientific Exploration

Code for scholarly annotation and scientific-exploration measurements.

```bash
python scripts/analysis/keyword_extraction/extract_keywords.py --help

python -m scripts.analysis.measurements.breadth --help
python -m scripts.analysis.measurements.distance --help
python -m scripts.analysis.measurements.frontier --help
python -m scripts.analysis.measurements.impact --help
python -m scripts.analysis.measurements.novelty --help
python -m scripts.analysis.measurements.plot --help
python -m scripts.analysis.measurements.table --help
python -m scripts.analysis.run_measurements --help
```

Keyword extraction writes `paper_annotations.jsonl` and
`idea_annotations.jsonl`. Each row contains the complete scholarly `analysis`,
and 5--12 `keywords`. Downstream users choose which annotation fields to embed.
The extraction model is selected with `--model`.

Novelty annotation files are supplied independently to the novelty measurement;
the pipeline does not prescribe the annotator models.

Input and output paths are command-line arguments. Local data and generated
outputs are ignored by Git.
