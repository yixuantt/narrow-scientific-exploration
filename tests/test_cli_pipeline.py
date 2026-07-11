from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, *arguments: object) -> None:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    subprocess.run(
        [sys.executable, "-m", module, *(str(argument) for argument in arguments)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class CommandPipelineTests(unittest.TestCase):
    def test_breadth_to_plot_and_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            idea_vectors = []
            idea_rows = []
            human_vectors = []
            human_rows = []
            bases = {
                "c1": np.asarray([1.0, 0.0, 0.0]),
                "c2": np.asarray([0.0, 1.0, 0.0]),
            }
            for context, base in bases.items():
                for agent_pos, agent in enumerate(("a1", "a2")):
                    for repeat in range(2):
                        vector = base + np.asarray([0.0, 0.0, 0.1 * (agent_pos + repeat + 1)])
                        idea_vectors.append(vector / np.linalg.norm(vector))
                        idea_rows.append(
                            {
                                "run_id": f"{context}-{agent}-{repeat}",
                                "context_id": context,
                                "primary_field": "field",
                                "agent": agent,
                                "model": f"m{agent_pos + 1}",
                            }
                        )
                for repeat in range(4):
                    vector = base + np.asarray([0.1 * (repeat % 2), 0.0, 0.15 * (repeat + 1)])
                    human_vectors.append(vector / np.linalg.norm(vector))
                    human_rows.append(
                        {
                            "paper_id": f"{context}-p{repeat}",
                            "context_id": context,
                            "primary_field": "field",
                        }
                    )
            idea_embeddings = root / "ideas.npy"
            human_embeddings = root / "humans.npy"
            idea_meta = root / "ideas.json"
            human_meta = root / "humans.json"
            np.save(idea_embeddings, np.asarray(idea_vectors, dtype=np.float32))
            np.save(human_embeddings, np.asarray(human_vectors, dtype=np.float32))
            idea_meta.write_text(json.dumps(idea_rows), encoding="utf-8")
            human_meta.write_text(json.dumps(human_rows), encoding="utf-8")
            out_dir = root / "out"

            run_module(
                "scripts.analysis.measurements.breadth",
                "--idea-embeddings",
                idea_embeddings,
                "--idea-meta",
                idea_meta,
                "--human-embeddings",
                human_embeddings,
                "--human-meta",
                human_meta,
                "--out-dir",
                out_dir,
                "--bootstrap-repetitions",
                20,
            )
            summary = out_dir / "exploration_breadth_summary.json"
            figure = root / "breadth.pdf"
            csv_path = root / "breadth.csv"
            tex_path = root / "breadth.tex"
            run_module(
                "scripts.analysis.measurements.plot",
                "--measure",
                "breadth",
                "--summary",
                summary,
                "--group",
                "agent",
                "--include-matrix",
                "--out",
                figure,
            )
            run_module(
                "scripts.analysis.measurements.table",
                "--measure",
                "breadth",
                "--summary",
                summary,
                "--group",
                "agent",
                "--csv-out",
                csv_path,
                "--tex-out",
                tex_path,
            )
            self.assertTrue(figure.exists())
            self.assertIn(r"\begin{tabular}", tex_path.read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 3)

    def test_novelty_vote_to_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = [
                ("r1", 1, 1),
                ("r2", 0, 1),
                ("r3", 1, 0),
                ("r4", 0, 0),
            ]
            files = []
            for annotator in range(3):
                path = root / f"annotator_{annotator}.jsonl"
                rows = []
                for run_id, question, method in base:
                    rows.append(
                        {
                            "run_id": run_id,
                            "agent": "a",
                            "model": "m",
                            "primary_field": "f",
                            "new_research_question": question,
                            "new_method": method if annotator < 2 or run_id != "r2" else 0,
                        }
                    )
                write_jsonl(path, rows)
                files.append(path)
            out_dir = root / "out"
            run_module(
                "scripts.analysis.measurements.novelty",
                "--annotator-files",
                *files,
                "--out-dir",
                out_dir,
            )
            summary_path = out_dir / "novelty_majority_summary.json"
            table_path = root / "novelty.csv"
            run_module(
                "scripts.analysis.measurements.table",
                "--measure",
                "novelty",
                "--summary",
                summary_path,
                "--group",
                "pooled",
                "--csv-out",
                table_path,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))["summaries"]["overall"]
            self.assertEqual(summary["new_research_question"]["share"], 0.5)
            self.assertEqual(summary["new_method"]["share"], 0.5)
            self.assertEqual(summary["both_new"]["share"], 0.25)

    def test_impact_uses_task_level_agent_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            human_vectors = np.asarray(
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], dtype=np.float32
            )
            human_vectors /= np.linalg.norm(human_vectors, axis=1, keepdims=True)
            human_rows = [
                {"paper_id": "p0", "context_id": "c", "year": 2019, "citation_count": 0},
                {"paper_id": "p1", "context_id": "c", "year": 2020, "citation_count": 2},
                {"paper_id": "p2", "context_id": "c", "year": 2020, "citation_count": 8},
                {"paper_id": "p3", "context_id": "c", "year": 2020, "citation_count": 4},
            ]
            idea_vectors = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
            )
            idea_vectors /= np.linalg.norm(idea_vectors, axis=1, keepdims=True)
            idea_rows = [
                {"run_id": "r0", "task_id": "t0", "context_id": "c", "seed_year": 2020, "agent": "a", "model": "m"},
                {"run_id": "r1", "task_id": "t0", "context_id": "c", "seed_year": 2020, "agent": "a", "model": "m"},
                {"run_id": "r2", "task_id": "t1", "context_id": "c", "seed_year": 2020, "agent": "a", "model": "m"},
            ]
            links = [
                {"task_id": "t0", "paper_id": "p2", "context_id": "c", "seed_year": 2020},
                {"task_id": "t1", "paper_id": "p3", "context_id": "c", "seed_year": 2020},
            ]
            np.save(root / "human.npy", human_vectors)
            np.save(root / "ideas.npy", idea_vectors)
            (root / "human.json").write_text(json.dumps(human_rows), encoding="utf-8")
            (root / "ideas.json").write_text(json.dumps(idea_rows), encoding="utf-8")
            write_jsonl(root / "links.jsonl", links)
            out_dir = root / "out"
            run_module(
                "scripts.analysis.measurements.impact",
                "--idea-embeddings", root / "ideas.npy",
                "--idea-meta", root / "ideas.json",
                "--human-embeddings", root / "human.npy",
                "--human-meta", root / "human.json",
                "--followon-embeddings", root / "human.npy",
                "--followon-meta", root / "human.json",
                "--followon-links", root / "links.jsonl",
                "--out-dir", out_dir,
                "--neighbors", 2,
                "--bootstrap-repetitions", 20,
                "--skip-validation",
            )
            summary_path = out_dir / "potential_impact_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["summaries"]["agent"],
                summary["task_level_summaries"]["agent"],
            )
            self.assertEqual(
                summary["summaries"]["pooled"],
                summary["idea_level_summaries"]["pooled"],
            )
            table_path = root / "impact.csv"
            run_module(
                "scripts.analysis.measurements.table",
                "--measure", "impact",
                "--summary", summary_path,
                "--group", "agent",
                "--csv-out", table_path,
            )
            with table_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
