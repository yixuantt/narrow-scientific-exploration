from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis.measurements.breadth import matched_context_records
from scripts.analysis.measurements.common import mean_pairwise_cosine
from scripts.analysis.measurements.distance import build_task_records
from scripts.analysis.measurements.frontier import build_frontiers, build_records
from scripts.analysis.measurements.impact import citation_baselines, score_followons
from scripts.analysis.measurements.novelty import category, pack, vote_for


class CommonTests(unittest.TestCase):
    def test_pairwise_cosine(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)
        mean, count = mean_pairwise_cosine(vectors)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(mean, 1 / 3)


class BreadthTests(unittest.TestCase):
    def test_equal_count_matching(self) -> None:
        ideas = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)
        humans = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        idea_rows = [
            {"context_id": "c1", "agent": "a", "primary_field": "f"}
            for _ in range(3)
        ]
        human_rows = [{"context_id": "c1", "primary_field": "f"} for _ in range(2)]
        records, matched = matched_context_records(
            ideas,
            idea_rows,
            humans,
            human_rows,
            dimension="agent",
            group="a",
            context_key="context_id",
            field_key="primary_field",
            seed=1,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["n_ai"], 2)
        self.assertEqual(len(matched["c1"]["human"]), 2)


class DistanceTests(unittest.TestCase):
    def test_distance_from_seed_centroid(self) -> None:
        seeds = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float)
        ideas = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        followons = np.asarray([[0.0, 1.0]], dtype=float)
        seed_rows = [{"task_id": "t"}, {"task_id": "t"}]
        idea_rows = [
            {"task_id": "t", "agent": "a", "model": "m", "seed_year": 2020},
            {"task_id": "t", "agent": "a", "model": "m", "seed_year": 2020},
        ]
        followon_rows = [{"task_id": "t", "paper_id": "p"}]
        records, _ = build_task_records(
            ideas,
            idea_rows,
            seeds,
            seed_rows,
            followons,
            followon_rows,
            task_key="task_id",
            year_key="seed_year",
            group_fields=["agent", "model"],
        )
        pooled = next(row for row in records if row["dimension"] == "pooled")
        self.assertAlmostEqual(pooled["ai_distance"], 0.5)
        self.assertAlmostEqual(pooled["human_distance"], 1.0)


class FrontierTests(unittest.TestCase):
    def test_frontier_excludes_followon(self) -> None:
        papers = [
            {"paper_id": "p1", "primary_field": "f", "year": 2021, "keywords": ["a", "b"]},
            {"paper_id": "p2", "primary_field": "f", "year": 2021, "keywords": ["a", "c"]},
        ]
        frontiers = build_frontiers(
            papers,
            field_key="primary_field",
            year_key="year",
            keyword_key="keywords",
            top_fraction=0.5,
            min_size=1,
            excluded_paper_ids={"p2"},
            paper_id_key="paper_id",
        )
        self.assertEqual(frontiers[("f", 2021)]["terms"], {"a"})
        ideas = [
            {
                "run_id": "r1",
                "task_id": "t1",
                "primary_field": "f",
                "seed_year": 2020,
                "agent": "x",
                "model": "y",
                "keywords": ["a"],
            }
        ]
        followons = [{"task_id": "t1", "paper_id": "p2", "keywords": ["c"]}]
        records, _ = build_records(
            ideas,
            followons,
            frontiers,
            field_key="primary_field",
            seed_year_key="seed_year",
            task_key="task_id",
            agent_key="agent",
            model_key="model",
            keyword_key="keywords",
        )
        self.assertEqual(records[0]["idea_frontier_coverage"], 1.0)
        self.assertEqual(records[0]["followon_frontier_coverage"], 0.0)


class ImpactTests(unittest.TestCase):
    def test_centered_citation_scores(self) -> None:
        rows = [
            {"context_id": "c", "year": 2020, "citation_count": 0},
            {"context_id": "c", "year": 2020, "citation_count": 3},
        ]
        baselines, scores = citation_baselines(
            rows, context_key="context_id", year_key="year", citation_key="citation_count"
        )
        self.assertAlmostEqual(float(scores.mean()), 0.0)
        followons = [
            {"task_id": "t", "paper_id": "p", "context_id": "c", "year": 2020, "citation_count": 3}
        ]
        by_task, skipped = score_followons(
            followons,
            baselines,
            context_key="context_id",
            year_key="year",
            citation_key="citation_count",
            task_key="task_id",
        )
        self.assertFalse(skipped)
        self.assertGreater(by_task["t"][0], 0)


class NoveltyTests(unittest.TestCase):
    def test_keyword_partition_fallback(self) -> None:
        self.assertEqual(vote_for({"task_new": ["new target"]}, "new_research_question"), 1)
        self.assertEqual(vote_for({"method_new": []}, "new_method"), 0)

    def test_overlap_is_counted_in_both_marginals(self) -> None:
        rows = [
            {"new_research_question": 1, "new_method": 1, "category": category(1, 1)},
            {"new_research_question": 0, "new_method": 1, "category": category(0, 1)},
        ]
        summary = pack(rows)
        self.assertEqual(summary["new_research_question"]["share"], 0.5)
        self.assertEqual(summary["new_method"]["share"], 1.0)
        self.assertEqual(summary["both_new"]["share"], 0.5)


if __name__ == "__main__":
    unittest.main()
