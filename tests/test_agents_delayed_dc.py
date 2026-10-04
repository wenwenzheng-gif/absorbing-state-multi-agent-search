#!/usr/bin/env python3
"""Tests for the ``first_round`` parameter added to ``critical_degree``.

The paper's *delayed* d_c (``references.delayed_dc``) sums the pooled
log-lineage-ratio from round ``t=2`` onward: round 1 has no neighbour
messages yet (communication is one-round delayed), so it carries no
d-dependence and is excluded from the crossing.  ``first_round`` (1-based)
generalises the old, always-from-round-1 cumulative sum: ``first_round=1``
must reproduce the old behaviour byte-for-byte, and ``first_round=2`` must
equal the hand-computed sum over rounds >= 2.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis.critical_degree import (  # noqa: E402
    pooled_log_ratio,
    scan,
)


def episode(old_pre, a_post, initial=10, task_seed=1, degree=0):
    return {
        "status": "ok",
        "d": degree,
        "task_seed": task_seed,
        "A_initial_incorrect": initial,
        "transitions": [
            {"old_pre": numerator, "old_post": posterior, "A_post": posterior}
            for numerator, posterior in zip(old_pre, a_post)
        ],
    }


class PooledLogRatioFirstRound(unittest.TestCase):
    """``first_round`` on a single hand-built episode."""

    def episodes(self):
        # Three rounds: A_initial_incorrect=10, then A_post after each round.
        return [episode([8, 4, 2], [10, 5, 2], initial=10, task_seed=1)]

    def test_default_first_round_is_the_old_full_sum(self) -> None:
        result = pooled_log_ratio(self.episodes())
        expected = math.log(8 / 10) + math.log(4 / 10) + math.log(2 / 5)
        self.assertAlmostEqual(result["value"], expected)
        self.assertEqual(result["rounds_used"], 3)

    def test_first_round_2_drops_round_1_from_the_sum(self) -> None:
        result = pooled_log_ratio(self.episodes(), first_round=2)
        expected = math.log(4 / 10) + math.log(2 / 5)
        self.assertAlmostEqual(result["value"], expected)
        self.assertEqual(result["rounds_used"], 2)
        self.assertEqual(result["first_round"], 2)

    def test_first_round_3_drops_the_first_two_rounds(self) -> None:
        result = pooled_log_ratio(self.episodes(), first_round=3)
        expected = math.log(2 / 5)
        self.assertAlmostEqual(result["value"], expected)
        self.assertEqual(result["rounds_used"], 1)

    def test_pooled_over_several_episodes_matches_the_sum_over_rounds_ge_2(self) -> None:
        rows = [
            episode([8, 4, 2], [10, 5, 2], initial=10, task_seed=1),
            episode([6, 3, 1], [10, 6, 3], initial=10, task_seed=2),
        ]
        result = pooled_log_ratio(rows, first_round=2)
        # round 2: numerator 4+3=7, denominator 10+10=20 (previous round's A_post)
        # round 3: numerator 2+1=3, denominator 5+6=11
        expected = math.log(7 / 20) + math.log(3 / 11)
        self.assertAlmostEqual(result["value"], expected)

    def test_first_round_1_is_the_default(self) -> None:
        default = pooled_log_ratio(self.episodes())
        explicit = pooled_log_ratio(self.episodes(), first_round=1)
        self.assertEqual(default["value"], explicit["value"])
        self.assertEqual(default["rounds_used"], explicit["rounds_used"])

    def test_first_round_must_be_at_least_one(self) -> None:
        with self.assertRaises(ValueError):
            pooled_log_ratio(self.episodes(), first_round=0)

    def test_ragged_episodes_pool_only_the_episodes_that_reach_each_round(self) -> None:
        rows = [
            episode([8, 4], [10, 5], initial=10, task_seed=1),
            episode([8, 4, 2], [10, 5, 2], initial=10, task_seed=2),
        ]
        result = pooled_log_ratio(rows, first_round=2)
        # round 2 (index 1): both episodes contribute: num 4+4=8, den 10+10=20
        # round 3 (index 2): only episode 2: num 2, den 5
        expected = math.log(8 / 20) + math.log(2 / 5)
        self.assertAlmostEqual(result["value"], expected)
        self.assertEqual(result["rounds_used"], 2)


class ScanFirstRoundDefaultIsUnchanged(unittest.TestCase):
    """``scan`` with ``first_round=1`` (the default) must be byte-identical."""

    def _rows(self):
        rows = []
        for degree, decay in ((0, 1.4), (4, 1.1), (8, 0.9), (16, 0.6), (24, 0.3)):
            for task in (1, 2, 3, 4):
                pre = [int(round(10 * decay ** (index + 1))) for index in range(3)]
                post = [10] * 3
                rows.append(episode(pre, post, initial=10, task_seed=task, degree=degree))
        return rows

    def test_default_json_and_csv_are_byte_identical_with_and_without_the_kwarg(self) -> None:
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            scan(run_dir, draws=20, seed=1)
            json_before = hashlib.sha256(
                (run_dir / "analysis" / "critical_degree.json").read_bytes()
            ).hexdigest()
            csv_before = hashlib.sha256(
                (run_dir / "analysis" / "log_R_by_degree.csv").read_bytes()
            ).hexdigest()

            scan(run_dir, draws=20, seed=1, first_round=1)
            json_after = hashlib.sha256(
                (run_dir / "analysis" / "critical_degree.json").read_bytes()
            ).hexdigest()
            csv_after = hashlib.sha256(
                (run_dir / "analysis" / "log_R_by_degree.csv").read_bytes()
            ).hexdigest()
        self.assertEqual(json_before, json_after)
        self.assertEqual(csv_before, csv_after)

    def test_first_round_1_report_has_no_first_round_key(self) -> None:
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            report = scan(run_dir, draws=20, seed=1)
        self.assertNotIn("first_round", report)

    def test_non_default_first_round_writes_suffixed_files_and_leaves_default_untouched(
        self,
    ) -> None:
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            scan(run_dir, draws=20, seed=1)
            default_json = (run_dir / "analysis" / "critical_degree.json").read_bytes()
            default_csv = (run_dir / "analysis" / "log_R_by_degree.csv").read_bytes()

            report2 = scan(run_dir, draws=20, seed=1, first_round=2)

            self.assertEqual(
                default_json, (run_dir / "analysis" / "critical_degree.json").read_bytes()
            )
            self.assertEqual(
                default_csv, (run_dir / "analysis" / "log_R_by_degree.csv").read_bytes()
            )
            self.assertTrue((run_dir / "analysis" / "critical_degree_t2.json").exists())
            self.assertTrue((run_dir / "analysis" / "log_R_by_degree_t2.csv").exists())
            self.assertEqual(report2["first_round"], 2)

            on_disk = json.loads(
                (run_dir / "analysis" / "critical_degree_t2.json").read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk["first_round"], 2)


if __name__ == "__main__":
    unittest.main()
