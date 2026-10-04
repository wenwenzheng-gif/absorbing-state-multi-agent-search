#!/usr/bin/env python3
"""Boundary behaviour of the lineage ratio, crossing search and bootstrap."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis.critical_degree import (  # noqa: E402
    CONVENTIONS,
    DEFAULT_CONVENTION,
    STATUS_EXTINCT,
    STATUS_NOT_ESTIMABLE,
    STATUS_OK,
    cluster_bootstrap,
    crossing_from_logs,
    find_crossings,
    pooled_log_ratio,
    pooled_step_log_ratio,
)
from src.agent_system.analysis.plots import _slope, compare  # noqa: E402


def episode(old_pre, a_post, initial=10, task_seed=1, degree=0, old_post=None):
    if old_post is None:
        old_post = list(a_post)
    transitions = [
        {"old_pre": numerator, "old_post": own, "A_post": posterior}
        for numerator, own, posterior in zip(old_pre, old_post, a_post)
    ]
    return {
        "status": "ok",
        "d": degree,
        "task_seed": task_seed,
        "A_initial_incorrect": initial,
        "transitions": transitions,
    }


class ConventionTests(unittest.TestCase):
    def test_default_is_the_frozen_pre_own_convention(self) -> None:
        self.assertEqual(DEFAULT_CONVENTION, "pre_own")
        self.assertEqual(CONVENTIONS["pre_own"], "old_pre")
        self.assertEqual(CONVENTIONS["post_own"], "old_post")

    def test_post_own_reads_the_other_numerator(self) -> None:
        rows = [episode([8, 8], [4, 4], initial=10, old_post=[2, 2])]
        pre = pooled_log_ratio(rows, "pre_own")["value"]
        post = pooled_log_ratio(rows, "post_own")["value"]
        self.assertAlmostEqual(pre, math.log(8 / 10) + math.log(8 / 4))
        self.assertAlmostEqual(post, math.log(2 / 10) + math.log(2 / 4))
        self.assertLess(post, pre)

    def test_unknown_convention_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pooled_log_ratio([episode([1], [1])], "after_lunch")


class PooledRatioTests(unittest.TestCase):
    def test_pooled_is_not_the_mean_of_per_episode_ratios(self) -> None:
        episodes = [episode([1, 1], [1, 1], initial=10), episode([9, 9], [9, 9], initial=10)]
        pooled = pooled_log_ratio(episodes)["value"]
        per_episode = [
            math.log(1 / 10) + math.log(1 / 1),
            math.log(9 / 10) + math.log(9 / 9),
        ]
        self.assertAlmostEqual(pooled, math.log(10 / 20) + math.log(10 / 10))
        self.assertNotAlmostEqual(pooled, sum(per_episode) / 2)

    def test_zero_denominator_is_not_estimable(self) -> None:
        result = pooled_log_ratio([episode([0, 0], [0, 0], initial=0)])
        self.assertEqual(result["status"], STATUS_NOT_ESTIMABLE)
        self.assertIsNone(result["value"])

    def test_positive_pool_with_no_descendants_is_extinction(self) -> None:
        result = pooled_log_ratio([episode([0, 0], [5, 5], initial=10)])
        self.assertEqual(result["status"], STATUS_EXTINCT)
        self.assertEqual(result["value"], -math.inf)

    def test_failed_episodes_are_excluded(self) -> None:
        broken = {"status": "policy_failed", "d": 0, "transitions": []}
        good = episode([5, 5], [5, 5], initial=10)
        self.assertEqual(pooled_log_ratio([broken, good])["episodes"], 1)


class CrossingTests(unittest.TestCase):
    def curve(self, values):
        return {
            degree: {"value": value, "status": STATUS_OK if value is not None else STATUS_NOT_ESTIMABLE}
            for degree, value in values.items()
        }

    def test_adjacent_sign_change_is_interpolated(self) -> None:
        result = find_crossings(self.curve({6: 0.2, 7: 0.2, 8: -0.3}))
        self.assertEqual(result["status"], "single_crossing")
        self.assertAlmostEqual(result["d_c"], 7.4)

    def test_a_sparse_grid_is_interpolated_between_measured_degrees(self) -> None:
        # Consecutive *measurements*, not adjacent integers: a step-6 grid
        # resolves, and the bracket says at what resolution.
        result = find_crossings(self.curve({0: 1.0, 5: 0.5, 11: -0.4}))
        self.assertEqual(result["status"], "single_crossing")
        self.assertAlmostEqual(result["d_c"], 5.0 + 0.5 * 6.0 / 0.9)
        self.assertEqual(result["bracket"], [5, 11])
        # Gaps of 5 and 6 tie; the finer one is reported as the resolution.
        self.assertEqual(result["grid_step"], 5)

    def test_a_censored_degree_still_blocks_the_pair_it_sits_in(self) -> None:
        curve = self.curve({0: 1.0, 5: 0.5, 8: None, 11: -0.4})
        self.assertEqual(find_crossings(curve)["status"], "no_adjacent_crossing")
        self.assertIsNone(find_crossings(curve)["d_c"])

    def test_multiple_crossings_are_reported_as_such(self) -> None:
        result = find_crossings(self.curve({0: 1.0, 1: -0.5, 2: 0.4, 3: -0.2}))
        self.assertEqual(result["status"], "multiple_crossings")
        self.assertEqual(len(result["crossings"]), 2)

    def test_an_all_positive_curve_reports_no_crossing(self) -> None:
        self.assertEqual(
            find_crossings(self.curve({0: 1.0, 1: 0.8}))["status"], "no_crossing_all_positive"
        )

    def test_an_unestimable_curve_is_not_forced_to_a_value(self) -> None:
        self.assertEqual(find_crossings(self.curve({0: None, 1: None}))["status"], "not_estimable")

    def test_infinite_endpoints_are_never_interpolated(self) -> None:
        curve = {
            0: {"value": 1.0, "status": STATUS_OK},
            1: {"value": -math.inf, "status": STATUS_EXTINCT},
        }
        self.assertIsNone(find_crossings(curve)["d_c"])

    def test_interpolation_rejects_bad_endpoints(self) -> None:
        # A wider bracket is now legitimate; a non-increasing one is not,
        # and neither is a pair that does not straddle zero.
        self.assertAlmostEqual(crossing_from_logs(7, 9, 0.2, -0.3), 7.8)
        with self.assertRaises(ValueError):
            crossing_from_logs(7, 7, 0.2, -0.3)
        with self.assertRaises(ValueError):
            crossing_from_logs(7, 8, -0.2, -0.3)


class BootstrapTests(unittest.TestCase):
    def test_reports_the_share_of_draws_that_produced_a_crossing(self) -> None:
        by_degree = {
            0: [episode([20, 20], [10, 10], initial=10, task_seed=task, degree=0) for task in (1, 2, 3)],
            1: [episode([1, 1], [1, 1], initial=10, task_seed=task, degree=1) for task in (1, 2, 3)],
        }
        result = cluster_bootstrap(by_degree, draws=50, seed=1)
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["draws"], 50)
        self.assertEqual(result["valid_fraction"], 1.0)
        self.assertIsNotNone(result["d_c_ci"])

    def test_no_crossing_gives_an_empty_interval_not_a_narrow_one(self) -> None:
        by_degree = {
            0: [episode([20, 20], [10, 10], initial=10, task_seed=task, degree=0) for task in (1, 2)],
            1: [episode([20, 20], [10, 10], initial=10, task_seed=task, degree=1) for task in (1, 2)],
        }
        result = cluster_bootstrap(by_degree, draws=20, seed=2)
        self.assertEqual(result["valid_draws"], 0)
        self.assertIsNone(result["d_c_ci"])
        self.assertEqual(result["valid_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()


class ComparisonTests(unittest.TestCase):
    """The slope of log R in d is the number the paired runs exist to give."""

    def _run_dir(self, root: Path, name: str, decay: float) -> Path:
        directory = root / name
        directory.mkdir(parents=True)
        lines = []
        for degree in (0, 4, 8, 12):
            # A_post falls by a factor exp(decay) per unit of degree, so the
            # pooled log R is linear in d with slope -decay.
            factor = math.exp(-decay * degree)
            lines.append(
                json.dumps(
                    {
                        "episode_id": f"d{degree}",
                        "status": "ok",
                        "d": degree,
                        "task_seed": 1,
                        "A_initial_incorrect": 100,
                        "transitions": [
                            {"old_pre": 200 * factor, "old_post": 200 * factor, "A_post": 100}
                        ],
                    }
                )
            )
        (directory / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    def test_slope_recovers_a_planted_decay_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fast = self._run_dir(root, "fast", 0.20)
            slow = self._run_dir(root, "slow", 0.05)
            report = compare([fast, slow], root / "out", ["fast", "slow"])
        self.assertAlmostEqual(report["slopes"]["fast"]["pre_own"], -0.20, places=6)
        self.assertAlmostEqual(report["slopes"]["slow"]["pre_own"], -0.05, places=6)

    def test_a_label_per_run_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = self._run_dir(root, "one", 0.1)
            with self.assertRaises(ValueError):
                compare([one], root / "out", ["a", "b"])

    def test_slope_is_undefined_without_two_distinct_degrees(self) -> None:
        self.assertIsNone(_slope([], []))
        self.assertIsNone(_slope([3.0], [1.0]))
        self.assertIsNone(_slope([3.0, 3.0], [1.0, 2.0]))


class StepwiseRatioTests(unittest.TestCase):
    """The step ratio isolates the last round from the uncommunicated first."""

    def test_step_reads_the_last_round_only(self) -> None:
        rows = [episode([6, 9], [3, 3], initial=2)]
        # cumulative: log(6/2) + log(9/3); step: log(9/3) alone.
        self.assertAlmostEqual(pooled_log_ratio(rows)["value"], math.log(3) + math.log(3))
        entry = pooled_step_log_ratio(rows)
        self.assertAlmostEqual(entry["value"], math.log(3))
        self.assertEqual(entry["round"], 2)

    def test_an_earlier_round_can_be_asked_for(self) -> None:
        rows = [episode([6, 9], [3, 3], initial=2)]
        entry = pooled_step_log_ratio(rows, DEFAULT_CONVENTION, 0)
        self.assertAlmostEqual(entry["value"], math.log(6 / 2))
        self.assertEqual(entry["round"], 1)

    def test_a_round_out_of_range_is_not_estimable(self) -> None:
        rows = [episode([6, 9], [3, 3], initial=2)]
        self.assertEqual(pooled_step_log_ratio(rows, DEFAULT_CONVENTION, 7)["status"], STATUS_NOT_ESTIMABLE)

    def test_an_extinct_last_round_is_reported_as_extinct(self) -> None:
        rows = [episode([6, 0], [3, 3], initial=2)]
        entry = pooled_step_log_ratio(rows)
        self.assertEqual(entry["status"], STATUS_EXTINCT)
        self.assertEqual(entry["value"], -math.inf)
