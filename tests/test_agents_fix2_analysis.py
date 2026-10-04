#!/usr/bin/env python3
"""Two failures of ``critical_degree`` on real runs.

* ragged episodes.  ``stop_on_success`` truncates some episodes, so a degree
  no longer has one round count.  The round count used to be read off the
  first episode of the group: too few rounds when a later episode was longer
  (silently dropping the last round) and an ``IndexError`` the other way.
* sparse grids.  The crossing was read only between *adjacent integers*, so
  every scan on a step-2 grid -- all the topology runs, every LLM run --
  returned ``no_adjacent_crossing`` and no ``d_c`` at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis.critical_degree import (  # noqa: E402
    STATUS_OK,
    find_crossings,
    pooled_log_ratio,
    pooled_step_log_ratio,
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


def curve_of(values):
    return {
        degree: {"value": value, "status": STATUS_OK, "episodes": 8}
        for degree, value in values.items()
    }


class RaggedEpisodes(unittest.TestCase):
    """A short episode and a long one pooled together, in both orders."""

    def rows(self):
        return [
            episode([8, 4], [10, 5], initial=10, task_seed=1),
            episode([8, 4, 2], [10, 5, 2], initial=10, task_seed=2),
        ]

    def test_the_long_episode_no_longer_raises(self) -> None:
        # The short one first: the old code sized the pools from it and the
        # long one's third round indexed past the end.
        result = pooled_log_ratio(self.rows())
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(result["rounds"], 3)

    def test_the_long_episode_is_not_silently_dropped(self) -> None:
        # The other order used to "work": three rounds were pooled but the
        # short episode's two contributed to only the first two, with no
        # record that the third stood on one episode alone.
        forwards = pooled_log_ratio(self.rows())
        backwards = pooled_log_ratio(list(reversed(self.rows())))
        self.assertEqual(forwards["value"], backwards["value"])
        self.assertEqual(forwards["episodes_per_round"], [2, 2, 1])

    def test_each_round_is_pooled_over_the_episodes_that_have_it(self) -> None:
        result = pooled_log_ratio(self.rows())
        self.assertAlmostEqual(
            result["value"],
            math.log(16 / 20) + math.log(8 / 20) + math.log(2 / 5),
        )

    def test_the_degree_is_flagged_ragged(self) -> None:
        self.assertTrue(pooled_log_ratio(self.rows())["ragged"])
        self.assertFalse(pooled_log_ratio(self.rows()[:1])["ragged"])

    def test_the_step_ratio_survives_a_round_only_some_episodes_reached(self) -> None:
        last = pooled_step_log_ratio(self.rows(), round_index=-1)
        self.assertEqual(last["round"], 3)
        self.assertEqual(last["episodes"], 1)
        self.assertTrue(last["ragged"])
        self.assertAlmostEqual(last["value"], math.log(2 / 5))

    def test_a_round_no_episode_reached_is_not_estimable(self) -> None:
        empty = pooled_step_log_ratio(self.rows(), round_index=7)
        self.assertIsNone(empty["value"])

    def test_a_square_group_is_unchanged(self) -> None:
        square = [episode([8, 4], [10, 5], task_seed=t) for t in (1, 2)]
        result = pooled_log_ratio(square)
        self.assertFalse(result["ragged"])
        self.assertEqual(result["episodes_per_round"], [2, 2])
        self.assertAlmostEqual(result["value"], math.log(16 / 20) + math.log(8 / 20))


class RaggedRunDirectory(unittest.TestCase):
    def test_a_whole_scan_runs_on_a_ragged_run(self) -> None:
        rows = []
        for degree, decay in ((0, 1.4), (1, 1.1), (2, 0.9), (3, 0.6)):
            for task in (1, 2, 3, 4):
                rounds = 2 if task % 2 else 3
                pre = [int(round(10 * decay ** (index + 1))) for index in range(rounds)]
                post = [10] * rounds
                rows.append(
                    episode(pre, post, initial=10, task_seed=task, degree=degree)
                )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            report = scan(run_dir, draws=20, seed=1)
        self.assertTrue(any(row["ragged"] for row in report["curve"]))
        self.assertTrue(all(row["episodes_per_round"] for row in report["curve"]))
        self.assertEqual(report["crossing"]["status"], "single_crossing")


class SparseGrids(unittest.TestCase):
    def test_a_step_two_grid_resolves_instead_of_giving_up(self) -> None:
        curve = curve_of({d: 2.1 - 0.2 * d for d in range(0, 16, 2)})
        result = find_crossings(curve)
        self.assertEqual(result["status"], "single_crossing")
        self.assertAlmostEqual(result["d_c"], 10.5)
        self.assertEqual(result["bracket"], [10, 12])
        self.assertEqual(result["grid_step"], 2)

    def test_the_dense_grid_answer_is_unchanged(self) -> None:
        dense = curve_of({d: 2.1 - 0.2 * d for d in range(0, 16)})
        result = find_crossings(dense)
        self.assertAlmostEqual(result["d_c"], 10.5)
        self.assertEqual(result["bracket"], [10, 11])
        self.assertEqual(result["grid_step"], 1)

    def test_thinning_a_grid_costs_resolution_not_validity(self) -> None:
        # A convex curve: the coarse bracket cannot give the same number, but
        # it must give one, and a close one.
        values = {d: 1.5 - 0.02 * d * d for d in range(0, 14)}
        dense = find_crossings(curve_of(values))["d_c"]
        sparse = find_crossings(curve_of({d: values[d] for d in range(0, 14, 2)}))
        self.assertEqual(sparse["status"], "single_crossing")
        self.assertLess(abs(sparse["d_c"] - dense), 0.3)

    def test_a_censored_degree_is_never_interpolated_across(self) -> None:
        curve = curve_of({0: 1.0, 2: 0.5, 6: -0.4})
        curve[4] = {"value": -math.inf, "status": "extinct_lineage", "episodes": 8}
        self.assertEqual(find_crossings(curve)["status"], "no_adjacent_crossing")

    def test_multiple_crossings_are_still_refused_a_point_estimate(self) -> None:
        result = find_crossings(curve_of({0: 1.0, 2: -0.5, 4: 0.4, 6: -0.2}))
        self.assertEqual(result["status"], "multiple_crossings")
        self.assertEqual(len(result["crossings"]), 2)


class PublishedNumberIsStable(unittest.TestCase):
    """The dense reference scan must still read 16.65 on v2_toy_b3."""

    RUN = ROOT / "runs" / "agents" / "v2_toy_b3"

    @unittest.skipUnless((ROOT / "runs" / "agents" / "v2_toy_b3").is_dir(),
                         "recorded run not present")
    def test_v2_toy_b3_still_crosses_at_16_65(self) -> None:
        from src.agent_system.analysis.run_cell import (
            episodes_by_degree,
            load_episodes,
        )

        by_degree = episodes_by_degree(load_episodes(self.RUN))
        curve = {
            degree: pooled_log_ratio(group) for degree, group in by_degree.items()
        }
        result = find_crossings(curve)
        self.assertEqual(result["status"], "single_crossing")
        self.assertAlmostEqual(result["d_c"], 16.652543302178092, places=6)
        self.assertEqual(result["grid_step"], 1)


if __name__ == "__main__":
    unittest.main()
