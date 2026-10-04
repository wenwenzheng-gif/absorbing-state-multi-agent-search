#!/usr/bin/env python3
"""Small deterministic checks for trajectories and the crossing estimator."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scaling_model import run_episode  # noqa: E402
from src.utils import crossing_from_logs, exact_analytic_dc  # noqa: E402


PARAMETERS = {
    "N": 12,
    "M": 8,
    "K": 4,
    "task_size": 4,
    "rounds": 3,
    "b": 2,
    "m": 1,
    "n0": 8,
    "w": 0,
}


class SmallReproductionTests(unittest.TestCase):
    def test_same_seed_is_exactly_reproducible(self) -> None:
        first = run_episode(PARAMETERS, degree=2, seed=12345)
        second = run_episode(PARAMETERS, degree=2, seed=12345)
        self.assertEqual(first, second)

    def test_different_seed_changes_trajectory(self) -> None:
        first = run_episode(PARAMETERS, degree=2, seed=12345)
        second = run_episode(PARAMETERS, degree=2, seed=12346)
        self.assertNotEqual(first, second)

    def test_adjacent_crossing_interpolation(self) -> None:
        self.assertAlmostEqual(crossing_from_logs(7, 8, 0.2, -0.3), 7.4)
        with self.assertRaises(ValueError):
            crossing_from_logs(7, 9, 0.2, -0.3)

    def test_exact_finite_T_formula(self) -> None:
        expected = -3.0 * math.log(2.0) / sum(
            math.log(1.0 - t / 64.0) for t in range(1, 4)
        )
        self.assertAlmostEqual(exact_analytic_dc(16, 4, 4, 2, 1), expected, places=14)


if __name__ == "__main__":
    unittest.main()

