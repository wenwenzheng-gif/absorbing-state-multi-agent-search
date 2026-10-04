#!/usr/bin/env python3
"""Golden regression: new diagnostics must not change scientific trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scaling_model import run_episode  # noqa: E402


NEW_TRANSITION_KEYS = {
    "total_budget_nominal",
    "total_candidate_capacity_start",
    "N_experiment_active_start",
    "N_frontier_active",
    "remaining_budget",
    "N_experiment_active_end",
    "total_candidate_capacity_end",
    "experiments_per_agent",
    "max_experiments_per_agent",
    "mean_experiments_per_experiment_active_agent",
    "allocation_passes",
    "N_frontier_active_end",
    "agents_exceeding_nominal_m",
    "budget_redistributed_above_m",
}

CASES = (
    (
        "m1_b2",
        {"N": 12, "M": 8, "K": 4, "task_size": 4, "rounds": 3, "b": 2, "m": 1, "n0": 8, "w": 0},
        2,
        12345,
        "d7b1f5f539636ccca837aa28dc29276d53a0decfc113fd680b846933e570b6cb",
    ),
    (
        "m2_b3",
        {"N": 12, "M": 8, "K": 4, "task_size": 4, "rounds": 3, "b": 3, "m": 2, "n0": 8, "w": 0},
        4,
        23456,
        "2ae263b51c064eb697c2a9916c12edb776852bf6cbbf92ef4a0c66f821a723a4",
    ),
    (
        "m3_T5",
        {"N": 12, "M": 12, "K": 4, "task_size": 5, "rounds": 4, "b": 2, "m": 3, "n0": 8, "w": 0},
        6,
        34567,
        "4bb3f629e3766b83b862cb723485fe70008c3f206159f6751aed43380f39af64",
    ),
)


def legacy_projection(result: dict) -> dict:
    projected = json.loads(json.dumps(result, allow_nan=True))
    for transition in projected["transitions"]:
        for key in NEW_TRANSITION_KEYS:
            transition.pop(key, None)
    return projected


def canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BudgetRegressionTests(unittest.TestCase):
    def test_legacy_projection_matches_prechange_golden_trajectories(self) -> None:
        for name, parameters, degree, seed, expected in CASES:
            with self.subTest(name=name):
                actual = run_episode(parameters, degree, seed)
                self.assertEqual(canonical_sha256(legacy_projection(actual)), expected)

    def test_required_diagnostics_and_invariants_hold(self) -> None:
        _, parameters, degree, seed, _ = CASES[-1]
        result = run_episode(parameters, degree, seed)
        for transition in result["transitions"]:
            for key in NEW_TRANSITION_KEYS:
                self.assertIn(key, transition)
            nominal = parameters["N"] * parameters["m"]
            allocation = transition["experiments_per_agent"]
            self.assertEqual(transition["total_budget_nominal"], nominal)
            self.assertLessEqual(transition["B_used"], nominal)
            self.assertEqual(sum(allocation), transition["B_used"])
            if transition["B_used"] < nominal:
                self.assertEqual(transition["total_candidate_capacity_end"], 0)


if __name__ == "__main__":
    unittest.main()
