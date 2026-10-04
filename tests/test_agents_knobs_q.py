#!/usr/bin/env python3
"""q_model: an agent that misses evidence, and the analytic reference for it."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis.references import (  # noqa: E402
    amplified_dc,
    delayed_dc,
    dilute_dc,
    exact_dc,
)
from src.utils import (  # noqa: E402
    delayed_boundary_dc,
    dilute_analytic_dc,
    exact_analytic_dc,
)
from tests.test_agents_knobs_defaults import build_runner, run_with  # noqa: E402


class ReferenceFormulas(unittest.TestCase):
    """At q=1 the new references must be the frozen formulas, to the bit."""

    GRID = [
        (M, K, T, b, m)
        for M in (8, 12, 16, 32)
        for K in (2, 4, 8)
        for T in (3, 4, 5, 6)
        for b in (2, 3, 4)
        for m in (1, 2, 3)
        if T <= M
    ]

    def test_exact_matches_frozen_at_q_one(self) -> None:
        for M, K, T, b, m in self.GRID:
            frozen = exact_analytic_dc(M, K, T, b, m)
            produced = exact_dc(M * K, T, b, m)
            with self.subTest(M=M, K=K, T=T, b=b, m=m):
                if math.isnan(frozen):
                    self.assertTrue(math.isnan(produced))
                else:
                    self.assertEqual(frozen, produced)

    def test_delayed_matches_frozen_at_q_one(self) -> None:
        for M, K, T, b, m in self.GRID:
            frozen = delayed_boundary_dc(M, K, T, b, m)
            produced = delayed_dc(M * K, T, b, m)
            with self.subTest(M=M, K=K, T=T, b=b, m=m):
                if math.isnan(frozen):
                    self.assertTrue(math.isnan(produced))
                else:
                    self.assertEqual(frozen, produced)

    def test_dilute_and_amplified_match_frozen_at_q_one(self) -> None:
        for M, K, T, b, m in self.GRID[:60]:
            self.assertAlmostEqual(
                dilute_analytic_dc(M, K, T, b, m), dilute_dc(M * K, T, b, m), places=12
            )
            first = exact_dc(M * K, T, b, m)
            second = amplified_dc(M * K, T, b, m, amplification=1.0)
            if math.isnan(first):
                self.assertTrue(math.isnan(second))
            else:
                self.assertEqual(first, second)

    def test_smaller_q_needs_a_larger_degree(self) -> None:
        # A weaker pruner buys less per message, so the crossing moves out;
        # in the dilute limit it moves out like 1/q.
        base = exact_dc(128, 4, 2, 1, 1.0)
        for q in (0.75, 0.5, 0.25):
            moved = exact_dc(128, 4, 2, 1, q)
            self.assertGreater(moved, base)
            self.assertAlmostEqual(moved / base, 1.0 / q, delta=0.05)

    def test_rejects_impossible_q(self) -> None:
        for q in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                exact_dc(128, 4, 2, 1, q)


class MissRates(unittest.TestCase):
    def _totals(self, overrides: dict, *, degree: int = 8, agents: int = 12) -> dict:
        totals = {"recv_unique": 0, "recv_missed": 0, "tests": 0, "own_missed": 0}
        for seed in (12345, 777, 20260918, 4242):
            produced = run_with(overrides, degree=degree, seed=seed, agents=agents)
            for log in produced["transitions"]:
                for key in totals:
                    totals[key] += log[key]
        return totals

    def test_half_of_the_received_labels_are_missed(self) -> None:
        totals = self._totals({"q_model_received": 0.5})
        self.assertGreater(totals["recv_unique"], 200)
        rate = totals["recv_missed"] / totals["recv_unique"]
        self.assertAlmostEqual(rate, 0.5, delta=0.08)
        # Own results are on a different stream and must be untouched.
        self.assertEqual(totals["own_missed"], 0)

    def test_half_of_the_own_results_are_missed(self) -> None:
        totals = self._totals({"q_model_own": 0.5})
        self.assertGreater(totals["tests"], 100)
        rate = totals["own_missed"] / totals["tests"]
        self.assertAlmostEqual(rate, 0.5, delta=0.10)
        self.assertEqual(totals["recv_missed"], 0)

    def test_q_model_drives_both_channels(self) -> None:
        totals = self._totals({"q_model": 0.5})
        self.assertGreater(totals["recv_missed"], 0)
        self.assertGreater(totals["own_missed"], 0)

    def test_a_missed_record_is_not_stored_and_prunes_nothing(self) -> None:
        # The definition of a miss: nothing enters the store, so the label
        # count is exactly the number of results the agent did take up, and
        # no correct hypothesis can be removed by evidence it never held.
        runner = build_runner({"q_model": 0.5}, degree=8, seed=12345)
        produced = asyncio.run(runner.run())
        tests = sum(log["tests"] for log in produced["transitions"])
        own_missed = sum(log["own_missed"] for log in produced["transitions"])
        recv_missed = sum(log["recv_missed"] for log in produced["transitions"])
        self.assertGreater(own_missed, 0)
        self.assertGreater(recv_missed, 0)
        held = sum(len(agent.evidence.records) for agent in runner.agent_states())
        own_held = sum(len(agent.evidence.own) for agent in runner.agent_states())
        self.assertEqual(own_held, tests - own_missed)
        self.assertGreater(held, own_held)
        for agent in runner.agent_states():
            for pair, record in agent.evidence.records.items():
                # Nothing false-negative: a stored label is the oracle's.
                self.assertEqual(record.positive, runner.task.is_positive(pair))

    def test_misses_reduce_the_amount_of_pruning(self) -> None:
        def pruned(overrides: dict) -> int:
            total = 0
            for seed in (12345, 777, 20260918, 4242):
                produced = run_with(overrides, degree=8, seed=seed)
                total += sum(
                    log["recv_pruned_hypotheses"] + log["own_pruned_hypotheses"]
                    for log in produced["transitions"]
                )
            return total

        self.assertLess(pruned({"q_model": 0.5}), pruned({}))

    def test_misses_leave_more_incorrect_lineage_alive(self) -> None:
        strict = 0
        lossy = 0
        for seed in (12345, 777, 20260918, 4242):
            strict += run_with({}, degree=8, seed=seed)["transitions"][-1]["A_post"]
            lossy += run_with({"q_model": 0.5}, degree=8, seed=seed)["transitions"][-1][
                "A_post"
            ]
        self.assertGreater(lossy, strict)

    def test_a_missed_own_result_is_still_forwarded(self) -> None:
        # The experiment happened, so the neighbours hear about it; this is
        # what separates q_model from a communication failure.
        produced = run_with({"q_model_own": 0.5}, degree=8, seed=12345)
        missed = sum(log["own_missed"] for log in produced["transitions"])
        self.assertGreater(missed, 0)
        for log in produced["transitions"]:
            self.assertEqual(
                log["outgoing_agents"],
                sum(1 for count in log["experiments_per_agent"] if count),
            )


if __name__ == "__main__":
    unittest.main()
