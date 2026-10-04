#!/usr/bin/env python3
"""Deterministic tests for the production system-wide budget allocator."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scaling_model import randomized_round_robin_allocate  # noqa: E402


def allocate(pools: list[set[int]], total_budget: int, seed: int = 7):
    mutable = [set(items) for items in pools]
    tested: list[tuple[int, int]] = []

    def candidates_for(agent_index: int) -> list[int]:
        return sorted(mutable[agent_index])

    def perform_one(agent_index: int, candidates: list[int], ordinal: int) -> None:
        del ordinal
        pair = candidates[0]
        assert pair in mutable[agent_index]
        mutable[agent_index].remove(pair)
        tested.append((agent_index, pair))

    result = randomized_round_robin_allocate(
        n_agents=len(mutable),
        total_budget=total_budget,
        seed=seed,
        round_number=1,
        candidates_for=candidates_for,
        perform_one=perform_one,
    )
    return result, tested


class BudgetRedistributionTests(unittest.TestCase):
    def test_A_all_four_agents_have_many_candidates(self) -> None:
        pools = [set(range(10 * index, 10 * index + 8)) for index in range(4)]
        result, _ = allocate(pools, total_budget=4)
        self.assertEqual(result["B_used"], 4)
        self.assertEqual(result["experiments_per_agent"], [1, 1, 1, 1])

    def test_B_only_two_agents_have_many_candidates(self) -> None:
        pools = [set(range(8)), set(range(20, 28)), set(), set()]
        result, _ = allocate(pools, total_budget=4)
        self.assertEqual(result["B_used"], 4)
        self.assertEqual(result["experiments_per_agent"], [2, 2, 0, 0])
        self.assertEqual(result["max_experiments_per_agent"], 2)

    def test_C_one_agent_can_use_all_eight_experiments(self) -> None:
        pools = [set(range(8)), set(), set(), set()]
        result, tested = allocate(pools, total_budget=8)
        self.assertEqual(result["B_used"], 8)
        self.assertEqual(result["experiments_per_agent"], [8, 0, 0, 0])
        self.assertEqual(result["max_experiments_per_agent"], 8)
        self.assertEqual(len(set(tested)), 8)

    def test_D_only_three_candidates_exist_system_wide(self) -> None:
        pools = [{10}, {20}, {30}, set()]
        result, _ = allocate(pools, total_budget=8)
        self.assertEqual(result["B_used"], 3)
        self.assertEqual(result["B_used"] / result["total_budget_nominal"], 3 / 8)
        self.assertEqual(result["total_candidate_capacity_end"], 0)

    def test_E_frontier_nonempty_does_not_imply_experiment_active(self) -> None:
        # Agents 0 and 2 are declared frontier-nonempty for the purpose of this
        # test, but their usable candidate pools are empty.  The allocator sees
        # candidate availability only and must assign them zero experiments.
        frontier_nonempty = [True, True, True, False]
        pools = [set(), {11, 12, 13}, set(), {31, 32, 33}]
        result, _ = allocate(pools, total_budget=4)
        self.assertTrue(frontier_nonempty[0] and frontier_nonempty[2])
        self.assertEqual(result["experiments_per_agent"][0], 0)
        self.assertEqual(result["experiments_per_agent"][2], 0)
        self.assertEqual(result["N_experiment_active_start"], 2)
        self.assertEqual(result["B_used"], 4)

    def test_seeded_order_is_reproducible(self) -> None:
        pools = [set(range(10 * index, 10 * index + 8)) for index in range(4)]
        first_result, first_trace = allocate(pools, total_budget=7, seed=991)
        second_result, second_trace = allocate(pools, total_budget=7, seed=991)
        self.assertEqual(first_result, second_result)
        self.assertEqual(first_trace, second_trace)


if __name__ == "__main__":
    unittest.main()
