#!/usr/bin/env python3
"""budget_mode, stop_on_success, evaluate_every_round, respawn and topology."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.budget import randomized_round_robin_allocate_async  # noqa: E402
from src.agent_system.communication import Router, build_graph  # noqa: E402
from src.agent_system.schemas import load_run_config  # noqa: E402
from tests.test_agents_knobs_defaults import BASE, build_runner, run_with  # noqa: E402


class HardCap(unittest.TestCase):
    def test_no_agent_exceeds_m(self) -> None:
        for degree in (0, 4, 8, 11):
            for seed in (12345, 777):
                produced = run_with(
                    {"budget_mode": "hard_cap"}, degree=degree, seed=seed
                )
                for log in produced["transitions"]:
                    self.assertLessEqual(log["max_experiments_per_agent"], 1)
                    self.assertEqual(log["agents_exceeding_nominal_m"], 0)
                    self.assertEqual(log["budget_redistributed_above_m"], 0)
                    self.assertEqual(log["per_agent_cap"], 1)

    def test_redistribution_does_exceed_m(self) -> None:
        # The control: without the cap the published rule really does hand an
        # extinct agent's slot to somebody else.
        over = 0
        for seed in (12345, 777, 20260918):
            produced = run_with({}, degree=8, seed=seed)
            over += sum(log["budget_redistributed_above_m"] for log in produced["transitions"])
        self.assertGreater(over, 0)

    def test_hard_cap_never_spends_more_budget(self) -> None:
        for seed in (12345, 777, 20260918):
            capped = run_with({"budget_mode": "hard_cap"}, degree=8, seed=seed)
            free = run_with({}, degree=8, seed=seed)
            capped_used = sum(log["B_used"] for log in capped["transitions"])
            free_used = sum(log["B_used"] for log in free["transitions"])
            self.assertLessEqual(capped_used, free_used)

    def test_allocator_honours_a_cap_of_two(self) -> None:
        pools = {index: [f"c{index}"] * 50 for index in range(4)}
        done: list[int] = []

        async def perform(index, candidates, ordinal):
            done.append(index)

        result = asyncio.run(
            randomized_round_robin_allocate_async(
                n_agents=4,
                total_budget=40,
                seed=7,
                round_number=1,
                candidates_for=lambda index: pools[index],
                perform_one=perform,
                per_agent_cap=2,
            )
        )
        self.assertEqual(result["experiments_per_agent"], [2, 2, 2, 2])
        self.assertEqual(result["B_used"], 8)
        self.assertEqual(result["remaining_budget"], 32)
        self.assertEqual(len(done), 8)


class StopOnSuccess(unittest.TestCase):
    def test_default_runs_every_round(self) -> None:
        produced = run_with({}, degree=8, seed=12345)
        self.assertEqual(produced["rounds_executed"], 3)
        self.assertIsNone(produced["stopped_round"])

    def test_solved_is_decidable_from_the_agent_s_own_evidence(self) -> None:
        # An agent is solved once it *holds* a depth-T candidate whose every
        # component carries a positive label -- the evaluator's
        # evidence_verified criterion.  Nothing hidden enters that test.
        runner = build_runner({}, degree=11, seed=12345)
        produced = asyncio.run(runner.run())
        depth = runner.config.task_size
        for agent in runner.agent_states():
            held = {
                components
                for components in agent.complete_candidates
                if len(components) == depth
            }
            held.update(
                runner.encoding.components(node)
                for node in agent.frontier
                if node.used.bit_count() == depth
            )
            self.assertEqual(
                agent.solved_round is not None,
                any(agent.evidence.covers_positively(c) for c in held),
            )
        self.assertEqual(
            produced["agents_solved"],
            sum(a.solved_round is not None for a in runner.agent_states()),
        )

    def test_at_the_published_horizon_the_knob_cannot_truncate(self) -> None:
        # agent_v1 fixes rounds = T - 1 and a hypothesis gains one component
        # per round, so a depth-T candidate first exists in the final round.
        # The knob can therefore only fire there, where stopping is a no-op.
        # It used to fire one round earlier, on T positive labels alone, and
        # the evaluator then scored a frontier of depth-(T-1) hypotheses and
        # reported P_succ = 0 for an episode that had just succeeded.
        for seed in (12345, 777, 20260918, 4242, 99, 1234567):
            produced = run_with(
                {"experiments_per_agent": 3, "stop_on_success": True},
                degree=11,
                seed=seed,
            )
            self.assertEqual(produced["rounds_executed"], 3)
            self.assertIn(produced["stopped_round"], (None, 3))
            if produced["stopped_round"] is not None:
                self.assertTrue(produced["transitions"][-1]["any_agent_solved"])

    def test_stopping_truncates_at_the_first_solved_round(self) -> None:
        for seed in (12345, 777, 20260918, 4242):
            stopped = run_with({"stop_on_success": True}, degree=11, seed=seed)
            rounds = stopped["rounds_executed"]
            if stopped["stopped_round"] is None:
                self.assertEqual(rounds, 3)
                continue
            self.assertEqual(rounds, stopped["stopped_round"])
            self.assertTrue(stopped["transitions"][-1]["any_agent_solved"])
            for log in stopped["transitions"][:-1]:
                self.assertFalse(log["any_agent_solved"])

    def test_stopping_changes_nothing_before_the_stop(self) -> None:
        for seed in (12345, 777):
            free = run_with({}, degree=11, seed=seed)
            stopped = run_with({"stop_on_success": True}, degree=11, seed=seed)
            for left, right in zip(free["transitions"], stopped["transitions"]):
                self.assertEqual(left["A_post"], right["A_post"])
                self.assertEqual(left["B_used"], right["B_used"])


class EvaluateEveryRound(unittest.TestCase):
    def test_flag_fills_the_candidate_pool_earlier(self) -> None:
        off = run_with({}, degree=8, seed=12345, evaluate=True)
        on = run_with(
            {"evaluate_every_round": True}, degree=8, seed=12345, evaluate=True
        )
        self.assertEqual(off["transitions"][0]["evaluation"]["mean_complete_candidates"], 0.0)
        self.assertGreater(on["transitions"][0]["evaluation"]["mean_complete_candidates"], 0.0)
        # The scientific trajectory is untouched: only what the hidden
        # evaluator is shown changes.
        for left, right in zip(off["transitions"], on["transitions"]):
            self.assertEqual(left["A_post"], right["A_post"])
            self.assertEqual(left["tests"], right["tests"])


class Respawn(unittest.TestCase):
    # A narrow space with a wide initial pool goes extinct often, which is
    # exactly the regime the knob exists for.
    SETTINGS = {
        "environment": {"kind": "synthetic", "families": 6, "variants": 2, "task_size": 4},
        "branching": 2,
        "initial_width": 8,
    }

    def _run(self, overrides: dict, seed: int, degree: int = 5) -> dict:
        return run_with({**self.SETTINGS, **overrides}, degree=degree, seed=seed)

    def test_default_never_respawns(self) -> None:
        produced = self._run({}, 12345)
        for log in produced["transitions"]:
            self.assertEqual(log["respawned_agents"], 0)

    def test_respawn_refills_extinct_agents(self) -> None:
        produced = self._run({"respawn": "evidence_consistent"}, 12345)
        respawned = sum(log["respawned_agents"] for log in produced["transitions"])
        self.assertGreater(respawned, 0)
        for log in produced["transitions"]:
            self.assertLessEqual(log["empty_after_respawn"], log["empty_after_receive"])

    def test_respawn_lowers_the_empty_frontier_fraction(self) -> None:
        off = 0.0
        on = 0.0
        for seed in (12345, 777, 20260918, 4242):
            off += self._run({}, seed)["transitions"][-1]["frontier_empty_fraction"]
            on += self._run({"respawn": "evidence_consistent"}, seed)["transitions"][-1][
                "frontier_empty_fraction"
            ]
        self.assertLess(on, off)

    def test_seeds_are_evidence_consistent_and_at_the_right_depth(self) -> None:
        runner = build_runner(
            {**self.SETTINGS, "respawn": "evidence_consistent"}, degree=5, seed=12345
        )
        original = runner._respawn_seeds
        seen: list[tuple[int, int]] = []

        def spy(agent, depth, round_number):
            seeds = original(agent, depth, round_number)
            for node in seeds:
                seen.append((node.used.bit_count(), depth))
                self.assertTrue(agent.evidence.compatible(node))
                self.assertFalse(node.from_wrong)
                for family, variant in runner.encoding.components(node):
                    self.assertTrue(agent.evidence.allowed[family] & (1 << variant))
            self.assertLessEqual(len(seeds), runner.config.initial_width)
            return seeds

        runner._respawn_seeds = spy
        asyncio.run(runner.run())
        self.assertTrue(seen)
        for depth_used, depth in seen:
            self.assertEqual(depth_used, depth)

    def test_respawned_lineage_is_never_counted_as_an_old_descendant(self) -> None:
        for seed in (12345, 777, 20260918, 4242):
            produced = self._run({"respawn": "evidence_consistent"}, seed)
            self.assertEqual(produced["audit"]["respawn_lineage_leak"], 0)
            for index, log in enumerate(produced["transitions"]):
                if not log["respawned_agents"]:
                    continue
                # Every edge leaving a seed is booked as a new source, so the
                # injected population cannot inflate the old-to-old ratio.
                self.assertGreater(log["respawn_source_children"], 0)
                self.assertGreaterEqual(log["old_incorrect_edges"], 0)
                self.assertEqual(
                    log["old_incorrect_edges"] + log["new_incorrect_source_edges"]
                    <= log["sampled_edges"],
                    True,
                    msg=f"round {index+1}",
                )

    def test_respawn_uses_known_positives_first(self) -> None:
        runner = build_runner(
            {**self.SETTINGS, "respawn": "evidence_consistent"}, degree=5, seed=777
        )
        original = runner._respawn_seeds

        def spy(agent, depth, round_number):
            positives = list(agent.evidence.known_positive())[:depth]
            seeds = original(agent, depth, round_number)
            for node in seeds:
                components = set(runner.encoding.components(node))
                for pair in positives:
                    self.assertIn(pair, components)
            return seeds

        runner._respawn_seeds = spy
        asyncio.run(runner.run())


class Topology(unittest.TestCase):
    def test_ring_lattice_is_regular_and_local(self) -> None:
        graph = build_graph(12, 4, 7, topology="ring_lattice")
        self.assertTrue(all(len(group) == 4 for group in graph))
        self.assertEqual(graph[0], (1, 2, 10, 11))

    def test_ring_lattice_rejects_an_odd_degree(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(12, 3, 7, topology="ring_lattice")
        with self.assertRaises(ValueError):
            load_run_config({**BASE, "topology": "ring_lattice", "degrees": [3]})

    def test_erdos_renyi_hits_the_nominal_degree_on_average(self) -> None:
        realised = [
            Router(60, 10, seed, topology="erdos_renyi").mean_degree
            for seed in range(12)
        ]
        self.assertAlmostEqual(sum(realised) / len(realised), 10.0, delta=1.0)

    def test_watts_strogatz_keeps_the_edge_count(self) -> None:
        ring = Router(40, 6, 3, topology="ring_lattice")
        rewired = Router(40, 6, 3, topology="watts_strogatz", rewire_p=0.3)
        self.assertAlmostEqual(ring.mean_degree, rewired.mean_degree, places=12)
        self.assertNotEqual(ring.graph, rewired.graph)

    def test_a_non_default_topology_changes_the_episode(self) -> None:
        default = run_with({}, degree=4, seed=12345)
        ring = run_with({"topology": "ring_lattice"}, degree=4, seed=12345)
        self.assertEqual(ring["realised_mean_degree"], 4.0)
        self.assertNotEqual(default["graph_digest"], ring["graph_digest"])

    def test_erdos_renyi_records_the_realised_mean_degree(self) -> None:
        produced = run_with({"topology": "erdos_renyi"}, degree=5, seed=12345, agents=12)
        self.assertNotEqual(produced["realised_mean_degree"], 5.0)
        self.assertGreater(produced["realised_mean_degree"], 2.0)
        self.assertLess(produced["realised_mean_degree"], 8.0)

    def test_odd_degree_is_allowed_for_erdos_renyi(self) -> None:
        config = load_run_config({**BASE, "topology": "erdos_renyi", "degrees": [3], "agents": 11})
        self.assertEqual(config.degrees, (3,))


if __name__ == "__main__":
    unittest.main()
