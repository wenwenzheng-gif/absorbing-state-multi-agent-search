#!/usr/bin/env python3
"""Protocol-level guarantees: canonical merging, exact pruning, messaging,
the dynamic budget, and the two candidate-pool conventions."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.budget import randomized_round_robin_allocate_async  # noqa: E402
from src.agent_system.communication import Router  # noqa: E402
from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.synthetic import SyntheticEnvironment  # noqa: E402
from src.agent_system.evidence import EvidenceStore  # noqa: E402
from src.agent_system.hypotheses import ChildCollector, Encoding  # noqa: E402
from src.agent_system.policies.uniform import UniformPolicy  # noqa: E402
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.schemas import (  # noqa: E402
    ActionError,
    EvidenceRecord,
    Hypothesis,
    load_run_config,
)
from src.agent_system.runner import validate_branch_decision  # noqa: E402
from src.agent_system.schemas import BranchDecision, ParentSlot, ParentUpdate  # noqa: E402

ENCODING = Encoding(8, 4)


def record(pair, positive, origin=0, round_number=1):
    return EvidenceRecord(
        event_id="e",
        pair=pair,
        outcome="positive" if positive else "negative",
        origin_agent=origin,
        created_round=round_number,
        experiment_id=None,
    )


class CanonicalisationTests(unittest.TestCase):
    def test_addition_order_does_not_matter(self) -> None:
        left = Hypothesis(((2, 1), (5, 0)))
        right = Hypothesis(((5, 0), (2, 1)))
        self.assertEqual(left, right)
        self.assertEqual(left.hid, right.hid)

    def test_confluent_merge_keeps_every_parent_origin(self) -> None:
        collector = ChildCollector()
        correct = ENCODING.node_from_pair((0, 0), wrong=False)
        incorrect = ENCODING.node_from_pair((1, 1), wrong=True)
        child_a = ENCODING.child(correct, (1, 1), wrong=True)
        child_b = ENCODING.child(incorrect, (0, 0), wrong=False)
        collector.add(correct, (1, 1), child_a)
        collector.add(incorrect, (0, 0), child_b)
        self.assertEqual(len(collector.children), 1)
        merged = collector.frontier()[0]
        self.assertTrue(merged.wrong)
        self.assertTrue(merged.from_wrong, "a single incorrect parent makes the child old lineage")
        self.assertEqual(collector.merge_loss, 1)
        self.assertEqual(len(collector.dual_origin), 1)

    def test_copies_in_different_agents_are_counted_separately(self) -> None:
        first, second = ChildCollector(), ChildCollector()
        parent = ENCODING.node_from_pair((0, 0), wrong=False)
        for collector in (first, second):
            collector.add(parent, (1, 1), ENCODING.child(parent, (1, 1), wrong=True))
        self.assertEqual(len(first.children) + len(second.children), 2)


class ExactPruningTests(unittest.TestCase):
    def test_positive_evidence_only_removes_rival_values(self) -> None:
        store = EvidenceStore(ENCODING)
        partial_truth = ENCODING.node_from_pair((3, 2), wrong=False)
        rival = ENCODING.node_from_pair((3, 1), wrong=True)
        unrelated = ENCODING.node_from_pair((5, 0), wrong=True)
        store.apply(record((3, 2), True), own=True)
        survivors = store.prune([partial_truth, rival, unrelated])
        self.assertIn(partial_truth, survivors)
        self.assertIn(unrelated, survivors, "a positive label must not delete hypotheses that lack the component")
        self.assertNotIn(rival, survivors)

    def test_negative_evidence_removes_only_that_value(self) -> None:
        store = EvidenceStore(ENCODING)
        store.apply(record((3, 1), False), own=True)
        self.assertFalse(store.value_allowed((3, 1)))
        self.assertTrue(store.value_allowed((3, 2)))
        self.assertEqual(store.known_negative(), ((3, 1),))

    def test_true_partial_mechanism_survives_oracle_evidence(self) -> None:
        task = build_task_instance(
            agents=4, encoding=ENCODING, task_size=4, seed=12345, initial_width=4
        )
        store = EvidenceStore(ENCODING)
        for pair in task.truth_pairs:
            store.apply(record(pair, True), own=True)
        for family in range(ENCODING.families):
            for variant in range(ENCODING.variants):
                if not task.is_positive((family, variant)):
                    store.apply(record((family, variant), False), own=True)
        node = ENCODING.node_from_pair(task.truth_pairs[0], wrong=False)
        node = ENCODING.child(node, task.truth_pairs[1], wrong=False)
        self.assertTrue(store.compatible(node))


class CommunicationTests(unittest.TestCase):
    def _path_router(self) -> Router:
        router = Router(3, 0, 1)
        router.graph = ((1,), (0, 2), (1,))
        return router

    def test_label_arrives_next_round_and_is_not_forwarded(self) -> None:
        router = self._path_router()
        router.stage(0, record((1, 1), False, origin=0))
        self.assertEqual(router.deliveries(1), [], "a label must not arrive in the round it was produced")
        router.commit_round()
        delivered = router.deliveries(1)
        self.assertEqual([item[1].pair for item in delivered], [(1, 1)])
        self.assertEqual(router.deliveries(2), [], "B must not forward A's label to C")

    def test_agent_cannot_send_a_label_it_did_not_produce(self) -> None:
        router = self._path_router()
        with self.assertRaises(ValueError):
            router.stage(1, record((1, 1), False, origin=0))

    def test_degree_zero_receives_nothing(self) -> None:
        router = Router(4, 0, 7)
        router.stage(0, record((1, 1), False, origin=0))
        router.commit_round()
        for agent in range(4):
            self.assertEqual(router.deliveries(agent), [])


class BudgetTests(unittest.TestCase):
    def test_one_agent_may_exceed_m_when_others_are_idle(self) -> None:
        pools = {0: ["a", "b", "c", "d"], 1: [], 2: []}
        performed: list[tuple[int, int]] = []

        async def perform(agent_index, candidates, ordinal):
            performed.append((agent_index, ordinal))
            pools[agent_index].pop()

        result = asyncio.run(
            randomized_round_robin_allocate_async(
                n_agents=3,
                total_budget=3,
                seed=11,
                round_number=1,
                candidates_for=lambda index: list(pools[index]),
                perform_one=perform,
            )
        )
        self.assertEqual(result["B_used"], 3)
        self.assertEqual(result["experiments_per_agent"], [3, 0, 0])
        self.assertEqual(result["max_experiments_per_agent"], 3)

    def test_budget_is_returned_when_candidates_run_out(self) -> None:
        pools = {0: ["a"], 1: ["b"]}

        async def perform(agent_index, candidates, ordinal):
            pools[agent_index].clear()

        result = asyncio.run(
            randomized_round_robin_allocate_async(
                n_agents=2,
                total_budget=10,
                seed=3,
                round_number=1,
                candidates_for=lambda index: list(pools[index]),
                perform_one=perform,
            )
        )
        self.assertEqual(result["B_used"], 2)
        self.assertEqual(result["total_candidate_capacity_end"], 0)

    def test_candidates_are_refreshed_between_slots(self) -> None:
        seen: list[int] = []
        pool = [1, 2, 3]

        async def perform(agent_index, candidates, ordinal):
            seen.append(len(candidates))
            pool.pop()

        asyncio.run(
            randomized_round_robin_allocate_async(
                n_agents=1,
                total_budget=3,
                seed=5,
                round_number=1,
                candidates_for=lambda index: list(pool),
                perform_one=perform,
            )
        )
        self.assertEqual(seen, [3, 2, 1])


class CandidatePoolTests(unittest.TestCase):
    """A pair can stay testable after every child carrying it was pruned."""

    def _runner(self, pool: str) -> EpisodeRunner:
        config = load_run_config(
            {
                "name": "pool",
                "agents": 2,
                "branching": 2,
                "experiments_per_agent": 1,
                "initial_width": 2,
                "degrees": [0],
                "candidate_pool": pool,
                "environment": {"kind": "synthetic", "families": 8, "variants": 4, "task_size": 4},
                "policy": {"kind": "uniform"},
            }
        )
        task = build_task_instance(
            agents=2, encoding=ENCODING, task_size=4, seed=99, initial_width=2
        )
        runner = EpisodeRunner(
            config=config,
            degree=0,
            seed=99,
            task=task,
            environment=SyntheticEnvironment(task),
            policy_factory=lambda: UniformPolicy(),
            episode_id="pool",
        )
        runner._build_agents()
        agent = runner.agents[0]
        agent.evidence = EvidenceStore(ENCODING)
        parent = ENCODING.node_from_pair((0, 0), wrong=True)
        agent.proposed_parents = {(1, 1): [parent], (2, 2): [parent]}
        # Every child was pruned by evidence about family 0, yet (1,1) and
        # (2,2) themselves are still untested and still allowed.
        agent.evidence.apply(record((0, 1), True), own=True)
        agent.frontier = []
        return runner

    def test_proposed_allowed_keeps_testing_after_the_frontier_empties(self) -> None:
        runner = self._runner("proposed_allowed_v1")
        pairs = [item.pair for item in runner._candidates_for(0)]
        self.assertEqual(pairs, [(1, 1), (2, 2)])

    def test_surviving_leaf_requires_a_live_child(self) -> None:
        runner = self._runner("surviving_leaf_v1")
        self.assertEqual(runner._candidates_for(0), [])


class BranchValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.slot = ParentSlot(
            hypothesis=Hypothesis(((0, 0),)),
            legal_additions=((1, 0), (1, 1), (2, 0)),
            required=2,
        )

    def _decision(self, additions, parent_id=None):
        return BranchDecision(
            updates=(
                ParentUpdate(parent_id=parent_id or self.slot.hypothesis.hid, additions=additions),
            )
        )

    def test_accepts_a_legal_update(self) -> None:
        ordered = validate_branch_decision(self._decision(((1, 0), (2, 0))), [self.slot])
        self.assertEqual(ordered[0][1], ((1, 0), (2, 0)))

    def test_rejects_wrong_count(self) -> None:
        with self.assertRaises(ActionError):
            validate_branch_decision(self._decision(((1, 0),)), [self.slot])

    def test_rejects_repeated_addition(self) -> None:
        with self.assertRaises(ActionError):
            validate_branch_decision(self._decision(((1, 0), (1, 0))), [self.slot])

    def test_rejects_illegal_component(self) -> None:
        with self.assertRaises(ActionError):
            validate_branch_decision(self._decision(((1, 0), (7, 3))), [self.slot])

    def test_rejects_unknown_parent(self) -> None:
        with self.assertRaises(ActionError):
            validate_branch_decision(self._decision(((1, 0), (2, 0)), "h_nope"), [self.slot])


if __name__ == "__main__":
    unittest.main()
