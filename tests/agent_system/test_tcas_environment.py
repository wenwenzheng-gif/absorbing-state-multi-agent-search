#!/usr/bin/env python3
"""The tcas parameter space, the seeded oracle and subset pruning."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.environments.tcas import (  # noqa: E402
    SubsetEvidenceStore,
    TcasEnvironment,
)
from src.agent_system.environments.tcas_dataset import (  # noqa: E402
    ASSIGNMENT_COUNT,
    CARDINALITIES,
    CONFIGURATION_COUNT,
    MAX_CARDINALITY,
    PARAMETERS,
    allowed_masks,
    seed_failure_combination,
    tcas_task_instance,
)
from src.agent_system.evidence import EvidenceStore  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.schemas import EvidenceRecord  # noqa: E402

ENCODING = Encoding(families=len(PARAMETERS), variants=MAX_CARDINALITY)


def record(configuration, *, failed, pair=(0, 0)):
    return EvidenceRecord(
        event_id="e",
        pair=pair,
        outcome="positive" if failed else "negative",
        origin_agent=0,
        created_round=1,
        experiment_id="cfg000",
        configuration=tuple(configuration),
    )


class SpaceTests(unittest.TestCase):
    def test_matches_the_published_counts(self) -> None:
        self.assertEqual(CONFIGURATION_COUNT, 2 ** 7 * 3 ** 2 * 4 * 10 ** 2)
        self.assertEqual(CONFIGURATION_COUNT, 460800)
        self.assertEqual(ASSIGNMENT_COUNT, 44)
        self.assertEqual(sorted(CARDINALITIES), [2] * 7 + [3] * 2 + [4] + [10] * 2)

    def test_masks_admit_exactly_the_real_values(self) -> None:
        for parameter, mask in enumerate(allowed_masks()):
            self.assertEqual(mask.bit_count(), CARDINALITIES[parameter])
            self.assertEqual(mask, (1 << CARDINALITIES[parameter]) - 1)


class OracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = seed_failure_combination(strength=3, seed=5)
        self.fixed = [
            (p, v) for p, v in enumerate(self.task.truth_values) if v >= 0
        ]

    def test_containing_the_combination_fails(self) -> None:
        configuration = [0] * len(PARAMETERS)
        for parameter, value in self.fixed:
            configuration[parameter] = value
        self.assertTrue(self.task.fails(tuple(configuration)))

    def test_breaking_one_assignment_passes(self) -> None:
        configuration = [0] * len(PARAMETERS)
        for parameter, value in self.fixed:
            configuration[parameter] = value
        parameter, value = self.fixed[0]
        configuration[parameter] = (value + 1) % CARDINALITIES[parameter]
        self.assertFalse(self.task.fails(tuple(configuration)))

    def test_the_mask_turns_some_failures_into_passes(self) -> None:
        masked = seed_failure_combination(strength=3, seed=5, mask=True)
        self.assertIsNotNone(masked.mask_parameter)
        self.assertGreater(masked.masked_fraction(), 0.0)
        configuration = [0] * len(PARAMETERS)
        for parameter, value in enumerate(masked.truth_values):
            if value >= 0:
                configuration[parameter] = value
        configuration[masked.mask_parameter] = masked.mask_value
        self.assertFalse(masked.fails(tuple(configuration)))
        other = (masked.mask_value + 1) % CARDINALITIES[masked.mask_parameter]
        configuration[masked.mask_parameter] = other
        self.assertTrue(masked.fails(tuple(configuration)))

    def test_mode_b_never_draws_a_padded_assignment(self) -> None:
        instance = tcas_task_instance(task=self.task, agents=16, seed=5, initial_width=8)
        for group in instance.initial_pairs:
            self.assertEqual(sum(pair in instance.truth_set for pair in group), 1)
            for family, variant in group:
                self.assertLess(variant, CARDINALITIES[family])


class SubsetPruningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SubsetEvidenceStore(ENCODING, masks=allowed_masks())

    def test_a_pass_refutes_every_combination_it_contains(self) -> None:
        passing = tuple([0] * len(PARAMETERS))
        self.store.apply(record(passing, failed=False), own=True)
        contained = ENCODING.child(
            ENCODING.node_from_pair((0, 0), wrong=True), (1, 0), wrong=True
        )
        self.assertFalse(self.store.compatible(contained))

    def test_a_pass_leaves_anything_it_does_not_contain(self) -> None:
        passing = tuple([0] * len(PARAMETERS))
        self.store.apply(record(passing, failed=False), own=True)
        outside = ENCODING.node_from_pair((0, 1), wrong=True)
        self.assertTrue(self.store.compatible(outside))

    def test_a_failing_configuration_refutes_nothing(self) -> None:
        failing = tuple([0] * len(PARAMETERS))
        self.store.apply(record(failing, failed=True), own=True)
        for family in range(len(PARAMETERS)):
            self.assertTrue(
                self.store.compatible(ENCODING.node_from_pair((family, 0), wrong=True))
            )

    def test_a_component_is_never_used_up(self) -> None:
        # Unlike the component-level store, a verdict settles a configuration,
        # not an assignment, so the same assignment stays testable.
        self.assertFalse(self.store.knows((0, 0)))
        self.store.apply(record([0] * len(PARAMETERS), failed=False), own=True)
        self.assertFalse(self.store.knows((0, 0)))
        self.assertFalse(SubsetEvidenceStore.consumes_candidates)
        self.assertTrue(EvidenceStore.consumes_candidates)

    def test_padded_values_are_never_allowed(self) -> None:
        for family, size in enumerate(CARDINALITIES):
            for variant in range(size, MAX_CARDINALITY):
                self.assertFalse(self.store.value_allowed((family, variant)))
            self.assertTrue(self.store.value_allowed((family, size - 1)))

    def test_verification_needs_own_support_and_no_refutation(self) -> None:
        components = ((0, 0), (1, 0))
        failing = tuple([0] * len(PARAMETERS))
        self.store.apply(record(failing, failed=True), own=True)
        self.assertTrue(self.store.covers_positively(components))
        other = list(failing)
        other[5] = 1
        self.store.apply(record(other, failed=False), own=True)
        self.assertFalse(self.store.covers_positively(components))


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = seed_failure_combination(strength=3, seed=17)
        self.instance = tcas_task_instance(
            task=self.task, agents=8, seed=17, initial_width=6
        )
        self.env = TcasEnvironment(
            self.instance, tcas_task=self.task, heldout_size=512, bank_size=64
        )

    def test_library_is_the_44_real_assignments(self) -> None:
        self.assertEqual(len(self.env.component_library()), ASSIGNMENT_COUNT)

    def test_the_bank_and_the_hold_out_are_disjoint(self) -> None:
        bank = set(self.env._bank_by_id.values())
        self.assertEqual(len(bank), 64)
        self.assertEqual(len(set(self.env._heldout)), 512)
        self.assertFalse(bank & set(self.env._heldout))

    def test_the_truth_scores_a_zero_hold_out_error(self) -> None:
        truth = tuple(sorted(self.instance.truth_pairs))
        self.assertEqual(self.env.heldout_error(truth), 0.0)

    def test_a_near_miss_is_penalised(self) -> None:
        truth = tuple(sorted(self.instance.truth_pairs))
        family, variant = truth[0]
        wrong = ((family, (variant + 1) % CARDINALITIES[family]),) + truth[1:]
        self.assertGreater(self.env.heldout_error(wrong), 0.0)

    def test_the_test_configuration_contains_the_hypothesis(self) -> None:
        truth = tuple(sorted(self.instance.truth_pairs))
        result = asyncio.run(
            self.env.execute(
                pair=truth[-1],
                experiment_id="cfg000",
                owner=0,
                round_number=1,
                event_id="e",
                parent_components=truth[:-1],
            )
        )
        configuration = result.evidence.configuration
        for family, variant in truth:
            self.assertEqual(configuration[family], variant)
        self.assertEqual(result.evidence.outcome, "positive")

    def test_an_unknown_bank_id_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            asyncio.run(
                self.env.execute(
                    pair=(0, 0),
                    experiment_id="nope",
                    owner=0,
                    round_number=1,
                    event_id="e",
                )
            )


if __name__ == "__main__":
    unittest.main()
