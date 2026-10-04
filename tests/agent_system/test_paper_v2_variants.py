#!/usr/bin/env python3
"""The dataset variants the revised paper specifies.

Three things changed: tcas moved to a larger input space and to component
level evidence, physics became SINDy-style support recovery over a flat term
library, and the success probability gained a closed form to check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis.success_model import final_incorrect, rows_by_degree  # noqa: E402
from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.component_library import (  # noqa: E402
    FAMILY_COUNT,
    VARIANT_COUNT,
    build_library,
    build_support_library,
)
from src.agent_system.environments.physics import PhysicsEnvironment  # noqa: E402
from src.agent_system.environments.tcas import TcasEnvironment  # noqa: E402
from src.agent_system.environments.tcas_dataset import (  # noqa: E402
    assignment_count,
    seed_failure_combination,
    space,
    space_size,
    tcas_task_instance,
)
from src.agent_system.evidence import EvidenceStore  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.schemas import EnvironmentConfig, load_run_config  # noqa: E402


class TcasSpaceTests(unittest.TestCase):
    def test_both_published_input_counts(self) -> None:
        self.assertEqual(space_size("v1_460800"), 2 ** 7 * 3 ** 2 * 4 * 10 ** 2)
        self.assertEqual(space_size("v1_460800"), 460800)
        self.assertEqual(assignment_count("v1_460800"), 44)
        # 3 * 2^3 * 3 * 2 * 4 * 10^2 * 3 * 2 * 3
        self.assertEqual(space_size("v2_1036800"), 1036800)
        self.assertEqual(assignment_count("v2_1036800"), 46)

    def test_an_unknown_space_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            space("v3")

    def test_the_combination_respects_the_space_cardinalities(self) -> None:
        for name in ("v1_460800", "v2_1036800"):
            task = seed_failure_combination(strength=4, seed=3, space_name=name)
            self.assertEqual(task.space_name, name)
            for parameter, value in enumerate(task.truth_values):
                if value >= 0:
                    self.assertLess(value, task.cardinalities[parameter])


class TcasEvidenceModelTests(unittest.TestCase):
    def _env(self, model: str) -> TcasEnvironment:
        task = seed_failure_combination(strength=3, seed=7, space_name="v2_1036800")
        instance = tcas_task_instance(task=task, agents=8, seed=7, initial_width=6)
        return TcasEnvironment(
            instance, tcas_task=task, heldout_size=128, bank_size=16, evidence_model=model
        )

    def test_component_model_uses_the_shared_store_and_no_bank(self) -> None:
        env = self._env("component")
        store = env.new_evidence_store(env.task.encoding)
        self.assertIsInstance(store, EvidenceStore)
        self.assertEqual(env.experiment_library(), ())
        self.assertFalse(env.requires_experiment_id())

    def test_subset_model_keeps_its_own_store_and_a_bank(self) -> None:
        env = self._env("subset")
        self.assertNotIsInstance(env.new_evidence_store(env.task.encoding), EvidenceStore)
        self.assertEqual(len(env.experiment_library()), 16)

    def test_masks_keep_the_padding_out_in_either_model(self) -> None:
        for model in ("component", "subset"):
            env = self._env(model)
            store = env.new_evidence_store(env.task.encoding)
            for family, size in enumerate(env.cardinalities):
                self.assertTrue(store.value_allowed((family, size - 1)), model)
                for variant in range(size, max(env.cardinalities)):
                    self.assertFalse(store.value_allowed((family, variant)), model)

    def test_an_unknown_evidence_model_is_refused(self) -> None:
        task = seed_failure_combination(strength=2, seed=1)
        instance = tcas_task_instance(task=task, agents=4, seed=1, initial_width=3)
        with self.assertRaises(ValueError):
            TcasEnvironment(instance, tcas_task=task, evidence_model="whatever")

    def test_component_model_labels_one_assignment(self) -> None:
        env = self._env("component")
        truth = tuple(sorted(env.task.truth_pairs))
        result = asyncio.run(
            env.execute(pair=truth[0], experiment_id=None, owner=0, round_number=1, event_id="e")
        )
        self.assertEqual(result.evidence.outcome, "positive")
        self.assertIsNone(result.evidence.configuration)


class PhysicsSupportTests(unittest.TestCase):
    def test_the_flat_library_is_the_factored_one_unrolled(self) -> None:
        flat = build_support_library()
        self.assertEqual(len(flat), FAMILY_COUNT * VARIANT_COUNT)
        self.assertEqual(
            [term.spec.expression for term in flat],
            [term.spec.expression for term in build_library()],
        )
        for index, term in enumerate(flat):
            self.assertEqual(term.pair, (index, 0))

    def test_asking_for_more_terms_than_exist_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_support_library(FAMILY_COUNT * VARIANT_COUNT + 1)

    def test_the_target_can_be_any_subset_and_scores_zero(self) -> None:
        encoding = Encoding(families=32, variants=1)
        config = EnvironmentConfig(
            kind="physics", families=32, variants=1, task_size=4,
            physics_mode="support_v1", heldout_size=8,
        )
        instance = build_task_instance(
            agents=6, encoding=encoding, task_size=4, seed=5, initial_width=4
        )
        env = PhysicsEnvironment(instance, config, 6, mode="support_v1")
        self.assertEqual(len(env.component_library()), 32)
        self.assertEqual(env.heldout_error(instance.truth_pairs), 0.0)
        # No family constraint: the truth is a plain subset of the 32 terms.
        self.assertEqual(len({pair[0] for pair in instance.truth_pairs}), 4)
        self.assertEqual({pair[1] for pair in instance.truth_pairs}, {0})

    def test_support_mode_requires_a_single_variant(self) -> None:
        payload = {
            "name": "bad", "agents": 4, "branching": 2, "experiments_per_agent": 1,
            "initial_width": 3, "degrees": [0], "task_seeds": [1],
            "environment": {"kind": "physics", "families": 32, "variants": 4,
                            "task_size": 3, "physics_mode": "support_v1"},
            "policy": {"kind": "uniform"},
        }
        with self.assertRaises(ValueError):
            load_run_config(payload).validate()


class SuccessModelTests(unittest.TestCase):
    def episode(self, degree, a_final, solved, agents=10):
        return {
            "status": "ok", "d": degree, "N": agents,
            "transitions": [{"A_post": 100}, {"A_post": a_final}],
            "evaluation": {"any_verified_predictive_success": solved},
        }

    def test_A_T_is_the_last_round(self) -> None:
        self.assertEqual(final_incorrect(self.episode(0, 7, False)), 7.0)
        self.assertIsNone(final_incorrect({"transitions": []}))

    def test_both_predictions_are_reported(self) -> None:
        rows = rows_by_degree([self.episode(0, 9, False, agents=10)])
        row = rows[0]
        self.assertAlmostEqual(row["mean_A_T"], 9.0)
        self.assertAlmostEqual(row["P_predicted"], 1 / 10)
        # per agent: A_T/N = 0.9, so 1/(1+0.9)
        self.assertAlmostEqual(row["P_predicted_per_agent"], 1 / 1.9)
        self.assertEqual(row["P_observed"], 0.0)

    def test_observed_is_the_share_of_solved_episodes(self) -> None:
        rows = rows_by_degree(
            [self.episode(3, 4, True), self.episode(3, 4, False)]
        )
        self.assertAlmostEqual(rows[0]["P_observed"], 0.5)

    def test_failed_episodes_are_excluded(self) -> None:
        bad = self.episode(1, 5, True)
        bad["status"] = "policy_failed"
        self.assertEqual(rows_by_degree([bad]), [])


if __name__ == "__main__":
    unittest.main()
