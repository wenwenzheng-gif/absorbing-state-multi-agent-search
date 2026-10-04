#!/usr/bin/env python3
"""DREAM4 loading, sign derivation and the task instance built from it."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.environments.grn import GRNEnvironment  # noqa: E402
from src.agent_system.environments.grn_dataset import (  # noqa: E402
    ACTIVATION,
    INHIBITION,
    build_grn_task,
    grn_task_instance,
    hypothesis_space_bits,
    load_network,
    select_targets,
)
from src.agent_system.schemas import load_run_config  # noqa: E402
from src.agent_system.shard import shard_configs  # noqa: E402

DATA = ROOT / "data" / "external" / "dream4"
HAVE_DATA = (DATA / "goldstandard_1.tsv").is_file()
skip_without_data = unittest.skipUnless(
    HAVE_DATA, "DREAM4 files are not present under data/external/dream4"
)


class HypothesisSpaceTests(unittest.TestCase):
    def test_matches_the_paper_formula(self) -> None:
        # H(K) = log2[C(99, K) * 2^K]
        for size in (1, 2, 3, 4, 5):
            self.assertAlmostEqual(
                hypothesis_space_bits(99, size),
                math.log2(math.comb(99, size) * 2 ** size),
                places=9,
            )


@skip_without_data
class NetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = load_network(1)

    def test_shape(self) -> None:
        self.assertEqual(len(self.network.genes), 100)
        self.assertEqual(len(self.network.knockout), 100)
        self.assertEqual(len(self.network.knockdown), 100)
        self.assertEqual(len(self.network.wildtype), 100)

    def test_sign_follows_the_knockout_direction(self) -> None:
        """Removing an activator lowers the target; an inhibitor raises it."""
        target_slot = self.network.position("G4")
        for regulator, sign in self.network.parents["G4"]:
            delta = (
                self.network.knockout[self.network.position(regulator)][target_slot]
                - self.network.wildtype[target_slot]
            )
            self.assertEqual(sign, INHIBITION if delta > 0 else ACTIVATION, regulator)

    def test_in_degree_selection_is_exact(self) -> None:
        for size in (1, 2, 3, 4):
            for target in self.network.targets_with_in_degree(size):
                self.assertEqual(len(self.network.parents[target]), size)


@skip_without_data
class TaskInstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = load_network(1)
        self.task = build_grn_task(self.network, "G4")

    def test_truth_marks_only_the_real_parents(self) -> None:
        named = {gene for gene, _sign in self.network.parents["G4"]}
        for family, value in enumerate(self.task.truth_values):
            expected = self.task.regulators[family] in named
            self.assertEqual(value >= 0, expected, self.task.regulators[family])

    def test_the_target_is_not_its_own_candidate_regulator(self) -> None:
        self.assertNotIn("G4", self.task.regulators)
        self.assertEqual(len(self.task.regulators), 99)

    def test_mode_b_gives_every_agent_exactly_one_true_component(self) -> None:
        instance = grn_task_instance(task=self.task, agents=40, seed=11, initial_width=8)
        for group in instance.initial_pairs:
            self.assertEqual(sum(pair in instance.truth_set for pair in group), 1)
        self.assertEqual(instance.initial_incorrect(), 40 * 8 - 40)

    def test_a_target_without_parents_is_refused(self) -> None:
        empty = build_grn_task(self.network, self.network.genes[0])
        if empty.task_size == 0:
            with self.assertRaises(ValueError):
                grn_task_instance(task=empty, agents=4, seed=1, initial_width=2)


@skip_without_data
class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = load_network(1)
        self.grn_task = build_grn_task(self.network, "G4")
        self.instance = grn_task_instance(
            task=self.grn_task, agents=8, seed=3, initial_width=4
        )
        self.env = GRNEnvironment(
            self.instance, network=self.network, grn_task=self.grn_task, observations=2
        )

    def test_library_covers_every_regulator_and_both_signs(self) -> None:
        library = self.env.component_library()
        self.assertEqual(len(library), 99 * 2)
        self.assertEqual(
            {(spec.family_id, spec.variant_id) for spec in library},
            {(f, v) for f in range(99) for v in (ACTIVATION, INHIBITION)},
        )

    def test_there_is_no_experiment_bank_and_no_hold_out(self) -> None:
        # The intervention is fixed by the component tested, and a thresholded
        # hold-out is not well posed on this data; see grn_dataset.
        self.assertEqual(self.env.experiment_library(), ())
        self.assertFalse(self.env.requires_experiment_id())
        self.assertFalse(self.env.supports_prediction())

    def test_each_agent_sees_a_different_slice_of_the_knockouts(self) -> None:
        seen = [
            tuple(obs.experiment_id for obs in self.env.initial_observations(index))
            for index in range(8)
        ]
        self.assertEqual(len(set(seen)), 8)

    def test_no_observation_leaks_the_target_membership(self) -> None:
        for index in range(8):
            for observation in self.env.initial_observations(index):
                blob = json.dumps(observation.public_dict())
                self.assertNotIn("parent", blob)
                self.assertNotIn("truth", blob)


@skip_without_data
class ShardSlicingTests(unittest.TestCase):
    def test_grn_targets_stay_parallel_to_task_seeds(self) -> None:
        targets = select_targets(3, 3)
        payload = {
            "name": "grn_shard_test",
            "agents": 8,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 4,
            "degrees": [0],
            "task_seeds": [1, 2, 3],
            "environment": {
                "kind": "grn",
                "families": 99,
                "variants": 2,
                "task_size": 3,
                "grn_targets": targets,
            },
            "policy": {"kind": "uniform"},
        }
        config = load_run_config(payload)
        with tempfile.TemporaryDirectory() as tmp:
            shards = shard_configs(config, Path(tmp))
            self.assertEqual(len(shards), 3)
            for position, (seed, path, _run_id) in enumerate(shards):
                body = json.loads(path.read_text())
                self.assertEqual(body["task_seeds"], [seed])
                self.assertEqual(body["environment"]["grn_targets"], [targets[position]])
                load_run_config(body).validate()


if __name__ == "__main__":
    unittest.main()
