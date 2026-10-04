#!/usr/bin/env python3
"""Numerics and data isolation for the physics environment."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.component_library import build_library  # noqa: E402
from src.agent_system.environments.physics import PhysicsEnvironment  # noqa: E402
from src.agent_system.environments.physics_dataset import (  # noqa: E402
    MIN_LIBRARY_DISTANCE,
    build_physics_data,
)
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.schemas import EnvironmentConfig, ExperimentCondition  # noqa: E402

ENCODING = Encoding(8, 4)
CONFIG = EnvironmentConfig(kind="physics", task_size=4)


def environment(seed: int = 12345, agents: int = 4) -> PhysicsEnvironment:
    task = build_task_instance(
        agents=agents, encoding=ENCODING, task_size=4, seed=seed, initial_width=6
    )
    return PhysicsEnvironment(task, CONFIG, agents)


class LibraryTests(unittest.TestCase):
    def test_eight_families_of_four_variants(self) -> None:
        library = build_library()
        self.assertEqual(len(library), 32)
        self.assertEqual(len({term.pair for term in library}), 32)

    def test_parameters_are_public_and_rendered(self) -> None:
        library = {term.pair: term for term in build_library()}
        self.assertEqual(library[(0, 2)].spec.parameters, {"alpha": 1.3})
        self.assertIn("1.3", library[(0, 2)].spec.expression)


class NumericsTests(unittest.TestCase):
    def test_linear_restoring_matches_the_analytic_solution(self) -> None:
        env = environment()
        alpha = 0.5
        omega = math.sqrt(alpha)
        condition = ExperimentCondition(
            experiment_id="probe", description="", settings={"x0": 1.0, "v0": -0.4, "t0": 0.0}
        )
        states = env.predict([(0, 0)], condition)
        times = states[:, 0]
        expected_x = 1.0 * np.cos(omega * times) + (-0.4 / omega) * np.sin(omega * times)
        expected_v = -1.0 * omega * np.sin(omega * times) + (-0.4) * np.cos(omega * times)
        self.assertLess(float(np.max(np.abs(states[:, 1] - expected_x))), 1e-8)
        self.assertLess(float(np.max(np.abs(states[:, 2] - expected_v))), 1e-8)

    def test_damped_oscillator_matches_the_analytic_solution(self) -> None:
        env = environment()
        alpha, gamma = 0.5, 0.1  # families (0,0) and (2,0)
        condition = ExperimentCondition(
            experiment_id="probe2", description="", settings={"x0": 1.5, "v0": 0.0, "t0": 0.0}
        )
        states = env.predict([(0, 0), (2, 0)], condition)
        times = states[:, 0]
        decay = gamma / 2.0
        omega = math.sqrt(alpha - decay * decay)
        amplitude = 1.5
        phase = math.atan2(decay * amplitude, omega * amplitude)
        scale = amplitude / math.cos(phase)
        expected = scale * np.exp(-decay * times) * np.cos(omega * times - phase)
        self.assertLess(float(np.max(np.abs(states[:, 1] - expected))), 1e-7)

    def test_tighter_tolerance_converges(self) -> None:
        condition = ExperimentCondition(
            experiment_id="probe3", description="", settings={"x0": 2.0, "v0": 1.0, "t0": 0.5}
        )
        task = build_task_instance(
            agents=2, encoding=ENCODING, task_size=4, seed=4242, initial_width=4
        )
        loose = PhysicsEnvironment(task, CONFIG, 2).predict(task.truth_pairs, condition)
        strict_config = EnvironmentConfig(kind="physics", task_size=4, rtol=1e-12, atol=1e-14)
        strict = PhysicsEnvironment(task, strict_config, 2).predict(task.truth_pairs, condition)
        self.assertLess(float(np.max(np.abs(loose[:, 1:] - strict[:, 1:]))), 1e-7)


class LabelChannelTests(unittest.TestCase):
    def test_label_does_not_depend_on_the_chosen_condition(self) -> None:
        env = environment()
        pair = env.task.truth_pairs[0]
        first = asyncio.run(
            env.execute(pair=pair, experiment_id="e_000", owner=0, round_number=1, event_id="a")
        )
        second = asyncio.run(
            env.execute(pair=pair, experiment_id="e_099", owner=1, round_number=1, event_id="b")
        )
        self.assertEqual(first.evidence.outcome, "positive")
        self.assertEqual(first.evidence.outcome, second.evidence.outcome)
        self.assertNotEqual(first.observation.states, second.observation.states)

    def test_trajectory_and_label_travel_in_separate_channels(self) -> None:
        env = environment()
        result = asyncio.run(
            env.execute(
                pair=(1, 1), experiment_id="e_010", owner=2, round_number=1, event_id="c"
            )
        )
        self.assertEqual(result.evidence.origin_agent, 2)
        self.assertIsNotNone(result.observation)
        self.assertFalse(hasattr(result.evidence, "states"))

    def test_hold_out_error_is_zero_for_the_truth_and_large_for_a_fragment(self) -> None:
        env = environment()
        self.assertAlmostEqual(env.heldout_error(env.task.truth_pairs), 0.0, places=12)
        self.assertGreater(env.heldout_error(env.task.truth_pairs[:2]), 1e-3)


class DataIsolationTests(unittest.TestCase):
    def test_hold_out_conditions_avoid_the_public_grid(self) -> None:
        data = build_physics_data(CONFIG, 12345, 4)
        self.assertEqual(len(data.heldout), CONFIG.heldout_size)
        for held in data.heldout:
            for public in data.library:
                distance = math.dist(
                    (held.settings["x0"], held.settings["v0"], held.settings["t0"]),
                    (public.settings["x0"], public.settings["v0"], public.settings["t0"]),
                )
                self.assertGreaterEqual(distance, MIN_LIBRARY_DISTANCE)

    def test_each_agent_gets_its_own_initial_observations(self) -> None:
        env = environment(agents=4)
        first = env.initial_observations(0)
        second = env.initial_observations(1)
        self.assertEqual(len(first), CONFIG.initial_observations)
        self.assertNotEqual(
            [obs.settings for obs in first], [obs.settings for obs in second]
        )

    def test_public_library_is_the_frozen_grid(self) -> None:
        env = environment()
        self.assertEqual(len(env.experiment_library()), 100)
        self.assertTrue(all(c.experiment_id.startswith("e_") for c in env.experiment_library()))

    def test_an_unpublished_condition_cannot_be_executed(self) -> None:
        env = environment()
        with self.assertRaises(Exception):
            asyncio.run(
                env.execute(
                    pair=(0, 0),
                    experiment_id="holdout_000",
                    owner=0,
                    round_number=1,
                    event_id="d",
                )
            )


if __name__ == "__main__":
    unittest.main()
