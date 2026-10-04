#!/usr/bin/env python3
"""tcas_selection: the long paper's argmax-containment experiment choice."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.environments.tcas import TcasEnvironment  # noqa: E402
from src.agent_system.environments.tcas_dataset import (  # noqa: E402
    seed_failure_combination,
    tcas_task_instance,
)
from src.agent_system.evaluation import HiddenEvaluator  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.policies.uniform import UniformPolicy  # noqa: E402
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.schemas import load_run_config  # noqa: E402

BASE = {
    "name": "tcas_knob",
    "agents": 12,
    "branching": 2,
    "experiments_per_agent": 1,
    "initial_width": 8,
    "environment": {
        "kind": "tcas",
        "families": 12,
        "variants": 10,
        "task_size": 3,
        "heldout_size": 64,
        "tcas_bank_size": 64,
        "tcas_space": "v2_1036800",
        "tcas_evidence": "subset",
    },
    "policy": {"kind": "uniform", "planner": "uniform"},
}


def build(overrides: dict, *, degree: int, seed: int) -> EpisodeRunner:
    payload = {**BASE, **overrides, "degrees": [degree], "task_seeds": [seed]}
    payload["environment"] = {**BASE["environment"], **overrides.get("environment", {})}
    config = load_run_config(payload)
    encoding = Encoding(config.environment.families, config.environment.variants)
    tcas_task = seed_failure_combination(
        strength=config.task_size,
        seed=seed,
        mask=config.environment.tcas_mask,
        space_name=config.environment.tcas_space,
    )
    task = tcas_task_instance(
        task=tcas_task,
        agents=config.agents,
        seed=seed,
        initial_width=config.initial_width,
    )
    environment = TcasEnvironment(
        task,
        tcas_task=tcas_task,
        heldout_size=config.environment.heldout_size,
        bank_size=config.environment.tcas_bank_size,
        evidence_model=config.environment.tcas_evidence,
        selection=config.environment.tcas_selection,
    )
    return EpisodeRunner(
        config=config,
        degree=degree,
        seed=seed,
        task=task,
        environment=environment,
        policy_factory=lambda: UniformPolicy(planner=config.policy.planner),
        episode_id=f"d{degree}_s{seed}",
        evaluator=HiddenEvaluator(
            task=task,
            environment=environment,
            encoding=encoding,
            epsilon=config.epsilon_predictive,
        ),
    )


class Configuration(unittest.TestCase):
    def test_default_is_uniform_and_overrides_nothing(self) -> None:
        runner = build({}, degree=4, seed=92300)
        produced = asyncio.run(runner.run())
        for log in produced["transitions"]:
            self.assertEqual(log["protocol_selected_experiments"], 0)
        self.assertEqual(produced["tcas_selection"], "uniform")

    def test_max_containment_needs_subset_evidence(self) -> None:
        with self.assertRaises(ValueError):
            load_run_config(
                {
                    **BASE,
                    "degrees": [2],
                    "environment": {
                        **BASE["environment"],
                        "tcas_evidence": "component",
                        "tcas_selection": "max_containment",
                    },
                }
            )


class Argmax(unittest.TestCase):
    """The rule itself, on a bank whose answer can be worked out by hand."""

    def _environment(self, selection: str = "max_containment") -> TcasEnvironment:
        tcas_task = seed_failure_combination(
            strength=3, seed=92300, mask=False, space_name="v2_1036800"
        )
        task = tcas_task_instance(
            task=tcas_task, agents=4, seed=92300, initial_width=4
        )
        return TcasEnvironment(
            task,
            tcas_task=tcas_task,
            heldout_size=8,
            bank_size=64,
            evidence_model="subset",
            selection=selection,
        )

    def test_picks_the_pattern_containing_most_surviving_hypotheses(self) -> None:
        environment = self._environment()
        bank = environment._bank_patterns
        # h fixes parameter 0; the frontier members differ only in parameters
        # the bank supplies, so the winner is decidable from the bank itself.
        pair = (0, 0)
        parents: tuple = ()
        frontier = tuple(
            ((1, bank[index][1]), (2, bank[index][2])) for index in range(len(bank))
        )
        chosen = environment.select_experiment(
            pair=pair,
            parent_components=parents,
            frontier_components=frontier,
            seed=1,
            round_number=1,
            agent_index=0,
            slot=0,
        )
        expected = _brute_force(environment, pair, parents, frontier)
        self.assertIn(chosen, expected)

    def test_agrees_with_a_brute_force_scan_on_random_frontiers(self) -> None:
        import random

        environment = self._environment()
        bank = environment._bank_patterns
        rng = random.Random(20260920)
        for _trial in range(30):
            pair = (0, rng.randrange(3))
            parents = ((1, rng.randrange(2)),)
            frontier = tuple(
                tuple(
                    sorted(
                        {
                            (parameter, bank[rng.randrange(len(bank))][parameter])
                            for parameter in rng.sample(range(2, 12), 3)
                        }
                    )
                )
                for _ in range(12)
            )
            chosen = environment.select_experiment(
                pair=pair,
                parent_components=parents,
                frontier_components=frontier,
                seed=3,
                round_number=2,
                agent_index=1,
                slot=0,
            )
            self.assertIn(chosen, _brute_force(environment, pair, parents, frontier))

    def test_the_choice_is_seeded_and_reproducible(self) -> None:
        environment = self._environment()
        arguments = dict(
            pair=(0, 0),
            parent_components=(),
            frontier_components=(),
            seed=11,
            round_number=1,
            agent_index=2,
            slot=0,
        )
        first = environment.select_experiment(**arguments)
        second = environment.select_experiment(**arguments)
        self.assertEqual(first, second)
        moved = environment.select_experiment(**{**arguments, "agent_index": 3})
        self.assertIsNotNone(moved)

    def test_uniform_selection_returns_no_override(self) -> None:
        environment = self._environment(selection="uniform")
        self.assertIsNone(
            environment.select_experiment(
                pair=(0, 0),
                parent_components=(),
                frontier_components=(),
                seed=1,
                round_number=1,
                agent_index=0,
                slot=0,
            )
        )


def _brute_force(environment, pair, parents, frontier) -> set[str]:
    """The same argmax, written the slow and obvious way."""
    best = -1
    winners: set[str] = set()
    for condition, base in zip(environment._bank, environment._bank_patterns):
        configuration = environment._configuration(base, tuple(parents) + (pair,))
        count = sum(
            all(configuration[family] == variant for family, variant in components)
            for components in frontier
        )
        if count > best:
            best = count
            winners = {condition.experiment_id}
        elif count == best:
            winners.add(condition.experiment_id)
    return winners


class Episode(unittest.TestCase):
    def test_max_containment_changes_which_configurations_are_run(self) -> None:
        overrides = {"environment": {"tcas_selection": "max_containment"}}
        runner = build(overrides, degree=4, seed=92300)
        produced = asyncio.run(runner.run())
        overridden = sum(
            log["protocol_selected_experiments"] for log in produced["transitions"]
        )
        self.assertGreater(overridden, 0)
        self.assertEqual(produced["tcas_selection"], "max_containment")

    def test_max_containment_prunes_at_least_as_hard(self) -> None:
        uniform = 0
        argmax = 0
        for seed in (92300, 92301, 92302, 92303):
            uniform += sum(
                log["recv_pruned_hypotheses"] + log["own_pruned_hypotheses"]
                for log in asyncio.run(build({}, degree=6, seed=seed).run())["transitions"]
            )
            argmax += sum(
                log["recv_pruned_hypotheses"] + log["own_pruned_hypotheses"]
                for log in asyncio.run(
                    build(
                        {"environment": {"tcas_selection": "max_containment"}},
                        degree=6,
                        seed=seed,
                    ).run()
                )["transitions"]
            )
        self.assertGreaterEqual(argmax, uniform)


if __name__ == "__main__":
    unittest.main()
