#!/usr/bin/env python3
"""Every WP1 knob at its default must reproduce agent_v1 exactly.

The frozen-parity test in ``tests/agent_system/test_parity.py`` is the
authority on what "exactly" means; this file checks the two things that test
cannot see -- that naming the defaults explicitly in a config changes nothing,
and that no default path touches the new random streams.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.synthetic import SyntheticEnvironment  # noqa: E402
from src.agent_system.evaluation import HiddenEvaluator  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.policies.uniform import UniformPolicy  # noqa: E402
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.schemas import load_run_config  # noqa: E402
from src.scaling_model import run_episode  # noqa: E402

PARAMETERS = {"N": 12, "M": 8, "K": 4, "task_size": 4, "rounds": 3, "b": 2, "m": 1, "n0": 8, "w": 0}

BASE = {
    "name": "knobs",
    "agents": 12,
    "branching": 2,
    "experiments_per_agent": 1,
    "initial_width": 8,
    "environment": {"kind": "synthetic", "families": 8, "variants": 4, "task_size": 4},
    "policy": {"kind": "uniform"},
}

EXPLICIT_DEFAULTS = {
    "q_model": 1.0,
    "q_model_own": None,
    "q_model_received": None,
    "q_model_draw": "per_label",
    "budget_mode": "redistribute",
    "stop_on_success": False,
    "evaluate_every_round": False,
    "respawn": "none",
    "topology": "random_regular",
    "topology_rewire_p": 0.0,
}


def build_runner(
    overrides: dict, *, degree: int, seed: int, agents: int = 12, evaluate: bool = False
) -> EpisodeRunner:
    """One synthetic episode with ``overrides`` merged into the base config."""
    payload = {**BASE, **overrides, "agents": agents, "degrees": [degree]}
    config = load_run_config(payload)
    encoding = Encoding(config.environment.families, config.environment.variants)
    task = build_task_instance(
        agents=agents,
        encoding=encoding,
        task_size=config.task_size,
        seed=seed,
        initial_width=config.initial_width,
    )
    environment = SyntheticEnvironment(task)
    evaluator = (
        HiddenEvaluator(
            task=task,
            environment=environment,
            encoding=encoding,
            epsilon=config.epsilon_predictive,
        )
        if evaluate
        else None
    )
    return EpisodeRunner(
        config=config,
        degree=degree,
        seed=seed,
        task=task,
        environment=environment,
        policy_factory=lambda: UniformPolicy(),
        episode_id=f"d{degree}_s{seed}",
        evaluator=evaluator,
    )


def run_with(
    overrides: dict, *, degree: int, seed: int, agents: int = 12, evaluate: bool = False
) -> dict:
    runner = build_runner(
        overrides, degree=degree, seed=seed, agents=agents, evaluate=evaluate
    )
    return asyncio.run(runner.run())


class DefaultsAreTheFrozenProtocol(unittest.TestCase):
    def test_explicit_defaults_match_the_frozen_reference(self) -> None:
        for degree in (0, 2, 5, 11):
            for seed in (12345, 777):
                with self.subTest(degree=degree, seed=seed):
                    reference = run_episode(PARAMETERS, degree=degree, seed=seed)
                    produced = run_with(EXPLICIT_DEFAULTS, degree=degree, seed=seed)
                    self.assertEqual(produced["graph_digest"], reference["graph_digest"])
                    for index in range(PARAMETERS["rounds"]):
                        left = reference["transitions"][index]
                        right = produced["transitions"][index]
                        for field in left:
                            if field in right:
                                self.assertEqual(
                                    left[field], right[field], msg=f"round {index+1} {field}"
                                )

    def test_defaults_record_no_knob_activity(self) -> None:
        produced = run_with({}, degree=5, seed=12345)
        for round_log in produced["transitions"]:
            self.assertEqual(round_log["recv_missed"], 0)
            self.assertEqual(round_log["own_missed"], 0)
            self.assertEqual(round_log["respawned_agents"], 0)
            self.assertEqual(round_log["respawned_hypotheses"], 0)
            self.assertEqual(round_log["protocol_selected_experiments"], 0)
        self.assertEqual(produced["audit"]["empty_frontier_replenished"], 0)
        self.assertEqual(produced["audit"]["respawn_lineage_leak"], 0)
        self.assertEqual(produced["rounds_executed"], 3)
        self.assertIsNone(produced["stopped_round"])
        self.assertEqual(produced["realised_mean_degree"], 5.0)

    def test_old_configs_still_validate_and_carry_defaults(self) -> None:
        """A config that never names a knob must still get the old behaviour.

        Configs that do name one are setting it on purpose, so only the knobs
        absent from the raw JSON are held to their defaults.
        """
        directory = ROOT / "configs" / "agents"
        seen = 0
        for path in sorted(directory.glob("*.json")):
            seen += 1
            raw = json.loads(path.read_text(encoding="utf-8"))
            environment = raw.get("environment") or {}
            config = load_run_config(raw)

            def unset(*keys: str, where: dict = raw) -> bool:
                return all(key not in where for key in keys)

            if unset("q_model", "q_model_own", "q_model_received"):
                self.assertEqual(config.q_model, 1.0, msg=path.name)
            if unset("budget_mode"):
                self.assertEqual(config.budget_mode, "redistribute", msg=path.name)
            if unset("respawn"):
                self.assertEqual(config.respawn, "none", msg=path.name)
            if unset("topology", "topology_rewire_p"):
                self.assertEqual(config.topology, "random_regular", msg=path.name)
            if unset("stop_on_success"):
                self.assertFalse(config.stop_on_success, msg=path.name)
            if unset("evaluate_every_round"):
                self.assertFalse(config.evaluate_every_round, msg=path.name)
            if unset("tcas_selection", where=environment):
                self.assertEqual(
                    config.environment.tcas_selection, "uniform", msg=path.name
                )
        self.assertGreater(seen, 0)

    def test_rejects_out_of_range_settings(self) -> None:
        for bad in (
            {"q_model": 0.0},
            {"q_model": 1.5},
            {"q_model_received": -0.1},
            {"budget_mode": "greedy"},
            {"respawn": "always"},
            {"topology": "star"},
            {"topology_rewire_p": 0.5},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_run_config({**BASE, **bad, "degrees": [2]})


if __name__ == "__main__":
    unittest.main()
