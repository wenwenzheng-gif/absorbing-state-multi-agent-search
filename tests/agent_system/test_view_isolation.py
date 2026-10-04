#!/usr/bin/env python3
"""Non-leakage: what a policy sees, and what it can never see."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.physics import PhysicsEnvironment  # noqa: E402
from src.agent_system.environments.synthetic import SyntheticEnvironment  # noqa: E402
from src.agent_system.evidence import EvidenceStore  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.policies.adaptive import AdaptivePolicy  # noqa: E402
from src.agent_system.policies.uniform import UniformPolicy  # noqa: E402
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.schemas import (  # noqa: E402
    AgentView,
    EnvironmentConfig,
    EvidenceRecord,
    load_run_config,
)
from src.agent_system.views import VIEW_FIELDS  # noqa: E402

ENCODING = Encoding(8, 4)
FORBIDDEN_KEYS = ("truth", "seed", "wrong", "from_wrong", "hidden", "holdout", "origin_agent")


def synthetic_runner(seed: int) -> EpisodeRunner:
    config = load_run_config(
        {
            "name": "view",
            "agents": 4,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 4,
            "degrees": [0],
            "environment": {"kind": "synthetic", "families": 8, "variants": 4, "task_size": 4},
            "policy": {"kind": "uniform"},
        }
    )
    task = build_task_instance(
        agents=4, encoding=ENCODING, task_size=4, seed=seed, initial_width=4
    )
    runner = EpisodeRunner(
        config=config,
        degree=0,
        seed=seed,
        task=task,
        environment=SyntheticEnvironment(task),
        policy_factory=lambda: UniformPolicy(),
        episode_id=f"view{seed}",
    )
    runner._build_agents()
    return runner


def force_public_state(runner: EpisodeRunner) -> None:
    """Give agent 0 a hidden-world-independent frontier and evidence."""
    agent = runner.agents[0]
    agent.evidence = EvidenceStore(ENCODING)
    agent.evidence.apply(
        EvidenceRecord(
            event_id="x",
            pair=(4, 1),
            outcome="negative",
            origin_agent=0,
            created_round=1,
            experiment_id=None,
        ),
        own=True,
    )
    agent.frontier = [
        ENCODING.node_from_pair((0, 0), wrong=False),
        ENCODING.node_from_pair((1, 2), wrong=True),
    ]
    agent.observations = []


class ViewWhitelistTests(unittest.TestCase):
    def test_view_fields_are_exactly_the_whitelist(self) -> None:
        self.assertEqual(tuple(AgentView.__dataclass_fields__), VIEW_FIELDS)

    def test_view_carries_no_hidden_key(self) -> None:
        runner = synthetic_runner(12345)
        view = runner._view(
            runner.agents[0], "branch", 1, parents=runner._parent_slots(runner.agents[0])
        )
        text = json.dumps(view.public_dict())
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(f'"{key}"', text, msg=f"{key} must not reach a policy")

    def test_two_hidden_worlds_produce_the_same_view(self) -> None:
        """Identical public state must give an identical prompt input."""
        first, second = synthetic_runner(12345), synthetic_runner(20260918)
        self.assertNotEqual(first.task.truth_pairs, second.task.truth_pairs)
        digests = []
        for runner in (first, second):
            force_public_state(runner)
            agent = runner.agents[0]
            view = runner._view(agent, "branch", 1, parents=runner._parent_slots(agent))
            digests.append(view.digest_payload())
        self.assertEqual(digests[0], digests[1])


class PredictorIsolationTests(unittest.TestCase):
    """Rule policies may simulate their own hypotheses, never the hold-out set."""

    def test_planner_and_scoring_never_touch_evaluator_data(self) -> None:
        agents = 4
        config = EnvironmentConfig(kind="physics", task_size=4, initial_observations=2)
        task = build_task_instance(
            agents=agents, encoding=ENCODING, task_size=4, seed=777, initial_width=4
        )
        environment = PhysicsEnvironment(task, config, agents)
        touched: list[str] = []

        def watched(components, condition):
            touched.append(condition.experiment_id)
            return environment.predict(components, condition)

        run_config = load_run_config(
            {
                "name": "isolation",
                "agents": agents,
                "branching": 2,
                "experiments_per_agent": 1,
                "initial_width": 4,
                "degrees": [0],
                "environment": {"kind": "physics", "families": 8, "variants": 4, "task_size": 4},
                "policy": {"kind": "adaptive", "beta": 1.0, "planner_max_conditions": 4},
            }
        )

        def factory():
            policy = AdaptivePolicy(beta=1.0, planner_max_conditions=4)
            policy.bind_predictor(watched)
            return policy

        runner = EpisodeRunner(
            config=run_config,
            degree=0,
            seed=777,
            task=task,
            environment=environment,
            policy_factory=factory,
            episode_id="isolation",
        )
        # bind_predictor in the runner would overwrite the watcher, so rebind.
        for policy in runner.policies:
            policy.bind_predictor(watched)
        asyncio.run(runner.run())
        self.assertTrue(touched, "the adaptive policy must actually simulate hypotheses")
        for experiment_id in touched:
            self.assertFalse(
                experiment_id.startswith(("holdout_", "diag_")),
                msg=f"policy read evaluator-only condition {experiment_id}",
            )

    def test_observations_stay_with_their_owner(self) -> None:
        agents = 3
        config = EnvironmentConfig(kind="physics", task_size=4, initial_observations=2)
        task = build_task_instance(
            agents=agents, encoding=ENCODING, task_size=4, seed=31337, initial_width=4
        )
        environment = PhysicsEnvironment(task, config, agents)
        run_config = load_run_config(
            {
                "name": "own",
                "agents": agents,
                "branching": 2,
                "experiments_per_agent": 1,
                "initial_width": 4,
                "degrees": [0],
                "environment": {"kind": "physics", "families": 8, "variants": 4, "task_size": 4},
                "policy": {"kind": "uniform", "planner": "fixed_first"},
            }
        )
        runner = EpisodeRunner(
            config=run_config,
            degree=0,
            seed=31337,
            task=task,
            environment=environment,
            policy_factory=lambda: UniformPolicy(planner="fixed_first"),
            episode_id="own",
        )
        asyncio.run(runner.run())
        owners = [
            {obs.observation_id for obs in agent.observations} for agent in runner.agents
        ]
        for left in range(agents):
            for right in range(left + 1, agents):
                self.assertFalse(
                    owners[left] & owners[right],
                    "no observation may appear in two agents' private sets",
                )


if __name__ == "__main__":
    unittest.main()
