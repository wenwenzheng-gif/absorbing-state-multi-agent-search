#!/usr/bin/env python3
"""E0: the new uniform/synthetic runner must match the frozen simulator.

The comparison is per round and per scientific field -- branch counts, the
budget allocation, evidence effects and the old/source lineage split -- not
just the final success flag.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.synthetic import SyntheticEnvironment  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.policies.adaptive import AdaptivePolicy  # noqa: E402
from src.agent_system.policies.uniform import UniformPolicy  # noqa: E402
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.schemas import load_run_config  # noqa: E402
from src.scaling_model import run_episode  # noqa: E402

PARAMETERS = {"N": 12, "M": 8, "K": 4, "task_size": 4, "rounds": 3, "b": 2, "m": 1, "n0": 8, "w": 0}

SCIENTIFIC_FIELDS = (
    "old_pre",
    "source_pre",
    "A_preown",
    "old_post",
    "source_post",
    "A_post",
    "tests",
    "tests_pos",
    "tests_neg",
    "B_used",
    "recv_messages",
    "recv_unique",
    "recv_duplicate",
    "recv_informative",
    "recv_already_explicit",
    "recv_already_implied",
    "recv_pruned_hypotheses",
    "own_pruned_hypotheses",
    "parents_attempted",
    "sampled_edges",
    "unique_children",
    "merge_loss",
    "dual_origin_unique_children",
    "incorrect_parents_after_receive",
    "old_incorrect_edges",
    "new_incorrect_source_edges",
    "total_proposed_pairs",
    "total_candidate_capacity",
    "final_candidate_capacity",
    "candidate_invalidated_without_test",
    "experiments_per_agent",
    "allocation_passes",
    "N_experiment_active_start",
    "N_experiment_active_end",
    "N_frontier_active",
    "N_frontier_active_end",
    "truth_compatible_hypotheses",
    "truth_compatible_agents",
    "mean_frontier_size",
    "mean_frontier_after_receive",
    "effective_branching",
    "unique_population_experiments",
    "outgoing_agents",
    "budget_redistributed_above_m",
    "agents_exceeding_nominal_m",
)


def build_runner(degree: int, seed: int, policy_factory):
    config = load_run_config(
        {
            "name": "parity",
            "agents": 12,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 8,
            "degrees": [degree],
            "environment": {"kind": "synthetic", "families": 8, "variants": 4, "task_size": 4},
            "policy": {"kind": "uniform"},
        }
    )
    encoding = Encoding(8, 4)
    task = build_task_instance(
        agents=12, encoding=encoding, task_size=4, seed=seed, initial_width=8
    )
    return EpisodeRunner(
        config=config,
        degree=degree,
        seed=seed,
        task=task,
        environment=SyntheticEnvironment(task),
        policy_factory=policy_factory,
        episode_id=f"d{degree}_s{seed}",
    )


class SyntheticParityTests(unittest.TestCase):
    def test_matches_reference_trajectory(self) -> None:
        for degree in (0, 2, 5, 11):
            for seed in (12345, 777, 20260918):
                with self.subTest(degree=degree, seed=seed):
                    reference = run_episode(PARAMETERS, degree=degree, seed=seed)
                    produced = asyncio.run(
                        build_runner(degree, seed, lambda: UniformPolicy()).run()
                    )
                    self.assertEqual(produced["truth"], list(reference["truth"]))
                    self.assertEqual(produced["graph_digest"], reference["graph_digest"])
                    self.assertEqual(
                        produced["initialization_digest"], reference["initialization_digest"]
                    )
                    self.assertEqual(
                        produced["A_initial_incorrect"], reference["A_initial_incorrect"]
                    )
                    self.assertEqual(
                        produced["agents_reached_truth"], reference["agents_reached_truth"]
                    )
                    for index in range(PARAMETERS["rounds"]):
                        left = reference["transitions"][index]
                        right = produced["transitions"][index]
                        for field in SCIENTIFIC_FIELDS:
                            self.assertEqual(
                                left[field], right[field], msg=f"round {index + 1} field {field}"
                            )

    def test_beta_zero_adaptive_equals_uniform(self) -> None:
        """beta=0 must share the uniform random stream, not merely its law."""
        uniform = asyncio.run(build_runner(2, 12345, lambda: UniformPolicy()).run())
        adaptive = asyncio.run(build_runner(2, 12345, lambda: AdaptivePolicy(beta=0.0)).run())
        for index, (left, right) in enumerate(
            zip(uniform["transitions"], adaptive["transitions"])
        ):
            for field in SCIENTIFIC_FIELDS:
                self.assertEqual(left[field], right[field], msg=f"round {index + 1} {field}")

    def test_no_audit_violations(self) -> None:
        produced = asyncio.run(build_runner(5, 12345, lambda: UniformPolicy()).run())
        self.assertEqual({k: v for k, v in produced["audit"].items() if v}, {})


if __name__ == "__main__":
    unittest.main()
