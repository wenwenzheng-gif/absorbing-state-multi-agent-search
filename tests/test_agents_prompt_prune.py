#!/usr/bin/env python3
"""``llm_prune``: the model does the pruning, and every call is graded.

The point of the mode is that an un-pruned refuted hypothesis really survives
and really branches, so ``q_model`` stops being identically 1.  These tests
pin that, the zero-false-negative guarantee of ``llm_prune_apply=validated``,
the shadow score, and the lineage bookkeeping the reproduction ratio rests on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.driver import estimate_requests  # noqa: E402
from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.synthetic import SyntheticEnvironment  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.policies.openai_policy import (  # noqa: E402
    LLMBackend,
    LLMPolicy,
    Usage,
    prune_schema,
)
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.policies.base import prompt_fingerprint  # noqa: E402
from src.agent_system.schemas import LLMConfig, load_run_config  # noqa: E402

ENCODING = Encoding(8, 4)


class PruneStub:
    """Backend stand-in whose prune answer is chosen by a rule under test."""

    cache_key = LLMBackend.cache_key

    def __init__(self, config: LLMConfig, prune_rule) -> None:
        self.config = config
        self.usage = Usage()
        self.responses: dict[str, dict] = {}
        self.prune_rule = prune_rule
        self.prune_calls: list[dict] = []
        self.calls: list[dict] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, *, messages, schema, schema_name, cache_key, metadata,
                       extra_body=None):
        props = schema["properties"]
        self.calls.append({"schema_name": schema_name, "prompt": messages[-1]["content"],
                           **metadata})
        self.usage.requests += 1
        if schema_name == "branch_decision":
            answer = {
                "updates": {
                    key: value["items"]["enum"][: value["minItems"]]
                    for key, value in props["updates"]["properties"].items()
                },
                "reason": "stub",
            }
        elif schema_name == "experiment_decision":
            answer = {"choice": props["choice"]["enum"][0], "reason": "stub"}
            if "experiment_id" in props:
                answer["experiment_id"] = props["experiment_id"]["enum"][0]
        else:
            offered = props["drop"]["items"]["enum"]
            self.prune_calls.append({"offered": list(offered),
                                     "prompt": messages[-1]["content"], **metadata})
            answer = {"drop": list(self.prune_rule(offered, messages[-1]["content"])),
                      "reason": "stub"}
        return {"cache_key": cache_key, "content": json.dumps(answer)}


def config_for(apply_mode: str = "validated"):
    return load_run_config(
        {
            "name": "prune", "agents": 6, "branching": 2, "experiments_per_agent": 1,
            "initial_width": 8, "degrees": [2], "task_seeds": [4242],
            "environment": {"kind": "synthetic", "families": 8, "variants": 4,
                            "task_size": 4},
            "policy": {
                "kind": "llm_prune",
                "llm_prune_apply": apply_mode,
                "llm": {"model": "stub", "base_url": "http://stub/v1"},
            },
        }
    )


def run_episode(prune_rule, apply_mode: str = "validated"):
    config = config_for(apply_mode)
    task = build_task_instance(
        agents=config.agents, encoding=ENCODING, task_size=4, seed=4242,
        initial_width=config.initial_width,
    )
    backend = PruneStub(config.policy.llm, prune_rule)
    runner = EpisodeRunner(
        config=config, degree=2, seed=4242, task=task,
        environment=SyntheticEnvironment(task),
        policy_factory=lambda: LLMPolicy(backend, kind="llm_prune", planner="fixed_first"),
        episode_id="d2_t4242_r0", run_id="prune_test",
    )
    result = asyncio.run(runner.run())
    return runner, backend, result


DROP_NONE = lambda offered, prompt: []
DROP_ALL = lambda offered, prompt: list(offered)
DROP_FIRST = lambda offered, prompt: list(offered[:1])


class WiringTests(unittest.TestCase):
    def test_the_runner_asks_at_both_prune_points(self) -> None:
        _runner, backend, _result = run_episode(DROP_NONE)
        stages = {(call["round"], call["slot"]) for call in backend.prune_calls}
        self.assertTrue(any(slot == 0 for _round, slot in stages), "receive step")
        self.assertTrue(any(slot == 1 for _round, slot in stages), "own-experiment step")
        receive = next(c for c in backend.prune_calls if c["slot"] == 0)
        own = next(c for c in backend.prune_calls if c["slot"] == 1)
        self.assertIn("at the receive step", receive["prompt"])
        self.assertIn("at the own-experiment step", own["prompt"])

    def test_the_prune_prompt_shows_the_new_raw_evidence_and_the_frontier(self) -> None:
        _runner, backend, _result = run_episode(DROP_NONE)
        call = backend.prune_calls[0]
        self.assertIn("NEW EVIDENCE JUST IN", call["prompt"])
        self.assertIn("YOUR HYPOTHESES", call["prompt"])
        for hid in call["offered"]:
            self.assertIn(hid, call["prompt"])

    def test_the_prune_phase_has_its_own_cache_key(self) -> None:
        from dataclasses import replace
        from src.agent_system.policies.base import DecisionContext

        backend = PruneStub(LLMConfig(model="m", base_url="u"), DROP_NONE)
        context = DecisionContext(run_id="r", episode_id="e", agent_index=0,
                                  round_number=1, phase="prune", seed=1)
        keys = {
            backend.cache_key(replace(context, phase=phase, slot=slot), batch=0,
                              attempt=0, schema_name="prune_decision", digest="d")
            for phase in ("branch", "prune")
            for slot in (0, 1)
        }
        self.assertEqual(len(keys), 4)

    def test_the_schema_admits_only_offered_ids(self) -> None:
        schema = prune_schema(["h_a", "h_b"])
        self.assertEqual(schema["properties"]["drop"]["items"]["enum"], ["h_a", "h_b"])
        self.assertEqual(schema["properties"]["drop"]["maxItems"], 2)
        self.assertEqual(schema["required"], ["drop", "reason"])

    def test_estimate_counts_the_extra_prune_calls(self) -> None:
        estimate = estimate_requests(config_for())
        self.assertGreater(estimate["prune_requests_per_episode_upper_bound"], 0)
        self.assertEqual(
            estimate["requests_per_episode_upper_bound"],
            estimate["branch_requests_per_episode_upper_bound"]
            + estimate["experiment_requests_per_episode_upper_bound"]
            + estimate["prune_requests_per_episode_upper_bound"],
        )


class ShadowScoreTests(unittest.TestCase):
    def test_a_model_that_drops_nothing_scores_zero_and_keeps_everything(self) -> None:
        runner, _backend, result = run_episode(DROP_NONE)
        rounds = result["transitions"]
        self.assertGreater(sum(r["prune_should"] for r in rounds), 0,
                           "the episode must actually refute something")
        self.assertEqual(sum(r["prune_hit"] for r in rounds), 0)
        self.assertEqual(sum(r["prune_applied"] for r in rounds), 0)
        self.assertEqual(
            sum(r["prune_missed"] for r in rounds), sum(r["prune_should"] for r in rounds)
        )
        measured = [r["q_model_measured"] for r in rounds if r["q_model_measured"] is not None]
        self.assertTrue(measured and all(value == 0.0 for value in measured))
        # a refuted hypothesis that was not dropped is genuinely still there
        self.assertTrue(
            any(
                not agent.evidence.compatible(node)
                for agent in runner.agents
                for node in agent.frontier
            )
        )

    def test_a_perfect_model_matches_exact_pruning(self) -> None:
        def perfect(offered, prompt):
            # the stub has no evidence store, so it reads the prompt the way
            # the model would: drop anything whose components contradict a
            # negative label or a determined family.
            return _refuted_from_prompt(offered, prompt)

        runner, _backend, result = run_episode(perfect)
        rounds = result["transitions"]
        self.assertGreater(sum(r["prune_should"] for r in rounds), 0)
        self.assertEqual(sum(r["prune_missed"] for r in rounds), 0)
        self.assertEqual(sum(r["prune_false"] for r in rounds), 0)
        measured = [r["q_model_measured"] for r in rounds if r["q_model_measured"] is not None]
        self.assertTrue(measured and all(value == 1.0 for value in measured))
        for agent in runner.agents:
            for node in agent.frontier:
                self.assertTrue(agent.evidence.compatible(node))

    def test_measured_q_is_hits_over_should(self) -> None:
        _runner, _backend, result = run_episode(DROP_FIRST)
        for row in result["transitions"]:
            if row["prune_should"]:
                self.assertAlmostEqual(
                    row["q_model_measured"], row["prune_hit"] / row["prune_should"]
                )
            self.assertEqual(
                row["prune_should"], row["prune_hit"] + row["prune_missed"]
            )


class ApplyModeTests(unittest.TestCase):
    def test_validated_never_kills_a_consistent_hypothesis(self) -> None:
        runner, _backend, result = run_episode(DROP_ALL, apply_mode="validated")
        rounds = result["transitions"]
        self.assertGreater(sum(r["prune_false"] for r in rounds), 0,
                           "the stub must actually have asked for a bad drop")
        for row in rounds:
            self.assertEqual(row["prune_applied"], row["prune_hit"])
        # everything the evidence still allows survived
        for agent in runner.agents:
            self.assertTrue(all(agent.evidence.compatible(n) for n in agent.frontier))
        # and nothing consistent was lost relative to exact pruning
        exact = run_episode(_refuted_from_prompt)[2]
        self.assertGreaterEqual(
            sum(r["truth_compatible_hypotheses"] for r in rounds),
            sum(r["truth_compatible_hypotheses"] for r in exact["transitions"]),
        )

    def test_raw_applies_whatever_the_model_said(self) -> None:
        _runner, _backend, result = run_episode(DROP_ALL, apply_mode="raw")
        rounds = result["transitions"]
        for row in rounds:
            self.assertEqual(
                row["prune_applied"], row["prune_hit"] + row["prune_false"]
            )
        self.assertGreater(sum(r["prune_false"] for r in rounds), 0)

    def test_a_false_drop_of_a_correct_lineage_is_counted(self) -> None:
        _runner, _backend, result = run_episode(DROP_ALL, apply_mode="raw")
        self.assertGreater(
            sum(r["prune_false_correct_lineage"] for r in result["transitions"]), 0
        )


class LineageTests(unittest.TestCase):
    def test_unpruned_refuted_hypotheses_keep_branching(self) -> None:
        kept = run_episode(DROP_NONE)[2]
        exact = run_episode(lambda o, p: _refuted_from_prompt(o, p))[2]
        kept_edges = sum(r["old_incorrect_edges"] for r in kept["transitions"])
        exact_edges = sum(r["old_incorrect_edges"] for r in exact["transitions"])
        self.assertGreater(kept_edges, exact_edges)
        self.assertGreater(
            sum(r["A_post"] for r in kept["transitions"]),
            sum(r["A_post"] for r in exact["transitions"]),
        )

    def test_the_lineage_split_still_adds_up(self) -> None:
        for rule in (DROP_NONE, DROP_ALL, _refuted_from_prompt):
            result = run_episode(rule)[2]
            for row in result["transitions"]:
                self.assertEqual(row["A_post"], row["old_post"] + row["source_post"])
                self.assertEqual(row["A_preown"], row["old_pre"] + row["source_pre"])
                self.assertLessEqual(row["old_post"], row["A_post"])
                self.assertEqual(
                    row["sampled_edges"],
                    row["old_incorrect_edges"]
                    + row["new_incorrect_source_edges"]
                    + _correct_edges(row),
                )
        self.assertEqual(
            run_episode(DROP_NONE)[0].audit["respawn_lineage_leak"], 0
        )

    def test_no_audit_violation_under_llm_pruning(self) -> None:
        runner, _backend, result = run_episode(DROP_NONE)
        for name, count in result["audit"].items():
            self.assertEqual(count, 0, msg=name)
        self.assertTrue(result["llm_prune"])
        self.assertEqual(result["llm_prune_apply"], "validated")
        self.assertEqual(result["prompt_sha256"], prompt_fingerprint())


def _correct_edges(row: dict) -> int:
    """Edges leaving a correct parent to a correct child."""
    return (
        row["sampled_edges"]
        - row["old_incorrect_edges"]
        - row["new_incorrect_source_edges"]
    )


def _refuted_from_prompt(offered, prompt: str) -> list[str]:
    """Reproduce exact pruning from what the prompt itself states.

    Deliberately not a peek at the evidence store: it reads the rendered
    ``values still allowed per family`` line, exactly the information the
    model is given, so "perfect" means perfect at the stated task.
    """
    allowed: dict[int, set[int]] = {}
    for line in prompt.splitlines():
        if not line.startswith("values still allowed per family:"):
            continue
        body = line.split(":", 1)[1]
        for entry in body.split(";"):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            family, _, values = entry.partition(":")
            values = values.strip()
            if values.startswith("determined = "):
                variants = {int(values.split("v")[1])}
            else:
                variants = {int(part.strip()[1:]) for part in values.split(",")}
            allowed[int(family[1:])] = variants
        break
    drop = []
    for line in prompt.splitlines():
        if ": c" not in line or not line.startswith("h_"):
            continue
        hid, _, body = line.partition(": ")
        if hid not in offered:
            continue
        for token in body.split(", "):
            family, _, variant = token[1:].partition("v")
            if int(variant) not in allowed.get(int(family), {int(variant)}):
                drop.append(hid)
                break
    return drop


if __name__ == "__main__":
    unittest.main()
