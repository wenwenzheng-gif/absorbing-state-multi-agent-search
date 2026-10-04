#!/usr/bin/env python3
"""Prompt switches: each one is separable, and the defaults are the final setting.

The tests build their policies from ``BARE`` (every switch off) so each
switch-isolation test measures exactly one block; ``DefaultsTests`` checks
that a config which names no switch gets the final experiment setting.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.policies.base import DecisionContext  # noqa: E402
from src.agent_system.policies.openai_policy import (  # noqa: E402
    LLMBackend,
    LLMPolicy,
    Usage,
    branch_schema,
    experiment_schema,
    prune_schema,
    prompt_key_extras,
    request_seed,
    sampling_payload,
    summarise_observation,
)
from src.agent_system.schemas import (  # noqa: E402
    AgentView,
    ComponentSpec,
    EnvironmentConfig,
    ExperimentCandidate,
    ExperimentCondition,
    Hypothesis,
    LabelRecord,
    LLMConfig,
    ParentSlot,
    PrivateObservation,
    RoundMemory,
    load_run_config,
)

# Every optional prompt block switched off.
BARE = {
    "context_identity": False,
    "team_objective": "off",
    "evidence_attribution": False,
    "shuffle_candidates": False,
    "per_agent_seed": False,
}

CONTEXT = DecisionContext(
    run_id="r", episode_id="e", agent_index=3, round_number=2, phase="branch", seed=7
)
SLOTS = (
    ParentSlot(
        hypothesis=Hypothesis(((0, 0), (2, 1))),
        legal_additions=((1, 0), (1, 1), (3, 0), (3, 1), (4, 2)),
        required=2,
    ),
    ParentSlot(
        hypothesis=Hypothesis(((0, 1), (2, 0))),
        legal_additions=((1, 0), (3, 1), (4, 0)),
        required=2,
    ),
)
CANDIDATES = (
    ExperimentCandidate(pair=(1, 0), parent_ids=("h_c0v0_c2v1", "h_c0v1_c2v0")),
    ExperimentCandidate(pair=(3, 1), parent_ids=("h_c0v0_c2v1",)),
    ExperimentCandidate(pair=(4, 2), parent_ids=("h_c0v1_c2v0",)),
)
LIBRARY = tuple(
    ExperimentCondition(experiment_id=f"e{i}", description=f"probe {i}", settings={"x0": i})
    for i in range(4)
)
LONG_OBSERVATION = PrivateObservation(
    observation_id="obs_long",
    kind="experiment",
    experiment_id="e1",
    settings={"x0": 1.0, "v0": -0.5, "t0": 0.0},
    times=tuple(round(0.0625 * i, 6) for i in range(33)),
    states=tuple(
        (round(1.234567 * math.sin(0.37 * i), 6), round(-0.987654 * math.cos(0.29 * i), 6))
        for i in range(33)
    ),
    variables=("x", "v"),
)
OBSERVATION = PrivateObservation(
    observation_id="obs_1",
    kind="initial",
    experiment_id="e0",
    settings={"x0": 1.0, "v0": -0.5},
    times=(0.0, 0.5, 1.0, 1.5),
    states=((1.0, -0.5), (0.6, -0.9), (-0.1, -1.0), (-0.7, -0.4)),
    variables=("x", "v"),
)


def view(**overrides) -> AgentView:
    payload = dict(
        phase="branch",
        round_number=2,
        total_rounds=3,
        task_size=4,
        families=5,
        variants=3,
        branching=2,
        component_library=tuple(
            ComponentSpec(f, v, f"component_{f}_{v}", f"expr[{f}][{v}]")
            for f in range(5)
            for v in range(3)
        ),
        known_positive=((2, 1),),
        known_negative=((4, 0),),
        allowed_values={0: (0, 1, 2), 1: (0, 1, 2), 2: (1,), 3: (0, 1, 2), 4: (1, 2)},
        parents=SLOTS,
        candidates=CANDIDATES,
        experiment_library=LIBRARY,
        environment_notes="A test environment.",
        agent_index=3,
        agent_count=12,
        degree=4,
        neighbours=(1, 5, 7, 9),
        round_budget=1,
        own_labels=(
            LabelRecord(pair=(2, 1), outcome="positive", source="own", sender=None,
                        round_number=1),
        ),
        received_labels=(
            LabelRecord(pair=(4, 0), outcome="negative", source="received", sender=5,
                        round_number=1),
        ),
        own_history=(RoundMemory(round_number=1, tested=((2, 1),), outcomes=("positive",)),),
    )
    payload.update(overrides)
    return AgentView(**payload)


def make_policy(**llm) -> LLMPolicy:
    config = LLMConfig(model="stub", base_url="http://stub/v1", **{**BARE, **llm})
    return LLMPolicy(_Stub(config), kind="llm_full", planner="fixed_first")


class _Stub:
    """Backend stand-in: records every call, answers the first legal way."""

    cache_key = LLMBackend.cache_key

    def __init__(self, config: LLMConfig, answers=None) -> None:
        self.config = config
        self.usage = Usage()
        self.responses: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.answers = list(answers or [])

    async def aclose(self) -> None:
        return None

    async def complete(self, *, messages, schema, schema_name, cache_key, metadata,
                       extra_body=None):
        self.calls.append(
            {
                "prompt": messages[-1]["content"],
                "schema": schema,
                "schema_name": schema_name,
                "cache_key": cache_key,
                "extra_body": extra_body,
                **metadata,
            }
        )
        if self.answers:
            payload = self.answers.pop(0)
        else:
            payload = _first_legal(schema, schema_name)
        self.usage.requests += 1
        return {"cache_key": cache_key, "content": json.dumps(payload)}


def _first_legal(schema, schema_name) -> dict:
    props = schema["properties"]
    answer: dict = {}
    if "analysis" in props:
        answer["analysis"] = "a"
    if schema_name == "branch_decision":
        answer["updates"] = {
            key: value["items"]["enum"][: value["minItems"]]
            for key, value in props["updates"]["properties"].items()
        }
    elif schema_name == "experiment_decision":
        answer["choice"] = props["choice"]["enum"][0]
        if "experiment_id" in props:
            answer["experiment_id"] = props["experiment_id"]["enum"][0]
    else:
        answer["drop"] = []
    answer["reason"] = "r"
    return answer


def render(policy: LLMPolicy, phase: str, current=None) -> str:
    if phase == "branch":
        return policy._branch_prompt(current or view(), SLOTS)
    return policy._experiment_prompt(current or view(phase="experiment"), CANDIDATES)


class DefaultsTests(unittest.TestCase):
    def test_defaults_are_the_final_setting(self) -> None:
        config = LLMConfig(model="m", base_url="u")
        self.assertTrue(config.context_identity)
        self.assertEqual(config.team_objective, "device")
        self.assertTrue(config.evidence_attribution)
        self.assertTrue(config.shuffle_candidates)
        self.assertTrue(config.per_agent_seed)
        self.assertEqual(config.reasoning, "none")
        self.assertEqual(config.physics_observations, "summary")

    def test_the_cache_key_names_every_switch(self) -> None:
        final = LLMConfig(model="m", base_url="u")
        bare = LLMConfig(model="m", base_url="u", **BARE)
        self.assertEqual(
            sorted(prompt_key_extras(final)),
            sorted([*BARE, "reasoning", "physics_observations", "communication_mode"]),
        )
        self.assertNotEqual(
            sampling_payload(final, CONTEXT, 0, 0), sampling_payload(bare, CONTEXT, 0, 0)
        )
        self.assertNotEqual(
            _Stub(final).cache_key(
                CONTEXT, batch=0, attempt=0, schema_name="branch_decision", digest="abc"
            ),
            _Stub(bare).cache_key(
                CONTEXT, batch=0, attempt=0, schema_name="branch_decision", digest="abc"
            ),
        )

    def test_plain_schemas_have_no_analysis_field(self) -> None:
        self.assertEqual(branch_schema(SLOTS)["required"], ["updates", "reason"])
        self.assertEqual(
            experiment_schema(CANDIDATES, LIBRARY, True)["required"],
            ["choice", "experiment_id", "reason"],
        )

    def test_bare_switches_add_no_optional_block(self) -> None:
        text = render(make_policy(), "branch")
        for marker in ("YOUR PLACE IN THE TEAM", "PROTOCOL", "TEAM OBJECTIVE",
                       "WHAT YOU YOURSELF DID", "SYMMETRY BREAKING", "analysis"):
            self.assertNotIn(marker, text, msg=marker)

    def test_default_switches_render_the_team_blocks(self) -> None:
        policy = LLMPolicy(
            _Stub(LLMConfig(model="stub", base_url="http://stub/v1")),
            kind="llm_full",
            planner="fixed_first",
        )
        text = render(policy, "branch")
        for marker in ("YOUR PLACE IN THE TEAM", "PROTOCOL", "TEAM OBJECTIVE",
                       "WHAT YOU YOURSELF DID"):
            self.assertIn(marker, text, msg=marker)


class SwitchIsolationTests(unittest.TestCase):
    def test_context_identity_adds_only_the_identity_blocks(self) -> None:
        off = render(make_policy(), "branch")
        on = render(make_policy(context_identity=True), "branch")
        added = set(on.splitlines()) - set(off.splitlines())
        self.assertTrue(any("You are agent 3 of 12" in line for line in added))
        self.assertTrue(any("direct neighbour(s): 1, 5, 7, 9" in line for line in added))
        self.assertTrue(any("START OF THE NEXT ROUND" in line for line in added))
        self.assertTrue(any("nominal budget is 1" in line for line in added))
        self.assertTrue(any("round(s) remain after this one" in line for line in added))
        # and it must not touch the action section
        self.assertEqual(off.split("WHAT TO RETURN")[1], on.split("WHAT TO RETURN")[1])

    def test_team_objective_soft_states_the_goal_without_the_device(self) -> None:
        soft = render(make_policy(team_objective="soft"), "branch")
        self.assertIn("TEAM OBJECTIVE", soft)
        self.assertIn("narrowed to a single value is determined", soft)
        self.assertNotIn("SYMMETRY BREAKING", soft)

    def test_team_objective_device_adds_executable_arithmetic_with_the_id(self) -> None:
        device = render(make_policy(team_objective="device"), "branch")
        self.assertIn("SYMMETRY BREAKING", device)
        self.assertIn("You are agent 3.", device)
        self.assertIn("(3 mod k)", device)

    def test_evidence_attribution_separates_own_from_received_and_adds_memory(self) -> None:
        off = render(make_policy(), "branch")
        on = render(make_policy(evidence_attribution=True), "branch")
        self.assertIn("labels you produced yourself: c2v1=positive (round 1)", on)
        self.assertIn("c4v0=negative (from agent 5, round 1)", on)
        self.assertIn("round 1: c2v1 -> positive", on)
        self.assertNotIn("from agent", off)

    def test_reasoning_reason_first_puts_analysis_before_the_action(self) -> None:
        policy = make_policy(reasoning="reason_first")
        schema = branch_schema(SLOTS, analysis_max_chars=policy._analysis_chars)
        self.assertEqual(schema["required"][0], "analysis")
        self.assertEqual(list(schema["properties"])[0], "analysis")
        self.assertEqual(schema["properties"]["analysis"]["maxLength"], 600)
        self.assertIn('Write "analysis" FIRST', render(policy, "branch"))

    def test_reasoning_thinking_turns_the_template_kwarg_on(self) -> None:
        config = LLMConfig(model="m", base_url="u", reasoning="thinking")
        self.assertIn("thinking", prompt_key_extras(config)["reasoning"])
        self.assertEqual(prompt_key_extras(config)["thinking_max_output_tokens"], 4096)

    def test_per_agent_seed_puts_a_seed_in_the_body_and_the_key(self) -> None:
        plain = make_policy()
        seeded = make_policy(per_agent_seed=True)
        self.assertEqual(plain._extra_body(CONTEXT, batch=0, attempt=0), {})
        body = seeded._extra_body(CONTEXT, batch=0, attempt=0)
        self.assertIn("seed", body)
        self.assertNotEqual(
            plain.backend.cache_key(CONTEXT, batch=0, attempt=0,
                                    schema_name="branch_decision", digest="d"),
            seeded.backend.cache_key(CONTEXT, batch=0, attempt=0,
                                     schema_name="branch_decision", digest="d"),
        )

    def test_request_seed_is_deterministic_and_separates_every_coordinate(self) -> None:
        base = request_seed(CONTEXT, batch=0, attempt=0)
        self.assertEqual(base, request_seed(CONTEXT, batch=0, attempt=0))
        seen = {base}
        from dataclasses import replace as _replace

        for field, value in (
            ("episode_id", "e2"), ("agent_index", 4), ("round_number", 3),
            ("phase", "experiment"), ("slot", 1),
        ):
            seen.add(request_seed(_replace(CONTEXT, **{field: value}), batch=0, attempt=0))
        seen.add(request_seed(CONTEXT, batch=1, attempt=0))
        seen.add(request_seed(CONTEXT, batch=0, attempt=1))
        self.assertEqual(len(seen), 8)


class ShuffleTests(unittest.TestCase):
    def _order(self, agent_index: int, round_number: int = 2) -> list[str]:
        from dataclasses import replace as _replace

        policy = make_policy(shuffle_candidates=True)
        context = _replace(CONTEXT, agent_index=agent_index, round_number=round_number)
        presented = policy._branch_order(SLOTS, context, 0)
        return [f"c{f}v{v}" for f, v in presented[0].legal_additions]

    def test_shuffling_is_a_permutation_of_the_same_legal_set(self) -> None:
        self.assertEqual(sorted(self._order(0)), sorted(self._order(1)))
        self.assertEqual(
            sorted(self._order(0)),
            sorted(f"c{f}v{v}" for f, v in SLOTS[0].legal_additions),
        )

    def test_shuffling_is_reproducible(self) -> None:
        self.assertEqual(self._order(3), self._order(3))

    def test_shuffling_differs_across_agents_and_rounds(self) -> None:
        orders = [self._order(index) for index in range(12)]
        self.assertGreater(len({tuple(order) for order in orders}), 6)
        # no position may be the same entry for every agent
        for position in range(len(orders[0])):
            self.assertGreater(len({order[position] for order in orders}), 1)
        self.assertNotEqual(self._order(3, 2), self._order(3, 3))

    def test_the_schema_enum_follows_the_shuffled_order(self) -> None:
        policy = make_policy(shuffle_candidates=True)
        presented = policy._branch_order(SLOTS, CONTEXT, 0)
        schema = branch_schema(presented)
        entry = schema["properties"]["updates"]["properties"][SLOTS[0].hypothesis.hid]
        self.assertEqual(
            entry["items"]["enum"],
            [f"c{f}v{v}" for f, v in presented[0].legal_additions],
        )
        self.assertNotEqual(
            entry["items"]["enum"],
            [f"c{f}v{v}" for f, v in SLOTS[0].legal_additions],
        )

    def test_candidate_and_library_order_are_shuffled_together(self) -> None:
        policy = make_policy(shuffle_candidates=True)
        current = view(phase="experiment")
        offered, library = policy._experiment_order(current, CANDIDATES, CONTEXT)
        self.assertEqual({c.pair for c in offered}, {c.pair for c in CANDIDATES})
        self.assertEqual(
            {c.experiment_id for c in library}, {c.experiment_id for c in LIBRARY}
        )
        schema = experiment_schema(offered, library, True)
        self.assertEqual(
            schema["properties"]["experiment_id"]["enum"],
            [item.experiment_id for item in library],
        )
        text = policy._experiment_prompt(current, offered, library)
        self.assertIn(json.dumps(offered[0].public_dict(), separators=(",", ":")), text)


class HygieneTests(unittest.TestCase):
    def test_no_empty_allowed_value_list_is_ever_rendered(self) -> None:
        current = view(allowed_values={0: (0, 1), 1: (), 2: (1,)})
        text = render(make_policy(), "branch")
        self.assertNotIn('"1":[]', text)
        text = make_policy()._labels_block(current)
        self.assertIn("c2: determined = v1", text)
        self.assertIn("no value remains for c1", text)
        self.assertNotIn("[]", text)

    def test_the_vacuous_observation_sentence_goes_when_there_is_nothing_to_see(self) -> None:
        text = render(make_policy(), "branch")
        self.assertNotIn("PRIVATE OBSERVATIONS", text)
        self.assertNotIn("make plausible", text)
        with_obs = render(
            make_policy(), "branch", view(private_observations=(OBSERVATION,))
        )
        self.assertIn("PRIVATE OBSERVATIONS", with_obs)
        self.assertIn("make plausible", with_obs)

    def test_observations_are_deduplicated_and_summarised(self) -> None:
        twice = (LONG_OBSERVATION, LONG_OBSERVATION)
        current = view(private_observations=twice)
        raw = make_policy(
            physics_observations="raw"
        )._observations_block(current, "branch")
        summary = make_policy()._observations_block(current, "branch")
        self.assertEqual(raw.count('"obs_long"'), 1)
        self.assertEqual(summary.count('"obs_long"'), 1)
        # a real 33-sample trajectory is an order of magnitude larger raw
        self.assertLess(len(summary), len(raw) / 2)
        self.assertLess(len(summary), 700)
        self.assertIn('"sign_changes"', summary)

    def test_summary_statistics_describe_the_trajectory(self) -> None:
        payload = summarise_observation(OBSERVATION)
        self.assertEqual(payload["samples"], 4)
        self.assertEqual(payload["t_range"], [0.0, 1.5])
        self.assertEqual(payload["x"]["first"], 1.0)
        self.assertEqual(payload["x"]["last"], -0.7)
        self.assertEqual(payload["x"]["sign_changes"], 1)
        self.assertEqual(payload["v"]["sign_changes"], 0)

    def test_everything_that_changes_per_round_sits_behind_the_marker(self) -> None:
        policy = make_policy(
            context_identity=True, team_objective="device",
            evidence_attribution=True,
        )
        marker = "--- everything below changes every round ---"
        first = policy._branch_prompt(
            view(round_number=1, private_observations=(OBSERVATION,)), SLOTS
        )
        later = policy._branch_prompt(
            view(round_number=3, private_observations=(OBSERVATION, OBSERVATION)), SLOTS
        )
        self.assertEqual(first.split(marker)[0], later.split(marker)[0])
        self.assertNotEqual(first.split(marker)[1], later.split(marker)[1])


class LeakTests(unittest.TestCase):
    """A prompt may never carry a label the agent does not hold."""

    def test_no_label_beyond_the_agents_own_evidence_store(self) -> None:
        from src.agent_system.environments.base import build_task_instance
        from src.agent_system.environments.synthetic import SyntheticEnvironment
        from src.agent_system.hypotheses import Encoding
        from src.agent_system.runner import EpisodeRunner

        encoding = Encoding(8, 4)
        config = load_run_config(
            {
                "name": "leak", "agents": 4, "branching": 2, "experiments_per_agent": 1,
                "initial_width": 4, "degrees": [2],
                "environment": {"kind": "synthetic", "families": 8, "variants": 4,
                                "task_size": 4},
                "policy": {"kind": "uniform"},
            }
        )
        task = build_task_instance(
            agents=4, encoding=encoding, task_size=4, seed=99, initial_width=4
        )
        runner = EpisodeRunner(
            config=config, degree=2, seed=99, task=task,
            environment=SyntheticEnvironment(task),
            policy_factory=lambda: __import__(
                "src.agent_system.policies.uniform", fromlist=["UniformPolicy"]
            ).UniformPolicy(),
            episode_id="leak",
        )
        runner._build_agents()
        agent = runner.agents[0]
        agent.evidence.apply(
            __import__("src.agent_system.schemas", fromlist=["EvidenceRecord"]).EvidenceRecord(
                event_id="x", pair=task.truth_pairs[0], outcome="positive",
                origin_agent=0, created_round=1, experiment_id=None,
            ),
            own=True,
        )
        current = runner._view(agent, "branch", 2, parents=runner._parent_slots(agent))
        policy = make_policy(
            context_identity=True, team_objective="device",
            evidence_attribution=True,
        )
        text = policy._branch_prompt(current, current.parents[:2])
        held = set(agent.evidence.records)
        for pair in task.truth_pairs:
            if pair in held:
                continue
            label = f"c{pair[0]}v{pair[1]}=positive"
            self.assertNotIn(label, text, msg=f"leaked the label of {pair}")
        # the hidden truth never appears as a set, either
        self.assertNotIn(
            ",".join(f"c{f}v{v}" for f, v in sorted(task.truth_pairs)), text
        )
        for forbidden in ('"truth"', "from_wrong", "holdout"):
            self.assertNotIn(forbidden, text)


class ConfigTests(unittest.TestCase):
    def test_every_switch_round_trips_through_the_run_config(self) -> None:
        payload = {
            "name": "cfg", "agents": 4, "branching": 2, "experiments_per_agent": 1,
            "initial_width": 4, "degrees": [0],
            "environment": {"kind": "synthetic", "families": 8, "variants": 4,
                            "task_size": 4},
            "policy": {
                "kind": "llm_prune",
                "llm_prune_apply": "raw",
                "llm_prune_batch": 8,
                "llm": {
                    "model": "m", "base_url": "u", "context_identity": True, "team_objective": "device",
                    "evidence_attribution": True, "shuffle_candidates": True,
                    "per_agent_seed": True, "reasoning": "reason_first",
                    "reasoning_max_chars": 400, "physics_observations": "raw",
                },
            },
        }
        config = load_run_config(payload)
        self.assertEqual(config.policy.kind, "llm_prune")
        self.assertEqual(config.policy.llm_prune_apply, "raw")
        self.assertTrue(config.policy.llm.context_identity)
        self.assertEqual(config.policy.llm.reasoning_max_chars, 400)

    def test_an_unknown_switch_value_is_rejected(self) -> None:
        for policy_patch in (
            {"llm": {"model": "m", "base_url": "u", "physics_observations": "pixels"}},
            {"llm": {"model": "m", "base_url": "u", "team_objective": "loud"}},
            {"llm": {"model": "m", "base_url": "u", "reasoning": "vibes"}},
            {"llm_prune_apply": "sometimes"},
            {"llm_prune_batch": 0},
        ):
            with self.assertRaises(ValueError):
                load_run_config(
                    {
                        "name": "bad", "agents": 4, "branching": 2,
                        "experiments_per_agent": 1, "initial_width": 4, "degrees": [0],
                        "environment": {"kind": "synthetic", "families": 8,
                                        "variants": 4, "task_size": 4},
                        "policy": {"kind": "uniform", **policy_patch},
                    }
                )


class _FakeResponse:
    status_code = 200

    def __init__(self, content: str) -> None:
        self._content = content

    def json(self) -> dict:
        return {
            "id": "resp",
            "model": "stub",
            "choices": [
                {"finish_reason": "stop", "message": {"content": self._content}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _FakeClient:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def post(self, _path, json):  # noqa: A002 - httpx's parameter name
        self.bodies.append(json)
        return _FakeResponse('{"updates": {}, "reason": "r"}')

    async def aclose(self) -> None:
        return None


class RequestBodyTests(unittest.TestCase):
    """The bytes actually sent to the server."""

    def _body(self, **llm) -> dict:
        backend = LLMBackend(
            LLMConfig(model="m", base_url="http://x/v1", **llm), "key"
        )
        client = _FakeClient()
        backend._client = client
        asyncio.run(
            backend.complete(
                messages=[{"role": "user", "content": "hi"}],
                schema={"type": "object"},
                schema_name="branch_decision",
                cache_key="k",
                metadata={},
                **({"extra_body": llm.pop("_extra")} if "_extra" in llm else {}),
            )
        )
        return client.bodies[0]

    def test_the_plain_body_has_no_optional_field(self) -> None:
        body = self._body()
        self.assertEqual(
            sorted(body),
            ["chat_template_kwargs", "max_tokens", "messages", "model",
             "response_format", "temperature"],
        )
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["max_tokens"], LLMConfig().max_output_tokens)
        self.assertNotIn("seed", body)

    def test_thinking_turns_the_flag_on_and_raises_the_ceiling(self) -> None:
        body = self._body(reasoning="thinking",
                          thinking_max_output_tokens=6000)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(body["max_tokens"], 6000)

    def test_reason_first_does_not_touch_the_template_kwargs(self) -> None:
        body = self._body(reasoning="reason_first")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_top_p_is_sent_only_when_configured(self) -> None:
        self.assertNotIn("top_p", self._body())
        self.assertEqual(self._body(top_p=0.9)["top_p"], 0.9)

    def test_an_extra_body_reaches_the_request(self) -> None:
        backend = LLMBackend(LLMConfig(model="m", base_url="http://x/v1"), "key")
        client = _FakeClient()
        backend._client = client
        asyncio.run(
            backend.complete(
                messages=[], schema={}, schema_name="branch_decision",
                cache_key="k", metadata={}, extra_body={"seed": 17},
            )
        )
        self.assertEqual(client.bodies[0]["seed"], 17)


if __name__ == "__main__":
    unittest.main()
