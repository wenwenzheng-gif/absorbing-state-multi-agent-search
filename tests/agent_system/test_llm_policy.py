#!/usr/bin/env python3
"""LLM action handling: repair, refusal, truncation, and replay identity.

A stub backend stands in for the service so these paths are exercised without
a network call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.policies.base import DecisionContext  # noqa: E402
from src.agent_system.policies.openai_policy import (  # noqa: E402
    LLMBackend,
    LLMPolicy,
    Usage,
    _parse_branch,
    branch_schema,
    choice_label,
    experiment_schema,
)
from src.agent_system.policies.replay import ReplayBackend  # noqa: E402
from src.agent_system.schemas import (  # noqa: E402
    ActionError,
    AgentView,
    ComponentSpec,
    ExperimentCandidate,
    ExperimentCondition,
    Hypothesis,
    IncompleteOutput,
    LLMConfig,
    ParentSlot,
    PolicyFailure,
)

CONFIG = LLMConfig(model="stub", base_url="http://stub/v1", repair_attempts=1)
CONTEXT = DecisionContext(
    run_id="r", episode_id="e", agent_index=0, round_number=1, phase="branch", seed=7
)
SLOT = ParentSlot(
    hypothesis=Hypothesis(((0, 0),)),
    legal_additions=((1, 0), (1, 1), (2, 0)),
    required=2,
)


def view(candidates=()) -> AgentView:
    return AgentView(
        phase="branch",
        round_number=1,
        total_rounds=3,
        task_size=4,
        families=8,
        variants=4,
        branching=2,
        component_library=(ComponentSpec(0, 0, "c", "expr"),),
        known_positive=(),
        known_negative=(),
        allowed_values={0: (0, 1, 2, 3)},
        parents=(SLOT,),
        candidates=tuple(candidates),
    )


class StubBackend:
    """Returns queued payloads; raises queued exceptions."""

    def __init__(self, items) -> None:
        self.config = CONFIG
        self.usage = Usage()
        self.items = list(items)
        self.calls: list[dict] = []
        self.responses: dict[str, dict] = {}

    cache_key = LLMBackend.cache_key

    async def aclose(self) -> None:
        return None

    async def complete(self, *, messages, schema, schema_name, cache_key, metadata,
                       extra_body=None):
        self.calls.append({"messages": messages, "cache_key": cache_key, **metadata})
        if not self.items:
            raise AssertionError("the policy asked for more attempts than were queued")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        self.usage.requests += 1
        return {"cache_key": cache_key, "content": json.dumps(item)}


def policy(backend, kind="llm_full") -> LLMPolicy:
    return LLMPolicy(backend, kind=kind, planner="fixed_first")


class ParsingTests(unittest.TestCase):
    def test_wrong_addition_count_is_an_action_error(self) -> None:
        payload = {"updates": {SLOT.hypothesis.hid: ["c1v0"]}, "reason": ""}
        with self.assertRaises(ActionError):
            _parse_branch(json.dumps(payload), [SLOT])

    def test_illegal_addition_is_an_action_error(self) -> None:
        payload = {"updates": {SLOT.hypothesis.hid: ["c1v0", "c7v3"]}, "reason": ""}
        with self.assertRaises(ActionError):
            _parse_branch(json.dumps(payload), [SLOT])

    def test_repeated_addition_is_an_action_error(self) -> None:
        payload = {"updates": {SLOT.hypothesis.hid: ["c1v0", "c1v0"]}, "reason": ""}
        with self.assertRaises(ActionError):
            _parse_branch(json.dumps(payload), [SLOT])

    def test_malformed_label_is_an_action_error(self) -> None:
        payload = {"updates": {SLOT.hypothesis.hid: ["c1v0", "family 2"]}, "reason": ""}
        with self.assertRaises(ActionError):
            _parse_branch(json.dumps(payload), [SLOT])

    def test_missing_parent_is_an_action_error(self) -> None:
        with self.assertRaises(ActionError):
            _parse_branch(json.dumps({"updates": {}, "reason": ""}), [SLOT])

    def test_unknown_parent_is_an_action_error(self) -> None:
        payload = {"updates": {SLOT.hypothesis.hid: ["c1v0", "c2v0"], "h_c9v9": []}, "reason": ""}
        with self.assertRaises(ActionError):
            _parse_branch(json.dumps(payload), [SLOT])

    def test_schema_pins_the_count_the_parents_and_the_legal_labels(self) -> None:
        schema = branch_schema([SLOT])
        updates = schema["properties"]["updates"]
        self.assertFalse(updates["additionalProperties"])
        self.assertEqual(updates["required"], [SLOT.hypothesis.hid])
        entry = updates["properties"][SLOT.hypothesis.hid]
        self.assertEqual(entry["minItems"], SLOT.required)
        self.assertEqual(entry["maxItems"], SLOT.required)
        self.assertEqual(
            entry["items"]["enum"], [f"c{f}v{v}" for f, v in SLOT.legal_additions]
        )

    def test_non_json_output_is_an_action_error(self) -> None:
        with self.assertRaises(ActionError):
            _parse_branch("not json at all", [SLOT])

    def test_prompts_put_every_constant_block_before_the_varying_ones(self) -> None:
        from src.agent_system.policies.openai_policy import PROMPT_DIR

        constant = (
            "{environment_notes}", "{identity_block}", "{protocol_block}",
            "{objective_block}", "{component_library}",
        )
        varying = (
            "{clock_block}", "{labels_block}", "{observations_block}", "{memory_block}",
        )
        for name, extra_constant, extra_varying in (
            ("branch.txt", (), ("{parents}",)),
            ("experiment.txt", ("{condition_block}",), ("{candidates}",)),
        ):
            text = (PROMPT_DIR / name).read_text(encoding="utf-8")
            last_constant = max(text.index(token) for token in constant + extra_constant)
            first_varying = min(text.index(token) for token in varying + extra_varying)
            self.assertLess(last_constant, first_varying, name)

    def test_experiment_schema_only_asks_for_a_condition_when_there_is_one(self) -> None:
        candidate = ExperimentCandidate(pair=(1, 0), parent_ids=(SLOT.hypothesis.hid,))
        self.assertNotIn(
            "experiment_id", experiment_schema([candidate], (), False)["properties"]
        )
        condition = ExperimentCondition(
            experiment_id="e0", settings={"x0": 0.0}, description="probe"
        )
        self.assertIn(
            "experiment_id",
            experiment_schema([candidate], [condition], True)["properties"],
        )

    def test_experiment_schema_enumerates_only_valid_component_parent_pairs(self) -> None:
        candidates = [
            ExperimentCandidate(pair=(1, 0), parent_ids=("h_a", "h_b")),
            ExperimentCandidate(pair=(2, 3), parent_ids=("h_b",)),
        ]
        enum = experiment_schema(candidates, (), False)["properties"]["choice"]["enum"]
        self.assertEqual(enum, ["c1v0|h_a", "c1v0|h_b", "c2v3|h_b"])


class RepairTests(unittest.TestCase):
    def _legal(self):
        return {"updates": {SLOT.hypothesis.hid: ["c1v0", "c2v0"]}, "reason": "ok"}

    def test_one_repair_recovers_an_illegal_answer(self) -> None:
        bad = {"updates": {SLOT.hypothesis.hid: []}, "reason": ""}
        backend = StubBackend([bad, self._legal()])
        decision = asyncio.run(policy(backend).propose(view(), [SLOT], CONTEXT))
        self.assertEqual(decision.updates[0].additions, ((1, 0), (2, 0)))
        self.assertEqual(backend.usage.repairs, 1)
        self.assertEqual(len(backend.calls), 2)

    def test_repair_prompt_states_the_constraint_without_the_answer(self) -> None:
        bad = {"updates": {SLOT.hypothesis.hid: []}, "reason": ""}
        backend = StubBackend([bad, self._legal()])
        asyncio.run(policy(backend).propose(view(), [SLOT], CONTEXT))
        complaint = backend.calls[1]["messages"][-1]["content"]
        self.assertIn("hard constraint", complaint)
        self.assertNotIn("truth", complaint.lower())

    def test_exhausted_repair_fails_loudly_with_no_random_substitute(self) -> None:
        bad = {"updates": {SLOT.hypothesis.hid: []}, "reason": ""}
        backend = StubBackend([bad, bad])
        with self.assertRaises(PolicyFailure):
            asyncio.run(policy(backend).propose(view(), [SLOT], CONTEXT))

    def test_truncated_output_is_repairable(self) -> None:
        backend = StubBackend([IncompleteOutput("cut off"), self._legal()])
        decision = asyncio.run(policy(backend).propose(view(), [SLOT], CONTEXT))
        self.assertEqual(len(decision.updates), 1)

    def test_refusal_is_not_repaired(self) -> None:
        backend = StubBackend([PolicyFailure("model refused")])
        with self.assertRaises(PolicyFailure):
            asyncio.run(policy(backend).propose(view(), [SLOT], CONTEXT))


class ExperimentChoiceTests(unittest.TestCase):
    def test_a_candidate_outside_the_list_is_rejected(self) -> None:
        candidate = ExperimentCandidate(pair=(1, 0), parent_ids=(SLOT.hypothesis.hid,))
        answer = {"choice": f"c6v3|{SLOT.hypothesis.hid}", "reason": ""}
        backend = StubBackend([answer, answer])
        with self.assertRaises(PolicyFailure):
            asyncio.run(
                policy(backend).choose_experiment(view([candidate]), [candidate], CONTEXT)
            )

    def test_a_parent_that_did_not_propose_the_pair_is_rejected(self) -> None:
        candidate = ExperimentCandidate(pair=(1, 0), parent_ids=("h_c9v9",))
        answer = {"choice": f"c1v0|{SLOT.hypothesis.hid}", "reason": ""}
        backend = StubBackend([answer, answer])
        with self.assertRaises(PolicyFailure):
            asyncio.run(
                policy(backend).choose_experiment(view([candidate]), [candidate], CONTEXT)
            )


    def test_a_legal_choice_resolves_to_its_component_and_parent(self) -> None:
        candidate = ExperimentCandidate(pair=(1, 0), parent_ids=("h_a", "h_b"))
        answer = {"choice": choice_label((1, 0), "h_b"), "reason": "ok"}
        backend = StubBackend([answer])
        decision = asyncio.run(
            policy(backend).choose_experiment(view([candidate]), [candidate], CONTEXT)
        )
        self.assertEqual(decision.pair, (1, 0))
        self.assertEqual(decision.parent_id, "h_b")


class CacheKeyTests(unittest.TestCase):
    def _key(self, backend, **overrides) -> str:
        context = DecisionContext(
            **{
                "run_id": "r",
                "episode_id": "e",
                "agent_index": 0,
                "round_number": 1,
                "phase": "branch",
                "seed": 7,
                **overrides,
            }
        )
        return backend.cache_key(
            context, batch=0, attempt=0, schema_name="branch_decision", digest="abc"
        )

    def test_identity_fields_change_the_key(self) -> None:
        backend = StubBackend([])
        base = self._key(backend)
        for field, value in (
            ("episode_id", "e2"),
            ("agent_index", 1),
            ("round_number", 2),
            ("phase", "experiment"),
            ("slot", 1),
        ):
            self.assertNotEqual(base, self._key(backend, **{field: value}), msg=field)

    def test_replay_keys_against_the_recorded_run(self) -> None:
        backend = ReplayBackend(CONFIG, {}, source_run_id="original")
        replayed = self._key(backend, run_id="original_replay")
        live = StubBackend([])
        self.assertEqual(replayed, self._key(live, run_id="original"))

    def test_replay_refuses_an_unrecorded_request(self) -> None:
        backend = ReplayBackend(CONFIG, {}, source_run_id="original")
        with self.assertRaises(PolicyFailure):
            asyncio.run(
                backend.complete(
                    messages=[],
                    schema={},
                    schema_name="branch_decision",
                    cache_key="missing",
                    metadata={},
                )
            )


if __name__ == "__main__":
    unittest.main()
