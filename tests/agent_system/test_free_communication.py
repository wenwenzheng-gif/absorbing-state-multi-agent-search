#!/usr/bin/env python3
"""C0/C1/I1: objective/protocol/opening-line text, the free-communication
schema, message delivery and conflict resolution, truthful scoring, and
replay of a recorded free-communication run.

A stub LLM backend stands in for the network throughout; nothing here talks
to a real model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent_system.communication import ClaimReport, CommunicationMessage, Router  # noqa: E402
from src.agent_system.environments.base import build_task_instance  # noqa: E402
from src.agent_system.environments.synthetic import SyntheticEnvironment  # noqa: E402
from src.agent_system.evaluation import HiddenEvaluator  # noqa: E402
from src.agent_system.hypotheses import Encoding  # noqa: E402
from src.agent_system.policies.base import DecisionContext  # noqa: E402
from src.agent_system.policies.openai_policy import (  # noqa: E402
    LLMBackend,
    LLMPolicy,
    Usage,
    _parse_communication,
    communication_schema,
)
from src.agent_system.policies.replay import ReplayBackend, ReplayPolicy  # noqa: E402
from src.agent_system.runner import EpisodeRunner  # noqa: E402
from src.agent_system.schemas import (  # noqa: E402
    ActionError,
    AgentView,
    ComponentSpec,
    EvidenceRecord,
    LLMConfig,
    ReceivedMessage,
    load_run_config,
)

ENCODING = Encoding(8, 4)


def make_policy(**overrides) -> LLMPolicy:
    config = LLMConfig(model="stub", base_url="http://stub/v1", **overrides)
    backend = _NullBackend(config)
    return LLMPolicy(backend, kind="llm_full", planner="fixed_first")


class _NullBackend:
    """Only good for rendering prompts; never actually completes a call."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config


def view(**overrides) -> AgentView:
    payload = dict(
        phase="branch",
        round_number=2,
        total_rounds=4,
        task_size=4,
        families=5,
        variants=3,
        branching=2,
        component_library=tuple(
            ComponentSpec(f, v, f"component_{f}_{v}", f"expr[{f}][{v}]")
            for f in range(5)
            for v in range(3)
        ),
        known_positive=(),
        known_negative=(),
        allowed_values={f: (0, 1, 2) for f in range(5)},
        agent_index=3,
        agent_count=12,
        degree=2,
        neighbours=(1, 5),
    )
    payload.update(overrides)
    return AgentView(**payload)


# ---------------------------------------------------------------------------
# Objective / protocol / opening-line text
# ---------------------------------------------------------------------------


class ObjectiveAndProtocolTextTests(unittest.TestCase):
    def _branch_text(self, **llm_overrides) -> str:
        policy = make_policy(context_identity=True, **llm_overrides)
        return policy._branch_prompt(view(), ())

    def test_simple_objective_has_no_strategy_and_no_device_text(self) -> None:
        text = self._branch_text(team_objective="simple", communication_mode="fixed")
        self.assertIn("OBJECTIVE", text)
        self.assertIn("team's performance", text)
        self.assertNotIn("SYMMETRY BREAKING", text)
        self.assertNotIn("informative for you when all three hold", text)
        self.assertNotIn("Prefer additions that are not already settled", text)

    def test_individual_objective_has_no_strategy_and_no_device_text(self) -> None:
        text = self._branch_text(team_objective="individual", communication_mode="free")
        self.assertIn("OBJECTIVE", text)
        self.assertIn("your own performance", text)
        self.assertNotIn("SYMMETRY BREAKING", text)
        self.assertNotIn("informative for you when all three hold", text)

    def test_c1_and_i1_prompts_differ_only_in_objective_block_and_opening_line(self) -> None:
        """C1 (simple, free) vs I1 (individual, free): same protocol, same
        everything else; only the OBJECTIVE block and the opening line
        change."""
        c1 = self._branch_text(team_objective="simple", communication_mode="free")
        i1 = self._branch_text(team_objective="individual", communication_mode="free")
        self.assertNotEqual(c1, i1)
        # Opening line differs exactly as specified.
        self.assertIn(
            "You are one scientist in a team that is identifying a hidden mechanism.",
            c1.splitlines()[0],
        )
        self.assertIn(
            "You are one scientist identifying a hidden mechanism.", i1.splitlines()[0]
        )
        self.assertNotIn("in a team", i1.splitlines()[0])
        # Strip the opening line and the OBJECTIVE paragraph; everything else
        # must be byte-for-byte identical (same protocol text, same
        # everything-below-changes-every-round content).
        def strip_objective(text: str) -> str:
            lines = text.splitlines()[1:]  # drop opening line
            out = []
            skipping = False
            for line in lines:
                if line == "OBJECTIVE":
                    skipping = True
                    continue
                if skipping and line == "":
                    skipping = False
                    continue
                if skipping:
                    continue
                out.append(line)
            return "\n".join(out)

        self.assertEqual(strip_objective(c1), strip_objective(i1))

    def test_free_protocol_text_differs_from_fixed_for_new_objectives(self) -> None:
        fixed = self._branch_text(team_objective="simple", communication_mode="fixed")
        free = self._branch_text(team_objective="simple", communication_mode="free")
        self.assertIn("sent automatically and truthfully", fixed)
        self.assertNotIn("may be inaccurate", fixed)
        self.assertIn("may be inaccurate", free)
        self.assertIn("can send", free)
        self.assertIn("nothing", free)
        self.assertNotIn("sent automatically and truthfully", free)

    def test_old_modes_keep_the_old_objective_and_protocol_text(self) -> None:
        """communication_mode must not change device/soft/off's PROTOCOL or
        OBJECTIVE text -- those two blocks are only redefined for
        team_objective in {"simple", "individual"}.  ``communication_mode``
        still legitimately adds the (empty, for a "fixed" episode) messages
        block, since that block's own gate is ``communication_mode``, not
        ``team_objective``."""
        device_fixed = self._branch_text(team_objective="device", communication_mode="fixed")
        device_free = self._branch_text(team_objective="device", communication_mode="free")
        for text in (device_fixed, device_free):
            self.assertIn("SYMMETRY BREAKING", text)
            self.assertIn("START OF THE NEXT ROUND", text)
            self.assertNotIn("may be inaccurate", text)
        protocol_and_objective = lambda text: text.split("COMPONENT LIBRARY")[0]
        self.assertEqual(
            protocol_and_objective(device_fixed), protocol_and_objective(device_free)
        )
        self.assertIn("SYMMETRY BREAKING", device_fixed)
        self.assertIn(
            "You are one scientist in a team that is identifying a hidden mechanism.",
            device_fixed.splitlines()[0],
        )

    def test_messages_block_only_appears_under_free_communication(self) -> None:
        fixed = self._branch_text(team_objective="simple", communication_mode="fixed")
        free = self._branch_text(
            team_objective="simple", communication_mode="free"
        )
        self.assertNotIn("MESSAGES FROM NEIGHBOURS", fixed)
        self.assertIn("MESSAGES FROM NEIGHBOURS", free)

    def test_received_message_text_appears_verbatim_and_labelled(self) -> None:
        policy = make_policy(
            context_identity=True, team_objective="simple", communication_mode="free"
        )
        current = view(
            received_messages=(
                ReceivedMessage(sender=1, round_number=1, text="watch out for c2v1"),
            )
        )
        text = policy._branch_prompt(current, ())
        self.assertIn("watch out for c2v1", text)
        self.assertIn("from agent 1", text)


# ---------------------------------------------------------------------------
# Communication schema / parsing
# ---------------------------------------------------------------------------


class CommunicationSchemaTests(unittest.TestCase):
    IDS = ["e1", "e2"]

    def test_schema_enumerates_only_this_rounds_event_ids(self) -> None:
        schema = communication_schema(self.IDS)
        item = schema["properties"]["reports"]["items"]
        self.assertEqual(item["properties"]["event_id"]["enum"], self.IDS)
        self.assertEqual(
            item["properties"]["claimed_outcome"]["enum"], ["positive", "negative"]
        )
        self.assertEqual(schema["properties"]["text"]["maxLength"], 500)

    def test_empty_reports_and_text_are_legal(self) -> None:
        decision = _parse_communication(
            json.dumps({"reports": [], "text": "", "reason": ""}), self.IDS
        )
        self.assertEqual(decision.reports, ())
        self.assertEqual(decision.text, "")

    def test_unknown_event_id_is_rejected(self) -> None:
        payload = {"reports": [{"event_id": "nope", "claimed_outcome": "positive"}], "text": ""}
        with self.assertRaises(ActionError):
            _parse_communication(json.dumps(payload), self.IDS)

    def test_duplicate_event_id_is_rejected(self) -> None:
        payload = {
            "reports": [
                {"event_id": "e1", "claimed_outcome": "positive"},
                {"event_id": "e1", "claimed_outcome": "negative"},
            ],
            "text": "",
        }
        with self.assertRaises(ActionError):
            _parse_communication(json.dumps(payload), self.IDS)

    def test_bad_outcome_is_rejected(self) -> None:
        payload = {"reports": [{"event_id": "e1", "claimed_outcome": "maybe"}], "text": ""}
        with self.assertRaises(ActionError):
            _parse_communication(json.dumps(payload), self.IDS)

    def test_text_over_500_chars_is_rejected(self) -> None:
        payload = {"reports": [], "text": "x" * 501}
        with self.assertRaises(ActionError):
            _parse_communication(json.dumps(payload), self.IDS)

    def test_a_lie_is_a_legal_report(self) -> None:
        """claimed_outcome may legitimately disagree with what was observed;
        the schema has no way to know the truth and must not try to."""
        payload = {"reports": [{"event_id": "e1", "claimed_outcome": "positive"}], "text": ""}
        decision = _parse_communication(json.dumps(payload), self.IDS)
        self.assertEqual(decision.reports[0].claimed_outcome, "positive")


# ---------------------------------------------------------------------------
# Router: message delivery, one hop, no forwarding
# ---------------------------------------------------------------------------


class MessageRouterTests(unittest.TestCase):
    def _path_router(self) -> Router:
        router = Router(3, 0, 1)
        router.graph = ((1,), (0, 2), (1,))
        return router

    def test_message_arrives_next_round_and_is_not_forwarded(self) -> None:
        router = self._path_router()
        message = CommunicationMessage(sender=0, round_number=1, text="hello")
        router.stage_message(0, message)
        self.assertEqual(router.message_deliveries(1), [])
        router.commit_round()
        delivered = router.message_deliveries(1)
        self.assertEqual([(sender, msg.text) for sender, msg in delivered], [(0, "hello")])
        self.assertEqual(router.message_deliveries(2), [], "B must not forward A's message to C")

    def test_agent_cannot_send_a_message_as_someone_else(self) -> None:
        router = self._path_router()
        with self.assertRaises(ValueError):
            router.stage_message(1, CommunicationMessage(sender=0, round_number=1))

    def test_degree_zero_receives_no_message(self) -> None:
        router = Router(4, 0, 7)
        router.stage_message(0, CommunicationMessage(sender=0, round_number=1, text="x"))
        router.commit_round()
        for agent in range(4):
            self.assertEqual(router.message_deliveries(agent), [])


# ---------------------------------------------------------------------------
# Receiver-side claim resolution: own beats claim, first-claim-wins, and the
# identity (including a subset-style configuration) travels through intact.
# ---------------------------------------------------------------------------


def _runner(agents: int = 3) -> EpisodeRunner:
    config = load_run_config(
        {
            "name": "comm",
            "agents": agents,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 2,
            "degrees": [0],
            "environment": {"kind": "synthetic", "families": 8, "variants": 4, "task_size": 4},
            "policy": {"kind": "uniform"},
        }
    )
    task = build_task_instance(
        agents=agents, encoding=ENCODING, task_size=4, seed=321, initial_width=2
    )
    from src.agent_system.policies.uniform import UniformPolicy

    runner = EpisodeRunner(
        config=config,
        degree=0,
        seed=321,
        task=task,
        environment=SyntheticEnvironment(task),
        policy_factory=lambda: UniformPolicy(),
        episode_id="comm_test",
    )
    runner._build_agents()
    return runner


class ClaimResolutionTests(unittest.TestCase):
    PAIR = (0, 0)

    def _star(self, runner: EpisodeRunner) -> None:
        # Agent 0 is the hub, connected to agents 1 and 2; 1 and 2 are not
        # connected to each other, so a claim never reaches a second hop.
        runner.router.graph = ((1, 2), (0,), (0,))

    def test_first_claim_wins_by_round_then_sender_then_event_id(self) -> None:
        runner = self._runner_with_star()
        receiver = runner.agents[0]
        runner._event_truth["ev1"] = EvidenceRecord(
            event_id="ev1", pair=self.PAIR, outcome="negative",
            origin_agent=1, created_round=1, experiment_id=None,
        )
        runner.router.stage_message(
            1, CommunicationMessage(
                sender=1, round_number=1,
                reports=(ClaimReport(event_id="ev1", claimed_outcome="positive"),),
            ),
        )
        runner.router.stage_message(
            2, CommunicationMessage(
                sender=2, round_number=1,
                reports=(ClaimReport(event_id="ev1", claimed_outcome="negative"),),
            ),
        )
        runner.router.commit_round()
        log = {"recv_false_reports": 0, "recv_conflicting_reports": 0}
        runner._receive_messages(1, receiver, log)
        # Sender 1 (lower id) is processed first and wins, even though its
        # claim is the false one.
        self.assertEqual(receiver.evidence.records[self.PAIR].outcome, "positive")
        self.assertEqual(log["recv_false_reports"], 1, "only sender 1's claim is false")
        self.assertEqual(log["recv_conflicting_reports"], 1, "sender 2's claim loses")
        self.assertNotIn(self.PAIR, receiver.verified_positive)
        self.assertNotIn(self.PAIR, receiver.verified_negative)

    def test_own_observation_beats_a_later_claim(self) -> None:
        runner = self._runner_with_star()
        receiver = runner.agents[0]
        receiver.evidence.apply(
            EvidenceRecord(
                event_id="own1", pair=self.PAIR, outcome="negative",
                origin_agent=0, created_round=1, experiment_id=None,
            ),
            own=True,
        )
        receiver.verified_negative.add(self.PAIR)
        runner._event_truth["ev1"] = EvidenceRecord(
            event_id="ev1", pair=self.PAIR, outcome="positive",
            origin_agent=1, created_round=1, experiment_id=None,
        )
        runner.router.stage_message(
            1, CommunicationMessage(
                sender=1, round_number=1,
                reports=(ClaimReport(event_id="ev1", claimed_outcome="positive"),),
            ),
        )
        runner.router.commit_round()
        log = {"recv_false_reports": 0, "recv_conflicting_reports": 0}
        runner._receive_messages(1, receiver, log)
        self.assertEqual(receiver.evidence.records[self.PAIR].outcome, "negative")
        self.assertEqual(log["recv_conflicting_reports"], 1)
        self.assertIn(self.PAIR, receiver.verified_negative)

    def test_configuration_identity_travels_with_the_claim(self) -> None:
        runner = self._runner_with_star()
        receiver = runner.agents[0]
        runner._event_truth["ev1"] = EvidenceRecord(
            event_id="ev1", pair=self.PAIR, outcome="positive",
            origin_agent=1, created_round=1, experiment_id="cfg007",
            configuration=(1, 2, 3, 0),
        )
        runner.router.stage_message(
            1, CommunicationMessage(
                sender=1, round_number=1,
                reports=(ClaimReport(event_id="ev1", claimed_outcome="positive"),),
            ),
        )
        runner.router.commit_round()
        log = {"recv_false_reports": 0, "recv_conflicting_reports": 0}
        runner._receive_messages(1, receiver, log)
        stored = receiver.evidence.records[self.PAIR]
        self.assertEqual(stored.configuration, (1, 2, 3, 0))
        self.assertEqual(stored.experiment_id, "cfg007")
        self.assertIn(self.PAIR, receiver.verified_positive)

    def _runner_with_star(self) -> EpisodeRunner:
        runner = _runner(agents=3)
        self._star(runner)
        return runner


# ---------------------------------------------------------------------------
# Scoring stays truthful: a false claim alone must never verify a hypothesis.
# ---------------------------------------------------------------------------


class TruthfulScoringTests(unittest.TestCase):
    def test_false_claim_does_not_make_evidence_verified_true(self) -> None:
        encoding = Encoding(2, 2)
        task = build_task_instance(
            agents=1, encoding=encoding, task_size=2, seed=101, initial_width=1
        )
        family0, variant0 = task.truth_pairs[0]
        family1 = 1 - family0
        true_variant1 = task.truth_values[family1]
        fake_variant1 = 1 - true_variant1
        fake_pair = (family1, fake_variant1)
        self.assertFalse(task.is_positive(fake_pair))
        candidate = tuple(sorted(((family0, variant0), fake_pair)))

        from src.agent_system.evidence import EvidenceStore

        evidence = EvidenceStore(encoding)
        evidence.apply(
            EvidenceRecord(
                event_id="own", pair=(family0, variant0), outcome="positive",
                origin_agent=0, created_round=1, experiment_id=None,
            ),
            own=True,
        )
        # A false claim: fake_pair is truly negative but is believed positive.
        evidence.apply(
            EvidenceRecord(
                event_id="claim", pair=fake_pair, outcome="positive",
                origin_agent=1, created_round=1, experiment_id=None,
            ),
            own=False,
        )
        self.assertTrue(
            evidence.covers_positively(candidate),
            "the raw belief store is fooled by the false claim -- this is the exploit",
        )

        class FakeAgent:
            pass

        agent = FakeAgent()
        agent.index = 0
        agent.evidence = evidence
        agent.complete_candidates = {candidate}
        agent.verified_positive = {(family0, variant0)}  # the false pair is absent
        agent.verified_negative = set()

        environment = SyntheticEnvironment(task)
        raw = HiddenEvaluator(
            task=task, environment=environment, encoding=encoding, epsilon=1e-3,
            truthful_scoring=False,
        )
        raw_summary = raw.evaluate([agent], round_number=1)["summary"]
        self.assertTrue(raw_summary["any_evidence_verified"])

        truthful = HiddenEvaluator(
            task=task, environment=environment, encoding=encoding, epsilon=1e-3,
            truthful_scoring=True,
        )
        truthful_summary = truthful.evaluate([agent], round_number=1)["summary"]
        self.assertFalse(truthful_summary["any_evidence_verified"])
        self.assertFalse(truthful_summary["any_verified_predictive_success"])


# ---------------------------------------------------------------------------
# End-to-end: call timing (no final round, no degree-0 agent) and stats.
# ---------------------------------------------------------------------------


class _FreeCommBackend:
    """Answers branch/experiment minimally and communication via a rule."""

    cache_key = LLMBackend.cache_key

    def __init__(self, config: LLMConfig, communication_rule=None) -> None:
        self.config = config
        self.usage = Usage()
        self.responses: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.communication_rule = communication_rule or (lambda **_: {"reports": [], "text": ""})

    async def aclose(self) -> None:
        return None

    async def complete(self, *, messages, schema, schema_name, cache_key, metadata,
                       extra_body=None):
        self.calls.append({"schema_name": schema_name, **metadata})
        props = schema["properties"]
        if schema_name == "branch_decision":
            answer = {
                "updates": {
                    key: value["items"]["enum"][: value["minItems"]]
                    for key, value in props["updates"]["properties"].items()
                },
                "reason": "s",
            }
        elif schema_name == "experiment_decision":
            answer = {"choice": props["choice"]["enum"][0], "reason": "s"}
            if "experiment_id" in props:
                answer["experiment_id"] = props["experiment_id"]["enum"][0]
        elif schema_name == "communication_decision":
            available = (
                props["reports"]["items"]
                .get("properties", {})
                .get("event_id", {})
                .get("enum", [])
            )
            answer = self.communication_rule(agent=metadata["agent"], round=metadata["round"],
                                              available=available)
            answer.setdefault("reason", "s")
        else:
            answer = {"drop": [], "reason": "s"}
        self.usage.requests += 1
        record = {"cache_key": cache_key, "content": json.dumps(answer), "usage": {}}
        self.responses[cache_key] = record
        return record


def _free_config(agents: int = 3, rounds_task_size: int = 3) -> dict:
    return {
        "name": "free_comm",
        "agents": agents,
        "branching": 2,
        "experiments_per_agent": 1,
        "initial_width": 2,
        "degrees": [0],
        "task_seeds": [909],
        "environment": {
            "kind": "synthetic", "families": 8, "variants": 4, "task_size": rounds_task_size,
        },
        "policy": {
            "kind": "llm_full",
            "planner": "fixed_first",
            "llm": {
                "model": "stub", "base_url": "http://stub/v1",
                "team_objective": "simple", "communication_mode": "free",
                "repair_attempts": 0,
            },
        },
    }


class EndToEndTimingTests(unittest.TestCase):
    def _run(self, agents: int, graph=None, communication_rule=None):
        config = load_run_config(_free_config(agents=agents))
        task = build_task_instance(
            agents=agents, encoding=ENCODING, task_size=config.task_size,
            seed=909, initial_width=config.initial_width,
        )
        backend = _FreeCommBackend(config.policy.llm, communication_rule)
        runner = EpisodeRunner(
            config=config, degree=0, seed=909, task=task,
            environment=SyntheticEnvironment(task),
            policy_factory=lambda: LLMPolicy(backend, kind="llm_full", planner="fixed_first"),
            episode_id="free_comm_ep", run_id="free_comm_run",
        )
        if graph is not None:
            runner.router.graph = graph
        result = asyncio.run(runner.run())
        return runner, backend, result

    def test_degree_zero_agents_never_get_a_communication_call(self) -> None:
        _runner, backend, result = self._run(agents=4)  # degree=0 config
        comm_calls = [c for c in backend.calls if c["schema_name"] == "communication_decision"]
        self.assertEqual(comm_calls, [])
        transitions = result["transitions"]
        self.assertGreater(len(transitions), 1, "need at least one non-final round")
        for transition in transitions[:-1]:  # the final round never attempts a call
            self.assertEqual(transition["comm_calls"], 0)
            # None of these 4 agents has a neighbour, so none is eligible to
            # send a message; a degree-0 agent must not inflate silent_agents.
            self.assertEqual(transition["silent_agents"], 0)

    def test_silent_agents_excludes_degree_zero_agents(self) -> None:
        """silent_agents counts only eligible agents (degree > 0, not the
        final round) that sent no message -- a degree-0 agent (no neighbour,
        never even attempted a call) must not be counted as silent."""
        # Agents 0 and 1 are connected to each other; agent 2 has no
        # neighbour at all.
        graph = ((1,), (0,), ())
        _runner, backend, result = self._run(agents=3, graph=graph)
        comm_calls = [c for c in backend.calls if c["schema_name"] == "communication_decision"]
        # Only the two eligible agents (0 and 1) ever get a communication call.
        called_agents = {c["agent"] for c in comm_calls}
        self.assertEqual(called_agents, {0, 1})
        transitions = result["transitions"]
        self.assertGreater(len(transitions), 1, "need at least one non-final round")
        for transition in transitions[:-1]:  # the final round never attempts a call
            self.assertEqual(transition["comm_calls"], 2)
            # The default communication rule sends nothing, so both eligible
            # agents stay silent -- but agent 2 (degree 0) must not be
            # counted, so the total is 2, not 3.
            self.assertEqual(transition["silent_agents"], 2)

    def test_no_communication_call_in_the_final_round(self) -> None:
        graph = ((1,), (0, 2), (1,))
        _runner, backend, result = self._run(agents=3, graph=graph)
        comm_rounds = sorted(
            {c["round"] for c in backend.calls if c["schema_name"] == "communication_decision"}
        )
        total_rounds = len(result["transitions"])
        self.assertNotIn(total_rounds, comm_rounds, "no call in the final round")
        if total_rounds > 1:
            self.assertIn(1, comm_rounds)

    def test_reports_and_silence_are_counted(self) -> None:
        graph = ((1,), (0, 2), (1,))

        def rule(agent, round, available):
            if agent == 1 and available:
                return {
                    "reports": [{"event_id": available[0], "claimed_outcome": "positive"}],
                    "text": "hi",
                }
            return {"reports": [], "text": ""}

        _runner, backend, result = self._run(agents=3, graph=graph, communication_rule=rule)
        transitions = result["transitions"]
        self.assertGreaterEqual(len(transitions), 2)
        first = transitions[0]
        self.assertGreaterEqual(first["outgoing_messages"], 0)
        # Every count is non-negative and the keys exist on every transition.
        for transition in transitions:
            for key in (
                "outgoing_reports", "outgoing_false_reports", "outgoing_messages",
                "silent_agents", "recv_false_reports", "recv_conflicting_reports",
            ):
                self.assertIn(key, transition)
                self.assertGreaterEqual(transition[key], 0)


# ---------------------------------------------------------------------------
# Replay reproduces a free-communication run from recorded responses.
# ---------------------------------------------------------------------------


class ReplayFreeCommTests(unittest.TestCase):
    def test_replay_matches_the_live_free_comm_run(self) -> None:
        agents = 3
        graph = ((1,), (0, 2), (1,))
        config = load_run_config(_free_config(agents=agents))
        task = build_task_instance(
            agents=agents, encoding=ENCODING, task_size=config.task_size,
            seed=909, initial_width=config.initial_width,
        )

        def rule(agent, round, available):
            if agent == 1 and available:
                return {
                    "reports": [{"event_id": available[0], "claimed_outcome": "positive"}],
                    "text": "note",
                }
            return {"reports": [], "text": ""}

        live_backend = _FreeCommBackend(config.policy.llm, rule)
        live_runner = EpisodeRunner(
            config=config, degree=0, seed=909, task=task,
            environment=SyntheticEnvironment(task),
            policy_factory=lambda: LLMPolicy(live_backend, kind="llm_full", planner="fixed_first"),
            episode_id="free_comm_ep", run_id="original",
        )
        live_runner.router.graph = graph
        live_result = asyncio.run(live_runner.run())

        replay_backend = ReplayBackend(config.policy.llm, live_backend.responses, "original")
        replay_runner = EpisodeRunner(
            config=config, degree=0, seed=909, task=task,
            environment=SyntheticEnvironment(task),
            policy_factory=lambda: ReplayPolicy(
                replay_backend, kind="replay", behaves_as="llm_full", planner="fixed_first",
            ),
            episode_id="free_comm_ep", run_id="original_replay",
        )
        replay_runner.router.graph = graph
        replay_result = asyncio.run(replay_runner.run())

        self.assertEqual(len(replay_backend.missing), 0)
        for key in ("outgoing_reports", "outgoing_messages", "silent_agents", "tests"):
            self.assertEqual(
                [t[key] for t in live_result["transitions"]],
                [t[key] for t in replay_result["transitions"]],
                msg=key,
            )


if __name__ == "__main__":
    unittest.main()
