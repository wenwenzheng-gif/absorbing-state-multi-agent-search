"""Synchronous episode state machine with pluggable policies.

Phase barriers are global: every agent finishes receiving and pruning before
any branch decision is made, and every branch decision is validated before the
experiment budget is allocated.  How fast a policy answers therefore cannot
change who receives a message or who wins an experiment slot.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import hashlib
import statistics
from typing import Callable, Sequence

from ..scaling_model import Node, derived_rng
from . import PROTOCOL_VERSION
from .budget import randomized_round_robin_allocate_async
from .communication import ClaimReport, CommunicationMessage, Router
from .evidence import EvidenceStore
from .environments.base import Environment, TaskInstance
from .evaluation import HiddenEvaluator
from .hypotheses import ChildCollector, Encoding
from .policies.base import DecisionContext, Policy, prompt_fingerprint
from .schemas import (
    ActionError,
    BranchDecision,
    EvidenceRecord,
    ExperimentCandidate,
    ExperimentDecision,
    LabelRecord,
    OwnExperimentSummary,
    Pair,
    ParentSlot,
    PolicyFailure,
    PrivateObservation,
    ReceivedMessage,
    RoundMemory,
    RunConfig,
)
from .views import build_view

SAMPLE_POINTS = (
    "initial",
    "before_receive",
    "after_receive",
    "after_branch_before_own",
    "after_own",
    "round_end",
)

# One SplitMix64 domain per knob, all disjoint from the frozen simulator's and
# from each other, so a knob at its default consumes no draw and a knob turned
# on cannot shift any other stream.
Q_OWN_DOMAIN = 0xBB67AE8584CAA73B
Q_RECEIVED_DOMAIN = 0x3C6EF372FE94F82B
# ``q_model_draw = "per_message"`` draws once per delivered copy instead of
# once per unique label, off its own domain so the two semantics cannot share
# a stream and the default consumes no draw from it.
Q_RECEIVED_COPY_DOMAIN = 0x510E527FADE682D1
RESPAWN_DOMAIN = 0xA54FF53A5F1D36F1

_FULL = 1 << 64


def _accepts(rng, probability: float) -> bool:
    """One Bernoulli(``probability``) draw off a SplitMix64 stream."""
    return rng.next() < int(probability * _FULL)


@dataclass
class AgentRuntime:
    index: int
    initial_pairs: tuple[Pair, ...]
    frontier: list[Node]
    evidence: EvidenceStore
    observations: list[PrivateObservation] = field(default_factory=list)
    reached_truth: bool = False
    proposed_parents: dict[Pair, list[Node]] = field(default_factory=dict)
    round_parents: tuple[ParentSlot, ...] = ()
    complete_candidates: set[tuple[Pair, ...]] = field(default_factory=set)
    # Keys of the hypotheses injected by ``respawn`` this round.  Their
    # children are recorded as a new source lineage rather than as offspring
    # of an old incorrect one, so the reproduction ratio stays clean.
    respawned_keys: set[int] = field(default_factory=set)
    solved_round: int | None = None
    # What the agent itself did, round by round: its only memory across the
    # stateless LLM calls, and never more than its own actions and labels.
    history: list[RoundMemory] = field(default_factory=list)
    round_tests: list[tuple[Pair, str]] = field(default_factory=list)
    # Labels that arrived since the last pruning decision.  Only filled when
    # the policy, not the runner, does the pruning.
    pending_received: list[LabelRecord] = field(default_factory=list)
    pending_own: list[LabelRecord] = field(default_factory=list)
    # -- free communication (communication_mode == "free") ------------------
    # The agent's own true experiment records this round, offered to it (and
    # only it) as candidates it may choose to report on.
    round_events: list[EvidenceRecord] = field(default_factory=list)
    # Free-text notes that arrived at the start of this round; ephemeral,
    # shown in this round's prompts and never carried into the next one.
    inbox_texts: list[ReceivedMessage] = field(default_factory=list)
    # Genuinely true labels only: own experiments, truthfully-delivered
    # evidence, and claims that happened to match the real outcome.  Read
    # only by the hidden evaluator (never by a policy or a prompt) so a false
    # claim alone can never make a success criterion true.
    verified_positive: set[Pair] = field(default_factory=set)
    verified_negative: set[Pair] = field(default_factory=set)


def _label(record: EvidenceRecord, *, own: bool) -> LabelRecord:
    return LabelRecord(
        pair=record.pair,
        outcome=record.outcome,
        source="own" if own else "received",
        sender=None if own else record.origin_agent,
        round_number=record.created_round,
        configuration=record.configuration,
    )


class EpisodeRunner:
    def __init__(
        self,
        *,
        config: RunConfig,
        degree: int,
        seed: int,
        task: TaskInstance,
        environment: Environment,
        policy_factory: Callable[[], Policy],
        episode_id: str,
        run_id: str = "adhoc",
        event_sink: Callable[[dict], None] | None = None,
        evaluator: HiddenEvaluator | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.degree = degree
        self.seed = seed
        self.task = task
        self.environment = environment
        self.episode_id = episode_id
        self.run_id = run_id
        self.event_sink = event_sink
        self.evaluator = evaluator
        self.encoding: Encoding = task.encoding
        self.agents_count = config.agents
        self.rounds = config.effective_rounds
        self.router = Router(
            config.agents,
            degree,
            seed,
            topology=config.topology,
            rewire_p=config.topology_rewire_p,
        )
        self.q_own = config.effective_q_own
        self.q_received = config.effective_q_received
        self.q_draw = config.q_model_draw
        self.communication_mode = config.policy.llm.communication_mode
        self.free_communication = self.communication_mode == "free"
        # event_id -> the real EvidenceRecord an experiment produced, kept
        # for the lifetime of the episode.  Only used to resolve a claim's
        # identity and to check it against the truth; never exposed to a
        # policy.
        self._event_truth: dict[str, EvidenceRecord] = {}
        self.per_agent_cap = (
            config.experiments_per_agent if config.budget_mode == "hard_cap" else None
        )
        self.policies = [policy_factory() for _ in range(config.agents)]
        for policy in self.policies:
            policy.bind_predictor(
                environment.predict if environment.supports_prediction() else None
            )
        # ``llm_prune`` moves both pruning points from exact code to the
        # model.  Everything else -- storing the label, tightening the allowed
        # set, sending the label on -- stays where it was, so the only thing
        # under test is whether the agent works out what the label refutes.
        self.llm_prune = bool(
            self.policies and getattr(self.policies[0], "prunes", False)
        )
        self.prune_apply = config.policy.llm_prune_apply
        self.audit = {
            "d0_messages": 0,
            "initialization_error": 0,
            "initial_merge": 0,
            "depth_nonsynchronous": 0,
            "empty_frontier_replenished": 0,
            "forwarded_evidence": 0,
            "nonproposed_experiment": 0,
            "budget_overflow": 0,
            "same_round_message_use": 0,
            "illegal_action": 0,
            "action_repairs": 0,
            # A respawned hypothesis must never be counted as the descendant
            # of an old incorrect lineage; this fires if one ever is.
            "respawn_lineage_leak": 0,
        }
        self.event_counter = 0
        self.agents: list[AgentRuntime] = []
        self.stopped_round: int | None = None

    # -- setup -------------------------------------------------------------
    def _build_agents(self) -> None:
        truth = self.task.truth_set
        for index in range(self.agents_count):
            store = self.environment.new_evidence_store(self.encoding)
            pairs = self.task.initial_pairs[index]
            frontier = [
                self.encoding.node_from_pair(pair, wrong=pair not in truth) for pair in pairs
            ]
            runtime = AgentRuntime(
                index=index,
                initial_pairs=pairs,
                frontier=frontier,
                evidence=store,
                observations=list(self.environment.initial_observations(index)),
            )
            if len(pairs) != self.config.initial_width or len(set(pairs)) != len(pairs):
                self.audit["initialization_error"] += 1
            if sum(pair in truth for pair in pairs) != 1:
                self.audit["initialization_error"] += 1
            if len({node.key for node in frontier}) != len(pairs):
                self.audit["initial_merge"] += 1
            if any(node.used.bit_count() != 1 for node in frontier):
                self.audit["depth_nonsynchronous"] += 1
            self.agents.append(runtime)

    def _event(self, kind: str, payload: dict) -> str:
        self.event_counter += 1
        event_id = f"{self.episode_id}:{self.event_counter:06d}"
        if self.event_sink is not None:
            self.event_sink({"event_id": event_id, "kind": kind, **payload})
        return event_id

    # -- main loop ----------------------------------------------------------
    async def run(self) -> dict:
        self._build_agents()
        transitions: list[dict] = []
        self.stopped_round = None
        for round_number in range(1, self.rounds + 1):
            log = await self._round(round_number)
            if self.evaluator is not None and round_number < self.rounds:
                log["evaluation"] = self.evaluator.evaluate(self.agents, round_number)["summary"]
            transitions.append(log)
            self.router.commit_round()
            # "The episode stops at the end of the first round in which at
            # least one agent is solved."  Stopping at the last round is a
            # no-op, so only an earlier one truncates the trajectory.
            if self.config.stop_on_success and log["agents_solved"]:
                self.stopped_round = round_number
                break
        result = self._finalise(transitions)
        for policy in self.policies:
            await policy.aclose()
        return result

    async def _round(self, round_number: int) -> dict:
        log: dict[str, object] = {
            "round": round_number,
            "total_budget_nominal": self.agents_count * self.config.experiments_per_agent,
            "B_nominal": self.agents_count * self.config.experiments_per_agent,
            "recv_messages": 0,
            "recv_unique": 0,
            "recv_duplicate": 0,
            "recv_already_explicit": 0,
            "recv_already_implied": 0,
            "recv_informative": 0,
            "recv_pruned_hypotheses": 0,
            "sender_edges_with_message": 0,
            "parents_attempted": 0,
            "sampled_edges": 0,
            "old_incorrect_edges": 0,
            "new_incorrect_source_edges": 0,
            "unique_children": 0,
            "merge_loss": 0,
            "dual_origin_unique_children": 0,
            "incorrect_parents_after_receive": 0,
            "old_pre": 0,
            "source_pre": 0,
            "tests": 0,
            "tests_pos": 0,
            "tests_neg": 0,
            "own_pruned_hypotheses": 0,
            # -- knob diagnostics.  All zero when the knobs are at default. --
            "recv_missed": 0,
            "recv_copies_missed": 0,
            "own_missed": 0,
            "respawned_agents": 0,
            "respawned_hypotheses": 0,
            "respawned_incorrect": 0,
            "respawn_source_children": 0,
            "protocol_selected_experiments": 0,
            "q_model_own": self.q_own,
            "q_model_received": self.q_received,
            # -- llm_prune shadow score.  All zero unless the model prunes. --
            "prune_calls": 0,
            "prune_hypotheses_shown": 0,
            "prune_should": 0,
            "prune_hit": 0,
            "prune_missed": 0,
            "prune_false": 0,
            "prune_false_correct_lineage": 0,
            "prune_applied": 0,
            # -- free communication (communication_mode == "free") ---------
            # All zero, and never touched, when communication_mode == "fixed".
            "outgoing_reports": 0,
            "outgoing_false_reports": 0,
            "outgoing_messages": 0,
            "silent_agents": 0,
            "recv_false_reports": 0,
            "recv_conflicting_reports": 0,
            "comm_calls": 0,
            "comm_failed": 0,
        }
        for agent in self.agents:
            agent.round_tests = []
            agent.pending_own = []
            agent.round_events = []
        self._receive(round_number, log)
        await self._prune_stage(round_number, log, "receive")
        self._respawn(round_number, log)
        await self._branch(round_number, log)
        await self._experiments(round_number, log)
        await self._prune_stage(round_number, log, "own")
        if self.free_communication and round_number < self.rounds:
            await self._communicate(round_number, log)
        self._close_round(round_number, log)
        return log

    # -- stage 1 ------------------------------------------------------------
    def _copy_accepted(
        self, round_number: int, receiver: int, sender: int, pair: Pair
    ) -> bool:
        """One ``q_model`` draw for one delivered copy of one label.

        Derived from a domain-separated hash of the episode seed, the round,
        the receiver, the sender and the label, so the outcome of a copy does
        not depend on how many other copies arrived, nor on the order the
        router happened to deliver them in.
        """
        rng = derived_rng(
            self.seed,
            Q_RECEIVED_COPY_DOMAIN,
            round_number,
            receiver,
            sender,
            pair[0],
            pair[1],
        )
        return _accepts(rng, self.q_received)

    def _per_message_learned(
        self, round_number: int, receiver: int, deliveries
    ) -> tuple[set[Pair], int]:
        """Labels learned under ``per_message``, and the copies that failed.

        Every distinct ``(sender, label)`` is an independent opportunity, so
        with ``c`` copies the label is learned with probability
        ``1 - (1 - q)^c`` -- the mean-field assumption that each neighbour is
        its own chance to prune.
        """
        seen_copies: set[tuple[int, Pair]] = set()
        learned: set[Pair] = set()
        failed = 0
        for sender, record in deliveries:
            key = (sender, record.pair)
            if key in seen_copies:
                continue
            seen_copies.add(key)
            if self._copy_accepted(round_number, receiver, sender, record.pair):
                learned.add(record.pair)
            else:
                failed += 1
        return learned, failed

    def _receive(self, round_number: int, log: dict) -> None:
        sizes = []
        misses = self.q_received < 1.0
        per_message = misses and self.q_draw == "per_message"
        for agent in self.agents:
            seen: set[Pair] = set()
            before = len(agent.frontier)
            ordinal = 0
            agent.pending_received = []
            agent.inbox_texts = []
            log["sender_edges_with_message"] += self.router.senders_with_message(agent.index)
            deliveries = self.router.deliveries(agent.index)
            learned: set[Pair] = set()
            copies: dict[Pair, int] = {}
            if per_message:
                learned, failed_copies = self._per_message_learned(
                    round_number, agent.index, deliveries
                )
                log["recv_copies_missed"] += failed_copies
            elif misses:
                # Copies of a label the agent misses are wasted too; count
                # them, so the two semantics can be compared on the same
                # denominator.  A duplicate of an *accepted* label is already
                # reported as ``recv_duplicate``.
                senders_by_pair: dict[Pair, set[int]] = {}
                for sender, record in deliveries:
                    senders_by_pair.setdefault(record.pair, set()).add(sender)
                copies = {pair: len(group) for pair, group in senders_by_pair.items()}
            for _sender, record in deliveries:
                log["recv_messages"] += 1
                if record.pair in seen:
                    log["recv_duplicate"] += 1
                    continue
                seen.add(record.pair)
                log["recv_unique"] += 1
                explicit = agent.evidence.knows(record.pair)
                informative = agent.evidence.would_be_informative(record.pair, record.positive)
                if explicit:
                    log["recv_already_explicit"] += 1
                elif not informative:
                    log["recv_already_implied"] += 1
                # "per_label": one q_model draw per *unique* incoming label.
                # The duplicate filter is about the message stream, not about
                # the agent, so the same label arriving twice is still a
                # single chance to act on it.  "per_message" has already
                # drawn once per delivered copy above.
                if per_message:
                    if record.pair not in learned:
                        log["recv_missed"] += 1
                        continue
                elif misses:
                    rng = derived_rng(
                        self.seed, Q_RECEIVED_DOMAIN, round_number, agent.index, ordinal
                    )
                    ordinal += 1
                    if not _accepts(rng, self.q_received):
                        log["recv_missed"] += 1
                        log["recv_copies_missed"] += copies.get(record.pair, 1)
                        continue
                agent.evidence.apply(record, own=False)
                (agent.verified_positive if record.positive else agent.verified_negative).add(
                    record.pair
                )
                if self.llm_prune:
                    agent.pending_received.append(_label(record, own=False))
                else:
                    agent.frontier = agent.evidence.prune(agent.frontier)
                log["recv_informative"] += int(informative)
            if self.free_communication:
                self._receive_messages(round_number, agent, log)
            log["recv_pruned_hypotheses"] += before - len(agent.frontier)
            sizes.append(len(agent.frontier))
        if self.degree == 0 and log["recv_messages"]:
            self.audit["d0_messages"] += int(log["recv_messages"])
        for agent in self.agents:
            if any(node.used.bit_count() != round_number for node in agent.frontier):
                self.audit["depth_nonsynchronous"] += 1
        log["empty_after_receive"] = sum(size == 0 for size in sizes)
        log["empty_after_receive_fraction"] = log["empty_after_receive"] / self.agents_count
        log["mean_frontier_after_receive"] = statistics.mean(sizes)
        log["median_frontier_after_receive"] = statistics.median(sizes)

    # -- stage 1, free communication -----------------------------------------
    def _receive_messages(self, round_number: int, agent: "AgentRuntime", log: dict) -> None:
        """Deliver one agent's inbox: free text plus claims about identities.

        Claims are resolved deterministically: an agent's own evidence always
        wins (it is never overwritten -- see ``EvidenceStore.apply``), and
        among several incoming claims about the same identity the first one
        in ``(round, sender, event_id)`` order is accepted; every later one
        is dropped, whether or not it agrees with the one that won.  A claim
        that matches the true outcome also updates ``verified_positive`` /
        ``verified_negative``, the evaluator's truthful-scoring surface; a
        false claim never does, however many hypotheses it goes on to prune.
        """
        deliveries = self.router.message_deliveries(agent.index)
        for sender, message in deliveries:
            if message.text:
                agent.inbox_texts.append(
                    ReceivedMessage(
                        sender=sender, round_number=message.round_number, text=message.text
                    )
                )
        flat: list[tuple[int, int, str, ClaimReport]] = []
        for sender, message in deliveries:
            for report in message.reports:
                flat.append((message.round_number, sender, report.event_id, report))
        flat.sort(key=lambda item: (item[0], item[1], item[2]))
        for msg_round, sender, _event_id, report in flat:
            truth = self._event_truth.get(report.event_id)
            if truth is None:
                continue  # defensive: identity must always be real
            pair = truth.pair
            is_false = report.claimed_outcome != truth.outcome
            log["recv_false_reports"] += int(is_false)
            if agent.evidence.knows(pair):
                if agent.evidence.records[pair].outcome != report.claimed_outcome:
                    log["recv_conflicting_reports"] += 1
                continue
            believed = EvidenceRecord(
                event_id=report.event_id,
                pair=pair,
                outcome=report.claimed_outcome,
                origin_agent=sender,
                created_round=msg_round,
                experiment_id=truth.experiment_id,
                configuration=truth.configuration,
            )
            agent.evidence.apply(believed, own=False)
            if self.llm_prune:
                agent.pending_received.append(_label(believed, own=False))
            else:
                agent.frontier = agent.evidence.prune(agent.frontier)
            if not is_false:
                target = (
                    agent.verified_positive
                    if report.claimed_outcome == "positive"
                    else agent.verified_negative
                )
                target.add(pair)

    # -- stage 1b -----------------------------------------------------------
    def _respawn(self, round_number: int, log: dict) -> None:
        """Re-seed agents whose frontier emptied, if the knob asks for it.

        The frozen simulator has no replenish mechanism: its
        ``empty_frontier_replenished`` counter is an *assertion* that children
        never appear without parents, and it stays at zero.  That name is
        therefore left to mean exactly what it means there, and the new
        behaviour is reported under ``respawned_*``.

        An extinct agent re-enters at the depth everyone else is at, so its
        seeds are depth-``t`` hypotheses (``t = round_number``) that its own
        evidence still allows: known positives first, because those are the
        components the agent has actually established, then uniformly drawn
        legal assignments.  Seeds are tagged so their children count as a new
        source lineage and never inflate ``old_pre``.
        """
        for agent in self.agents:
            agent.respawned_keys = set()
        if self.config.respawn == "none":
            return
        depth = round_number
        for agent in self.agents:
            if agent.frontier:
                continue
            seeds = self._respawn_seeds(agent, depth, round_number)
            if not seeds:
                continue
            agent.frontier = seeds
            agent.respawned_keys = {node.key for node in seeds}
            log["respawned_agents"] += 1
            log["respawned_hypotheses"] += len(seeds)
            log["respawned_incorrect"] += sum(node.wrong for node in seeds)
            if any(node.from_wrong for node in seeds):
                self.audit["respawn_lineage_leak"] += 1
        log["empty_after_respawn"] = sum(not agent.frontier for agent in self.agents)
        log["empty_after_respawn_fraction"] = log["empty_after_respawn"] / self.agents_count

    def _respawn_seeds(
        self, agent: AgentRuntime, depth: int, round_number: int
    ) -> list[Node]:
        positives = list(agent.evidence.known_positive())[:depth]
        collected: dict[int, Node] = {}
        # A bounded number of tries: the legal space can be far smaller than
        # n0 once evidence has narrowed it, and a failed draw must not spin.
        for attempt in range(self.config.initial_width * 8):
            if len(collected) >= self.config.initial_width:
                break
            rng = derived_rng(
                self.seed, RESPAWN_DOMAIN, round_number, agent.index, attempt
            )
            components = list(positives)
            used = {family for family, _ in components}
            while len(components) < depth:
                choices = [
                    (family, variant)
                    for family in range(self.encoding.families)
                    if family not in used
                    for variant in range(self.encoding.variants)
                    if agent.evidence.allowed[family] & (1 << variant)
                ]
                if not choices:
                    break
                family, variant = choices[rng.below(len(choices))]
                components.append((family, variant))
                used.add(family)
            if len(components) != depth:
                break
            node = self._respawn_node(components)
            if node.key in collected or not agent.evidence.compatible(node):
                continue
            collected[node.key] = node
        return [collected[key] for key in sorted(collected)]

    def _respawn_node(self, components: Sequence[Pair]) -> Node:
        """A depth-t seed carrying a *new source* lineage tag."""
        truth = self.task.truth_set
        key = 0
        used = 0
        for family, variant in components:
            key |= (variant + 1) << (self.encoding.bits * family)
            used |= 1 << family
        return Node(
            key=key,
            used=used,
            wrong=any(pair not in truth for pair in components),
            from_wrong=False,
        )

    # -- stage 2 ------------------------------------------------------------
    def _parent_slots(self, agent: AgentRuntime) -> tuple[ParentSlot, ...]:
        slots = []
        for node in sorted(agent.frontier, key=lambda item: item.key):
            choices = self.encoding.legal_extensions(agent.evidence.allowed, node)
            slots.append(
                ParentSlot(
                    hypothesis=self.encoding.hypothesis(node),
                    legal_additions=tuple(choices),
                    required=min(self.config.branching, len(choices)),
                )
            )
        return tuple(slots)

    def _view(self, agent: AgentRuntime, phase: str, round_number: int, **extra):
        return build_view(
            phase=phase,
            round_number=round_number,
            total_rounds=self.rounds,
            task_size=self.config.task_size,
            families=self.encoding.families,
            variants=self.encoding.variants,
            branching=self.config.branching,
            component_library=self.environment.component_library(),
            evidence=agent.evidence,
            experiment_library=self.environment.experiment_library(),
            private_observations=tuple(agent.observations),
            environment_kind=self.environment.kind,
            environment_notes=self.environment.notes,
            # Its own place in the team and its own past: legitimate context.
            agent_index=agent.index,
            agent_count=self.agents_count,
            degree=self.degree,
            neighbours=self.router.neighbours(agent.index),
            round_budget=self.config.experiments_per_agent,
            own_history=tuple(agent.history),
            received_messages=tuple(agent.inbox_texts),
            **extra,
        )

    # -- stage 1c / 3b: pruning by the policy --------------------------------
    async def _prune_stage(self, round_number: int, log: dict, stage: str) -> None:
        """Hand the two pruning points to the policy, and grade every call.

        The runner keeps applying the label and tightening the allowed set;
        what moves to the model is only the inference "which of my hypotheses
        does this label kill".  A hypothesis the model fails to drop really
        does stay in the frontier and really does branch next round -- that
        is the whole point of measuring ``q_model``.
        """
        if not self.llm_prune:
            return
        slot = 0 if stage == "receive" else 1
        work: list[tuple[AgentRuntime, list, tuple[LabelRecord, ...]]] = []
        for agent in self.agents:
            new_labels = (
                agent.pending_received if stage == "receive" else agent.pending_own
            )
            if not new_labels or not agent.frontier:
                continue
            nodes = sorted(agent.frontier, key=lambda item: item.key)
            work.append((agent, nodes, tuple(new_labels)))
        if work:
            views = [
                self._view(
                    agent,
                    "prune",
                    round_number,
                    prune_hypotheses=tuple(
                        self.encoding.hypothesis(node) for node in nodes
                    ),
                    new_evidence=labels,
                )
                for agent, nodes, labels in work
            ]
            decisions = await asyncio.gather(
                *[
                    self.policies[agent.index].prune(
                        view,
                        view.prune_hypotheses,
                        DecisionContext(
                            run_id=self.run_id,
                            episode_id=self.episode_id,
                            agent_index=agent.index,
                            round_number=round_number,
                            phase="prune",
                            seed=self.seed,
                            slot=slot,
                        ),
                    )
                    for (agent, _nodes, _labels), view in zip(work, views)
                ]
            )
            counter = (
                "recv_pruned_hypotheses" if stage == "receive" else "own_pruned_hypotheses"
            )
            for (agent, nodes, _labels), dropped in zip(work, decisions):
                log["prune_calls"] += 1
                before = len(agent.frontier)
                self._apply_prune(agent, nodes, dropped, log)
                log[counter] += before - len(agent.frontier)
        for agent in self.agents:
            if stage == "receive":
                agent.pending_received = []
            else:
                agent.pending_own = []
        if stage == "receive":
            sizes = [len(agent.frontier) for agent in self.agents]
            log["empty_after_receive"] = sum(size == 0 for size in sizes)
            log["empty_after_receive_fraction"] = (
                log["empty_after_receive"] / self.agents_count
            )
            log["mean_frontier_after_receive"] = statistics.mean(sizes)
            log["median_frontier_after_receive"] = statistics.median(sizes)

    def _apply_prune(
        self, agent: AgentRuntime, nodes: Sequence, dropped: Sequence[str], log: dict
    ) -> None:
        """Score the model against exact pruning, then apply its verdict.

        ``validated`` keeps the paper's zero-false-negative oracle: a drop is
        applied only when the hypothesis really is refuted by evidence the
        agent holds, so the model can miss a prune but can never kill a
        consistent hypothesis.  ``raw`` applies whatever it said.
        """
        by_id = {self.encoding.hypothesis(node).hid: node for node in nodes}
        should = {
            hid for hid, node in by_id.items() if not agent.evidence.compatible(node)
        }
        asked = {hid for hid in dropped if hid in by_id}
        hit = asked & should
        wrongly = asked - should
        log["prune_hypotheses_shown"] += len(by_id)
        log["prune_should"] += len(should)
        log["prune_hit"] += len(hit)
        log["prune_missed"] += len(should - asked)
        log["prune_false"] += len(wrongly)
        # A consistent hypothesis it tried to drop that is also still on the
        # correct lineage: the failure mode "validated" exists to prevent.
        log["prune_false_correct_lineage"] += sum(
            not by_id[hid].wrong for hid in wrongly
        )
        applied = hit if self.prune_apply == "validated" else asked
        log["prune_applied"] += len(applied)
        if not applied:
            return
        agent.frontier = [
            node
            for node in agent.frontier
            if self.encoding.hypothesis(node).hid not in applied
        ]

    async def _branch(self, round_number: int, log: dict) -> None:
        slot_table = [self._parent_slots(agent) for agent in self.agents]
        views = [
            self._view(agent, "branch", round_number, parents=slot_table[agent.index])
            for agent in self.agents
        ]
        tasks = [
            self.policies[agent.index].propose(
                views[agent.index],
                slot_table[agent.index],
                DecisionContext(
                    run_id=self.run_id,
                    episode_id=self.episode_id,
                    agent_index=agent.index,
                    round_number=round_number,
                    phase="branch",
                    seed=self.seed,
                ),
            )
            for agent in self.agents
        ]
        decisions = await asyncio.gather(*tasks)

        preown_sizes = []
        proposed_sizes = []
        for agent, slots, decision in zip(self.agents, slot_table, decisions):
            collector = ChildCollector()
            node_by_id = {
                self.encoding.hypothesis(node).hid: node
                for node in sorted(agent.frontier, key=lambda item: item.key)
            }
            log["incorrect_parents_after_receive"] += sum(node.wrong for node in agent.frontier)
            updates = validate_branch_decision(decision, slots)
            agent.proposed_parents = {}
            # Respawn empties and refills the whole frontier at once, so a
            # round either has no respawned parents or none but respawned
            # ones; no child can mix the two lineages.
            respawn_wrong_edges = 0
            for slot, additions in updates:
                parent = node_by_id[slot.hypothesis.hid]
                respawned = parent.key in agent.respawned_keys
                for pair in additions:
                    child = self.encoding.child(
                        parent, pair, wrong=not self.task.is_positive(pair)
                    )
                    if respawned:
                        # The seed was injected this round, so its offspring
                        # are the *first* generation of that lineage, not the
                        # continuation of an old incorrect one.  Clearing
                        # from_wrong keeps them out of old_pre/old_post.
                        child = replace(child, from_wrong=False)
                        log["respawn_source_children"] += 1
                        respawn_wrong_edges += int(parent.wrong)
                    collector.add(parent, pair, child)
                    agent.proposed_parents.setdefault(pair, []).append(parent)
            if not slots and collector.children:
                self.audit["empty_frontier_replenished"] += 1
            log["parents_attempted"] += len(slots)
            log["sampled_edges"] += collector.sampled_edges
            log["unique_children"] += len(collector.children)
            log["merge_loss"] += collector.merge_loss
            log["dual_origin_unique_children"] += len(collector.dual_origin)
            # An edge leaving a respawned incorrect seed is the birth of that
            # lineage, not the reproduction of an old one, so it is moved from
            # the old-incorrect column to the new-source column.
            log["old_incorrect_edges"] += collector.old_incorrect_edges - respawn_wrong_edges
            log["new_incorrect_source_edges"] += (
                collector.new_incorrect_source_edges + respawn_wrong_edges
            )
            agent.frontier = collector.frontier()
            agent.round_parents = tuple(
                ParentSlot(hypothesis=slot.hypothesis, legal_additions=(), required=0)
                for slot in slots
            )
            preown_sizes.append(len(agent.frontier))
            proposed_sizes.append(len(agent.proposed_parents))
            log["old_pre"] += sum(node.wrong and node.from_wrong for node in agent.frontier)
            log["source_pre"] += sum(node.wrong and not node.from_wrong for node in agent.frontier)
            # The pool the hidden evaluator scores.  ``evaluate_every_round``
            # fills it at every round instead of only the last, which lets
            # ``first_rounds`` be measured -- at the cost of admitting
            # depth-``t`` hypotheses, so it changes recorded evaluation
            # fields and is off by default.
            if round_number == self.rounds or self.config.evaluate_every_round:
                for node in agent.frontier:
                    agent.complete_candidates.add(self.encoding.components(node))

        log["total_proposed_pairs"] = sum(proposed_sizes)
        log["mean_preexperiment_frontier"] = statistics.mean(preown_sizes)
        log["median_preexperiment_frontier"] = statistics.median(preown_sizes)
        log["mean_unique_proposed_pairs"] = statistics.mean(proposed_sizes)
        log["median_unique_proposed_pairs"] = statistics.median(proposed_sizes)
        log["effective_branching"] = (
            log["unique_children"] / log["parents_attempted"] if log["parents_attempted"] else 0.0
        )

    # -- stage 3 ------------------------------------------------------------
    def _candidates_for(self, agent_index: int) -> list[ExperimentCandidate]:
        agent = self.agents[agent_index]
        surviving = None
        if self.config.candidate_pool == "surviving_leaf_v1":
            surviving = set()
            for node in agent.frontier:
                surviving.update(self.encoding.components(node))
        result = []
        for pair in sorted(agent.proposed_parents):
            if agent.evidence.knows(pair) or not agent.evidence.value_allowed(pair):
                continue
            if surviving is not None and pair not in surviving:
                continue
            parents = agent.proposed_parents[pair]
            result.append(
                ExperimentCandidate(
                    pair=pair,
                    parent_ids=tuple(
                        sorted({self.encoding.hypothesis(node).hid for node in parents})
                    ),
                )
            )
        return result

    async def _experiments(self, round_number: int, log: dict) -> None:
        capacity = [len(self._candidates_for(index)) for index in range(self.agents_count)]
        log["total_candidate_capacity"] = sum(capacity)
        log["experiment_active_agents"] = sum(bool(value) for value in capacity)
        log["total_candidate_capacity_start"] = sum(capacity)
        log["N_experiment_active_start"] = sum(bool(value) for value in capacity)
        log["N_frontier_active"] = sum(bool(agent.frontier) for agent in self.agents)
        action_sets: list[set[Pair]] = [set() for _ in range(self.agents_count)]

        async def perform_one(
            agent_index: int, candidates: list[ExperimentCandidate], ordinal: int
        ) -> None:
            agent = self.agents[agent_index]
            view = self._view(
                agent,
                "experiment",
                round_number,
                candidates=tuple(candidates),
                parents=agent.round_parents,
            )
            context = DecisionContext(
                run_id=self.run_id,
                episode_id=self.episode_id,
                agent_index=agent_index,
                round_number=round_number,
                phase="experiment",
                seed=self.seed,
                slot=ordinal,
            )
            decision = await self.policies[agent_index].choose_experiment(
                view, tuple(candidates), context
            )
            validate_experiment_decision(
                decision,
                candidates,
                agent.proposed_parents,
                self.environment.experiment_library(),
                self.encoding,
            )
            parent = next(
                (
                    node
                    for node in agent.proposed_parents.get(decision.pair, ())
                    if self.encoding.hypothesis(node).hid == decision.parent_id
                ),
                None,
            )
            parent_components = (
                self.encoding.components(parent) if parent is not None else ()
            )
            # A selection rule the paper states as part of the protocol, not
            # of the agent's reasoning, may override the chosen condition.
            # Every environment returns None unless a knob says otherwise.
            override = self.environment.select_experiment(
                pair=decision.pair,
                parent_components=parent_components,
                frontier_components=tuple(
                    self.encoding.components(node) for node in agent.frontier
                ),
                seed=self.seed,
                round_number=round_number,
                agent_index=agent_index,
                slot=ordinal,
            )
            if override is not None and override != decision.experiment_id:
                decision = replace(decision, experiment_id=override)
                log["protocol_selected_experiments"] += 1
            event_id = self._event(
                "experiment",
                {
                    "episode": self.episode_id,
                    "round": round_number,
                    "agent": agent_index,
                    "slot": ordinal,
                    "pair": list(decision.pair),
                    "parent_id": decision.parent_id,
                    "experiment_id": decision.experiment_id,
                    "reason": decision.reason,
                },
            )
            outcome = await self.environment.execute(
                pair=decision.pair,
                experiment_id=decision.experiment_id,
                owner=agent_index,
                round_number=round_number,
                event_id=event_id,
                parent_components=parent_components,
            )
            # Kept for the lifetime of the episode: this is how a claim's
            # identity is resolved and checked against the truth later,
            # whether or not the owner ever reports it, and however the
            # neighbour that receives a claim eventually treats it.
            self._event_truth[event_id] = outcome.evidence
            taken = True
            if self.q_own < 1.0:
                rng = derived_rng(
                    self.seed, Q_OWN_DOMAIN, round_number, agent_index, ordinal
                )
                taken = _accepts(rng, self.q_own)
            if taken:
                before = len(agent.frontier)
                agent.evidence.apply(outcome.evidence, own=True)
                (
                    agent.verified_positive
                    if outcome.evidence.positive
                    else agent.verified_negative
                ).add(outcome.evidence.pair)
                if self.llm_prune:
                    agent.pending_own.append(_label(outcome.evidence, own=True))
                else:
                    agent.frontier = agent.evidence.prune(agent.frontier)
                log["own_pruned_hypotheses"] += before - len(agent.frontier)
                agent.round_events.append(outcome.evidence)
            else:
                log["own_missed"] += 1
            agent.round_tests.append((decision.pair, outcome.evidence.outcome))
            if outcome.observation is not None:
                agent.observations.append(outcome.observation)
            if not self.free_communication:
                # The experiment happened whether or not the agent read its
                # own result, so the true record still goes to the
                # neighbours.  The alternative -- an agent that cannot report
                # what it ran -- would confound q_model with a communication
                # failure.  Under communication_mode == "free" this automatic,
                # always-truthful channel does not run at all: reporting is
                # the agent's own, possibly inaccurate, choice instead (see
                # ``_communicate``).
                self.router.stage(agent_index, outcome.evidence)
            action_sets[agent_index].add(decision.pair)
            log["tests"] += 1
            log["tests_pos" if outcome.evidence.positive else "tests_neg"] += 1

        allocation = await randomized_round_robin_allocate_async(
            n_agents=self.agents_count,
            total_budget=self.agents_count * self.config.experiments_per_agent,
            seed=self.seed,
            round_number=round_number,
            candidates_for=self._candidates_for,
            perform_one=perform_one,
            per_agent_cap=self.per_agent_cap,
        )
        per_agent = list(allocation["experiments_per_agent"])
        log.update(allocation)
        assert int(log["B_used"]) == int(log["tests"])
        log["final_candidate_capacity"] = int(allocation["total_candidate_capacity_end"])
        log["N_frontier_active_end"] = sum(bool(agent.frontier) for agent in self.agents)
        log["agents_exceeding_nominal_m"] = sum(
            count > self.config.experiments_per_agent for count in per_agent
        )
        log["budget_redistributed_above_m"] = sum(
            max(0, count - self.config.experiments_per_agent) for count in per_agent
        )
        # Only meaningful where a tested component leaves the candidate pool;
        # with configuration-level verdicts it does not, so capacity is not
        # expected to fall by the number of tests run.
        exhaustible = getattr(self.agents[0].evidence, "consumes_candidates", True)
        log["candidate_invalidated_without_test"] = (
            int(log["total_candidate_capacity"])
            - int(log["B_used"])
            - int(log["final_candidate_capacity"])
            if exhaustible
            else None
        )
        log["unique_population_experiments"] = len(set().union(*action_sets)) if action_sets else 0
        if int(log["B_used"]) > self.agents_count * self.config.experiments_per_agent:
            self.audit["budget_overflow"] += 1
        if exhaustible:
            # A missed own result leaves its candidate in the pool although
            # the budget was spent, so the identity only holds up to the
            # number of misses.
            assert int(log["candidate_invalidated_without_test"]) >= -int(log["own_missed"])

    # -- stage 3b: free communication -----------------------------------------
    async def _communicate(self, round_number: int, log: dict) -> None:
        """One extra, optional LLM call per agent with a neighbour.

        Never called in the final round (a message sent there would arrive
        after the episode ends) and never called for a degree-0 agent (it has
        no one to send to).  A policy failure that survives repair does not
        abort the episode here: the agent simply sends nothing this round,
        which is recorded, not raised.
        """
        # Counts every *eligible* agent (has at least one neighbour -- this
        # round is already guaranteed not the final round by the caller) that
        # ends this round having sent no message, for any reason: chose to
        # send nothing, or the call failed after repair.  A degree-0 agent
        # (no neighbour to send to) is never eligible and is not counted
        # here at all.  Decremented below for each eligible agent that
        # actually stages a message.
        log["silent_agents"] = sum(
            1 for agent in self.agents if self.router.neighbours(agent.index)
        )

        async def one(agent: AgentRuntime) -> None:
            if not self.router.neighbours(agent.index):
                return
            own = tuple(
                OwnExperimentSummary(
                    event_id=record.event_id,
                    pair=record.pair,
                    outcome=record.outcome,
                    experiment_id=record.experiment_id,
                    configuration=record.configuration,
                )
                for record in agent.round_events
            )
            view = self._view(agent, "communication", round_number, own_experiments=own)
            context = DecisionContext(
                run_id=self.run_id,
                episode_id=self.episode_id,
                agent_index=agent.index,
                round_number=round_number,
                phase="communication",
                seed=self.seed,
            )
            try:
                decision = await self.policies[agent.index].communicate(view, context)
            except PolicyFailure:
                log["comm_failed"] += 1
                return
            log["comm_calls"] += 1
            events_by_id = {record.event_id: record for record in agent.round_events}
            reports: list[ClaimReport] = []
            seen: set[str] = set()
            for item in decision.reports:
                truth = events_by_id.get(item.event_id)
                if truth is None or item.event_id in seen:
                    continue
                seen.add(item.event_id)
                reports.append(
                    ClaimReport(event_id=item.event_id, claimed_outcome=item.claimed_outcome)
                )
            text = (decision.text or "").strip()[:500]
            if not reports and not text:
                return
            log["outgoing_reports"] += len(reports)
            log["outgoing_false_reports"] += sum(
                1
                for item in reports
                if item.claimed_outcome != events_by_id[item.event_id].outcome
            )
            log["outgoing_messages"] += 1
            log["silent_agents"] -= 1
            self.router.stage_message(
                agent.index,
                CommunicationMessage(
                    sender=agent.index,
                    round_number=round_number,
                    reports=tuple(reports),
                    text=text,
                ),
            )

        await asyncio.gather(*(one(agent) for agent in self.agents))

    # -- stage 4 ------------------------------------------------------------
    def _solved(self, agent: AgentRuntime) -> bool:
        """Whether the agent can itself certify that it holds the target.

        Deliberately decidable from the agent's own evidence, never from the
        hidden truth.  The condition mirrors the hidden evaluator's
        ``evidence_verified``: the agent must actually *hold* a depth-``T``
        candidate -- in its frontier, or in the pool the evaluator scores --
        every component of which carries a positive label.  Under the
        subset-level oracle ``covers_positively`` additionally requires a
        failing configuration that contains them and no passing one.

        Holding ``T`` positive labels is not enough and used to be the test:
        the labels are known one round before a depth-``T`` hypothesis can
        exist, so the episode was stopped a round early and the evaluator,
        scoring a frontier of depth-``(T-1)`` hypotheses, reported
        ``P_succ = 0`` for a run that had in fact just succeeded.
        """
        depth = self.config.task_size
        candidates = {
            components
            for components in agent.complete_candidates
            if len(components) == depth
        }
        candidates.update(
            self.encoding.components(node)
            for node in agent.frontier
            if node.used.bit_count() == depth
        )
        return any(
            agent.evidence.covers_positively(components) for components in candidates
        )

    def _close_round(self, round_number: int, log: dict) -> None:
        for agent in self.agents:
            agent.history.append(
                RoundMemory(
                    round_number=round_number,
                    tested=tuple(pair for pair, _ in agent.round_tests),
                    outcomes=tuple(outcome for _, outcome in agent.round_tests),
                )
            )
        # Measured q_model: the share of the prunes exact reasoning would have
        # made that the model actually made.  None when nothing was refuted.
        log["q_model_measured"] = (
            log["prune_hit"] / log["prune_should"] if log["prune_should"] else None
        )
        post_sizes = [len(agent.frontier) for agent in self.agents]
        truth_counts = [
            sum(not node.wrong for node in agent.frontier) for agent in self.agents
        ]
        log["old_post"] = sum(
            node.wrong and node.from_wrong for agent in self.agents for node in agent.frontier
        )
        log["source_post"] = sum(
            node.wrong and not node.from_wrong for agent in self.agents for node in agent.frontier
        )
        log["A_post"] = int(log["old_post"]) + int(log["source_post"])
        log["A_preown"] = int(log["old_pre"]) + int(log["source_pre"])
        truth_components = tuple(sorted(self.task.truth_pairs))
        for agent in self.agents:
            if any(
                self.encoding.components(node) == truth_components for node in agent.frontier
            ):
                agent.reached_truth = True
            if agent.solved_round is None and self._solved(agent):
                agent.solved_round = round_number
        log["agents_solved"] = sum(
            agent.solved_round is not None and agent.solved_round <= round_number
            for agent in self.agents
        )
        log["any_agent_solved"] = bool(log["agents_solved"])
        log["frontier_nonempty"] = sum(size > 0 for size in post_sizes)
        log["frontier_empty"] = self.agents_count - int(log["frontier_nonempty"])
        log["frontier_empty_fraction"] = log["frontier_empty"] / self.agents_count
        log["active_agent_fraction"] = log["frontier_nonempty"] / self.agents_count
        log["mean_frontier_size"] = statistics.mean(post_sizes)
        log["median_frontier_size"] = statistics.median(post_sizes)
        log["truth_compatible_hypotheses"] = sum(truth_counts)
        log["truth_compatible_agents"] = sum(count > 0 for count in truth_counts)
        nominal = self.agents_count * self.config.experiments_per_agent
        log["budget_utilization"] = log["B_used"] / nominal if nominal else 0.0
        log["message_availability"] = (
            log["sender_edges_with_message"] / (self.agents_count * self.degree)
            if self.degree
            else 0.0
        )
        log["outgoing_agents"] = self.router.pending_outgoing_agents()
        log["outgoing_message_availability"] = log["outgoing_agents"] / self.agents_count
        log["candidate_capacity_ratio"] = (
            log["total_candidate_capacity"] / nominal if nominal else 0.0
        )
        log["duplicate_evidence_fraction"] = (
            log["recv_duplicate"] / log["recv_messages"] if log["recv_messages"] else 0.0
        )
        log["already_known_evidence_fraction"] = (
            (log["recv_already_explicit"] + log["recv_already_implied"]) / log["recv_unique"]
            if log["recv_unique"]
            else 0.0
        )

    # -- output --------------------------------------------------------------
    def _finalise(self, transitions: list[dict]) -> dict:
        graph_digest = hashlib.sha256(repr(self.router.graph).encode()).hexdigest()[:20]
        usage: dict[str, float] = {}
        for policy in self.policies:
            for key, value in policy.usage().items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        return {
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "policy": self.policies[0].kind if self.policies else "none",
            "environment": self.environment.kind,
            "candidate_pool": self.config.candidate_pool,
            "seed": self.seed,
            "N": self.agents_count,
            "M": self.encoding.families,
            "K": self.encoding.variants,
            "task_size": self.config.task_size,
            "rounds": self.rounds,
            "b": self.config.branching,
            "m": self.config.experiments_per_agent,
            "n0": self.config.initial_width,
            "w": 0,
            "d": self.degree,
            # Knob settings, recorded so a run is readable without its config.
            "q_model": self.config.q_model,
            "q_model_own": self.q_own,
            "q_model_received": self.q_received,
            "q_model_draw": self.config.q_model_draw,
            "budget_mode": self.config.budget_mode,
            "stop_on_success": self.config.stop_on_success,
            "evaluate_every_round": self.config.evaluate_every_round,
            "respawn": self.config.respawn,
            "llm_prune": self.llm_prune,
            "llm_prune_apply": self.prune_apply if self.llm_prune else None,
            "prompt_sha256": (
                prompt_fingerprint()
                if self.config.policy.kind.startswith("llm_")
                else None
            ),
            "topology": self.config.topology,
            "topology_rewire_p": self.config.topology_rewire_p,
            "realised_mean_degree": self.router.mean_degree,
            "tcas_selection": self.config.environment.tcas_selection,
            "rounds_executed": len(transitions),
            "stopped_round": self.stopped_round,
            "agents_solved": sum(
                agent.solved_round is not None for agent in self.agents
            ),
            "first_solved_round": min(
                (
                    agent.solved_round
                    for agent in self.agents
                    if agent.solved_round is not None
                ),
                default=None,
            ),
            "truth": list(self.task.truth_values),
            "truth_pairs": [list(pair) for pair in self.task.truth_pairs],
            "graph_digest": graph_digest,
            "initialization_digest": hashlib.sha256(
                repr([tuple(self.encoding.index(p) for p in group) for group in self.task.initial_pairs]).encode()
            ).hexdigest()[:20],
            "A_initial_incorrect": self.task.initial_incorrect(),
            "transitions": transitions,
            "agents_reached_truth": sum(agent.reached_truth for agent in self.agents),
            "any_agent_reached_truth": any(agent.reached_truth for agent in self.agents),
            "audit": dict(self.audit),
            "policy_usage": usage,
            "environment_diagnostics": self.environment.diagnostics(),
            "evaluation": (
                self.evaluator.final_report(
                    self.agents, self.stopped_round or self.rounds
                )
                if self.evaluator is not None
                else None
            ),
        }

    # -- state exposed to the evaluator ---------------------------------------
    def agent_states(self) -> list[AgentRuntime]:
        return self.agents


# --------------------------------------------------------------------------
# Action validation (plan section 5.3)
# --------------------------------------------------------------------------


def validate_branch_decision(
    decision: BranchDecision, slots: Sequence[ParentSlot]
) -> list[tuple[ParentSlot, tuple[Pair, ...]]]:
    by_id = {slot.hypothesis.hid: slot for slot in slots}
    if len(decision.updates) != len(slots):
        raise ActionError(
            f"expected exactly {len(slots)} parent updates, got {len(decision.updates)}"
        )
    seen: set[str] = set()
    ordered: list[tuple[ParentSlot, tuple[Pair, ...]]] = []
    for update in decision.updates:
        slot = by_id.get(update.parent_id)
        if slot is None:
            raise ActionError(f"unknown parent_id {update.parent_id!r}")
        if update.parent_id in seen:
            raise ActionError(f"duplicate update for parent {update.parent_id!r}")
        seen.add(update.parent_id)
        additions = tuple(update.additions)
        if len(additions) != slot.required:
            raise ActionError(
                f"parent {update.parent_id!r} needs exactly {slot.required} additions, "
                f"got {len(additions)}"
            )
        if len(set(additions)) != len(additions):
            raise ActionError(f"parent {update.parent_id!r} repeats an addition")
        legal = set(slot.legal_additions)
        for pair in additions:
            if pair not in legal:
                raise ActionError(
                    f"addition {pair} is not a legal extension of {update.parent_id!r}"
                )
        ordered.append((slot, additions))
    # Apply in the frontier's canonical order, not the order the policy replied in.
    index = {slot.hypothesis.hid: position for position, slot in enumerate(slots)}
    ordered.sort(key=lambda item: index[item[0].hypothesis.hid])
    return ordered


def validate_experiment_decision(
    decision: ExperimentDecision,
    candidates: Sequence[ExperimentCandidate],
    proposed_parents: dict[Pair, list[Node]],
    library: Sequence,
    encoding: Encoding,
) -> None:
    by_pair = {item.pair: item for item in candidates}
    candidate = by_pair.get(decision.pair)
    if candidate is None:
        raise ActionError(f"{decision.pair} is not a testable candidate this slot")
    if decision.parent_id not in candidate.parent_ids:
        raise ActionError(
            f"parent {decision.parent_id!r} did not propose {decision.pair} this round"
        )
    if library:
        allowed = {condition.experiment_id for condition in library}
        if decision.experiment_id not in allowed:
            raise ActionError(
                f"experiment_id {decision.experiment_id!r} is not in the public library"
            )
    elif decision.experiment_id is not None:
        raise ActionError("this environment has no experiment library")
