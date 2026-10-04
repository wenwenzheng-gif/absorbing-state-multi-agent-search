"""Explicit whitelist construction of :class:`AgentView`.

Only fields listed here can ever reach a policy.  Hidden truth, other agents'
state, unreceived evidence, hold-out data and seeds are structurally absent.

The team and history fields at the end of the whitelist are all things the
agent itself holds: its own index, how many agents there are, its own degree
and the ids of its own neighbours, its own budget for the round, the
provenance of the labels it was handed, what it itself did in earlier rounds,
and -- in the ``prune`` phase -- its own hypotheses and the labels that just
arrived.  Nothing here is derived from the hidden mechanism.
"""

from __future__ import annotations

from typing import Sequence

from .evidence import EvidenceStore
from .schemas import (
    AgentView,
    ComponentSpec,
    ExperimentCandidate,
    ExperimentCondition,
    Hypothesis,
    LabelRecord,
    OwnExperimentSummary,
    ParentSlot,
    PrivateObservation,
    ReceivedMessage,
    RoundMemory,
)

VIEW_FIELDS = (
    "phase",
    "round_number",
    "total_rounds",
    "task_size",
    "families",
    "variants",
    "branching",
    "component_library",
    "known_positive",
    "known_negative",
    "allowed_values",
    "parents",
    "candidates",
    "experiment_library",
    "private_observations",
    "environment_kind",
    "environment_notes",
    "agent_index",
    "agent_count",
    "degree",
    "neighbours",
    "round_budget",
    "own_labels",
    "received_labels",
    "own_history",
    "prune_hypotheses",
    "new_evidence",
    "own_experiments",
    "received_messages",
)


def label_records(evidence: EvidenceStore) -> tuple[
    tuple[LabelRecord, ...], tuple[LabelRecord, ...]
]:
    """Split the labels an agent holds into the ones it made and the ones it got.

    ``EvidenceStore.own`` already distinguishes the two; the sender of a
    received label is its producer, which is a direct neighbour because
    ``agent_v1`` never forwards evidence.
    """
    own: list[LabelRecord] = []
    received: list[LabelRecord] = []
    for pair, record in sorted(evidence.records.items()):
        is_own = pair in evidence.own
        item = LabelRecord(
            pair=pair,
            outcome=record.outcome,
            source="own" if is_own else "received",
            sender=None if is_own else record.origin_agent,
            round_number=record.created_round,
            configuration=record.configuration,
        )
        (own if is_own else received).append(item)
    return tuple(own), tuple(received)


def build_view(
    *,
    phase: str,
    round_number: int,
    total_rounds: int,
    task_size: int,
    families: int,
    variants: int,
    branching: int,
    component_library: Sequence[ComponentSpec],
    evidence: EvidenceStore,
    parents: Sequence[ParentSlot] = (),
    candidates: Sequence[ExperimentCandidate] = (),
    experiment_library: Sequence[ExperimentCondition] = (),
    private_observations: Sequence[PrivateObservation] = (),
    environment_kind: str = "synthetic",
    environment_notes: str = "",
    agent_index: int = -1,
    agent_count: int = 0,
    degree: int = 0,
    neighbours: Sequence[int] = (),
    round_budget: int = 0,
    own_history: Sequence[RoundMemory] = (),
    prune_hypotheses: Sequence[Hypothesis] = (),
    new_evidence: Sequence[LabelRecord] = (),
    own_experiments: Sequence[OwnExperimentSummary] = (),
    received_messages: Sequence[ReceivedMessage] = (),
) -> AgentView:
    own_labels, received_labels = label_records(evidence)
    return AgentView(
        phase=phase,  # type: ignore[arg-type]
        round_number=round_number,
        total_rounds=total_rounds,
        task_size=task_size,
        families=families,
        variants=variants,
        branching=branching,
        component_library=tuple(component_library),
        known_positive=evidence.known_positive(),
        known_negative=evidence.known_negative(),
        allowed_values=evidence.allowed_values(),
        parents=tuple(parents),
        candidates=tuple(candidates),
        experiment_library=tuple(experiment_library),
        private_observations=tuple(private_observations),
        environment_kind=environment_kind,
        environment_notes=environment_notes,
        agent_index=agent_index,
        agent_count=agent_count,
        degree=degree,
        neighbours=tuple(neighbours),
        round_budget=round_budget,
        own_labels=own_labels,
        received_labels=received_labels,
        own_history=tuple(own_history),
        prune_hypotheses=tuple(prune_hypotheses),
        new_evidence=tuple(new_evidence),
        own_experiments=tuple(own_experiments),
        received_messages=tuple(received_messages),
    )
