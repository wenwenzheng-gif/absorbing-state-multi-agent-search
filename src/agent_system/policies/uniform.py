"""Reference uniform policy.

Consumes exactly the random draws that :func:`src.scaling_model.run_episode`
consumes, in the same order, so a synthetic episode reproduces the frozen
trajectory step for step.
"""

from __future__ import annotations

from typing import Sequence

from ...scaling_model import ACTION_DOMAIN, derived_rng
from ..hypotheses import pack_key, uniform_branch_sample
from ..schemas import (
    AgentView,
    BranchDecision,
    ExperimentCandidate,
    ExperimentDecision,
    ParentSlot,
    ParentUpdate,
)
from .base import DecisionContext, Policy
from .planner import plan_experiment


class UniformPolicy(Policy):
    kind = "uniform"

    def __init__(
        self,
        *,
        planner: str = "disagreement_v1",
        planner_max_conditions: int | None = None,
        planner_max_parents: int | None = None,
    ) -> None:
        super().__init__()
        self.planner = planner
        self.planner_max_conditions = planner_max_conditions
        self.planner_max_parents = planner_max_parents

    async def propose(
        self,
        view: AgentView,
        parents: Sequence[ParentSlot],
        context: DecisionContext,
    ) -> BranchDecision:
        bits = view.variants.bit_length()
        updates = []
        for slot in parents:
            key = pack_key(slot.hypothesis.components, bits)
            key_node = _KeyOnly(key)
            additions = uniform_branch_sample(
                context.seed,
                context.agent_index,
                view.round_number,
                key_node,
                list(slot.legal_additions),
                slot.required,
            )
            updates.append(ParentUpdate(parent_id=slot.hypothesis.hid, additions=tuple(additions)))
        return BranchDecision(updates=tuple(updates))

    async def choose_experiment(
        self,
        view: AgentView,
        candidates: Sequence[ExperimentCandidate],
        context: DecisionContext,
    ) -> ExperimentDecision:
        rng = derived_rng(
            context.seed, ACTION_DOMAIN, view.round_number, context.agent_index, context.slot
        )
        chosen = candidates[rng.below(len(candidates))]
        return ExperimentDecision(
            pair=chosen.pair,
            parent_id=chosen.parent_ids[0],
            experiment_id=self._plan(view, chosen, context),
            reason="uniform pair draw from the reference action stream",
        )

    def _plan(
        self, view: AgentView, candidate: ExperimentCandidate, context: DecisionContext
    ) -> str | None:
        if not view.experiment_library:
            return None
        parents = [
            slot.hypothesis for slot in view.parents if slot.hypothesis.hid in candidate.parent_ids
        ]
        return plan_experiment(
            planner=self.planner,
            pair=candidate.pair,
            parents=parents,
            library=view.experiment_library,
            predictor=self.predictor,
            seed=context.seed,
            round_number=view.round_number,
            agent_index=context.agent_index,
            slot=context.slot,
            max_conditions=self.planner_max_conditions,
            max_parents=self.planner_max_parents,
        )


class _KeyOnly:
    """Minimal stand-in exposing the ``key`` field the branch stream labels on."""

    __slots__ = ("key",)

    def __init__(self, key: int) -> None:
        self.key = key
