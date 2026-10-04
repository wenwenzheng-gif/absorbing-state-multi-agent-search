"""Observation-driven adaptive rule policy.

Branch weights come from the improvement each candidate term makes to the
agent's own trajectory-prediction loss.  Nothing here may touch the hidden
vector field, true acceleration labels or hold-out data.

``beta = 0`` delegates to :class:`UniformPolicy` so the two conditions share
one random stream and align step by step, rather than merely matching in
distribution.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...scaling_model import derived_rng
from ..hypotheses import pack_key
from ..schemas import (
    AgentView,
    BranchDecision,
    ExperimentCondition,
    Pair,
    ParentSlot,
    ParentUpdate,
    PrivateObservation,
)
from .base import DecisionContext
from .uniform import UniformPolicy

ADAPTIVE_DOMAIN = 0xC3A5C85C97CB3127
EPSILON_STD = 1e-8


class AdaptivePolicy(UniformPolicy):
    kind = "adaptive"

    def __init__(
        self,
        *,
        beta: float = 1.0,
        scale_x: float = 1.0,
        scale_v: float = 1.0,
        planner: str = "disagreement_v1",
        planner_max_conditions: int | None = None,
        planner_max_parents: int | None = None,
    ) -> None:
        super().__init__(
            planner=planner,
            planner_max_conditions=planner_max_conditions,
            planner_max_parents=planner_max_parents,
        )
        self.beta = float(beta)
        self.scale_x = float(scale_x)
        self.scale_v = float(scale_v)
        self.scoring_failures = 0
        self.scored_parents = 0
        self.failed_parents = 0

    async def propose(
        self,
        view: AgentView,
        parents: Sequence[ParentSlot],
        context: DecisionContext,
    ) -> BranchDecision:
        if self.beta == 0.0:
            return await super().propose(view, parents, context)
        if self.predictor is None or not view.private_observations:
            raise ValueError(
                "the adaptive policy needs a predictor and private observations; "
                "use beta=0 or the uniform policy for the synthetic environment"
            )
        updates: list[ParentUpdate] = []
        for slot in parents:
            weights = self._weights(view, slot)
            rng = derived_rng(
                context.seed,
                ADAPTIVE_DOMAIN,
                context.agent_index,
                view.round_number,
                pack_key(slot.hypothesis.components, view.variants.bit_length()),
            )
            additions = _sample_without_replacement(
                rng, list(slot.legal_additions), weights, slot.required
            )
            updates.append(
                ParentUpdate(parent_id=slot.hypothesis.hid, additions=tuple(additions))
            )
        return BranchDecision(
            updates=tuple(updates), reason=f"softmax over local loss improvement, beta={self.beta}"
        )

    # -- scoring ----------------------------------------------------------
    def _weights(self, view: AgentView, slot: ParentSlot) -> list[float]:
        base = self._loss(view, slot.hypothesis.components)
        scores: list[float] = []
        valid: list[float] = []
        for addition in slot.legal_additions:
            extended = self._loss(view, tuple(slot.hypothesis.components) + (addition,))
            if base is None or extended is None or not math.isfinite(extended):
                self.scoring_failures += 1
                scores.append(math.nan)
                continue
            score = base - extended
            scores.append(score)
            valid.append(score)
        self.scored_parents += 1
        if not valid:
            self.failed_parents += 1
            return [1.0] * len(scores)
        mean = float(np.mean(valid))
        std = max(float(np.std(valid)), EPSILON_STD)
        exponents = [
            self.beta * (score - mean) / std if math.isfinite(score) else -math.inf
            for score in scores
        ]
        peak = max(value for value in exponents if math.isfinite(value))
        return [math.exp(value - peak) if math.isfinite(value) else 0.0 for value in exponents]

    def _loss(self, view: AgentView, components: Sequence[Pair]) -> float | None:
        assert self.predictor is not None
        total = 0.0
        used = 0
        for observation in view.private_observations:
            condition = _condition_of(observation)
            predicted = self.predictor(tuple(sorted(components)), condition)
            if predicted is None:
                return None
            observed = np.asarray(observation.states, dtype=np.float64)
            if predicted.shape[0] != observed.shape[0]:
                return None
            dx = (predicted[:, 1] - observed[:, 0]) / self.scale_x
            dv = (predicted[:, 2] - observed[:, 1]) / self.scale_v
            value = float(np.mean(dx * dx + dv * dv))
            if not math.isfinite(value):
                return None
            total += value
            used += 1
        return total / used if used else None

    def usage(self) -> dict[str, object]:
        return {
            "adaptive_scoring_failures": self.scoring_failures,
            "adaptive_scored_parents": self.scored_parents,
            "adaptive_failed_parents": self.failed_parents,
        }


def _condition_of(observation: PrivateObservation) -> ExperimentCondition:
    return ExperimentCondition(
        experiment_id=observation.experiment_id or observation.observation_id,
        description="reconstructed from a private observation",
        settings=dict(observation.settings),
    )


def _sample_without_replacement(rng, items: list, weights: list[float], draws: int) -> list:
    """Sequential weight-proportional sampling with a deterministic fallback."""
    remaining = list(range(len(items)))
    current = list(weights)
    chosen: list = []
    while remaining and len(chosen) < draws:
        total = sum(current[index] for index in remaining)
        if total <= 0.0:
            position = rng.below(len(remaining))
        else:
            target = (rng.next() / float(1 << 64)) * total
            cumulative = 0.0
            position = len(remaining) - 1
            for offset, index in enumerate(remaining):
                cumulative += current[index]
                if target < cumulative:
                    position = offset
                    break
        chosen.append(items[remaining[position]])
        remaining.pop(position)
    return chosen
