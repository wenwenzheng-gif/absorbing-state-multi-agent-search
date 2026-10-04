"""Rule-based physics experiment-condition planners.

A planner never calls the hidden environment: it compares the predictions of
an agent's own hypotheses and picks the public condition that separates them
most.  All non-LLM conditions share this planner so the comparison isolates
branch and component choice.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...scaling_model import derived_rng
from ..schemas import ExperimentCondition, Hypothesis, Pair
from .base import Predictor

PLANNER_DOMAIN = 0x1BD11BDAA9FC1A22


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = left[:, 1:] - right[:, 1:]
    return float(np.mean(difference * difference) * 2.0)


def plan_experiment(
    *,
    planner: str,
    pair: Pair,
    parents: Sequence[Hypothesis],
    library: Sequence[ExperimentCondition],
    predictor: Predictor | None,
    seed: int,
    round_number: int,
    agent_index: int,
    slot: int,
    max_conditions: int | None = None,
    max_parents: int | None = None,
) -> str | None:
    """Return the chosen ``experiment_id`` or ``None`` when there is no library."""
    if not library:
        return None
    if planner == "fixed_first":
        return library[0].experiment_id
    if planner == "uniform":
        rng = derived_rng(seed, PLANNER_DOMAIN, round_number, agent_index, slot)
        return library[rng.below(len(library))].experiment_id
    if planner != "disagreement_v1":
        raise ValueError(f"unknown planner {planner!r}")
    if predictor is None or not parents:
        return library[0].experiment_id

    considered = sorted(parents, key=lambda item: item.hid)
    if max_parents is not None:
        considered = considered[:max_parents]
    conditions = list(library)
    if max_conditions is not None and max_conditions < len(conditions):
        step = len(conditions) / max_conditions
        conditions = [conditions[int(index * step)] for index in range(max_conditions)]

    best_id = conditions[0].experiment_id
    best_score = -1.0
    for condition in conditions:
        score = 0.0
        usable = False
        for parent in considered:
            base = predictor(parent.components, condition)
            extended = predictor(tuple(parent.components) + (pair,), condition)
            if base is None or extended is None:
                continue
            usable = True
            score += _distance(base, extended)
        if not usable:
            continue
        if score > best_score:
            best_score = score
            best_id = condition.experiment_id
    return best_id
