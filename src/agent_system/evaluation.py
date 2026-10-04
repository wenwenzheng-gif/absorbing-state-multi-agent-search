"""Hidden, read-only evaluation of what agents actually achieved.

The evaluator may read the hidden mechanism and the hold-out set; it never
returns anything to an agent and never influences the search.  The four
criteria are reported separately because they answer different questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .environments.base import Environment, TaskInstance
from .hypotheses import Encoding
from .schemas import Pair

CRITERIA = (
    "symbolic_recovered",
    "evidence_verified",
    "predictive_success",
    "verified_predictive_success",
)


class _TruthfulEvidence:
    """A ``covers_positively`` surface built only from genuinely true labels.

    Used in place of ``agent.evidence`` when ``communication_mode == "free"``:
    under that mode ``agent.evidence`` may hold a claim the agent believed but
    that was in fact false, and scoring must never credit that.  ``agent``
    keeps its own ``verified_positive`` / ``verified_negative`` sets, filled
    by :class:`~src.agent_system.runner.EpisodeRunner` only from its own
    experiments, truthfully-delivered evidence, and claims that happened to
    match the real outcome -- never from a false claim.  This object is read
    only here, at scoring time, and never reaches a policy or a prompt.
    """

    __slots__ = ("_agent",)

    def __init__(self, agent) -> None:
        self._agent = agent

    def covers_positively(self, components) -> bool:
        positive = self._agent.verified_positive
        return all(pair in positive for pair in components)


@dataclass
class EvaluationState:
    first_rounds: dict[str, int | None] = field(
        default_factory=lambda: {name: None for name in CRITERIA}
    )
    per_round: list[dict] = field(default_factory=list)


class HiddenEvaluator:
    def __init__(
        self,
        *,
        task: TaskInstance,
        environment: Environment,
        encoding: Encoding,
        epsilon: float,
        truthful_scoring: bool = False,
    ) -> None:
        self.task = task
        self.environment = environment
        self.encoding = encoding
        self.epsilon = epsilon
        # Set only for communication_mode == "free" runs: see
        # ``_TruthfulEvidence``.  Every other run keeps scoring directly off
        # ``agent.evidence``, unchanged.
        self.truthful_scoring = truthful_scoring
        self.truth = tuple(sorted(task.truth_pairs))
        self.state = EvaluationState()
        self._error_cache: dict[tuple[Pair, ...], float] = {}
        self.evaluated_candidates = 0

    def heldout_error(self, components: Sequence[Pair]) -> float:
        key = tuple(sorted(components))
        cached = self._error_cache.get(key)
        if cached is None:
            cached = self.environment.heldout_error(key)
            self._error_cache[key] = cached
            self.evaluated_candidates += 1
        return cached

    def evaluate(self, agents, round_number: int) -> dict:
        """Score every complete candidate each agent has actually generated."""
        predictive_capable = self.environment.scores_heldout()
        agent_rows: list[dict] = []
        errors: list[float] = []
        for agent in agents:
            source = _TruthfulEvidence(agent) if self.truthful_scoring else agent.evidence
            complete = sorted(agent.complete_candidates)
            symbolic = any(candidate == self.truth for candidate in complete)
            verified = any(
                source.covers_positively(candidate) for candidate in complete
            )
            predictive = False
            verified_predictive = False
            best_error = math.inf
            if predictive_capable:
                for candidate in complete:
                    error = self.heldout_error(candidate)
                    if math.isfinite(error):
                        errors.append(error)
                        best_error = min(best_error, error)
                    if error < self.epsilon:
                        predictive = True
                        if source.covers_positively(candidate):
                            verified_predictive = True
            else:
                predictive = symbolic
                verified_predictive = symbolic and verified
            agent_rows.append(
                {
                    "agent": agent.index,
                    "complete_candidates": len(complete),
                    "symbolic_recovered": symbolic,
                    "evidence_verified": verified,
                    "predictive_success": predictive,
                    "verified_predictive_success": verified_predictive,
                    "best_heldout_error": None if math.isinf(best_error) else best_error,
                }
            )
        summary = {
            "round": round_number,
            "predictive_metric": "heldout" if predictive_capable else "symbolic_proxy",
            "epsilon_predictive": self.epsilon,
        }
        for name in CRITERIA:
            achieved = [row[name] for row in agent_rows]
            summary[f"agents_{name}"] = sum(achieved)
            summary[f"any_{name}"] = any(achieved)
            if any(achieved) and self.state.first_rounds[name] is None:
                self.state.first_rounds[name] = round_number
        summary["mean_complete_candidates"] = (
            sum(row["complete_candidates"] for row in agent_rows) / len(agent_rows)
            if agent_rows
            else 0.0
        )
        summary["heldout_error_samples"] = len(errors)
        summary["heldout_error_min"] = min(errors) if errors else None
        summary["heldout_error_median"] = (
            sorted(errors)[len(errors) // 2] if errors else None
        )
        self.state.per_round.append(summary)
        return {"summary": summary, "agents": agent_rows}

    def final_report(self, agents, round_number: int) -> dict:
        detail = self.evaluate(agents, round_number)
        report = dict(detail["summary"])
        report["agents_detail"] = detail["agents"]
        report["first_rounds"] = dict(self.state.first_rounds)
        report["evaluated_candidates"] = self.evaluated_candidates
        report["truth_components"] = [list(pair) for pair in self.truth]
        return report
