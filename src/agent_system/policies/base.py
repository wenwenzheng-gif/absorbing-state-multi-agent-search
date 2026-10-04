"""The policy protocol shared by rule, LLM and replay decision makers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..schemas import (
    AgentView,
    BranchDecision,
    ExperimentCandidate,
    ExperimentCondition,
    ExperimentDecision,
    Pair,
    ParentSlot,
)

Predictor = Callable[[Sequence[Pair], ExperimentCondition], "np.ndarray | None"]

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = ("branch.txt", "experiment.txt", "prune.txt", "communication.txt")


def prompt_fingerprint() -> str:
    """SHA-256 over the prompt templates, recorded with every LLM episode."""
    digest = hashlib.sha256()
    for name in PROMPT_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((PROMPT_DIR / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class DecisionContext:
    """Scheduler metadata.

    This is deliberately *not* part of :class:`AgentView`: it carries the seed
    that rule policies need to reproduce the reference random stream and the
    identity fields that make an LLM response cacheable and replayable.  It is
    never serialised into a prompt.
    """

    run_id: str
    episode_id: str
    agent_index: int
    round_number: int
    phase: str
    seed: int
    slot: int = 0
    batch: int = 0
    attempt: int = 0


class Policy(ABC):
    kind: str = "abstract"
    uses_llm: bool = False
    # Whether the runner must hand its two pruning points to this policy
    # instead of doing the exact pruning itself.  Only ``llm_prune`` does.
    prunes: bool = False

    def __init__(self) -> None:
        self.predictor: Predictor | None = None

    def bind_predictor(self, predictor: Predictor | None) -> None:
        """Give the policy a way to simulate its own hypotheses (free compute)."""
        self.predictor = predictor

    @abstractmethod
    async def propose(
        self,
        view: AgentView,
        parents: Sequence[ParentSlot],
        context: DecisionContext,
    ) -> BranchDecision:
        ...

    @abstractmethod
    async def choose_experiment(
        self,
        view: AgentView,
        candidates: Sequence[ExperimentCandidate],
        context: DecisionContext,
    ) -> ExperimentDecision:
        ...

    def usage(self) -> dict[str, object]:
        return {}

    async def aclose(self) -> None:
        return None
