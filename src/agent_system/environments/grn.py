"""Gene regulatory network environment over DREAM4 In Silico Size 100.

A component is ``(regulator, sign)``.  One experiment is one gene
intervention: the agent knocks out a candidate regulator and learns whether
that regulator, with that sign, belongs to the target's upstream mechanism.
Evidence is therefore component level and reusable across search paths,
exactly as in the physics rule experiment.

There is no experiment bank: the intervention is fixed by the component being
tested, so no separate condition has to be chosen.  There is also no
predictive hold-out; see :mod:`.grn_dataset` for why one is not well posed on
this data, and :class:`~..evaluation.HiddenEvaluator` for the symbolic proxy
it falls back to.
"""

from __future__ import annotations

from typing import Sequence

from ..schemas import ComponentSpec, Pair, PrivateObservation
from .base import Environment, TaskInstance
from .grn_dataset import ACTIVATION, INHIBITION, GRNTask, Network, SIGN_NAMES

LIBRARY_VERSION = "dream4_insilico_size100_v1"


class GRNEnvironment(Environment):
    kind = "grn"

    def __init__(
        self,
        task: TaskInstance,
        *,
        network: Network,
        grn_task: GRNTask,
        observations: int = 2,
    ) -> None:
        super().__init__(task)
        self.network = network
        self.grn_task = grn_task
        self.observations = int(observations)
        self.notes = (
            f"DREAM4 In Silico Size 100, network {network.index}. The hidden "
            f"mechanism is the complete upstream regulation of {grn_task.target}: "
            f"which of the other {len(grn_task.regulators)} genes regulate it and "
            "whether each activates or inhibits it. One experiment knocks out one "
            "candidate regulator and returns whether that regulator, with that "
            "sign, is part of the mechanism."
        )
        self._library = tuple(
            ComponentSpec(
                family_id=family,
                variant_id=variant,
                name=f"{gene} {SIGN_NAMES[variant]} {grn_task.target}",
                expression=f"{gene} {'->' if variant == ACTIVATION else '-|'} {grn_task.target}",
                parameters={},
            )
            for family, gene in enumerate(grn_task.regulators)
            for variant in (ACTIVATION, INHIBITION)
        )

    def component_library(self) -> tuple[ComponentSpec, ...]:
        return self._library

    def initial_observations(self, agent_index: int) -> tuple[PrivateObservation, ...]:
        """A few knockout rows the agent starts with, one stride per agent.

        Every agent sees the same wild-type level and a different, disjoint
        slice of single-gene knockouts, which is the DREAM4 analogue of the
        different short trajectories the physics agents receive.
        """
        target = self.network.position(self.grn_task.target)
        baseline = self.network.wildtype[target]
        rows = []
        count = len(self.grn_task.regulators)
        for step in range(self.observations):
            family = (agent_index * self.observations + step) % count
            gene = self.grn_task.regulators[family]
            level = self.network.knockout[self.network.position(gene)][target]
            rows.append(
                PrivateObservation(
                    observation_id=f"init:a{agent_index}:{gene}",
                    kind="initial",
                    experiment_id=f"knockout:{gene}",
                    settings={"regulator_family": float(family)},
                    times=(0.0,),
                    states=((baseline, level),),
                    variables=("wildtype_level", "knockout_level"),
                )
            )
        return tuple(rows)

    def supports_prediction(self) -> bool:
        return False
