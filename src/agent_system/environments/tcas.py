"""Configuration fault localisation over the tcas parameter space.

This environment differs from the others in what one experiment says.  A test
runs a *complete* configuration and returns a pass/fail verdict, so evidence
is not a label on one component:

* a **passing** configuration ``x`` refutes every hypothesis ``h`` with
  ``h subset of x``, across all paths and agents, because a failure-inducing
  combination contained in ``x`` would have made ``x`` fail;
* a **failing** configuration eliminates nothing on its own -- it only says
  that some surviving combination is inside ``x`` -- so it is kept as positive
  support and never prunes.

:class:`SubsetEvidenceStore` implements that, and the environment hands it to
the runner in place of the component-level store the other environments use.
"""

from __future__ import annotations

from typing import Sequence

from ...scaling_model import Node
from ..hypotheses import Encoding
from ..schemas import (
    ComponentSpec,
    EvidenceRecord,
    ExperimentCondition,
    ExperimentResult,
    Pair,
)
from .base import Environment, TaskInstance
from ...scaling_model import derived_rng
from ..evidence import EvidenceStore
from .tcas_dataset import (
    BANK_DOMAIN,
    DEFAULT_SPACE,
    HELDOUT_DOMAIN,
    TcasTask,
    allowed_masks,
    assignment_count,
    space,
    space_size,
)

# Disjoint from every domain in src/scaling_model.py and from the bank and
# hold-out domains, so turning the knob on consumes no draw any other stream
# would have taken.
SELECTION_DOMAIN = 0x6A09E667F3BCC909


class SubsetEvidenceStore:
    """Pruning by containment in a passing configuration.

    The interface is the one the runner already uses for the component-level
    store.  ``allowed`` stays structural here: a pass does not rule out an
    assignment on its own, only the specific sub-combinations it contains, so
    narrowing the per-parameter masks would over-prune.
    """

    # A verdict belongs to a configuration, not to a component, so testing a
    # component inside one configuration leaves it testable inside another.
    consumes_candidates = False

    def __init__(self, encoding: Encoding, *, masks: Sequence[int]) -> None:
        self.encoding = encoding
        self.allowed: list[int] = list(masks)
        self.records: dict[Pair, EvidenceRecord] = {}
        self.own: set[Pair] = set()
        self.passing: list[tuple[int, ...]] = []
        self.failing: list[tuple[int, ...]] = []
        self._seen: set[tuple[int, ...]] = set()

    # -- queries ---------------------------------------------------------
    def knows(self, pair: Pair) -> bool:
        # A single assignment is never settled by a configuration verdict, so a
        # candidate is never skipped for being already known.
        return False

    def value_allowed(self, pair: Pair) -> bool:
        family, variant = pair
        return bool(self.allowed[family] & (1 << variant))

    def known_positive(self) -> tuple[Pair, ...]:
        return tuple(sorted(p for p, r in self.records.items() if r.positive))

    def known_negative(self) -> tuple[Pair, ...]:
        return tuple(sorted(p for p, r in self.records.items() if not r.positive))

    def allowed_values(self) -> dict[int, tuple[int, ...]]:
        return {
            family: tuple(
                variant
                for variant in range(self.encoding.variants)
                if self.allowed[family] & (1 << variant)
            )
            for family in range(self.encoding.families)
        }

    def would_be_informative(self, pair: Pair, positive: bool) -> bool:
        # The verdict of a configuration cannot be predicted from the labels
        # held, so every proposed test is treated as worth running.
        return True

    # -- updates ---------------------------------------------------------
    def apply(self, record: EvidenceRecord, *, own: bool) -> int:
        self.records.setdefault(record.pair, record)
        if own:
            self.own.add(record.pair)
        configuration = record.configuration
        if configuration is None or configuration in self._seen:
            return 0
        self._seen.add(configuration)
        if record.positive:
            self.failing.append(configuration)
        else:
            self.passing.append(configuration)
        return 1

    def compatible(self, node: Node) -> bool:
        bits = self.encoding.bits
        mask = (1 << bits) - 1
        components = []
        used = node.used
        while used:
            family = (used & -used).bit_length() - 1
            used &= used - 1
            components.append((family, ((node.key >> (bits * family)) & mask) - 1))
        for configuration in self.passing:
            if all(configuration[family] == variant for family, variant in components):
                return False
        return True

    def prune(self, frontier: list[Node]) -> list[Node]:
        return [node for node in frontier if self.compatible(node)]

    def covers_positively(self, components: tuple[Pair, ...]) -> bool:
        """The agent itself ran a failing configuration containing the candidate.

        A failing configuration is the only positive support this oracle
        gives, so a candidate counts as verified when the agent holds one that
        contains it and none that passes.
        """
        supported = any(
            all(configuration[family] == variant for family, variant in components)
            for configuration in self.failing
        )
        refuted = any(
            all(configuration[family] == variant for family, variant in components)
            for configuration in self.passing
        )
        return supported and not refuted

    def snapshot(self) -> dict[str, object]:
        return {
            "allowed": list(self.allowed),
            "passing": [list(c) for c in self.passing],
            "failing": [list(c) for c in self.failing],
            "own": sorted(f"{p[0]}:{p[1]}" for p in self.own),
        }


class TcasEnvironment(Environment):
    kind = "tcas"

    def __init__(
        self,
        task: TaskInstance,
        *,
        tcas_task: TcasTask,
        heldout_size: int = 256,
        bank_size: int = 64,
        evidence_model: str = "subset",
        selection: str = "uniform",
    ) -> None:
        super().__init__(task)
        self.tcas_task = tcas_task
        self.heldout_size = int(heldout_size)
        if evidence_model not in ("subset", "component"):
            raise ValueError(f"unknown tcas evidence model {evidence_model!r}")
        self.evidence_model = evidence_model
        if selection not in ("uniform", "max_containment"):
            raise ValueError(f"unknown tcas selection {selection!r}")
        self.selection = selection
        self.space_name = tcas_task.space_name
        self.cardinalities = tcas_task.cardinalities
        self.parameter_names = tcas_task.parameter_names
        shared = (
            "Configuration fault localisation over the tcas parameter space: "
            f"{len(self.cardinalities)} categorical parameters, "
            f"{space_size(self.space_name)} complete configurations and "
            f"{assignment_count(self.space_name)} atomic assignments. The hidden "
            f"mechanism is a failure-inducing combination of {tcas_task.strength} "
            "assignments; a configuration fails exactly when it contains all of them. "
        )
        self.notes = shared + (
            "One experiment runs one complete configuration and returns pass or "
            "fail. A passing configuration refutes every combination it contains; "
            "a failing one refutes nothing by itself."
            if evidence_model == "subset"
            else "One experiment asks whether a single parameter-value assignment "
            "belongs to the combination, and a refuted assignment prunes every "
            "candidate that contains it."
        )
        self._bank = tuple(
            ExperimentCondition(
                experiment_id=f"cfg{index:03d}",
                settings={
                    self.parameter_names[parameter]: float(value)
                    for parameter, value in enumerate(pattern)
                },
                description="bank pattern completed by the hypothesis under test",
            )
            for index, pattern in enumerate(
                _patterns(BANK_DOMAIN, task.seed, bank_size, self.cardinalities)
            )
        )
        self._bank_by_id = {
            condition.experiment_id: tuple(
                int(condition.settings[name]) for name in self.parameter_names
            )
            for condition in self._bank
        }
        self._bank_patterns = tuple(
            self._bank_by_id[condition.experiment_id] for condition in self._bank
        )
        # One bitmask over bank positions per assignment, so "which bank
        # patterns already carry (parameter, value)" is a single lookup and a
        # hypothesis is matched by ANDing its assignments' masks.  Bank sizes
        # reach a few thousand and a frontier a few hundred, so the query has
        # to be a handful of big-integer operations, not a double loop.
        bank_bits: dict[Pair, int] = {}
        for position, pattern in enumerate(self._bank_patterns):
            for parameter, value in enumerate(pattern):
                key = (parameter, value)
                bank_bits[key] = bank_bits.get(key, 0) | (1 << position)
        self._bank_bits = bank_bits
        self._all_bank_bits = (1 << len(self._bank_patterns)) - 1
        # Drawn from a domain the bank never touches, so a configuration an
        # agent can run is never one it is scored on.
        self._heldout = _patterns(
            HELDOUT_DOMAIN, task.seed, self.heldout_size, self.cardinalities
        )
        self._library = tuple(
            ComponentSpec(
                family_id=parameter,
                variant_id=value,
                name=f"{self.parameter_names[parameter]}={value}",
                expression=f"{self.parameter_names[parameter]} == {value}",
                parameters={},
            )
            for parameter, size in enumerate(self.cardinalities)
            for value in range(size)
        )

    def component_library(self) -> tuple[ComponentSpec, ...]:
        return self._library

    def experiment_library(self) -> tuple[ExperimentCondition, ...]:
        # A component-level oracle answers about the assignment itself, so
        # there is no configuration left for the agent to choose.
        return self._bank if self.evidence_model == "subset" else ()

    def new_evidence_store(self, encoding: Encoding):
        masks = allowed_masks(self.space_name)
        if self.evidence_model == "component":
            return EvidenceStore(encoding, masks=masks)
        return SubsetEvidenceStore(encoding, masks=masks)

    def supports_prediction(self) -> bool:
        # There is nothing for an agent to simulate: a verdict is a lookup.
        return False

    def scores_heldout(self) -> bool:
        return True

    # -- experiment selection ---------------------------------------------
    def select_experiment(
        self,
        *,
        pair: Pair,
        parent_components: tuple[Pair, ...],
        frontier_components: tuple[tuple[Pair, ...], ...],
        seed: int,
        round_number: int,
        agent_index: int,
        slot: int,
    ) -> str | None:
        """The long paper's ``x* = argmax_{x in bank, h subset x} I(x; H_i)``.

        ``h`` is the refinement about to be tested, ``H_i`` the agent's
        surviving frontier.  As in :meth:`_configuration`, the bank pattern is
        *completed* by ``h`` rather than filtered for already containing it:
        a random bank of 64--4000 patterns over this space almost never
        contains a given depth-``t+1`` combination outright, so filtering
        would usually leave the argmax undefined.  Completing keeps
        ``h subset of x`` true by construction and leaves the free parameters
        -- which is exactly what the argmax ranges over -- to the bank.

        Ties, including the common case where no other surviving hypothesis
        fits any pattern, are broken by a seeded draw on a domain of its own.
        """
        if self.selection != "max_containment" or self.evidence_model != "subset":
            return None
        if not self._bank:
            return None
        forced = dict(tuple(parent_components) + (pair,))
        counts: dict[int, int] = {}
        for components in frontier_components:
            free: list[Pair] = []
            consistent = True
            for family, variant in components:
                pinned = forced.get(family)
                if pinned is None:
                    free.append((family, variant))
                elif pinned != variant:
                    # The completed configuration fixes this parameter to a
                    # different value, so no bank choice can contain it.
                    consistent = False
                    break
            if not consistent:
                continue
            if not free:
                # Contained in every completion; a constant offset that cannot
                # change the argmax, so it is not accumulated.
                continue
            matched = self._all_bank_bits
            for key in free:
                matched &= self._bank_bits.get(key, 0)
                if not matched:
                    break
            while matched:
                lowest = matched & -matched
                position = lowest.bit_length() - 1
                counts[position] = counts.get(position, 0) + 1
                matched ^= lowest
        rng = derived_rng(seed, SELECTION_DOMAIN, round_number, agent_index, slot)
        if not counts:
            return self._bank[rng.below(len(self._bank))].experiment_id
        best = max(counts.values())
        winners = sorted(position for position, value in counts.items() if value == best)
        return self._bank[winners[rng.below(len(winners))]].experiment_id

    # -- experiments -------------------------------------------------------
    def _configuration(self, base: tuple[int, ...], fixed: Sequence[Pair]) -> tuple[int, ...]:
        """One bank pattern, overridden by the hypothesis under test.

        The paper picks a bank member that already contains the proposed
        hypothesis; completing a pattern instead guarantees containment for
        every hypothesis without needing a bank large enough to cover them all.
        The free parameters still come from the fixed bank, so what varies
        between tests of the same hypothesis is exactly the bank choice.
        """
        configuration = list(base)
        for family, variant in fixed:
            configuration[family] = variant
        return tuple(configuration)

    async def execute(
        self,
        *,
        pair: Pair,
        experiment_id: str | None,
        owner: int,
        round_number: int,
        event_id: str,
        parent_components: tuple[Pair, ...] = (),
    ) -> ExperimentResult:
        if self.evidence_model == "component":
            return await super().execute(
                pair=pair,
                experiment_id=experiment_id,
                owner=owner,
                round_number=round_number,
                event_id=event_id,
                parent_components=parent_components,
            )
        self.experiments_executed += 1
        base = self._bank_by_id.get(experiment_id or "")
        if base is None:
            self.failures["unknown_experiment_id"] = (
                self.failures.get("unknown_experiment_id", 0) + 1
            )
            raise KeyError(f"experiment_id {experiment_id!r} is not in the tcas bank")
        configuration = self._configuration(base, tuple(parent_components) + (pair,))
        failed = self.tcas_task.fails(configuration)
        record = EvidenceRecord(
            event_id=event_id,
            pair=pair,
            # A failing configuration is the only positive support this oracle
            # gives; a passing one refutes every combination it contains.
            outcome="positive" if failed else "negative",
            origin_agent=owner,
            created_round=round_number,
            experiment_id=experiment_id,
            configuration=configuration,
        )
        return ExperimentResult(evidence=record, observation=None)

    # -- hidden, evaluator-only -------------------------------------------
    def heldout_error(self, components: Sequence[Pair]) -> float:
        """The paper's ``D(h, H*)`` over held-out configurations.

        A hypothesis predicts failure exactly for the configurations that
        contain it; the error is how often that disagrees with the oracle.
        """
        wrong = 0
        for configuration in self._heldout:
            predicted = all(
                configuration[family] == variant for family, variant in components
            )
            if predicted != self.tcas_task.fails(configuration):
                wrong += 1
        return wrong / len(self._heldout) if self._heldout else float("nan")

    def heldout_conditions(self) -> tuple[ExperimentCondition, ...]:
        return ()


def _patterns(
    domain: int, seed: int, count: int, cardinalities: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """``count`` distinct complete configurations from one seeded stream."""
    rng = derived_rng(seed, domain)
    seen: set[tuple[int, ...]] = set()
    drawn: list[tuple[int, ...]] = []
    attempts = 0
    while len(drawn) < count and attempts < count * 100:
        attempts += 1
        candidate = tuple(rng.below(size) for size in cardinalities)
        if candidate in seen:
            continue
        seen.add(candidate)
        drawn.append(candidate)
    return tuple(drawn)
