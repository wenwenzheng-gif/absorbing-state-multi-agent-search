"""Per-agent evidence store and exact pruning.

Positive evidence fixes a family's value; negative evidence removes one value.
A positive label never deletes a partial hypothesis that simply lacks the
component -- only conflicting values are pruned.
"""

from __future__ import annotations

from typing import Sequence

from ..scaling_model import Node
from .hypotheses import Encoding
from .schemas import EvidenceRecord, Pair


class EvidenceStore:
    """Allowed-value masks plus the labels this agent actually holds."""

    # Testing a component settles its label, so it leaves the candidate pool.
    consumes_candidates = True

    def __init__(self, encoding: Encoding, *, masks: Sequence[int] | None = None) -> None:
        self.encoding = encoding
        # ``masks`` narrows the starting allowed set per family, for a space
        # whose families do not all have the same number of values and whose
        # encoding is therefore padded to the widest one.
        self.allowed: list[int] = (
            list(masks) if masks is not None else [encoding.full_mask] * encoding.families
        )
        if len(self.allowed) != encoding.families:
            raise ValueError("one allowed-value mask per family is required")
        self.records: dict[Pair, EvidenceRecord] = {}
        self.own: set[Pair] = set()

    # -- queries ---------------------------------------------------------
    def knows(self, pair: Pair) -> bool:
        return pair in self.records

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
        family, variant = pair
        mask = self.allowed[family]
        if positive:
            return mask != (1 << variant)
        return bool(mask & (1 << variant))

    # -- updates ---------------------------------------------------------
    def apply(self, record: EvidenceRecord, *, own: bool) -> int:
        """Store a label and tighten the allowed set.  Returns values removed.

        ``own`` always overwrites whatever is on record for this pair: an
        agent's own experiment is ground truth and must take precedence over
        anything it was told about the same pair, including a stale or false
        claim accepted under ``communication_mode == "free"`` before the
        agent tested it itself.  A ``received`` record only fills a gap
        (``setdefault``): the caller is responsible for not handing this
        method a second, conflicting ``received`` record for a pair it
        already knows -- see the deterministic first-claim-wins resolution
        in ``EpisodeRunner._receive``.
        """
        family, variant = record.pair
        before = self.allowed[family].bit_count()
        if own:
            self.records[record.pair] = record
        else:
            self.records.setdefault(record.pair, record)
        if own:
            self.own.add(record.pair)
        if record.positive:
            self.allowed[family] = 1 << variant
        else:
            self.allowed[family] &= ~(1 << variant)
        return before - self.allowed[family].bit_count()

    def compatible(self, node: Node) -> bool:
        bits = self.encoding.bits
        mask = (1 << bits) - 1
        used = node.used
        while used:
            family = (used & -used).bit_length() - 1
            used &= used - 1
            variant = ((node.key >> (bits * family)) & mask) - 1
            if not (self.allowed[family] & (1 << variant)):
                return False
        return True

    def prune(self, frontier: list[Node]) -> list[Node]:
        return [node for node in frontier if self.compatible(node)]

    def covers_positively(self, components: tuple[Pair, ...]) -> bool:
        """Every component of a complete candidate carries a positive label."""
        return all(
            pair in self.records and self.records[pair].positive for pair in components
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "allowed": list(self.allowed),
            "records": {
                f"{p[0]}:{p[1]}": {
                    "event_id": r.event_id,
                    "outcome": r.outcome,
                    "origin_agent": r.origin_agent,
                    "created_round": r.created_round,
                    "experiment_id": r.experiment_id,
                }
                for p, r in sorted(self.records.items())
            },
            "own": sorted(f"{p[0]}:{p[1]}" for p in self.own),
        }
