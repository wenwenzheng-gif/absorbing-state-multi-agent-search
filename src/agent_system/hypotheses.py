"""Canonical hypothesis encoding, legal extensions and confluent merging.

The bit-packed representation and the branch random stream are taken from the
validated simulator so that a ``uniform`` policy reproduces
:func:`src.scaling_model.run_episode` step for step.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scaling_model import (
    BRANCH_DOMAIN,
    Node,
    choose_without_replacement,
    derived_rng,
)
from .schemas import Hypothesis, Pair


@dataclass(frozen=True)
class Encoding:
    """Maps between ``(family, variant)`` pairs and the packed node key."""

    families: int
    variants: int

    @property
    def bits(self) -> int:
        return self.variants.bit_length()

    @property
    def pair_count(self) -> int:
        return self.families * self.variants

    @property
    def full_mask(self) -> int:
        return (1 << self.variants) - 1

    # The packed key is a Python int, so its width is not a representational
    # limit; this only catches a configuration that would make keys absurd.
    # DREAM4 needs 99 families x 2 bits and tcas 12 x 4, so the old cap of 120
    # was below what the paper's own datasets require.
    MAX_KEY_BITS = 512

    def check(self) -> None:
        if self.bits * self.families > self.MAX_KEY_BITS:
            raise ValueError(
                f"hypothesis encoding needs {self.bits * self.families} bits, "
                f"over the {self.MAX_KEY_BITS}-bit guard"
            )

    def index(self, pair: Pair) -> int:
        return pair[0] * self.variants + pair[1]

    def pair(self, index: int) -> Pair:
        return index // self.variants, index % self.variants

    def node_from_pair(self, pair: Pair, *, wrong: bool) -> Node:
        family, variant = pair
        return Node(
            key=(variant + 1) << (self.bits * family),
            used=1 << family,
            wrong=wrong,
        )

    def child(self, parent: Node, pair: Pair, *, wrong: bool) -> Node:
        family, variant = pair
        return Node(
            key=parent.key | ((variant + 1) << (self.bits * family)),
            used=parent.used | (1 << family),
            wrong=parent.wrong or wrong,
            from_wrong=parent.wrong,
        )

    def components(self, node: Node) -> tuple[Pair, ...]:
        result: list[Pair] = []
        used = node.used
        while used:
            family = (used & -used).bit_length() - 1
            used &= used - 1
            variant = ((node.key >> (self.bits * family)) & ((1 << self.bits) - 1)) - 1
            result.append((family, variant))
        return tuple(sorted(result))

    def hypothesis(self, node: Node) -> Hypothesis:
        return Hypothesis(self.components(node))

    def legal_extensions(self, allowed: list[int], node: Node) -> list[Pair]:
        """All single-component extensions consistent with current evidence.

        Ordering is ascending pair index, matching the reference simulator.
        """
        choices: list[Pair] = []
        for family in range(self.families):
            if node.used & (1 << family):
                continue
            mask = allowed[family]
            for variant in range(self.variants):
                if mask & (1 << variant):
                    choices.append((family, variant))
        return choices


def pack_key(components, bits: int) -> int:
    """Rebuild the packed node key from a canonical component set."""
    key = 0
    for family, variant in components:
        key |= (variant + 1) << (bits * family)
    return key


def uniform_branch_sample(
    seed: int,
    agent_index: int,
    round_number: int,
    parent: Node,
    choices: list[Pair],
    draws: int,
) -> list[Pair]:
    """The reference branch draw: fixed domain, fixed labels, fixed order."""
    rng = derived_rng(seed, BRANCH_DOMAIN, agent_index, round_number, parent.key)
    positions = choose_without_replacement(rng, len(choices), draws)
    return [choices[position] for position in positions]


class ChildCollector:
    """Confluent merge of children inside one agent, with lineage bookkeeping."""

    def __init__(self) -> None:
        self.children: dict[int, Node] = {}
        self.dual_origin: set[int] = set()
        self.parents_of_pair: dict[Pair, list[Node]] = {}
        self.sampled_edges = 0
        self.old_incorrect_edges = 0
        self.new_incorrect_source_edges = 0

    def add(self, parent: Node, pair: Pair, child: Node) -> None:
        self.sampled_edges += 1
        self.old_incorrect_edges += int(parent.wrong)
        self.new_incorrect_source_edges += int((not parent.wrong) and child.wrong)
        self.parents_of_pair.setdefault(pair, []).append(parent)
        prior = self.children.get(child.key)
        if prior is None:
            self.children[child.key] = child
            return
        if prior.from_wrong != child.from_wrong:
            self.dual_origin.add(child.key)
        self.children[child.key] = Node(
            key=child.key,
            used=child.used,
            wrong=child.wrong,
            from_wrong=prior.from_wrong or child.from_wrong,
        )

    @property
    def merge_loss(self) -> int:
        return self.sampled_edges - len(self.children)

    def frontier(self) -> list[Node]:
        return [self.children[key] for key in sorted(self.children)]

    def proposed_pairs(self) -> set[Pair]:
        return set(self.parents_of_pair)
