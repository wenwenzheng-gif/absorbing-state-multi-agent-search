"""Fixed topology with one-round delayed, non-forwarded messages.

``random_regular`` is the published protocol and is produced by the frozen
:func:`src.scaling_model.fixed_graph`.  The other topologies exist for the
ablation the long paper asks for: they keep every other rule intact and only
change *which* ``d`` neighbours an agent has, so the realised mean degree is
recorded alongside the nominal one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scaling_model import derived_rng, fixed_graph
from .schemas import EvidenceRecord, Outcome

# Disjoint from every domain in src/scaling_model.py, so building a
# non-default graph consumes no draw from any existing stream.
TOPOLOGY_DOMAIN = 0x510E527FADE682D1


def build_graph(
    agents: int,
    degree: int,
    seed: int,
    *,
    topology: str = "random_regular",
    rewire_p: float = 0.0,
) -> tuple[tuple[int, ...], ...]:
    """Adjacency lists for one episode's fixed graph."""
    if not 0 <= degree < agents:
        raise ValueError(f"degree {degree} outside [0, N)")
    if degree == 0:
        return tuple(() for _ in range(agents))
    if topology == "random_regular":
        return fixed_graph(agents, degree, seed)
    if topology == "ring_lattice":
        return _ring_lattice(agents, degree)
    if topology == "watts_strogatz":
        return _watts_strogatz(agents, degree, seed, rewire_p)
    if topology == "erdos_renyi":
        return _erdos_renyi(agents, degree, seed)
    raise ValueError(f"unknown topology {topology!r}")


def _edges_to_graph(agents: int, edges: set[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    neighbours: list[set[int]] = [set() for _ in range(agents)]
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    return tuple(tuple(sorted(group)) for group in neighbours)


def _ring_lattice(agents: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Each node linked to the ``d/2`` nearest nodes on each side.

    ``d`` must be even: an odd degree on a ring would need the diametrically
    opposite node, which only exists for even ``N`` and makes the construction
    N-dependent.  :meth:`RunConfig.validate` rejects odd ``d`` here rather
    than silently rounding it.
    """
    if degree % 2:
        raise ValueError("a ring lattice needs an even degree")
    half = degree // 2
    if 2 * half >= agents:
        raise ValueError(f"degree {degree} is too large for a ring of {agents} nodes")
    edges = {
        tuple(sorted((node, (node + step) % agents)))
        for node in range(agents)
        for step in range(1, half + 1)
    }
    return _edges_to_graph(agents, edges)  # type: ignore[arg-type]


def _watts_strogatz(
    agents: int, degree: int, seed: int, rewire_p: float
) -> tuple[tuple[int, ...], ...]:
    """Ring lattice whose clockwise edges are rewired with probability ``p``.

    The canonical construction: for each node in order and each clockwise
    shift, the far end is moved to a uniformly drawn node that keeps the graph
    simple.  A rewiring that cannot be placed is left alone, so the edge count
    -- and therefore the mean degree -- is exactly the ring's.
    """
    if degree % 2:
        raise ValueError("a ring lattice needs an even degree")
    half = degree // 2
    if 2 * half >= agents:
        raise ValueError(f"degree {degree} is too large for a ring of {agents} nodes")
    edges = {
        tuple(sorted((node, (node + step) % agents)))
        for node in range(agents)
        for step in range(1, half + 1)
    }
    rng = derived_rng(seed, TOPOLOGY_DOMAIN, agents, degree, 1)
    threshold = _threshold(rewire_p)
    for step in range(1, half + 1):
        for node in range(agents):
            old = tuple(sorted((node, (node + step) % agents)))
            if old not in edges:
                continue
            if rng.next() >= threshold:
                continue
            for _attempt in range(20):
                target = rng.below(agents)
                new = tuple(sorted((node, target)))
                if target == node or new in edges:
                    continue
                edges.discard(old)
                edges.add(new)
                break
    return _edges_to_graph(agents, edges)  # type: ignore[arg-type]


def _erdos_renyi(agents: int, degree: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """G(N, p) with ``p = d/(N-1)``, so the *expected* degree is ``d``."""
    rng = derived_rng(seed, TOPOLOGY_DOMAIN, agents, degree, 2)
    threshold = _threshold(degree / (agents - 1))
    edges = {
        (left, right)
        for left in range(agents)
        for right in range(left + 1, agents)
        if rng.next() < threshold
    }
    return _edges_to_graph(agents, edges)


def _threshold(probability: float) -> int:
    """SplitMix64 acceptance threshold for a Bernoulli draw."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    return int(probability * (1 << 64))


@dataclass(frozen=True)
class ClaimReport:
    """One claim inside a :class:`CommunicationMessage`.

    Only ``event_id`` and ``claimed_outcome`` ever come from the model; the
    runner resolves ``event_id`` back to the real experiment's identity
    (pair and, where relevant, configuration) when the claim is *received*,
    so the model has no way to alter what it claims to be reporting on.
    """

    event_id: str
    claimed_outcome: Outcome


@dataclass(frozen=True)
class CommunicationMessage:
    """One agent's free-communication output for one round.

    Delivered with exactly the same timing and one-hop, non-forwarding rule
    as an :class:`EvidenceRecord` under ``communication_mode == "fixed"``:
    produced at the end of a round, arriving at the start of the next, never
    forwarded past a direct neighbour.
    """

    sender: int
    round_number: int
    reports: tuple[ClaimReport, ...] = ()
    text: str = ""


class Router:
    """Delivers only labels an agent produced itself, to direct neighbours."""

    def __init__(
        self,
        agents: int,
        degree: int,
        seed: int,
        *,
        topology: str = "random_regular",
        rewire_p: float = 0.0,
    ) -> None:
        regular = topology in ("random_regular", "ring_lattice")
        if not 0 <= degree < agents or (regular and (agents * degree) % 2):
            raise ValueError("a simple undirected d-regular graph does not exist")
        self.agents = agents
        self.degree = degree
        self.topology = topology
        self.graph = build_graph(
            agents, degree, seed, topology=topology, rewire_p=rewire_p
        )
        self._outbox: list[list[EvidenceRecord]] = [[] for _ in range(agents)]
        self._pending: list[list[EvidenceRecord]] = [[] for _ in range(agents)]
        # A second, independent channel for ``communication_mode == "free"``.
        # It shares the graph and the commit timing but never the evidence
        # channel's payload, so a run that never uses it costs nothing extra.
        self._message_outbox: list[list[CommunicationMessage]] = [[] for _ in range(agents)]
        self._message_pending: list[list[CommunicationMessage]] = [[] for _ in range(agents)]

    @property
    def mean_degree(self) -> float:
        """Realised mean degree, which only equals ``d`` for a regular graph."""
        return sum(len(group) for group in self.graph) / self.agents if self.agents else 0.0

    def neighbours(self, agent_index: int) -> tuple[int, ...]:
        return self.graph[agent_index]

    def stage(self, agent_index: int, record: EvidenceRecord) -> None:
        """Queue one own label for delivery in the next round."""
        if record.origin_agent != agent_index:
            raise ValueError("an agent may only send labels it produced itself")
        self._pending[agent_index].append(record)

    def commit_round(self) -> None:
        """Rotate: this round's staged labels become next round's outbox."""
        self._outbox = self._pending
        self._pending = [[] for _ in range(self.agents)]
        self._message_outbox = self._message_pending
        self._message_pending = [[] for _ in range(self.agents)]

    # -- free communication (communication_mode == "free") ------------------
    def stage_message(self, agent_index: int, message: CommunicationMessage) -> None:
        """Queue one agent's communication decision for next-round delivery."""
        if message.sender != agent_index:
            raise ValueError("an agent may only send a message as itself")
        self._message_pending[agent_index].append(message)

    def message_deliveries(self, receiver: int) -> list[tuple[int, CommunicationMessage]]:
        """Sender-ordered messages reaching ``receiver`` this round."""
        result: list[tuple[int, CommunicationMessage]] = []
        for sender in self.graph[receiver]:
            for message in self._message_outbox[sender]:
                if message.sender != sender:
                    raise ValueError("forwarded messages are not allowed in agent_v1")
                result.append((sender, message))
        return result

    def outgoing(self, agent_index: int) -> list[EvidenceRecord]:
        return self._outbox[agent_index]

    def deliveries(self, receiver: int) -> list[tuple[int, EvidenceRecord]]:
        """Sender-ordered messages reaching ``receiver`` this round."""
        result: list[tuple[int, EvidenceRecord]] = []
        for sender in self.graph[receiver]:
            for record in self._outbox[sender]:
                if record.origin_agent != sender:
                    raise ValueError("forwarded evidence is not allowed in agent_v1")
                result.append((sender, record))
        return result

    def senders_with_message(self, receiver: int) -> int:
        return sum(1 for sender in self.graph[receiver] if self._outbox[sender])

    def total_outgoing_agents(self) -> int:
        return sum(1 for box in self._outbox if box)

    def pending_outgoing_agents(self) -> int:
        """Agents that produced at least one label for next round's delivery."""
        return sum(1 for box in self._pending if box)
