#!/usr/bin/env python3
"""Validated Mode-B width-only simulator with a system-wide experiment budget.

State transitions and random streams are unchanged from the validated
v3 simulator.  The round budget is ``N*m`` and is redistributed by seeded
randomized round-robin over agents with valid, untested candidates; ``m`` is
not a per-agent cap.  Read-only diagnostics use disjoint random domains and
never alter agent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import statistics
from typing import Callable

import networkx as nx
import numpy as np


MASK = (1 << 64) - 1
TRUTH_DOMAIN = 0x13A5BA1D73293617
ASSIGNMENT_DOMAIN = 0x8CB92BA72F3D8DD7
WIDTH_DOMAIN = 0xA77038F23A14E51B
BRANCH_DOMAIN = 0x632BE59BD9B4E019
ALLOCATOR_DOMAIN = 0xA24BAED4963EE407
ACTION_DOMAIN = 0x9D18BC31EA47265F
NEIGHBOR_DIAGNOSTIC_DOMAIN = 0x3E4932C0C87F4D61
NEIGHBOR_PAIR_DOMAIN = 0x54C9A15590A97BD3


class RNG:
    """Portable SplitMix64 stream."""

    def __init__(self, state: int):
        self.state = state & MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        return value ^ (value >> 31)

    def below(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        threshold = ((-n) & MASK) % n
        while True:
            value = self.next()
            if value >= threshold:
                return value % n

    def shuffle(self, values: list) -> None:
        for right in range(len(values) - 1, 0, -1):
            left = self.below(right + 1)
            values[left], values[right] = values[right], values[left]


def mix(seed: int) -> int:
    return RNG(seed).next()


def fold_integer(value: int) -> int:
    result = 0
    value = int(value)
    while value:
        result = mix(result ^ (value & MASK))
        value >>= 64
    return result


def derived_rng(seed: int, domain: int, *labels: int) -> RNG:
    state = mix((int(seed) ^ int(domain)) & MASK)
    for label in labels:
        state = mix((state ^ fold_integer(int(label)) ^ 0xD1B54A32D192ED03) & MASK)
    return RNG(state)


@lru_cache(maxsize=100_000)
def fixed_graph(n: int, degree: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """One fixed simple undirected d-regular graph per paired episode."""
    if not 0 <= degree < n or (n * degree) % 2:
        raise ValueError("a simple undirected d-regular graph does not exist")
    if degree == 0:
        return tuple(() for _ in range(n))
    if degree == n - 1:
        return tuple(tuple(j for j in range(n) if j != i) for i in range(n))
    graph_seed = (seed * 1_000_003 + n * 1_009 + degree * 9_176 + 71) & ((1 << 63) - 1)
    sparse_degree = min(degree, n - 1 - degree)
    graph = nx.random_regular_graph(sparse_degree, n, seed=graph_seed)
    if degree > (n - 1) / 2:
        graph = nx.complement(graph)
    assert not list(nx.selfloop_edges(graph))
    assert all(graph.degree(i) == degree for i in range(n))
    return tuple(tuple(sorted(graph.neighbors(i))) for i in range(n))


@dataclass(frozen=True)
class Node:
    key: int
    used: int
    wrong: bool
    from_wrong: bool = False


@dataclass
class Agent:
    initial_pairs: tuple[int, ...]
    frontier: list[Node]
    allowed: list[int]
    evidence: set[int]
    own_evidence: set[int]
    reached_truth: bool = False


def pair_component(pair: int, k_values: int) -> int:
    return pair // k_values


def pair_value(pair: int, k_values: int) -> int:
    return pair % k_values


def node_value(key: int, component: int, bits: int) -> int:
    return ((key >> (bits * component)) & ((1 << bits) - 1)) - 1


def compatible(node: Node, allowed: list[int], bits: int) -> bool:
    used = node.used
    while used:
        component = (used & -used).bit_length() - 1
        used &= used - 1
        value = node_value(node.key, component, bits)
        if not (allowed[component] & (1 << value)):
            return False
    return True


def prune(agent: Agent, bits: int) -> None:
    agent.frontier = [node for node in agent.frontier if compatible(node, agent.allowed, bits)]


def choose_without_replacement(rng: RNG, count: int, draws: int) -> list[int]:
    chosen: list[int] = []
    while len(chosen) < draws:
        candidate = rng.below(count)
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen


def observed_jaccard_summary(sets: list[set[int]]) -> dict[str, float | int]:
    observed_sum = 0.0
    valid = 0
    empty = 0
    total = 0
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            total += 1
            union_size = len(sets[left] | sets[right])
            if union_size == 0:
                empty += 1
                continue
            observed_sum += len(sets[left] & sets[right]) / union_size
            valid += 1
    return {
        "jaccard_sum": observed_sum,
        "valid_pairs": valid,
        "empty_pairs": empty,
        "total_pairs": total,
        "jaccard": observed_sum / valid if valid else math.nan,
    }


def size_matched_jaccard_summary(
    sets: list[set[int]], universe_size: int
) -> dict[str, float | int]:
    """Pair-mean Jaccard and its uniform, size-matched finite-universe null."""
    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    count = len(sets)
    sizes = np.fromiter((len(items) for items in sets), dtype=np.float64, count=count)
    intersections = np.zeros((count, count), dtype=np.int32)
    owners: dict[int, list[int]] = {}
    for agent_index, items in enumerate(sets):
        for item in items:
            owners.setdefault(item, []).append(agent_index)
    for members in owners.values():
        indices = np.asarray(members, dtype=np.intp)
        intersections[np.ix_(indices, indices)] += 1
    left, right = np.triu_indices(count, 1)
    observed_intersection = intersections[left, right].astype(np.float64)
    union = sizes[left] + sizes[right] - observed_intersection
    valid = union > 0
    total_pairs = len(left)
    empty_pairs = int((~valid).sum())
    if not np.any(valid):
        return {
            "jaccard_sum": 0.0,
            "null_jaccard_sum": 0.0,
            "excess_jaccard_sum": 0.0,
            "valid_pairs": 0,
            "empty_pairs": empty_pairs,
            "total_pairs": total_pairs,
            "jaccard": math.nan,
            "null_jaccard": math.nan,
            "excess_jaccard": math.nan,
            "max_pair_excess_jaccard": math.nan,
        }
    observed = observed_intersection[valid] / union[valid]
    null_intersection = sizes[left][valid] * sizes[right][valid] / universe_size
    null_union = sizes[left][valid] + sizes[right][valid] - null_intersection
    null = np.divide(
        null_intersection,
        null_union,
        out=np.zeros_like(null_intersection),
        where=null_union > 0,
    )
    excess = observed - null
    return {
        "jaccard_sum": float(observed.sum()),
        "null_jaccard_sum": float(null.sum()),
        "excess_jaccard_sum": float(excess.sum()),
        "valid_pairs": int(valid.sum()),
        "empty_pairs": empty_pairs,
        "total_pairs": total_pairs,
        "jaccard": float(observed.mean()),
        "null_jaccard": float(null.mean()),
        "excess_jaccard": float(excess.mean()),
        "max_pair_excess_jaccard": float(excess.max()),
    }


def sampled_neighbor_jaccard(
    graph: tuple[tuple[int, ...], ...], seed: int, maximum_pairs: int = 512
) -> dict[str, float | int]:
    """Read-only graph-overlap diagnostic on a deterministic pair sample."""
    n = len(graph)
    degree = len(graph[0]) if graph else 0
    total_pairs = n * (n - 1) // 2
    target = min(maximum_pairs, total_pairs)
    if target == 0 or degree == 0:
        return {
            "sample_pairs": target,
            "jaccard": 0.0,
            "random_set_null": 0.0,
            "excess_jaccard": 0.0,
        }
    rng = derived_rng(seed, NEIGHBOR_PAIR_DOMAIN, n, degree)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < target:
        left = rng.below(n)
        right = rng.below(n - 1)
        if right >= left:
            right += 1
        pairs.add((min(left, right), max(left, right)))
    neighbor_sets = [set(items) for items in graph]
    values = []
    for left, right in sorted(pairs):
        intersection = len(neighbor_sets[left] & neighbor_sets[right])
        union = 2 * degree - intersection
        values.append(intersection / union if union else 0.0)
    # Size-matched random-subset approximation; reported as a diagnostic only.
    null = degree / (2 * n - degree) if degree else 0.0
    observed = statistics.mean(values)
    return {
        "sample_pairs": target,
        "jaccard": observed,
        "random_set_null": null,
        "excess_jaccard": observed - null,
    }


def make_truth(
    n: int, m_components: int, task_size: int, k_values: int, seed: int, bits: int
) -> tuple[list[int], int, int, list[int], list[int]]:
    truth_rng = derived_rng(seed, TRUTH_DOMAIN)
    pool = list(range(m_components))
    truth_values = [-1] * m_components
    for index in range(task_size):
        chosen = index + truth_rng.below(m_components - index)
        pool[index], pool[chosen] = pool[chosen], pool[index]
        truth_values[pool[index]] = truth_rng.below(k_values)
    truth_key = 0
    truth_used = 0
    truth_pairs = []
    for component, value in enumerate(truth_values):
        if value >= 0:
            truth_key |= (value + 1) << (bits * component)
            truth_used |= 1 << component
            truth_pairs.append(component * k_values + value)

    group_order = list(range(task_size))
    assignment_rng = derived_rng(seed, ASSIGNMENT_DOMAIN)
    assignment_rng.shuffle(group_order)
    primary_groups = [group_order[index % task_size] for index in range(n)]
    assignment_rng.shuffle(primary_groups)
    return truth_values, truth_key, truth_used, truth_pairs, primary_groups


def make_initial_pairs(
    n: int,
    truth_pairs: list[int],
    primary_groups: list[int],
    pair_count: int,
    seed: int,
    initial_width: int,
) -> list[tuple[int, ...]]:
    truth_set = set(truth_pairs)
    distractor_pool = [pair for pair in range(pair_count) if pair not in truth_set]
    distractor_count = initial_width - 1
    if distractor_count < 0 or distractor_count > len(distractor_pool):
        raise ValueError("invalid width-only initial frontier size")
    result = []
    for agent_index, group in enumerate(primary_groups):
        rng = derived_rng(seed, WIDTH_DOMAIN, agent_index)
        positions = choose_without_replacement(rng, len(distractor_pool), distractor_count)
        pairs = [truth_pairs[group], *(distractor_pool[position] for position in positions)]
        result.append(tuple(sorted(pairs)))
    return result


def randomized_round_robin_allocate(
    *,
    n_agents: int,
    total_budget: int,
    seed: int,
    round_number: int,
    candidates_for: Callable[[int], list[int]],
    perform_one: Callable[[int, list[int], int], None],
) -> dict[str, object]:
    """Spend a system-wide budget fairly while candidate sets evolve.

    This is the production allocation rule extracted verbatim from the
    previously validated inline loop.  Each pass shuffles all agent indices in
    the existing ``ALLOCATOR_DOMAIN`` stream and skips ineligible agents.  The
    relative order induced on the eligible subset is therefore a uniform
    random permutation, while preserving the historical seeded trajectory
    exactly.  Candidate availability is recomputed immediately before every
    experiment; ``perform_one`` may prune candidates before the next slot.

    No additional random draw is introduced by the extraction or diagnostics.
    """
    if n_agents < 0 or total_budget < 0:
        raise ValueError("agent count and total budget must be nonnegative")

    start_candidates = [tuple(candidates_for(index)) for index in range(n_agents)]
    start_capacity = sum(len(items) for items in start_candidates)
    active_start = sum(bool(items) for items in start_candidates)
    inactive_start = {index for index, items in enumerate(start_candidates) if not items}

    remaining = total_budget
    experiments_per_agent = [0] * n_agents
    allocation_pass = 0
    while remaining:
        # Preserve the validated RNG stream exactly: shuffle all agents, then
        # skip those with no current candidate.  Restricted to eligible agents,
        # this is the same distribution as shuffling the eligible set directly.
        order = list(range(n_agents))
        derived_rng(seed, ALLOCATOR_DOMAIN, round_number, allocation_pass).shuffle(order)
        progressed = False
        for agent_index in order:
            if not remaining:
                break
            candidates = candidates_for(agent_index)
            if not candidates:
                continue
            experiment_ordinal = experiments_per_agent[agent_index]
            perform_one(agent_index, candidates, experiment_ordinal)
            experiments_per_agent[agent_index] += 1
            remaining -= 1
            progressed = True
        allocation_pass += 1
        if not progressed:
            break

    final_candidates = [tuple(candidates_for(index)) for index in range(n_agents)]
    final_capacity = sum(len(items) for items in final_candidates)
    active_end = sum(bool(items) for items in final_candidates)
    used = total_budget - remaining

    assert used <= total_budget
    assert sum(experiments_per_agent) == used
    assert all(experiments_per_agent[index] == 0 for index in inactive_start)
    # If any valid candidate survives, the only legal stopping condition is
    # exhaustion of the global budget.
    assert remaining == 0 or final_capacity == 0

    return {
        "total_budget_nominal": total_budget,
        "B_used": used,
        "remaining_budget": remaining,
        "N_experiment_active_start": active_start,
        "N_experiment_active_end": active_end,
        "total_candidate_capacity_start": start_capacity,
        "total_candidate_capacity_end": final_capacity,
        "experiments_per_agent": experiments_per_agent,
        "max_experiments_per_agent": max(experiments_per_agent, default=0),
        "mean_experiments_per_experiment_active_agent": (
            used / active_start if active_start else 0.0
        ),
        "allocation_passes": allocation_pass,
    }


def run_episode(parameters: dict, degree: int, seed: int, mode: str = "width_only") -> dict:
    if mode != "width_only":
        raise ValueError("this scaling experiment uses width_only only")
    n = int(parameters.get("N", 80))
    m_components = int(parameters["M"])
    k_values = int(parameters.get("K", 4))
    task_size = int(parameters.get("task_size", parameters.get("T", 7)))
    rounds = int(parameters.get("rounds", task_size - 1))
    branching = int(parameters.get("b", 2))
    budget_per_agent = int(parameters.get("m", 1))
    initial_width = int(parameters.get("n0", 8))
    if parameters.get("warmup", parameters.get("w", 0)) != 0:
        raise ValueError("this experiment fixes w=0")
    if rounds != task_size - 1:
        raise ValueError("depth-synchronous run must use rounds=T-1")
    if initial_width < 1:
        raise ValueError("n0 must be positive")
    if not (2 <= task_size <= m_components and 0 <= degree < n):
        raise ValueError("invalid task or graph parameters")
    bits = k_values.bit_length()
    if bits * m_components > 120:
        raise ValueError("hypothesis encoding exceeds 120 bits")
    pair_count = m_components * k_values
    full_mask = (1 << k_values) - 1

    truth_values, truth_key, truth_used, truth_pairs, primary_groups = make_truth(
        n, m_components, task_size, k_values, seed, bits
    )
    graph = fixed_graph(n, degree, seed)
    neighbor_overlap = sampled_neighbor_jaccard(graph, seed)
    initial_pairs = make_initial_pairs(
        n, truth_pairs, primary_groups, pair_count, seed, initial_width
    )
    truth_set = set(truth_pairs)

    def is_positive(pair: int) -> bool:
        return truth_values[pair_component(pair, k_values)] == pair_value(pair, k_values)

    def node_from_pair(pair: int) -> Node:
        component = pair_component(pair, k_values)
        value = pair_value(pair, k_values)
        return Node(
            key=(value + 1) << (bits * component),
            used=1 << component,
            wrong=not is_positive(pair),
        )

    agents = [
        Agent(
            initial_pairs=pairs,
            frontier=[node_from_pair(pair) for pair in pairs],
            allowed=[full_mask] * m_components,
            evidence=set(),
            own_evidence=set(),
        )
        for pairs in initial_pairs
    ]
    initial_evidence_empty = all(not agent.evidence and not agent.own_evidence for agent in agents)
    outbox: list[list[int]] = [[] for _ in range(n)]
    transitions: list[dict] = []
    audit = {
        "d0_messages": 0,
        "private_seed_unexperimented_leak": 0,
        "initialization_error": 0,
        "initial_merge": 0,
        "depth_nonsynchronous": 0,
        "empty_frontier_replenished": 0,
        "parent_retained": 0,
        "retry_events": 0,
        "budget_overflow": 0,
        "nonproposed_experiment": 0,
        "forwarded_evidence": 0,
        "same_round_message_use": 0,
    }

    for agent in agents:
        correct = sum(pair in truth_set for pair in agent.initial_pairs)
        if len(agent.initial_pairs) != initial_width or len(set(agent.initial_pairs)) != initial_width:
            audit["initialization_error"] += 1
        if correct != 1:
            audit["initialization_error"] += 1
        if len({node.key for node in agent.frontier}) != initial_width:
            audit["initial_merge"] += 1
        if any(node.used.bit_count() != 1 for node in agent.frontier):
            audit["depth_nonsynchronous"] += 1

    initial_component_counts = {str(pair): 0 for pair in truth_pairs}
    for pairs in initial_pairs:
        for pair in pairs:
            if pair in truth_set:
                initial_component_counts[str(pair)] += 1

    def record(agent: Agent, pair: int, own: bool) -> tuple[bool, int]:
        component = pair_component(pair, k_values)
        value = pair_value(pair, k_values)
        positive = truth_values[component] == value
        before = agent.allowed[component].bit_count()
        agent.evidence.add(pair)
        if own:
            agent.own_evidence.add(pair)
        if positive:
            agent.allowed[component] = 1 << value
        else:
            agent.allowed[component] &= ~(1 << value)
        return positive, before - agent.allowed[component].bit_count()

    def legal_extensions(agent: Agent, node: Node) -> list[int]:
        choices = []
        for component in range(m_components):
            if node.used & (1 << component):
                continue
            for value in range(k_values):
                if agent.allowed[component] & (1 << value):
                    choices.append(component * k_values + value)
        return choices

    def make_child(parent: Node, pair: int) -> Node:
        component = pair_component(pair, k_values)
        value = pair_value(pair, k_values)
        return Node(
            key=parent.key | ((value + 1) << (bits * component)),
            used=parent.used | (1 << component),
            wrong=parent.wrong or truth_values[component] != value,
            from_wrong=parent.wrong,
        )

    for round_number in range(1, rounds + 1):
        log: dict[str, object] = {
            "round": round_number,
            "total_budget_nominal": n * budget_per_agent,
            "B_nominal": n * budget_per_agent,
            "B_used": 0,
            "recv_messages": 0,
            "recv_unique": 0,
            "recv_duplicate": 0,
            "recv_already_explicit": 0,
            "recv_already_implied": 0,
            "recv_informative": 0,
            "recv_pruned_hypotheses": 0,
            "sender_edges_with_message": 0,
            "parents_attempted": 0,
            "sampled_edges": 0,
            "old_incorrect_edges": 0,
            "new_incorrect_source_edges": 0,
            "unique_children": 0,
            "merge_loss": 0,
            "dual_origin_unique_children": 0,
            "incorrect_parents_after_receive": 0,
            "old_pre": 0,
            "source_pre": 0,
            "A_preown": 0,
            "old_post": 0,
            "source_post": 0,
            "A_post": 0,
            "total_proposed_pairs": 0,
            "total_candidate_capacity": 0,
            "final_candidate_capacity": 0,
            "candidate_invalidated_without_test": 0,
            "experiment_active_agents": 0,
            "tests": 0,
            "tests_pos": 0,
            "tests_neg": 0,
            "own_pruned_hypotheses": 0,
            "frontier_nonempty": 0,
            "frontier_empty": 0,
            "truth_compatible_hypotheses": 0,
            "truth_compatible_agents": 0,
            "outgoing_agents": 0,
            "unique_population_experiments": 0,
        }

        # Read-only neighbor-order counterfactual.  Index k stores the pooled
        # number of already-incorrect parents surviving after the first k
        # randomly ordered neighbors.  It is evaluated on cloned state only.
        log["neighbor_wrong_survivors_by_k"] = [0] * (degree + 1)
        log["neighbor_sender_messages_by_rank"] = [0] * degree
        log["neighbor_unique_new_by_rank"] = [0] * degree
        log["neighbor_informative_by_rank"] = [0] * degree
        log["neighbor_receivers_with_wrong_parent"] = 0

        # Stage 1: receive only the previous round's own experimental messages.
        sizes_receive = []
        for receiver, agent in enumerate(agents):
            diagnostic_agent = Agent(
                initial_pairs=agent.initial_pairs,
                frontier=list(agent.frontier),
                allowed=list(agent.allowed),
                evidence=set(agent.evidence),
                own_evidence=set(agent.own_evidence),
                reached_truth=agent.reached_truth,
            )
            diagnostic_senders = list(graph[receiver])
            derived_rng(
                seed, NEIGHBOR_DIAGNOSTIC_DOMAIN, round_number, receiver
            ).shuffle(diagnostic_senders)
            diagnostic_seen: set[int] = set()
            wrong_before = sum(node.wrong for node in diagnostic_agent.frontier)
            log["neighbor_wrong_survivors_by_k"][0] += wrong_before
            log["neighbor_receivers_with_wrong_parent"] += int(wrong_before > 0)
            for rank, sender in enumerate(diagnostic_senders):
                for pair in outbox[sender]:
                    log["neighbor_sender_messages_by_rank"][rank] += 1
                    if pair in diagnostic_seen:
                        continue
                    diagnostic_seen.add(pair)
                    log["neighbor_unique_new_by_rank"][rank] += 1
                    component = pair_component(pair, k_values)
                    value = pair_value(pair, k_values)
                    before_allowed = diagnostic_agent.allowed[component]
                    informative = (
                        before_allowed != (1 << value)
                        if is_positive(pair)
                        else bool(before_allowed & (1 << value))
                    )
                    log["neighbor_informative_by_rank"][rank] += int(informative)
                    record(diagnostic_agent, pair, own=False)
                    prune(diagnostic_agent, bits)
                log["neighbor_wrong_survivors_by_k"][rank + 1] += sum(
                    node.wrong for node in diagnostic_agent.frontier
                )

            seen: set[int] = set()
            before_size = len(agent.frontier)
            for sender in graph[receiver]:
                messages = outbox[sender]
                if messages:
                    log["sender_edges_with_message"] += 1
                for pair in messages:
                    log["recv_messages"] += 1
                    if pair in seen:
                        log["recv_duplicate"] += 1
                        continue
                    seen.add(pair)
                    log["recv_unique"] += 1
                    explicit = pair in agent.evidence
                    component = pair_component(pair, k_values)
                    value = pair_value(pair, k_values)
                    before_allowed = agent.allowed[component]
                    informative = (
                        before_allowed != (1 << value)
                        if is_positive(pair)
                        else bool(before_allowed & (1 << value))
                    )
                    if explicit:
                        log["recv_already_explicit"] += 1
                    elif not informative:
                        log["recv_already_implied"] += 1
                    record(agent, pair, own=False)
                    prune(agent, bits)
                    log["recv_informative"] += int(informative)
            log["recv_pruned_hypotheses"] += before_size - len(agent.frontier)
            sizes_receive.append(len(agent.frontier))

        if degree == 0 and log["recv_messages"]:
            audit["d0_messages"] += int(log["recv_messages"])
        expected_parent_depth = round_number
        for agent in agents:
            if any(node.used.bit_count() != expected_parent_depth for node in agent.frontier):
                audit["depth_nonsynchronous"] += 1
        log["empty_after_receive"] = sum(size == 0 for size in sizes_receive)
        log["empty_after_receive_fraction"] = log["empty_after_receive"] / n
        log["mean_frontier_after_receive"] = statistics.mean(sizes_receive)
        log["median_frontier_after_receive"] = statistics.median(sizes_receive)
        frontier_receive = [{node.key for node in agent.frontier} for agent in agents]
        receive_universe = math.comb(m_components, round_number) * (k_values ** round_number)
        receive_overlap = size_matched_jaccard_summary(frontier_receive, receive_universe)
        for key, value in receive_overlap.items():
            log[f"frontier_receive_{key}"] = value

        # Stage 2: every surviving parent branches; empty stays empty.
        proposed_sets: list[set[int]] = [set() for _ in range(n)]
        preown_sizes = []
        proposed_sizes = []
        for agent_index, agent in enumerate(agents):
            parents = sorted(agent.frontier, key=lambda node: node.key)
            log["incorrect_parents_after_receive"] += sum(node.wrong for node in parents)
            children: dict[int, Node] = {}
            dual_origin_keys: set[int] = set()
            proposed: set[int] = set()
            sampled_edges = 0
            for parent in parents:
                choices = legal_extensions(agent, parent)
                draws = min(branching, len(choices))
                rng = derived_rng(seed, BRANCH_DOMAIN, agent_index, round_number, parent.key)
                sampled = [choices[position] for position in choose_without_replacement(rng, len(choices), draws)]
                for pair in sampled:
                    child = make_child(parent, pair)
                    proposed.add(pair)
                    log["old_incorrect_edges"] += int(parent.wrong)
                    log["new_incorrect_source_edges"] += int((not parent.wrong) and child.wrong)
                    if child.key in children:
                        prior = children[child.key]
                        if prior.from_wrong != child.from_wrong:
                            dual_origin_keys.add(child.key)
                        children[child.key] = Node(
                            key=child.key,
                            used=child.used,
                            wrong=child.wrong,
                            from_wrong=prior.from_wrong or child.from_wrong,
                        )
                    else:
                        children[child.key] = child
                    sampled_edges += 1
            if not parents and children:
                audit["empty_frontier_replenished"] += 1
            log["parents_attempted"] += len(parents)
            log["sampled_edges"] += sampled_edges
            log["unique_children"] += len(children)
            log["merge_loss"] += sampled_edges - len(children)
            log["dual_origin_unique_children"] += len(dual_origin_keys)
            agent.frontier = [children[key] for key in sorted(children)]
            proposed_sets[agent_index] = proposed
            preown_sizes.append(len(children))
            proposed_sizes.append(len(proposed))
            log["old_pre"] += sum(node.wrong and node.from_wrong for node in agent.frontier)
            log["source_pre"] += sum(node.wrong and not node.from_wrong for node in agent.frontier)

        log["total_proposed_pairs"] = sum(proposed_sizes)
        log["mean_preexperiment_frontier"] = statistics.mean(preown_sizes)
        log["median_preexperiment_frontier"] = statistics.median(preown_sizes)
        log["mean_unique_proposed_pairs"] = statistics.mean(proposed_sizes)
        log["median_unique_proposed_pairs"] = statistics.median(proposed_sizes)
        log["effective_branching"] = (
            log["unique_children"] / log["parents_attempted"] if log["parents_attempted"] else 0.0
        )

        def candidates_for(agent_index: int) -> list[int]:
            agent = agents[agent_index]
            return sorted(
                pair
                for pair in proposed_sets[agent_index]
                if pair not in agent.evidence
                and agent.allowed[pair_component(pair, k_values)] & (1 << pair_value(pair, k_values))
            )

        candidate_sets = [set(candidates_for(index)) for index in range(n)]
        log["total_candidate_capacity"] = sum(len(items) for items in candidate_sets)
        log["experiment_active_agents"] = sum(bool(items) for items in candidate_sets)
        log["total_candidate_capacity_start"] = int(log["total_candidate_capacity"])
        log["N_experiment_active_start"] = int(log["experiment_active_agents"])
        log["N_frontier_active"] = sum(bool(agent.frontier) for agent in agents)

        # Stage 3: redistribute total Nm budget over proposed-only pools.
        nextout: list[list[int]] = [[] for _ in range(n)]
        action_sets: list[set[int]] = [set() for _ in range(n)]

        def perform_one(
            agent_index: int, candidates: list[int], experiment_ordinal: int
        ) -> None:
            agent = agents[agent_index]
            rng = derived_rng(
                seed, ACTION_DOMAIN, round_number, agent_index, experiment_ordinal
            )
            pair = candidates[rng.below(len(candidates))]
            # The candidate list was recomputed for this exact slot, after all
            # previous own-pruning updates in the round.
            assert pair in candidates_for(agent_index)
            before_size = len(agent.frontier)
            positive, _ = record(agent, pair, own=True)
            prune(agent, bits)
            log["own_pruned_hypotheses"] += before_size - len(agent.frontier)
            nextout[agent_index].append(pair)
            action_sets[agent_index].add(pair)
            log["tests"] += 1
            log["tests_pos" if positive else "tests_neg"] += 1
            # record() marks the candidate as tested evidence immediately.
            assert pair not in candidates_for(agent_index)

        allocation = randomized_round_robin_allocate(
            n_agents=n,
            total_budget=n * budget_per_agent,
            seed=seed,
            round_number=round_number,
            candidates_for=candidates_for,
            perform_one=perform_one,
        )
        experiments_per_agent = list(allocation["experiments_per_agent"])
        log.update(allocation)
        assert int(log["B_used"]) == int(log["tests"])
        log["final_candidate_capacity"] = int(allocation["total_candidate_capacity_end"])
        log["N_experiment_active_end"] = int(allocation["N_experiment_active_end"])
        log["N_frontier_active_end"] = sum(bool(agent.frontier) for agent in agents)
        log["agents_exceeding_nominal_m"] = sum(
            count > budget_per_agent for count in experiments_per_agent
        )
        log["budget_redistributed_above_m"] = sum(
            max(0, count - budget_per_agent) for count in experiments_per_agent
        )
        log["candidate_invalidated_without_test"] = (
            int(log["total_candidate_capacity"]) - int(log["B_used"]) - int(log["final_candidate_capacity"])
        )
        log["outgoing_agents"] = sum(bool(messages) for messages in nextout)
        if int(log["B_used"]) > n * budget_per_agent:
            audit["budget_overflow"] += 1
        assert int(log["B_used"]) <= n * budget_per_agent
        assert sum(experiments_per_agent) == int(log["B_used"])
        assert all(len(nextout[index]) == experiments_per_agent[index] for index in range(n))
        assert all(len(nextout[index]) == len(set(nextout[index])) for index in range(n))
        assert all(
            pair in proposed_sets[index]
            for index in range(n)
            for pair in nextout[index]
        )
        assert int(log["candidate_invalidated_without_test"]) >= 0
        if int(log["B_used"]) < n * budget_per_agent:
            assert int(log["final_candidate_capacity"]) == 0

        # Stage 4: post-own state and requested observables.
        evidence_post = [set(agent.evidence) for agent in agents]
        post_sizes = [len(agent.frontier) for agent in agents]
        truth_counts = [sum(not node.wrong for node in agent.frontier) for agent in agents]
        log["old_post"] = sum(node.wrong and node.from_wrong for agent in agents for node in agent.frontier)
        log["source_post"] = sum(node.wrong and not node.from_wrong for agent in agents for node in agent.frontier)
        log["A_post"] = int(log["old_post"]) + int(log["source_post"])
        log["A_preown"] = int(log["old_pre"]) + int(log["source_pre"])
        for agent in agents:
            if any(node.key == truth_key and node.used == truth_used for node in agent.frontier):
                agent.reached_truth = True

        log["frontier_nonempty"] = sum(size > 0 for size in post_sizes)
        assert int(log["N_frontier_active_end"]) == int(log["frontier_nonempty"])
        log["frontier_empty"] = n - int(log["frontier_nonempty"])
        log["frontier_empty_fraction"] = log["frontier_empty"] / n
        log["active_agent_fraction"] = log["frontier_nonempty"] / n
        log["mean_frontier_size"] = statistics.mean(post_sizes)
        log["median_frontier_size"] = statistics.median(post_sizes)
        log["truth_compatible_hypotheses"] = sum(truth_counts)
        log["truth_compatible_agents"] = sum(count > 0 for count in truth_counts)
        log["budget_utilization"] = log["B_used"] / (n * budget_per_agent)
        log["message_availability"] = log["sender_edges_with_message"] / (n * degree) if degree else 0.0
        log["outgoing_message_availability"] = log["outgoing_agents"] / n
        log["candidate_capacity_ratio"] = log["total_candidate_capacity"] / (n * budget_per_agent)
        log["duplicate_evidence_fraction"] = (
            log["recv_duplicate"] / log["recv_messages"] if log["recv_messages"] else 0.0
        )
        log["already_known_evidence_fraction"] = (
            (log["recv_already_explicit"] + log["recv_already_implied"]) / log["recv_unique"]
            if log["recv_unique"] else 0.0
        )
        population_actions = set().union(*action_sets)
        log["unique_population_experiments"] = len(population_actions)
        evidence_overlap = size_matched_jaccard_summary(evidence_post, pair_count)
        for key, value in evidence_overlap.items():
            log[f"evidence_post_{key}"] = value
        frontier_post = [{node.key for node in agent.frontier} for agent in agents]
        frontier_universe = math.comb(m_components, round_number + 1) * (k_values ** (round_number + 1))
        frontier_overlap = size_matched_jaccard_summary(frontier_post, frontier_universe)
        for key, value in frontier_overlap.items():
            log[f"frontier_post_{key}"] = value
        action_overlap = size_matched_jaccard_summary(action_sets, pair_count)
        for key, value in action_overlap.items():
            log[f"action_{key}"] = value

        transitions.append(log)
        outbox = nextout

    return {
        "status": "ok",
        "mode": mode,
        "seed": seed,
        "N": n,
        "M": m_components,
        "K": k_values,
        "task_size": task_size,
        "rounds": rounds,
        "b": branching,
        "m": budget_per_agent,
        "n0": initial_width,
        "w": 0,
        "d": degree,
        "truth": truth_values,
        "truth_key": str(truth_key),
        "truth_pairs": truth_pairs,
        "graph_digest": hashlib.sha256(repr(graph).encode()).hexdigest()[:20],
        "neighbor_set_sample_pairs": neighbor_overlap["sample_pairs"],
        "neighbor_set_jaccard": neighbor_overlap["jaccard"],
        "neighbor_set_random_null": neighbor_overlap["random_set_null"],
        "neighbor_set_excess_jaccard": neighbor_overlap["excess_jaccard"],
        "neighbor_independence_diagnostic": (
            "read-only cloned-state replay in a disjoint deterministic sender-order RNG domain"
        ),
        "primary_assignment_digest": hashlib.sha256(repr(primary_groups).encode()).hexdigest()[:20],
        "initialization_digest": hashlib.sha256(repr(initial_pairs).encode()).hexdigest()[:20],
        "initial_pairs": [list(pairs) for pairs in initial_pairs],
        "initial_component_counts": initial_component_counts,
        "initial_evidence_empty": initial_evidence_empty,
        "A_initial_incorrect": sum(pair not in truth_set for pairs in initial_pairs for pair in pairs),
        "lineage_convention": "merged child is A->A iff any direct generating parent was already incorrect",
        "transitions": transitions,
        "agents_reached_truth": sum(agent.reached_truth for agent in agents),
        "any_agent_reached_truth": any(agent.reached_truth for agent in agents),
        "audit": audit,
    }
