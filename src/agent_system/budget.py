"""System-wide experiment budget with an asynchronous round-robin allocator.

The permutation stream, eligibility test and candidate refresh semantics are
identical to :func:`src.scaling_model.randomized_round_robin_allocate`; only
``perform_one`` becomes a coroutine so LLM decisions can be awaited.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from ..scaling_model import ALLOCATOR_DOMAIN, derived_rng


async def randomized_round_robin_allocate_async(
    *,
    n_agents: int,
    total_budget: int,
    seed: int,
    round_number: int,
    candidates_for: Callable[[int], list],
    perform_one: Callable[[int, list, int], Awaitable[None]],
    per_agent_cap: int | None = None,
) -> dict[str, object]:
    """Allocate the round budget.

    ``per_agent_cap`` is the ``budget_mode='hard_cap'`` knob: an agent that has
    already run that many experiments this round is skipped even when budget
    remains, so the share of an extinct agent is simply not spent.  ``None``
    is the published rule, where ``m`` is a density and not a per-agent quota.
    """
    if n_agents < 0 or total_budget < 0:
        raise ValueError("agent count and total budget must be nonnegative")
    if per_agent_cap is not None and per_agent_cap < 0:
        raise ValueError("per_agent_cap must be nonnegative")

    start_candidates = [tuple(candidates_for(index)) for index in range(n_agents)]
    start_capacity = sum(len(items) for items in start_candidates)
    active_start = sum(bool(items) for items in start_candidates)
    inactive_start = {index for index, items in enumerate(start_candidates) if not items}

    remaining = total_budget
    experiments_per_agent = [0] * n_agents
    allocation_pass = 0
    while remaining:
        order = list(range(n_agents))
        derived_rng(seed, ALLOCATOR_DOMAIN, round_number, allocation_pass).shuffle(order)
        progressed = False
        for agent_index in order:
            if not remaining:
                break
            if per_agent_cap is not None and experiments_per_agent[agent_index] >= per_agent_cap:
                continue
            candidates = candidates_for(agent_index)
            if not candidates:
                continue
            experiment_ordinal = experiments_per_agent[agent_index]
            await perform_one(agent_index, candidates, experiment_ordinal)
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
    if per_agent_cap is None:
        assert remaining == 0 or final_capacity == 0
    else:
        assert max(experiments_per_agent, default=0) <= per_agent_cap

    return {
        "total_budget_nominal": total_budget,
        "per_agent_cap": per_agent_cap,
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
