import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.agent_system.communication import CommunicationMessage, Router
from src.agent_system.no_transition.environments import (
    ChainEnvironment, NeedleEnvironment, candidate_order, coverage_probability, reference_batch,
)


def test_needle_exactly_one_target_atomic_and_no_cross_candidate_effect():
    env = NeedleEnvironment(100, 771)
    target = env._target
    rejected = (target + 1) % 100
    assert env.verify(rejected, round_number=1) == "NOT TARGET"
    assert len(env.tested) == 1
    assert all(env.children(i) == () for i in range(100))
    results = [env.verify(i, round_number=2) for i in range(100)]
    assert results.count("TARGET") == 1
    assert env.success and env.round_of_success == 2
    assert len(env.tested) == 100 and env.total_tests == 101 and env.duplicate_tests == 1


@pytest.mark.parametrize("candidate", [-1, 8, None, "3", True, 2.0])
def test_needle_illegal_candidates_do_not_consume_verification(candidate):
    env = NeedleEnvironment(8, 4)
    assert env.verify(candidate, round_number=1) == "INVALID_CANDIDATE"
    assert not env.tested and env.total_tests == 0 and not env.success


def test_coverage_baseline_by_exhaustive_targets():
    order = candidate_order(120, 7)
    scheduled = [i for r in range(1, 3) for a in range(6)
                 for i in reference_batch(order, 6, a, r, 4)]
    assert len(scheduled) == len(set(scheduled)) == 48
    successes = 0
    for target in range(120):
        env = NeedleEnvironment(120, 7)
        env._target = target  # enumerate the uniform target space, no simulation shortcut
        for candidate in scheduled:
            env.verify(candidate, round_number=1)
        successes += env.success
    assert successes / 120 == coverage_probability(48, 120)
    assert coverage_probability(200, 120) == 1
    assert coverage_probability(0, 120) == 0


def test_seed_control_and_independent_order():
    assert candidate_order(50, 17) == candidate_order(50, 17)
    assert candidate_order(50, 17) != candidate_order(50, 18)
    assert NeedleEnvironment(50, 17)._target == NeedleEnvironment(50, 17)._target


def test_chain_future_key_guess_rejected_and_unlock_hidden_until_barrier():
    env = ChainEnvironment(8, 6, 2, 10)
    root = env.observe(0)
    env.begin_round(1)
    future = env._keys[1]
    assert env.read(0, 2, future)["status"] == "LOCKED_OR_STALE"
    assert not env.receive_pointer(0, {"level": 2, "key": future})
    assert env.submit(1, 2, future)["status"] == "LOCKED_OR_STALE"
    assert env.submit(0, 1, root["key"])["status"] == "ACCEPTED_PENDING_BARRIER"
    assert env.depth == 0 and env.observe(0)["level"] == 1
    assert env.read(0, 2, future)["status"] == "LOCKED_OR_STALE"
    assert env.submit(0, 2, future)["status"] == "ALREADY_ACTED"
    receipts = env.commit_round()
    assert env.depth == 1
    assert receipts[0]["next_pointer"] == {"level": 2, "key": future}
    assert env.observe(0)["level"] == 2
    assert env.observe(1)["status"] == "NO_CAPABILITY"
    with pytest.raises(ValueError):
        env.commit_round()


def test_concurrent_requests_cannot_race_levels():
    env = ChainEnvironment(8, 40, 5, 8)
    root = env.observe(0)
    env.begin_round(1)
    def submit(i):
        return env.submit(i % 40, 1, root["key"])
    with ThreadPoolExecutor(max_workers=16) as pool:
        responses = list(pool.map(submit, range(200)))
    assert sum(r["status"] == "ACCEPTED_PENDING_BARRIER" for r in responses) == 40
    assert env.depth == 0
    assert len(env.commit_round()) == 40
    assert env.depth == 1


@pytest.mark.parametrize("horizon,expected", [(6, False), (8, True), (10, True)])
@pytest.mark.parametrize("degree", [0, 2, 5])
def test_depth_limit_and_reference_completion(horizon, expected, degree):
    env = ChainEnvironment(8, 6, 10, horizon)
    router = Router(6, degree, 500)
    assert all(len(v) == degree for v in router.graph)
    for r in range(1, horizon + 1):
        if env.success:
            break
        env.begin_round(r)
        for agent in range(6):
            node = env.observe(agent)
            env.submit(agent, node["level"], node["key"])
        env.commit_round()
    assert env.success is expected
    assert env.depth_history == list(range(1, min(8, horizon) + 1))


def test_communication_delayed_truthful_no_forwarding_no_pruning():
    router = Router(6, 2, 101)
    message = CommunicationMessage(sender=0, round_number=1, text='{"candidate":47,"result":"NOT TARGET"}')
    router.stage_message(0, message)
    assert all(not router.message_deliveries(a) for a in range(6))
    router.commit_round()
    assert {a for a in range(6) if router.message_deliveries(a)} == set(router.neighbours(0))
    env = NeedleEnvironment(60, 101)
    assert not env.tested  # routing does not mutate or prune a task
    with pytest.raises(ValueError):
        router.stage_message(1, message)
    router.commit_round()
    assert all(not router.message_deliveries(a) for a in range(6))


def test_shared_pointer_cannot_skip_a_chain_level():
    env = ChainEnvironment(8, 2, 4, 8)
    root = env.observe(0)
    env.begin_round(1)
    env.submit(0, 1, root["key"])
    receipt = env.commit_round()[0]
    assert env.receive_pointer(1, receipt["next_pointer"])
    env.begin_round(2)
    assert env.submit(1, 3, env._keys[2])["status"] == "LOCKED_OR_STALE"
    env.commit_round()
    assert env.depth == 1


def test_round_and_horizon_enforced_by_environment():
    env = ChainEnvironment(3, 2, 4, 1)
    assert env.submit(0, 1, env._keys[0])["status"] == "ROUND_CLOSED"
    with pytest.raises(ValueError):
        env.begin_round(2)
    env.begin_round(1)
    with pytest.raises(ValueError):
        env.begin_round(1)
    env.commit_round()
    with pytest.raises(ValueError):
        env.begin_round(2)
