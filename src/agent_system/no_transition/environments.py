"""Environment-enforced atomicity and round-barrier sequential capabilities.

These environments deliberately do not inherit the old literal/subtree task:
its encoding, candidate reproduction and pruning are inapplicable here.
Hidden state is never supplied to a policy. No future pointer is public.
"""
from __future__ import annotations

from threading import RLock

from ...scaling_model import derived_rng

NEEDLE_TARGET = 0x4E434E4545444C01
NEEDLE_ORDER = 0x4E434E4545444C02
NEEDLE_REPEAT_ORDER = 0x4E434E4545444C03
CHAIN_KEYS = 0x4E43434841494E01


def coverage_probability(unique_budget: int, candidates: int) -> float:
    if unique_budget < 0 or candidates <= 0:
        raise ValueError("invalid coverage budget")
    return min(unique_budget, candidates) / candidates


def candidate_order(size: int, seed: int) -> list[int]:
    values = list(range(size))
    derived_rng(seed, NEEDLE_ORDER).shuffle(values)
    return values


def reference_batch(order: list[int], agents: int, agent: int, round_number: int, batch: int) -> list[int]:
    start = ((round_number - 1) * agents + agent) * batch
    return order[start:start + batch]


def public_schedule(condition: dict, seed: int) -> list[int]:
    """Public, target-independent complete passes; v1's first pass is unchanged."""
    size = condition["candidates"]
    order = candidate_order(size, seed)
    for pass_number in range(1, condition.get("coverage_passes", 1)):
        repeat = list(range(size))
        derived_rng(seed, NEEDLE_REPEAT_ORDER, pass_number).shuffle(repeat)
        order.extend(repeat)
    return order


class NeedleEnvironment:
    """Exactly one uniform target; verification returns no relational evidence."""
    def __init__(self, candidates: int, seed: int):
        if candidates <= 0:
            raise ValueError("empty candidate set")
        self.size = candidates
        self._target = derived_rng(seed, NEEDLE_TARGET).below(candidates)
        self.tested: set[int] = set()
        self.total_tests = 0
        self.duplicate_tests = 0
        self.success = False
        self.round_of_success = None

    def children(self, candidate: int) -> tuple:
        return ()

    def verify(self, candidate, *, round_number: int) -> str:
        if type(candidate) is not int or not 0 <= candidate < self.size:
            return "INVALID_CANDIDATE"
        self.total_tests += 1
        self.duplicate_tests += int(candidate in self.tested)
        self.tested.add(candidate)
        if candidate == self._target:
            self.success = True
            if self.round_of_success is None:
                self.round_of_success = round_number
            return "TARGET"
        return "NOT TARGET"

    def snapshot(self) -> dict:
        return {"hidden_target": self._target, "tested": sorted(self.tested),
                "total_tests": self.total_tests, "duplicate_tests": self.duplicate_tests,
                "success": self.success, "round_of_success": self.round_of_success}

    def restore(self, data: dict) -> None:
        assert data["hidden_target"] == self._target
        self.tested = set(data["tested"])
        for key in ("total_tests", "duplicate_tests", "success", "round_of_success"):
            setattr(self, key, data[key])


class ChainEnvironment:
    """One shared chain with per-agent possession and atomic global commits.

Each successful submitter receives its own next pointer at the end barrier.
Agents missing it can receive a truthful neighbor's pointer, but a pointer
can never advance more than the currently unlocked level. Even a correct
guess of an unreleased pointer is rejected by the environment.
"""
    def __init__(self, length: int, agents: int, seed: int, horizon: int):
        if min(length, agents, horizon) <= 0:
            raise ValueError("positive dimensions required")
        self.length, self.agents, self.horizon = length, agents, horizon
        rng = derived_rng(seed, CHAIN_KEYS)
        self._keys = [f"k_{rng.next():016x}{rng.next():016x}" for _ in range(length)]
        self._owned = [{self._keys[0]} for _ in range(agents)]
        self.depth = 0
        self.round_number = 0
        self.depth_history: list[int] = []
        self._open = False
        self._attempted: set[int] = set()
        self._accepted: set[int] = set()
        self._lock = RLock()

    @property
    def success(self) -> bool:
        return self.depth == self.length

    def begin_round(self, number: int) -> None:
        with self._lock:
            if self._open or number != self.round_number + 1 or number > self.horizon or self.success:
                raise ValueError("invalid global round")
            self.round_number = number
            self._open = True
            self._attempted, self._accepted = set(), set()

    def receive_pointer(self, receiver: int, pointer: dict) -> bool:
        with self._lock:
            level, key = pointer.get("level"), pointer.get("key")
            # Keys for an unlocked current or past node only; never a future key.
            if type(level) is not int or not 1 <= level <= min(self.depth + 1, self.length):
                return False
            if key != self._keys[level - 1]:
                return False
            self._owned[receiver].add(key)
            return True

    def read(self, agent: int, level, key) -> dict:
        with self._lock:
            if self.success:
                return {"status": "COMPLETE"}
            if type(level) is not int or level != self.depth + 1:
                return {"status": "LOCKED_OR_STALE"}
            if key != self._keys[self.depth] or key not in self._owned[agent]:
                return {"status": "NO_CAPABILITY"}
            return {"status": "AVAILABLE", "level": level, "key": key}

    def observe(self, agent: int) -> dict:
        with self._lock:
            if self.success:
                return {"status": "COMPLETE"}
            key = self._keys[self.depth]
            return self.read(agent, self.depth + 1, key if key in self._owned[agent] else None)

    def submit(self, agent: int, level, key) -> dict:
        with self._lock:
            if not self._open:
                return {"status": "ROUND_CLOSED"}
            if agent in self._attempted:
                return {"status": "ALREADY_ACTED"}
            self._attempted.add(agent)
            result = self.read(agent, level, key)
            if result["status"] != "AVAILABLE":
                return result
            self._accepted.add(agent)
            # No key for the next node is returned before the global barrier.
            return {"status": "ACCEPTED_PENDING_BARRIER", "level": level}

    def commit_round(self) -> dict[int, dict]:
        with self._lock:
            if not self._open:
                raise ValueError("round already committed")
            previous = self.depth
            receipts = {}
            if self._accepted:
                self.depth += 1
                for agent in sorted(self._accepted):
                    if self.success:
                        receipts[agent] = {"status": "COMPLETE", "completed_level": self.depth}
                    else:
                        key = self._keys[self.depth]
                        self._owned[agent].add(key)
                        receipts[agent] = {"status": "UNLOCKED", "completed_level": self.depth,
                                           "next_pointer": {"level": self.depth + 1, "key": key}}
            assert 0 <= self.depth - previous <= 1
            assert self.depth <= self.round_number
            self.depth_history.append(self.depth)
            self._open = False
            return receipts

    def snapshot(self) -> dict:
        with self._lock:
            assert not self._open, "checkpoint only at a global barrier"
            return {"depth": self.depth, "round_number": self.round_number,
                    "depth_history": self.depth_history[:],
                    "owned": [sorted(group) for group in self._owned]}

    def restore(self, data: dict) -> None:
        with self._lock:
            assert not self._open
            self.depth, self.round_number = data["depth"], data["round_number"]
            self.depth_history = data["depth_history"][:]
            self._owned = [set(group) for group in data["owned"]]
            assert self.depth <= min(self.round_number, self.length)
