#!/usr/bin/env python3
"""``q_model_draw``: one chance per label, or one chance per neighbour.

The published semantics ("per_label") filters duplicates first and then makes
a single Bernoulli(q) draw per unique incoming label, so however many
neighbours deliver the same label the agent gets one chance at it and pruning
saturates at ``q``.  The mean-field term ``(1 - m q_overlap q_model)^d`` of
the paper instead treats every neighbour as an independent opportunity, which
is what "per_message" implements.  Both are kept, the default is the
published one, and ``q_model = 1`` draws nothing under either.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.schemas import load_run_config  # noqa: E402
from tests.test_agents_knobs_defaults import BASE, build_runner, run_with  # noqa: E402


@dataclass
class _Copy:
    """The only field ``_per_message_learned`` reads off a delivery."""

    pair: tuple[int, int]


def _deliveries(pair, senders):
    return [(sender, _Copy(pair)) for sender in senders]


class DrawSemantics(unittest.TestCase):
    """With ``c`` copies of one label: q under per_label, 1-(1-q)^c under per_message."""

    Q = 0.5
    TRIALS = 4000

    def _runner(self, mode: str, q: float = 0.5):
        return build_runner(
            {"q_model_received": q, "q_model_draw": mode}, degree=8, seed=12345
        )

    def _per_message_rate(self, copies: int, q: float = 0.5) -> float:
        runner = self._runner("per_message", q)
        learned = 0
        for trial in range(self.TRIALS):
            round_number = 1 + trial % 3
            receiver = trial % runner.agents_count
            pair = (trial % 8, (trial // 8) % 4)
            # A fresh (round, receiver, label) per trial, so the trials are
            # independent draws of the same experiment.
            got, _failed = runner._per_message_learned(
                round_number + 7 * trial, receiver, _deliveries(pair, range(copies))
            )
            learned += int(pair in got)
        return learned / self.TRIALS

    def test_one_copy_is_the_same_experiment_under_both_semantics(self) -> None:
        self.assertAlmostEqual(self._per_message_rate(1), self.Q, delta=0.03)

    def test_the_learn_probability_is_one_minus_one_minus_q_to_the_copies(self) -> None:
        for copies in (2, 3, 5, 8):
            expected = 1.0 - (1.0 - self.Q) ** copies
            with self.subTest(copies=copies):
                self.assertAlmostEqual(
                    self._per_message_rate(copies), expected, delta=0.03
                )

    def test_a_smaller_q_still_follows_the_same_law(self) -> None:
        self.assertAlmostEqual(
            self._per_message_rate(4, 0.25), 1.0 - 0.75 ** 4, delta=0.03
        )

    def test_per_label_gives_one_chance_however_many_copies_arrive(self) -> None:
        # End to end, in the regime the two semantics are about: a small
        # component space and m = 3, so the same label really does reach an
        # agent from several neighbours.  The per-label miss rate sits at
        # 1 - q whatever the degree; the per-message one falls well below it.
        per_label = self._rates({"q_model_received": 0.5})
        per_message = self._rates(
            {"q_model_received": 0.5, "q_model_draw": "per_message"}
        )
        self.assertGreater(per_label["recv_unique"], 500)
        self.assertGreater(per_label["recv_duplicate"], per_label["recv_unique"])
        self.assertAlmostEqual(
            per_label["recv_missed"] / per_label["recv_unique"], 0.5, delta=0.06
        )
        self.assertLess(
            per_message["recv_missed"] / per_message["recv_unique"], 0.40
        )

    def _rates(self, overrides: dict, degree: int = 11) -> dict:
        totals = {"recv_unique": 0, "recv_missed": 0, "recv_copies_missed": 0,
                  "recv_messages": 0, "recv_duplicate": 0, "own_missed": 0}
        dense = {
            **overrides,
            "experiments_per_agent": 3,
            "environment": {
                "kind": "synthetic", "families": 8, "variants": 2, "task_size": 4
            },
        }
        for seed in (12345, 777, 20260918, 4242):
            produced = run_with(dense, degree=degree, seed=seed)
            for log in produced["transitions"]:
                for key in totals:
                    totals[key] += log[key]
        return totals


class DrawBookkeeping(unittest.TestCase):
    def test_both_counters_are_silent_at_the_default(self) -> None:
        for mode in ("per_label", "per_message"):
            produced = run_with({"q_model_draw": mode}, degree=8, seed=12345)
            for log in produced["transitions"]:
                with self.subTest(mode=mode, round=log["round"]):
                    self.assertEqual(log["recv_missed"], 0)
                    self.assertEqual(log["recv_copies_missed"], 0)

    def test_per_message_at_q_one_is_the_default_run_field_for_field(self) -> None:
        for seed in (12345, 777):
            plain = run_with({}, degree=8, seed=seed)
            switched = run_with({"q_model_draw": "per_message"}, degree=8, seed=seed)
            for left, right in zip(plain["transitions"], switched["transitions"]):
                self.assertEqual(left, right)

    def test_missed_labels_and_missed_copies_are_both_reported(self) -> None:
        for mode in ("per_label", "per_message"):
            produced = run_with(
                {"q_model_received": 0.5, "q_model_draw": mode}, degree=8, seed=12345
            )
            missed = sum(log["recv_missed"] for log in produced["transitions"])
            copies = sum(log["recv_copies_missed"] for log in produced["transitions"])
            with self.subTest(mode=mode):
                self.assertGreater(missed, 0)
                # Every missed label costs at least the copy that carried it.
                self.assertGreaterEqual(copies, missed)

    def test_the_mode_is_recorded_on_the_episode(self) -> None:
        produced = run_with({"q_model_draw": "per_message"}, degree=8, seed=12345)
        self.assertEqual(produced["q_model_draw"], "per_message")
        self.assertEqual(run_with({}, degree=8, seed=12345)["q_model_draw"], "per_label")

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_run_config({**BASE, "degrees": [0], "q_model_draw": "per_agent"})


class DrawDeterminism(unittest.TestCase):
    def test_a_copy_draw_does_not_depend_on_delivery_order(self) -> None:
        runner = build_runner(
            {"q_model_received": 0.5, "q_model_draw": "per_message"},
            degree=8,
            seed=12345,
        )
        rng = random.Random(7)
        deliveries = [
            (sender, _Copy((family, 1)))
            for sender in range(9)
            for family in range(4)
        ]
        reference = runner._per_message_learned(2, 3, deliveries)
        for _ in range(20):
            shuffled = deliveries[:]
            rng.shuffle(shuffled)
            self.assertEqual(runner._per_message_learned(2, 3, shuffled), reference)

    def test_a_copy_draw_does_not_depend_on_how_many_copies_arrive(self) -> None:
        # Each (sender, label) is its own hash, so adding neighbours cannot
        # change the verdict on the neighbours already there.
        runner = build_runner(
            {"q_model_received": 0.5, "q_model_draw": "per_message"},
            degree=8,
            seed=12345,
        )
        pair = (2, 3)
        verdicts = [runner._copy_accepted(2, 3, sender, pair) for sender in range(6)]
        for copies in range(1, 7):
            learned, failed = runner._per_message_learned(
                2, 3, _deliveries(pair, range(copies))
            )
            self.assertEqual(pair in learned, any(verdicts[:copies]))
            self.assertEqual(failed, sum(not v for v in verdicts[:copies]))

    def test_the_same_config_replays_to_the_same_numbers(self) -> None:
        first = run_with(
            {"q_model_received": 0.5, "q_model_draw": "per_message"},
            degree=8,
            seed=12345,
        )
        second = run_with(
            {"q_model_received": 0.5, "q_model_draw": "per_message"},
            degree=8,
            seed=12345,
        )
        self.assertEqual(first["transitions"], second["transitions"])

    def test_the_two_modes_do_not_share_a_stream(self) -> None:
        # Different domains: the per-message run is not the per-label run
        # with the draws relabelled.
        per_label = run_with({"q_model_received": 0.5}, degree=8, seed=12345)
        per_message = run_with(
            {"q_model_received": 0.5, "q_model_draw": "per_message"},
            degree=8,
            seed=12345,
        )
        self.assertNotEqual(
            [log["recv_missed"] for log in per_label["transitions"]],
            [log["recv_missed"] for log in per_message["transitions"]],
        )


class SolvedCondition(unittest.TestCase):
    """``stop_on_success`` must mean what the evaluator means by success."""

    def test_solved_requires_a_held_depth_T_candidate(self) -> None:
        runner = build_runner({}, degree=11, seed=12345)
        asyncio.run(runner.run())
        depth = runner.config.task_size
        for agent in runner.agent_states():
            held = {
                components
                for components in agent.complete_candidates
                if len(components) == depth
            }
            held.update(
                runner.encoding.components(node)
                for node in agent.frontier
                if node.used.bit_count() == depth
            )
            verified = any(agent.evidence.covers_positively(c) for c in held)
            with self.subTest(agent=agent.index):
                self.assertEqual(agent.solved_round is not None, verified)

    def test_solved_agrees_with_the_evaluator_s_evidence_verified(self) -> None:
        # The knob's criterion and the criterion the run is scored on are the
        # same sentence; they used to differ by one round, and the evaluator
        # then scored a truncated episode as P_succ = 0.
        runner = build_runner({}, degree=11, seed=12345, evaluate=True)
        produced = asyncio.run(runner.run())
        evaluation = produced["evaluation"]
        self.assertEqual(
            evaluation["agents_evidence_verified"],
            produced["transitions"][-1]["agents_solved"],
        )

    def test_holding_T_positive_labels_is_no_longer_enough(self) -> None:
        # The old test: T positive labels anywhere in the store.  It fires a
        # round early and for agents that never built the hypothesis, so it
        # must be strictly weaker than what is now recorded.
        runner = build_runner({}, degree=11, seed=777)
        produced = asyncio.run(runner.run())
        labelled = sum(
            len(agent.evidence.known_positive()) == runner.config.task_size
            for agent in runner.agent_states()
        )
        # Every one of the twelve agents ends the episode holding the four
        # positive labels; exactly one of them ever built the hypothesis.
        self.assertEqual(labelled, 12)
        self.assertEqual(produced["agents_solved"], 1)


if __name__ == "__main__":
    unittest.main()
