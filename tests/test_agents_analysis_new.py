#!/usr/bin/env python3
"""Unit tests for the analysis modules added by WP0-NEW.

Every case is built from a hand-written run directory whose answer is known
in closed form, so a failure points at the estimator rather than at the data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis import (  # noqa: E402
    diagnostics,
    llm_diversity,
    order_parameter,
    overlap_measured,
    regimes,
    success_crossover,
)
from src.agent_system.analysis.run_cell import (  # noqa: E402
    cell_parameters,
    cluster_bootstrap,
    linear_fit,
)
from src.utils import delayed_boundary_dc  # noqa: E402


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def manifest(
    directory: Path, *, agents=10, branching=3, m=1, families=8, variants=4, task_size=4,
    kind="synthetic", policy="uniform",
) -> None:
    payload = {
        "run_id": directory.name,
        "config": {
            "agents": agents,
            "branching": branching,
            "experiments_per_agent": m,
            "policy": {"kind": policy},
            "environment": {
                "kind": kind,
                "families": families,
                "variants": variants,
                "task_size": task_size,
            },
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def transition(round_number: int, **fields) -> dict:
    row = {
        "round": round_number,
        "old_pre": 1.0,
        "old_post": 1.0,
        "A_post": 1.0,
        "A_preown": 1.0,
        "tests": 10,
        "tests_pos": 1,
        "tests_neg": 9,
        "recv_messages": 0,
        "recv_duplicate": 0,
        "recv_already_explicit": 0,
        "budget_utilization": 1.0,
        "outgoing_message_availability": 1.0,
        "candidate_capacity_ratio": 10.0,
        "active_agent_fraction": 1.0,
        "frontier_empty_fraction": 0.0,
        "empty_after_receive_fraction": 0.0,
        "agents_exceeding_nominal_m": 0,
        "budget_redistributed_above_m": 0,
        "B_used": 10,
        "N_experiment_active_start": 10,
        "max_experiments_per_agent": 1,
    }
    row.update(fields)
    return row


def episode(
    *, degree: int, task_seed: int, initial: float, a_post, success=False,
    old_pre=None, extra_rounds=None, episode_id=None,
) -> dict:
    rounds = len(a_post)
    old_pre = old_pre if old_pre is not None else list(a_post)
    transitions = []
    for index in range(rounds):
        fields = {"old_pre": old_pre[index], "old_post": old_pre[index],
                  "A_post": a_post[index]}
        if extra_rounds:
            fields.update(extra_rounds[index])
        transitions.append(transition(index + 1, **fields))
    return {
        "status": "ok",
        "episode_id": episode_id or f"d{degree}_t{task_seed}_r0",
        "d": degree,
        "task_seed": task_seed,
        "seed": task_seed,
        "N": 10,
        "A_initial_incorrect": initial,
        "transitions": transitions,
        "evaluation": {f"any_{name}": success for name in
                       ("symbolic_recovered", "evidence_verified", "predictive_success",
                        "verified_predictive_success")},
    }


class RunCellTests(unittest.TestCase):
    def test_parameters_come_back_from_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "cell"
            manifest(run, families=8, variants=4, task_size=5, branching=2, m=3)
            cell = cell_parameters(run)
            self.assertEqual((cell["M"], cell["K"], cell["V"]), (8, 4.0, 32.0))
            self.assertEqual((cell["T"], cell["rounds"], cell["b"], cell["m"]), (5, 4, 2, 3))

    def test_missing_manifest_is_none_not_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cell_parameters(Path(tmp)))

    def test_linear_fit_is_exact_on_a_line(self) -> None:
        fit = linear_fit([(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)])
        self.assertAlmostEqual(fit["slope"], 2.0)
        self.assertAlmostEqual(fit["intercept"], 1.0)

    def test_bootstrap_resamples_tasks_not_episodes(self) -> None:
        groups = {
            0: [{"task_seed": 1, "value": 0.0}] * 5 + [{"task_seed": 2, "value": 10.0}] * 5
        }
        result = cluster_bootstrap(
            groups,
            lambda sample: sum(e["value"] for e in sample[0]) / len(sample[0]),
            draws=200,
            seed=7,
        )
        # Resampling two clusters can only give 0, 5 or 10; episode-level
        # resampling would produce a continuum around 5.
        self.assertEqual(result["clusters"], 2)
        self.assertTrue(result["ci"][0] in {0.0, 5.0, 10.0})
        self.assertTrue(result["ci"][1] in {0.0, 5.0, 10.0})


class OrderParameterTests(unittest.TestCase):
    def _run(self, tmp: str) -> Path:
        run = Path(tmp) / "order"
        manifest(run)
        rows = []
        for task in (1, 2):
            # d = 0 grows by 2x a round, d = 4 decays by 1/2 a round.
            rows.append(episode(degree=0, task_seed=task, initial=8.0, a_post=[16.0, 32.0]))
            rows.append(episode(degree=4, task_seed=task, initial=8.0, a_post=[4.0, 2.0]))
        write_jsonl(run / "episodes.jsonl", rows)
        return run

    def test_trajectory_starts_at_the_initial_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp)
            report = order_parameter.report(run, draws=20)
            self.assertEqual(report["degrees_growing"], [0])
            self.assertEqual(report["degrees_decaying"], [4])
            self.assertTrue(report["separation_is_monotone"])

    def test_endpoint_crossing_interpolates_the_log_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp)
            report = order_parameter.report(run, draws=20)
            crossing = report["endpoint_crossing"]
            # +log 2 per round at d = 0 and -log 2 at d = 4: the zero is the midpoint.
            self.assertEqual(crossing["status"], "crossing")
            self.assertAlmostEqual(crossing["d_c"], 2.0)

    def test_A_t_is_A_post_not_the_lineage_numerator(self) -> None:
        values = order_parameter.trajectory(
            episode(degree=0, task_seed=1, initial=5.0, a_post=[7.0, 9.0], old_pre=[1.0, 1.0])
        )
        self.assertEqual(values, [5.0, 7.0, 9.0])


class SuccessCrossoverTests(unittest.TestCase):
    def test_fit_recovers_a_known_logistic(self) -> None:
        midpoint, scale = 12.0, 2.5
        points = []
        for degree in range(0, 25):
            probability = 1.0 / (1.0 + math.exp(-(degree - midpoint) / scale))
            successes = round(probability * 400)
            points += [(float(degree), 1)] * successes
            points += [(float(degree), 0)] * (400 - successes)
        fit = success_crossover.logistic_fit(points)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["d_half"], midpoint, places=1)
        self.assertAlmostEqual(fit["w"], scale, places=1)

    def test_constant_outcomes_are_reported_not_fitted(self) -> None:
        self.assertEqual(
            success_crossover.logistic_fit([(1.0, 0), (2.0, 0)])["status"], "all_failure"
        )
        self.assertEqual(
            success_crossover.logistic_fit([(1.0, 1), (2.0, 1)])["status"], "all_success"
        )
        self.assertEqual(success_crossover.logistic_fit([])["status"], "no_episodes")

    def test_width_10_90_is_the_scale_times_log_81(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "succ"
            manifest(run)
            rows = []
            for task in (1, 2, 3, 4):
                for degree in (0, 5, 10, 15, 20):
                    rows.append(
                        episode(degree=degree, task_seed=task, initial=8.0,
                                a_post=[8.0, 8.0], success=degree >= 10,
                                episode_id=f"d{degree}_t{task}_r0")
                    )
            write_jsonl(run / "episodes.jsonl", rows)
            report = success_crossover.report(run, criteria=("verified_predictive_success",),
                                              draws=20)
            fit = report["fits"][0]
            self.assertAlmostEqual(fit["width_10_90"], fit["w"] * math.log(81.0))
            self.assertTrue(5.0 <= fit["d_half"] <= 10.0)


class OverlapMeasuredTests(unittest.TestCase):
    def _run(self, tmp: str, slopes, log_b=1.0, variants=4, families=8, task_size=4) -> Path:
        run = Path(tmp) / "overlap"
        manifest(run, families=families, variants=variants, task_size=task_size,
                 branching=round(math.exp(log_b)))
        rows = []
        for task in (1, 2):
            for degree in range(0, 9):
                denominators, numerators = [], []
                previous = 1000.0
                for slope in slopes:
                    denominators.append(previous)
                    value = previous * math.exp(log_b + slope * degree)
                    numerators.append(value)
                    previous = value
                rows.append(
                    episode(degree=degree, task_seed=task, initial=denominators[0],
                            a_post=numerators, old_pre=numerators,
                            episode_id=f"d{degree}_t{task}_r0")
                )
        write_jsonl(run / "episodes.jsonl", rows)
        return run

    def test_slope_and_overlap_are_recovered_exactly(self) -> None:
        slopes = [0.0, -0.08, -0.12]
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp, slopes)
            measurement = overlap_measured.measure(run)
            for row, slope in zip(measurement["rounds"], slopes):
                self.assertAlmostEqual(row["slope_t"], slope, places=9)
                self.assertAlmostEqual(row["p_nbr_t"], 1.0 - math.exp(slope), places=9)
                self.assertAlmostEqual(row["q_measured_t"], 1.0 - math.exp(slope), places=9)
            self.assertAlmostEqual(measurement["rounds"][1]["q_theory_t"], 2.0 / 32.0)
            self.assertFalse(measurement["rounds"][0]["communicated"])
            self.assertTrue(measurement["rounds"][1]["communicated"])

    def test_reconstruction_solves_the_measured_lines(self) -> None:
        slopes = [0.0, -0.08, -0.12]
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp, slopes, log_b=1.0)
            measurement = overlap_measured.measure(run)
            self.assertAlmostEqual(
                measurement["summary"]["reconstruction"], 3.0 / 0.20, places=6
            )

    def test_amplified_boundary_reduces_to_the_frozen_one_at_amp_one(self) -> None:
        cell = {"V": 32.0, "T": 5, "b": 3, "m": 1}
        self.assertAlmostEqual(
            overlap_measured._delayed_with_amplification(cell, 1.0),
            delayed_boundary_dc(32, 1, 5, 3, 1),
        )

    def test_sibling_prediction_falls_below_one_for_a_flat_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp, [0.0, -0.05, -0.08], families=32, variants=1)
            measurement = overlap_measured.measure(run)
            # tests_pos / tests is 1/10 in the fixture, so (1-f) + f(K-1) = 0.9.
            self.assertAlmostEqual(measurement["summary"]["amp_predicted_from_K"], 0.9)


class RegimeTests(unittest.TestCase):
    def test_thresholds_are_the_frozen_ones(self) -> None:
        from src.experiment_config import INDEPENDENCE_LIMITS, SUPPLY_LIMITS

        self.assertEqual(SUPPLY_LIMITS["min_budget_utilization"], 0.85)
        self.assertEqual(INDEPENDENCE_LIMITS["max_duplicate_evidence_fraction"], 0.2)

    def test_clean_cell_is_theory_clean(self) -> None:
        metrics = {
            "min_budget_utilization": 1.0,
            "min_message_availability": 1.0,
            "min_candidate_capacity_ratio": 5.0,
            "max_frontier_empty_fraction": 0.0,
            "max_duplicate_evidence_fraction": 0.05,
        }
        self.assertEqual(regimes.classify(metrics, 0.1)["classification"], regimes.THEORY_CLEAN)

    def test_duplicate_evidence_makes_it_redundant(self) -> None:
        metrics = {
            "min_budget_utilization": 1.0,
            "min_message_availability": 1.0,
            "min_candidate_capacity_ratio": 5.0,
            "max_frontier_empty_fraction": 0.0,
            "max_duplicate_evidence_fraction": 0.5,
        }
        verdict = regimes.classify(metrics, 0.1)
        self.assertEqual(verdict["classification"], regimes.REDUNDANT)
        self.assertIn("duplicate evidence", verdict["independence_reasons"])

    def test_supply_failure_outranks_redundancy(self) -> None:
        metrics = {
            "min_budget_utilization": 0.2,
            "min_message_availability": 1.0,
            "min_candidate_capacity_ratio": 0.1,
            "max_frontier_empty_fraction": 0.9,
            "max_duplicate_evidence_fraction": 0.9,
        }
        verdict = regimes.classify(metrics, 0.9)
        self.assertEqual(verdict["classification"], regimes.DEPLETED)
        self.assertIn("candidate capacity", verdict["supply_reasons"])

    def test_end_to_end_labels_a_depleted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "regime"
            manifest(run)
            starved = [
                {"budget_utilization": 0.1, "frontier_empty_fraction": 0.8,
                 "candidate_capacity_ratio": 0.2},
                {"budget_utilization": 0.1, "frontier_empty_fraction": 0.8,
                 "candidate_capacity_ratio": 0.2},
            ]
            rows = [
                episode(degree=degree, task_seed=task, initial=8.0, a_post=[8.0, 4.0],
                        extra_rounds=starved, episode_id=f"d{degree}_t{task}_r0")
                for degree in (0, 1) for task in (1, 2)
            ]
            write_jsonl(run / "episodes.jsonl", rows)
            report = regimes.report(run)
            self.assertEqual(report["cell"]["classification"], regimes.DEPLETED)
            self.assertEqual(
                report["thresholds"]["source"],
                "configs/clean_criteria.json via src.experiment_config",
            )


class DiagnosticsTests(unittest.TestCase):
    def test_realised_m_exceeds_nominal_when_the_budget_is_redistributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "diag"
            manifest(run, agents=10, m=1)
            redistributed = [
                {"B_used": 10, "N_experiment_active_start": 4,
                 "agents_exceeding_nominal_m": 3, "max_experiments_per_agent": 4,
                 "frontier_empty_fraction": 0.6},
                {"B_used": 10, "N_experiment_active_start": 4,
                 "agents_exceeding_nominal_m": 3, "max_experiments_per_agent": 4,
                 "frontier_empty_fraction": 0.6},
            ]
            rows = [
                episode(degree=degree, task_seed=task, initial=8.0, a_post=[8.0, 4.0],
                        extra_rounds=redistributed, episode_id=f"d{degree}_t{task}_r0")
                for degree in (0, 1) for task in (1, 2)
            ]
            write_jsonl(run / "episodes.jsonl", rows)
            report = diagnostics.report(run)
            at_crossing = report["at_crossing"]
            self.assertAlmostEqual(at_crossing["max_realised_m_per_active_agent"], 2.5)
            self.assertAlmostEqual(at_crossing["max_fraction_exceeding_nominal_m"], 0.3)
            self.assertAlmostEqual(at_crossing["max_frontier_empty_fraction"], 0.6)


class LLMDiversityTests(unittest.TestCase):
    def _run(self, tmp: str) -> Path:
        run = Path(tmp) / "llm"
        manifest(run, policy="llm_full")
        write_jsonl(
            run / "episodes.jsonl",
            [episode(degree=0, task_seed=1, initial=8.0, a_post=[8.0],
                     episode_id="d0_t1_r0")],
        )
        # Four agents, three of them on the same component.
        events = [
            {"kind": "experiment", "episode": "d0_t1_r0", "round": 1, "agent": agent,
             "pair": pair, "experiment_id": experiment}
            for agent, pair, experiment in (
                (0, [1, 1], "e_00"), (1, [1, 1], "e_00"),
                (2, [1, 1], "e_01"), (3, [2, 0], "e_00"),
            )
        ]
        write_jsonl(run / "events.jsonl", events)
        return run

    def test_collision_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp)
            rows = llm_diversity.collision_table(run)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["tests"], 4)
            self.assertEqual(row["distinct_components"], 2)
            self.assertAlmostEqual(row["distinct_per_test"], 0.5)
            self.assertAlmostEqual(row["mean_max_multiplicity"], 3.0)
            # Three of one and one of the other: H = -(3/4)log2(3/4) - (1/4)log2(1/4).
            self.assertAlmostEqual(row["mean_entropy_bits"], 0.8112781244591328)
            self.assertEqual(row["distinct_experiment_ids"], 2)
            self.assertAlmostEqual(row["experiment_id_top2_share"], 1.0)

    def test_position_and_singleton_bias_are_parsed_from_the_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp)
            offered = json.dumps([
                {"family_id": 5, "variant_id": 0, "parent_ids": ["h"]},
                {"family_id": 1, "variant_id": 1, "parent_ids": ["h"]},
            ])
            allowed = json.dumps({"1": [1], "5": [0, 1, 2]})
            prompt = (
                "CANDIDATES YOU MAY TEST THIS SLOT\n" + offered + "\n"
                "values still allowed per family: " + allowed + "\n"
                "WHAT TO RETURN\n"
            )
            write_jsonl(run / "requests.jsonl", [
                {"cache_key": "k", "schema_name": "experiment_decision",
                 "messages": [{"role": "user", "content": prompt}]},
            ])
            write_jsonl(run / "responses.jsonl", [
                {"cache_key": "k", "episode": "d0_t1_r0", "round": 1, "agent": 0,
                 "phase": "experiment", "content": json.dumps({"choice": "c1v1|h"})},
            ])
            rows, overall, histogram = llm_diversity.position_table(run)
            self.assertEqual(overall["decisions"], 1)
            self.assertEqual(overall["P_first"], 0.0)
            self.assertAlmostEqual(overall["uniform_baseline_P_first"], 0.5)
            # Family 1 has a single allowed value and is one of two offers.
            self.assertAlmostEqual(overall["singleton_family_chosen"], 1.0)
            self.assertAlmostEqual(overall["singleton_family_available"], 0.5)
            self.assertEqual(rows[0]["d"], 0)
            self.assertEqual(histogram[0]["position"], 1)

    def test_a_run_without_a_transcript_is_reported_not_crashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(tmp)
            _rows, overall, _histogram = llm_diversity.position_table(run)
            self.assertEqual(overall["status"], "no_llm_transcript")

    def test_both_response_shapes_yield_the_component(self) -> None:
        self.assertEqual(
            llm_diversity._chosen_component(json.dumps({"choice": "c3v2|h_x"})), (3, 2)
        )
        self.assertEqual(
            llm_diversity._chosen_component(
                json.dumps({"family_id": 3, "variant_id": 2, "experiment_id": "e_0"})
            ),
            (3, 2),
        )
        self.assertIsNone(llm_diversity._chosen_component("not json"))


if __name__ == "__main__":
    unittest.main()
