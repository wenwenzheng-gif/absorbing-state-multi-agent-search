#!/usr/bin/env python3
"""Regression tests for the WP0 fixes to ``src/agent_system/analysis``.

Every input here is synthetic and written to a temporary directory, so none
of it depends on ``runs/`` existing or on any particular run being current.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis import scaling  # noqa: E402
from src.agent_system.analysis import success_model  # noqa: E402
from src.agent_system.analysis.critical_degree import (  # noqa: E402
    CONVENTIONS,
    STATUS_EXTINCT,
    STATUS_NOT_ESTIMABLE,
    STATUS_OK,
    STATUS_PARTIAL,
    cluster_bootstrap,
    convention_available,
    find_crossings,
    fit_log_ratio,
    fit_window_degrees,
    pooled_log_ratio,
    scan,
)
from src.agent_system.analysis.plots import build_tables  # noqa: E402


def episode(old_pre, a_post, initial=10, task_seed=1, degree=0, old_post=None):
    if old_post is None:
        old_post = list(a_post)
    transitions = [
        {"old_pre": numerator, "old_post": own, "A_post": posterior}
        for numerator, own, posterior in zip(old_pre, old_post, a_post)
    ]
    return {
        "status": "ok",
        "d": degree,
        "task_seed": task_seed,
        "A_initial_incorrect": initial,
        "transitions": transitions,
    }


def curve_of(values: dict[int, float | None]) -> dict[int, dict]:
    out = {}
    for degree, value in values.items():
        if value is None:
            out[degree] = {"value": None, "status": STATUS_NOT_ESTIMABLE, "episodes": 4}
        elif value == -math.inf:
            out[degree] = {"value": -math.inf, "status": STATUS_EXTINCT, "episodes": 4}
        else:
            out[degree] = {"value": value, "status": STATUS_OK, "episodes": 4}
    return out


def write_run(
    root: Path,
    name: str,
    *,
    agents: int,
    families: int,
    variants: int,
    task_size: int,
    branching: int,
    d_c: float,
    crossing_status: str = "single_crossing",
    kind: str = "synthetic",
) -> Path:
    """A run directory carrying only what ``scaling`` reads back."""
    directory = root / name
    (directory / "analysis").mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "agents": agents,
                    "branching": branching,
                    "experiments_per_agent": 1,
                    "environment": {
                        "kind": kind,
                        "families": families,
                        "variants": variants,
                        "task_size": task_size,
                        "tcas_space": "v1_460800",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (directory / "analysis" / "critical_degree.json").write_text(
        json.dumps(
            {
                "convention": "pre_own",
                "crossing": {"status": crossing_status, "d_c": d_c},
                "bootstrap": {
                    "clusters": 8,
                    "d_c_ci": [d_c - 1.0, d_c + 1.0] if d_c is not None else None,
                    "fit": {"d_c_fit_se": 0.5},
                },
                "fit": {"d_c_fit": d_c, "status": "ok"} if d_c is not None else {},
            }
        ),
        encoding="utf-8",
    )
    return directory


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class DedupKeyTests(unittest.TestCase):
    """Fix 1: two ``N`` at the same (M, K, T, b, m) are two cells, not one."""

    def _pair(self, root: Path) -> None:
        write_run(root, "small_N", agents=40, families=8, variants=4,
                  task_size=4, branching=3, d_c=16.65)
        write_run(root, "zz_big_N", agents=80, families=8, variants=4,
                  task_size=4, branching=3, d_c=18.00)

    def test_both_agent_counts_survive_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._pair(root)
            summary = scaling.report(
                [root / "small_N", root / "zz_big_N"], root / "out", depletion_headroom=1.0
            )
            rows = read_csv(root / "out" / "cells.csv")
        self.assertEqual(summary["cells"], 2)
        self.assertEqual(summary["cells_by_agents"], {"40": 1, "80": 1})
        self.assertEqual(sorted(row["N"] for row in rows), ["40", "80"])

    def test_the_agents_filter_keeps_one_N_and_says_why_the_other_went(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._pair(root)
            summary = scaling.report(
                [root / "small_N", root / "zz_big_N"], root / "out",
                depletion_headroom=1.0, agents=80,
            )
        self.assertEqual(summary["cells"], 1)
        self.assertEqual(summary["agents_filter"], 80)
        reasons = [entry["reason"] for entry in summary["dropped"]]
        self.assertIn("N=40 filtered out by --agents", reasons)

    def test_a_true_duplicate_is_still_dropped_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, "a_run", agents=80, families=8, variants=4,
                      task_size=4, branching=3, d_c=18.0)
            write_run(root, "b_run", agents=80, families=8, variants=4,
                      task_size=4, branching=3, d_c=18.4)
            summary = scaling.report(
                [root / "a_run", root / "b_run"], root / "out", depletion_headroom=1.0
            )
        self.assertEqual(summary["cells"], 1)
        self.assertEqual(summary["dropped_cells"], 1)
        self.assertIn("duplicate cell, kept a_run", summary["dropped"][0]["reason"])

    def test_each_N_is_fitted_separately_and_the_pool_records_its_N(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for index, (task_size, d_c) in enumerate(
                [(3, 34.0), (4, 24.0), (5, 18.0), (6, 15.0), (7, 13.0)]
            ):
                runs.append(write_run(root, f"n80_{index}", agents=80, families=32,
                                      variants=1, task_size=task_size, branching=3, d_c=d_c))
            for index, (task_size, d_c) in enumerate(
                [(3, 20.0), (4, 16.0), (5, 13.0), (6, 11.0), (7, 9.0)]
            ):
                runs.append(write_run(root, f"n40_{index}", agents=40, families=32,
                                      variants=2, task_size=task_size, branching=3, d_c=d_c))
            summary = scaling.report(runs, root / "out", depletion_headroom=1.0)
        self.assertEqual(sorted(summary["fits_by_agents_all_cells"]), ["40", "80"])
        self.assertEqual(summary["fits_all_cells"]["collapsed_into_V"]["agents"], [40, 80])
        self.assertEqual(
            summary["fits_by_agents_all_cells"]["80"]["collapsed_into_V"]["agents"], [80]
        )


class NoFakeKTests(unittest.TestCase):
    """Fix 2: tcas has no ``K``, so it stays out of the separate fit."""

    def _cells(self, root: Path) -> list[Path]:
        runs = []
        for index, (task_size, d_c) in enumerate(
            [(3, 34.0), (4, 24.0), (5, 18.0), (6, 15.0), (7, 13.0), (8, 12.0)]
        ):
            runs.append(write_run(root, f"toy_{index}", agents=80, families=32,
                                  variants=1, task_size=task_size, branching=3, d_c=d_c))
        for index, (task_size, d_c) in enumerate([(4, 25.3), (5, 19.0), (6, 16.4)]):
            runs.append(write_run(root, f"tcas_{index}", agents=80, families=12,
                                  variants=4, task_size=task_size, branching=3,
                                  d_c=d_c, kind="tcas"))
        return runs

    def test_tcas_is_marked_as_having_no_M_times_K_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = scaling.cell_of(
                write_run(root, "tcas", agents=40, families=12, variants=4,
                          task_size=4, branching=3, d_c=25.3, kind="tcas")
            )
        self.assertFalse(cell["mk_structure"])
        # |V| is still the real component count -- the sum of the twelve
        # parameters' unequal value counts -- only the split into M x K is
        # fictional.
        self.assertAlmostEqual(cell["V"], 44.0)
        self.assertEqual(cell["M"], 12)

    def test_tcas_enters_the_collapsed_fit_but_not_the_separate_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = scaling.report(self._cells(root), root / "out", depletion_headroom=1.0)
        self.assertEqual(summary["cells"], 9)
        self.assertEqual(summary["cells_without_MK_structure"], 3)
        fit = summary["fits_all_cells"]
        self.assertEqual(fit["collapsed_into_V"]["n_cells"], 9)
        self.assertEqual(fit["M_and_K_separate"]["n_cells"], 6)
        self.assertEqual(fit["M_and_K_separate"]["cells_without_MK_structure_excluded"], 3)

    def test_a_synthetic_cell_keeps_its_M_times_K_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = scaling.cell_of(
                write_run(root, "toy", agents=80, families=8, variants=4,
                          task_size=4, branching=3, d_c=18.0)
            )
        self.assertTrue(cell["mk_structure"])


class CrossingStatusTests(unittest.TestCase):
    """Fix 3: a crossing that is not clean is listed, not silently fitted."""

    def test_a_multiple_crossing_cell_is_dropped_with_its_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = write_run(root, "good", agents=80, families=32, variants=1,
                             task_size=4, branching=3, d_c=24.0)
            messy = write_run(root, "messy", agents=80, families=32, variants=1,
                              task_size=5, branching=3, d_c=18.0,
                              crossing_status="multiple_crossings")
            summary = scaling.report([good, messy], root / "out", depletion_headroom=1.0)
            dropped = read_csv(root / "out" / "dropped_cells.csv")
        self.assertEqual(summary["cells"], 1)
        self.assertEqual(len(dropped), 1)
        self.assertIn("multiple_crossings", dropped[0]["drop_reason"])

    def test_a_no_crossing_cell_is_listed_rather_than_lost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = write_run(root, "good", agents=80, families=32, variants=1,
                             task_size=4, branching=3, d_c=24.0)
            none = write_run(root, "flat", agents=80, families=32, variants=1,
                             task_size=5, branching=3, d_c=None,
                             crossing_status="no_crossing_all_positive")
            summary = scaling.report([good, none], root / "out", depletion_headroom=1.0)
        self.assertEqual(summary["cells"], 1)
        self.assertEqual(
            [entry["reason"] for entry in summary["dropped"]],
            ["no crossing (no_crossing_all_positive)"],
        )

    def test_a_looser_status_list_lets_the_messy_cell_back_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = write_run(root, "good", agents=80, families=32, variants=1,
                             task_size=4, branching=3, d_c=24.0)
            messy = write_run(root, "messy", agents=80, families=32, variants=1,
                              task_size=5, branching=3, d_c=18.0,
                              crossing_status="multiple_crossings")
            summary = scaling.report(
                [good, messy], root / "out", depletion_headroom=1.0,
                crossing_statuses=("single_crossing", "multiple_crossings"),
            )
        self.assertEqual(summary["cells"], 2)

    def test_cell_of_hides_an_unresolved_crossing_unless_asked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(root, "flat", agents=80, families=32, variants=1,
                                  task_size=5, branching=3, d_c=None,
                                  crossing_status="no_crossing_all_positive")
            self.assertIsNone(scaling.cell_of(directory))
            cell = scaling.cell_of(directory, include_unresolved=True)
        self.assertEqual(cell["crossing_status"], "no_crossing_all_positive")
        self.assertIsNone(cell["d_c"])


class FitEstimatorTests(unittest.TestCase):
    """Fix 4: the fitted estimator the paper's wording describes."""

    def test_a_planted_log_linear_curve_is_recovered_exactly(self) -> None:
        # log R = 2.0 - 0.2 d crosses zero at d = 10.
        curve = curve_of({d: 2.0 - 0.2 * d for d in range(0, 16, 2)})
        fit = fit_log_ratio(curve)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["d_c_fit"], 10.0)
        self.assertAlmostEqual(fit["slope"], -0.2)
        self.assertAlmostEqual(fit["intercept"], 2.0)
        self.assertFalse(fit["extrapolated"])

    def test_the_fit_uses_the_whole_window_not_just_the_straddling_pair(self) -> None:
        # One noisy degree next to the crossing moves the interpolation a lot
        # and the fitted line hardly at all.
        values = {d: 2.0 - 0.2 * d for d in range(0, 16)}
        values[10] = -0.35
        curve = curve_of(values)
        interpolated = find_crossings(curve)["d_c"]
        fitted = fit_log_ratio(curve)["d_c_fit"]
        self.assertLess(abs(fitted - 10.0), abs(interpolated - 10.0))

    def test_extinct_and_unestimable_degrees_never_enter_the_fit(self) -> None:
        curve = curve_of({0: 2.0, 2: 1.6, 4: 1.2, 6: 0.8, 8: -math.inf, 10: None})
        self.assertEqual(fit_window_degrees(curve), [0, 2, 4, 6])
        self.assertEqual(fit_log_ratio(curve)["n_points"], 4)

    def test_a_degree_past_extinction_does_not_reopen_the_window(self) -> None:
        curve = curve_of({0: 2.0, 1: 1.0, 2: -math.inf, 3: 0.4})
        self.assertEqual(fit_window_degrees(curve), [0, 1])
        self.assertEqual(fit_window_degrees(curve, "all"), [0, 1, 3])

    def test_a_magnitude_bound_trims_the_window(self) -> None:
        curve = curve_of({0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0, 4: -1.0, 5: -3.0})
        self.assertEqual(fit_window_degrees(curve, 1.5), [2, 3, 4])

    def test_a_flat_or_rising_curve_has_no_fitted_crossing(self) -> None:
        fit = fit_log_ratio(curve_of({0: 1.0, 1: 1.1, 2: 1.2, 3: 1.3}))
        self.assertEqual(fit["status"], "no_decay_in_degree")
        self.assertIsNone(fit["d_c_fit"])

    def test_two_points_are_not_enough_for_a_fit(self) -> None:
        fit = fit_log_ratio(curve_of({0: 1.0, 1: -1.0}))
        self.assertEqual(fit["status"], "too_few_points")
        self.assertIsNone(fit["d_c_fit"])

    def test_weighting_by_episodes_shifts_the_line_towards_the_heavy_degrees(self) -> None:
        curve = curve_of({0: 2.0, 1: 1.8, 2: 1.6, 3: 0.0})
        curve[3]["episodes"] = 1000
        plain = fit_log_ratio(curve)
        weighted = fit_log_ratio(curve, weighted=True)
        self.assertNotAlmostEqual(plain["slope"], weighted["slope"])
        self.assertTrue(weighted["weighted"])

    def test_the_bootstrap_reports_a_standard_error_for_the_fit(self) -> None:
        by_degree = {}
        for degree in range(0, 8):
            group = []
            for task in range(1, 7):
                # log R falls by 0.3 per degree; a per-task jitter gives the
                # bootstrap something to resample.
                numerator = 10.0 * math.exp(-0.3 * degree) * (1.0 + 0.05 * task)
                group.append(
                    episode([numerator], [10.0], initial=5.0, task_seed=task, degree=degree)
                )
            by_degree[degree] = group
        result = cluster_bootstrap(by_degree, draws=60, seed=7)
        self.assertEqual(result["clusters"], 6)
        self.assertGreater(result["fit"]["valid_fraction"], 0.9)
        self.assertIsNotNone(result["fit"]["d_c_fit_se"])
        self.assertLess(result["fit"]["d_c_fit_ci"][0], result["fit"]["d_c_fit_median"])
        self.assertGreater(result["fit"]["d_c_fit_ci"][1], result["fit"]["d_c_fit_median"])


class LiteralConventionTests(unittest.TestCase):
    """Fix 4: the paper's literal ``A_{t+1} / A_t`` as a third convention."""

    def test_literal_reads_the_whole_incorrect_population(self) -> None:
        self.assertEqual(CONVENTIONS["literal"], "A_post")
        rows = [episode([8, 8], [4, 6], initial=10, old_post=[2, 2])]
        literal = pooled_log_ratio(rows, "literal")["value"]
        self.assertAlmostEqual(literal, math.log(4 / 10) + math.log(6 / 4))

    def test_literal_differs_from_both_lineage_conventions(self) -> None:
        rows = [episode([8, 8], [4, 6], initial=10, old_post=[2, 2])]
        values = {
            name: pooled_log_ratio(rows, name)["value"]
            for name in ("pre_own", "post_own", "literal")
        }
        self.assertEqual(len(set(values.values())), 3)

    def test_a_missing_field_is_reported_as_absent_not_as_zero(self) -> None:
        stripped = {
            "status": "ok", "d": 0, "task_seed": 1, "A_initial_incorrect": 10,
            "transitions": [{"old_pre": 8}],
        }
        self.assertFalse(convention_available([stripped], "literal"))
        self.assertTrue(convention_available([stripped], "pre_own"))


class PartialRoundTests(unittest.TestCase):
    """Fix 5: one dead round is recorded, and never silently dropped."""

    def test_a_dead_round_is_named_with_its_reason(self) -> None:
        rows = [episode([8, 0, 5], [4, 0, 5], initial=10)]
        result = pooled_log_ratio(rows)
        self.assertEqual(result["status"], STATUS_NOT_ESTIMABLE)
        self.assertEqual([entry["round"] for entry in result["failed_rounds"]], [3])
        self.assertIn("round(s) 3", result["reason"])
        self.assertEqual(result["failed_rounds"][0]["denominator"], 0.0)

    def test_the_partial_sum_is_opt_in_and_flagged(self) -> None:
        rows = [episode([8, 4, 5], [4, 0, 5], initial=10)]
        partial = pooled_log_ratio(rows, allow_partial=True)
        self.assertEqual(partial["status"], STATUS_PARTIAL)
        self.assertEqual(partial["rounds_used"], 2)
        self.assertAlmostEqual(partial["value"], math.log(8 / 10) + math.log(4 / 4))

    def test_a_partial_degree_is_not_treated_as_a_measurement(self) -> None:
        # Nothing downstream may read a partial value as if it were a full
        # product: it is a sum over fewer rounds and would sit too high.
        curve = {
            0: {"value": 1.0, "status": STATUS_OK, "episodes": 4},
            1: {"value": 0.5, "status": STATUS_PARTIAL, "episodes": 4},
            2: {"value": -1.0, "status": STATUS_OK, "episodes": 4},
        }
        self.assertEqual(fit_window_degrees(curve), [0])
        self.assertIsNone(find_crossings(curve)["d_c"])

    def test_every_round_dead_stays_not_estimable_even_when_partial(self) -> None:
        rows = [episode([0, 0], [0, 0], initial=0)]
        result = pooled_log_ratio(rows, allow_partial=True)
        self.assertEqual(result["status"], STATUS_NOT_ESTIMABLE)
        self.assertIsNone(result["value"])

    def test_a_degenerate_high_degree_does_not_throw_the_fit(self) -> None:
        curve = curve_of({0: 2.0, 1: 1.6, 2: 1.2, 3: 0.8, 4: 0.4})
        curve[5] = {"value": None, "status": STATUS_NOT_ESTIMABLE, "episodes": 4}
        fit = fit_log_ratio(curve)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["d_c_fit"], 5.0)


class BootstrapQuantileTests(unittest.TestCase):
    """Fix 6: interpolated percentiles, and the sample size stated up front."""

    def test_quantiles_interpolate_rather_than_snapping_to_an_order_statistic(self) -> None:
        from src.agent_system.analysis.critical_degree import _quantile

        values = [0.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(_quantile(values, 0.5), 1.5)
        self.assertAlmostEqual(_quantile(values, 0.25), 0.75)
        self.assertIsNone(_quantile([], 0.5))

    def test_clusters_and_valid_fraction_are_reported(self) -> None:
        by_degree = {
            0: [episode([20, 20], [10, 10], initial=10, task_seed=t, degree=0) for t in (1, 2, 3)],
            1: [episode([1, 1], [1, 1], initial=10, task_seed=t, degree=1) for t in (1, 2, 3)],
        }
        result = cluster_bootstrap(by_degree, draws=40, seed=3)
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["valid_fraction"], 1.0)
        self.assertIn("fit", result)

    def test_no_tasks_still_reports_the_cluster_count(self) -> None:
        result = cluster_bootstrap({}, draws=10, seed=1)
        self.assertEqual(result["clusters"], 0)
        self.assertEqual(result["draws"], 0)
        self.assertIsNone(result["d_c_ci"])


class ScanOutputTests(unittest.TestCase):
    """The scan writes both estimators for every available convention."""

    def _run_dir(self, root: Path) -> Path:
        directory = root / "run"
        directory.mkdir(parents=True)
        lines = []
        for degree in range(0, 10):
            for task in (1, 2, 3, 4):
                factor = math.exp(-0.25 * degree)
                lines.append(
                    json.dumps(
                        {
                            "episode_id": f"d{degree}_t{task}",
                            "status": "ok",
                            "d": degree,
                            "task_seed": task,
                            "N": 40,
                            "A_initial_incorrect": 100.0,
                            "transitions": [
                                {
                                    "old_pre": 250.0 * factor * (1.0 + 0.02 * task),
                                    "old_post": 200.0 * factor,
                                    "A_post": 100.0,
                                }
                            ],
                            "evaluation": {
                                "any_symbolic_recovered": degree > 4,
                                "any_verified_predictive_success": False,
                            },
                        }
                    )
                )
        (directory / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    def test_every_convention_gets_both_estimators_in_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self._run_dir(root)
            report = scan(directory, draws=20)
            rows = read_csv(directory / "analysis" / "d_c_estimates.csv")
        self.assertEqual(sorted(report["alternate_conventions"]), ["literal", "post_own"])
        pairs = {(row["convention"], row["estimator"]) for row in rows}
        for name in ("pre_own", "post_own", "literal"):
            self.assertIn((name, "interpolation"), pairs)
            self.assertIn((name, "fit"), pairs)
        self.assertIsNotNone(report["fit"]["d_c_fit"])
        self.assertEqual(report["bootstrap_clusters"], 4)

    def test_the_curve_csv_says_which_degrees_the_fit_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self._run_dir(root)
            scan(directory, draws=5)
            rows = read_csv(directory / "analysis" / "log_R_by_degree.csv")
        self.assertIn("in_fit_window", rows[0])
        self.assertTrue(any(row["in_fit_window"] == "True" for row in rows))


class JensenTests(unittest.TestCase):
    """Fix 7: the mean of the prediction, not the prediction at the mean."""

    def _episodes(self, finals, degree=0, agents=4, criteria=None):
        criteria = criteria or {}
        return [
            {
                "status": "ok",
                "d": degree,
                "N": agents,
                "transitions": [{"A_post": value}],
                "evaluation": {f"any_{k}": v for k, v in criteria.items()},
            }
            for value in finals
        ]

    def test_the_prediction_is_averaged_over_episodes(self) -> None:
        rows = success_model.rows_by_degree(
            self._episodes([0.0, 99.0]), criterion="symbolic_recovered"
        )
        row = rows[0]
        self.assertAlmostEqual(row["P_predicted"], (1.0 + 0.01) / 2)
        self.assertAlmostEqual(row["P_pred_of_mean_A"], 1.0 / (1.0 + 49.5))
        self.assertGreater(row["P_predicted"], 10 * row["P_pred_of_mean_A"])

    def test_the_per_agent_and_any_agent_forms_are_averaged_too(self) -> None:
        finals = [0.0, 99.0]
        agents = 4
        rows = success_model.rows_by_degree(
            self._episodes(finals, agents=agents), criterion="symbolic_recovered"
        )
        row = rows[0]
        expected = sum(1.0 / (1.0 + value / agents) for value in finals) / len(finals)
        self.assertAlmostEqual(row["P_predicted_per_agent"], expected)
        expected_any = sum(
            1.0 - (1.0 - 1.0 / (1.0 + value / agents)) ** agents for value in finals
        ) / len(finals)
        self.assertAlmostEqual(row["P_predicted_any_agent"], expected_any)
        self.assertAlmostEqual(
            row["P_pred_per_agent_of_mean_A"], 1.0 / (1.0 + 49.5 / agents)
        )

    def test_a_constant_A_T_makes_the_two_readings_agree(self) -> None:
        rows = success_model.rows_by_degree(
            self._episodes([7.0, 7.0, 7.0]), criterion="symbolic_recovered"
        )
        self.assertAlmostEqual(rows[0]["P_predicted"], rows[0]["P_pred_of_mean_A"])

    def test_every_recorded_criterion_gets_its_own_column(self) -> None:
        episodes = self._episodes(
            [3.0], criteria={"symbolic_recovered": True, "verified_predictive_success": False}
        )
        rows = success_model.rows_by_degree(episodes)
        row = rows[0]
        self.assertEqual(row["P_observed_symbolic_recovered"], 1.0)
        self.assertEqual(row["P_observed_verified_predictive_success"], 0.0)
        # The named criterion is still the one the headline columns use.
        self.assertEqual(row["criterion"], success_model.PRIMARY)
        self.assertEqual(row["P_observed"], row[f"P_observed_{success_model.PRIMARY}"])

    def test_the_criteria_are_read_off_the_episodes(self) -> None:
        episodes = self._episodes([1.0], criteria={"symbolic_recovered": True})
        self.assertEqual(success_model.criteria_present(episodes), ["symbolic_recovered"])

    def test_the_report_names_each_criterion_and_its_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            directory.mkdir(parents=True)
            episodes = []
            for degree in (0, 1):
                episodes += self._episodes(
                    [1.0 * (degree + 1), 3.0 * (degree + 1)],
                    degree=degree,
                    criteria={"symbolic_recovered": degree == 1,
                              "verified_predictive_success": False},
                )
            (directory / "episodes.jsonl").write_text(
                "\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8"
            )
            summary = success_model.report(directory)
            rows = read_csv(directory / "analysis" / "success_model.csv")
        self.assertEqual(
            sorted(summary["by_criterion"]),
            ["symbolic_recovered", "verified_predictive_success"],
        )
        self.assertIn("P_observed_symbolic_recovered", rows[0])
        self.assertIn("P_pred_of_mean_A", rows[0])
        self.assertIn("P_predicted", summary["columns"])
        self.assertIsNotNone(summary["mean_absolute_error_of_mean_A"])


class PlotConventionTests(unittest.TestCase):
    """Fix 8: the convention is an argument, and cost is not one episode."""

    def _run_dir(self, root: Path) -> Path:
        directory = root / "run"
        directory.mkdir(parents=True)
        lines = []
        for degree in (0, 2):
            for task, tokens in ((1, 100), (2, 300)):
                lines.append(
                    json.dumps(
                        {
                            "episode_id": f"d{degree}_t{task}",
                            "status": "ok",
                            "d": degree,
                            "task_seed": task,
                            "A_initial_incorrect": 100.0,
                            "transitions": [
                                {"old_pre": 150.0, "old_post": 80.0, "A_post": 100.0},
                                {"old_pre": 120.0, "old_post": 60.0, "A_post": 100.0},
                            ],
                            "llm_usage": {"llm_total_tokens": tokens},
                            "evaluation": {"any_predictive_success": True},
                        }
                    )
                )
        (directory / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    def test_the_per_round_table_follows_the_requested_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._run_dir(Path(tmp))
            pre = build_tables(directory)["per_round_response"]
            post = build_tables(directory, "post_own")["per_round_response"]
        self.assertTrue(all(row["convention"] == "pre_own" for row in pre))
        self.assertTrue(all(row["convention"] == "post_own" for row in post))
        self.assertAlmostEqual(pre[0]["log_r"], math.log(150.0 / 100.0))
        self.assertAlmostEqual(post[0]["log_r"], math.log(80.0 / 100.0))

    def test_the_pooled_curve_follows_it_as_well(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._run_dir(Path(tmp))
            tables = build_tables(directory, "post_own")
        self.assertAlmostEqual(
            tables["log_R_vs_degree"][0]["log_R"],
            math.log(80.0 / 100.0) + math.log(60.0 / 100.0),
        )

    def test_token_cost_is_the_mean_and_the_sum_over_the_degree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._run_dir(Path(tmp))
            cost = build_tables(directory)["cost_vs_success"]
        self.assertEqual(cost[0]["mean_llm_total_tokens"], 200.0)
        self.assertEqual(cost[0]["sum_llm_total_tokens"], 400.0)
        self.assertEqual(cost[0]["llm_episodes"], 2)
        self.assertNotIn("llm_total_tokens", cost[0])


if __name__ == "__main__":
    unittest.main()
