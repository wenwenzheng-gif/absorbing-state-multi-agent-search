#!/usr/bin/env python3
"""Tests for the local fit window added to ``critical_degree.fit_log_ratio``.

``log R(d)`` is steep near the crossing and flattens or plateaus at high
``d``, so the global least-squares window (``fit_window="estimable"``) lets
distant points bias ``d_c_fit`` away from the published interpolation.  The
new default, ``"local"``, restricts the fit to the measured degrees around
the interpolated crossing (see ``fit_window_degrees_local`` in
``critical_degree.py``).  These tests check: an exactly log-linear curve
recovers the same crossing under every estimator (a); a curve that bends and
plateaus at high ``d`` is where local and global disagree, with local staying
close to the true crossing (b); sparse grids still resolve (c); a curve with
no crossing falls back to the old global window and says so (d); and the fit
payload keeps every field it had before this change (e).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.analysis.critical_degree import (  # noqa: E402
    STATUS_OK,
    find_crossings,
    fit_log_ratio,
    fit_window_degrees,
    fit_window_degrees_local,
)


def curve_of(values: dict[int, float]) -> dict[int, dict]:
    return {
        degree: {"value": value, "status": STATUS_OK, "episodes": 8}
        for degree, value in values.items()
    }


# The fields ``fit_log_ratio`` returned before the local window was added,
# for a curve with enough points to reach a clean ``d_c_fit`` (the fields
# present on "too_few_points" or "no_decay_in_degree" were, and still are, a
# strict subset of these -- the code returns early before computing them).
PRE_CHANGE_FIT_FIELDS = {
    "window", "weighted", "degrees", "n_points", "d_c_fit", "intercept",
    "slope", "R2", "residual_sd", "status", "d_min", "d_max", "extrapolated",
}


class ExactlyLinearCurve(unittest.TestCase):
    """(a) No bend at all: every estimator must agree with the interpolation."""

    def test_local_and_global_fits_agree_with_interpolation(self) -> None:
        # slope -0.29, not -0.3: the crossing must not land exactly on an
        # integer degree, or no *strict* sign change is ever measured.
        true_dc = 3.0 / 0.29
        curve = curve_of({d: 3.0 - 0.29 * d for d in range(0, 41)})
        interpolated = find_crossings(curve)["d_c"]
        local_fit = fit_log_ratio(curve)  # default window="local"
        global_fit = fit_log_ratio(curve, "estimable")
        self.assertEqual(local_fit["fit_window_mode"], "local")
        self.assertAlmostEqual(interpolated, true_dc)
        self.assertAlmostEqual(local_fit["d_c_fit"], true_dc, places=6)
        self.assertAlmostEqual(global_fit["d_c_fit"], true_dc, places=6)


class BendingAndPlateauingCurve(unittest.TestCase):
    """(b) Steep near the crossing, flat far from it: the case the fix is for."""

    def curve(self):
        values = {d: 2.0 - 0.21 * d for d in range(0, 13)}
        values.update({d: -1.0 for d in range(13, 41)})  # plateau past the bend
        return curve_of(values), 2.0 / 0.21

    def test_local_fit_is_close_global_fit_is_not(self) -> None:
        curve, true_dc = self.curve()
        crossing = find_crossings(curve)
        self.assertEqual(crossing["status"], "single_crossing")
        local_fit = fit_log_ratio(curve)
        global_fit = fit_log_ratio(curve, "estimable")
        local_error = abs(local_fit["d_c_fit"] - true_dc) / true_dc
        global_error = abs(global_fit["d_c_fit"] - true_dc) / true_dc
        self.assertLess(local_error, 0.02, f"local fit off by {local_error:.1%}")
        self.assertGreater(global_error, 0.10, f"global fit only off by {global_error:.1%}")

    def test_local_window_is_anchored_on_the_bracket_not_the_whole_curve(self) -> None:
        curve, _ = self.curve()
        crossing = find_crossings(curve)
        degrees, mode = fit_window_degrees_local(curve, crossing, k=3, bound=1.0)
        self.assertEqual(mode, "local")
        self.assertEqual(crossing["bracket"], [9, 10])
        # k=3 measured degrees each side of the bracket.
        self.assertEqual(degrees, list(range(6, 14)))


class SparseGrids(unittest.TestCase):
    """(c) Coarser grids must still resolve, at both fit windows."""

    def test_step_four_and_step_eight_grids_resolve(self) -> None:
        for step in (4, 8):
            with self.subTest(step=step):
                curve = curve_of({d: 3.0 - 0.3 * d for d in range(0, 41, step)})
                crossing = find_crossings(curve)
                self.assertEqual(crossing["status"], "single_crossing")
                local_fit = fit_log_ratio(curve)
                self.assertEqual(local_fit["status"], "ok")
                self.assertEqual(local_fit["fit_window_mode"], "local")
                self.assertAlmostEqual(local_fit["d_c_fit"], 10.0, places=6)
                self.assertAlmostEqual(crossing["d_c"], 10.0, places=6)


class NoCrossingFallback(unittest.TestCase):
    """(d) No crossing to anchor on: fall back to the old global window."""

    def test_all_positive_curve_falls_back_and_says_so(self) -> None:
        curve = curve_of({d: 1.0 + 0.05 * d for d in range(0, 10)})
        crossing = find_crossings(curve)
        self.assertIsNone(crossing["d_c"])
        fit = fit_log_ratio(curve)
        self.assertEqual(fit["fit_window_mode"], "estimable_fallback")
        self.assertEqual(fit["degrees"], fit_window_degrees(curve, "estimable"))

    def test_all_negative_curve_falls_back_and_says_so(self) -> None:
        curve = curve_of({d: -1.0 - 0.05 * d for d in range(0, 10)})
        crossing = find_crossings(curve)
        self.assertIsNone(crossing["d_c"])
        fit = fit_log_ratio(curve)
        self.assertEqual(fit["fit_window_mode"], "estimable_fallback")
        self.assertEqual(fit["degrees"], fit_window_degrees(curve, "estimable"))


class OutputFieldsAreNotLost(unittest.TestCase):
    """(e) Every field the fit payload had before this change is still there."""

    def test_local_fit_payload_keeps_every_old_field(self) -> None:
        curve = curve_of({d: 3.0 - 0.3 * d for d in range(0, 41)})
        fit = fit_log_ratio(curve)
        self.assertTrue(PRE_CHANGE_FIT_FIELDS.issubset(fit.keys()))
        # And the new fields this change adds, additively.
        for field in ("fit_window_mode", "local_k", "local_bound", "slope_low_d",
                      "n_points_low_d", "degrees_low_d"):
            self.assertIn(field, fit)

    def test_global_fit_payload_keeps_every_old_field(self) -> None:
        curve = curve_of({d: 3.0 - 0.3 * d for d in range(0, 41)})
        fit = fit_log_ratio(curve, "estimable")
        self.assertTrue(PRE_CHANGE_FIT_FIELDS.issubset(fit.keys()))

    def test_slope_low_d_is_reported_regardless_of_window(self) -> None:
        # log R falls twice as fast below d=8 as above it -- slope_low_d must
        # read the low-degree rate, not whatever the chosen fit window used.
        values = {d: 2.0 - 0.4 * d for d in range(0, 9)}
        values.update({d: values[8] - 0.1 * (d - 8) for d in range(9, 30)})
        curve = curve_of(values)
        fit_local = fit_log_ratio(curve)
        fit_global = fit_log_ratio(curve, "estimable")
        self.assertIsNotNone(fit_local["slope_low_d"])
        self.assertAlmostEqual(fit_local["slope_low_d"], -0.4, places=6)
        self.assertAlmostEqual(fit_global["slope_low_d"], -0.4, places=6)
        self.assertLessEqual(fit_local["n_points_low_d"], 9)


if __name__ == "__main__":
    unittest.main()
