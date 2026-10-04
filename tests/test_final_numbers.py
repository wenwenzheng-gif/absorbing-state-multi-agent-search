#!/usr/bin/env python3
"""Regression checks for the final frozen scientific outputs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FinalNumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.master = pd.read_csv(ROOT / "data" / "master_resolved_cells.csv")
        cls.parameters = pd.read_csv(ROOT / "data" / "global_fit_parameters.csv").set_index("parameter")
        cls.theory = pd.read_csv(ROOT / "data" / "theory_error_summary.csv").set_index("group")
        cls.one_factor = pd.read_csv(ROOT / "data" / "one_factor_fits.csv").set_index("fit_name")

    def test_master_counts(self) -> None:
        self.assertEqual(len(self.master), 53)
        self.assertEqual(int(self.master.theory_clean.sum()), 24)
        self.assertEqual(
            self.master.final_classification.value_counts().to_dict(),
            {"THEORY-CLEAN": 24, "SUPPLY-CLEAN-BUT-REDUNDANT": 20, "SUPPLY-DEPLETED": 9},
        )

    def test_primary_fit(self) -> None:
        expected = {
            "C_ref": 23.613687,
            "alpha_M": 1.078947,
            "alpha_K": 0.903770,
            "alpha_m": -1.009077,
            "alpha_T": -1.488607,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(float(self.parameters.loc[name, "estimate"]), value, places=6)
        self.assertFalse(bool(self.parameters.loc["alpha_b", "identifiable"]))
        self.assertTrue(pd.isna(self.parameters.loc["alpha_b", "estimate"]))

    def test_theory_error_summary(self) -> None:
        row = self.theory.loc["theory_clean"]
        self.assertEqual(int(row.n_cells), 24)
        self.assertAlmostEqual(float(row.median_absolute_relative_error), 0.0791024956162049, places=14)
        self.assertAlmostEqual(float(row.mean_absolute_relative_error), 0.0777795083366258, places=14)

    def test_representative_critical_degrees(self) -> None:
        expected = {
            "v2::K3_N160": 18.431688,
            "v2::M24_N160": 36.642487,
            "v3_width::b2_T5_n0_32": 16.722357,
            "logb::B_b5": 51.656516,
        }
        values = self.master.set_index("cell_uid").d_c_data
        for cell_uid, value in expected.items():
            self.assertAlmostEqual(float(values.loc[cell_uid]), value, places=6)

    def test_b_status_is_descriptive_only(self) -> None:
        for baseline in "ABC":
            clean = self.one_factor.loc[f"b_clean_{baseline}"]
            descriptive = self.one_factor.loc[f"b_descriptive_{baseline}"]
            self.assertFalse(bool(clean.identifiable))
            self.assertEqual(int(clean.n_cells), 1)
            self.assertEqual(int(clean.levels), 2)
            self.assertTrue(bool(descriptive.identifiable))
            self.assertIn("diagnostic only", str(descriptive.fit_role))


if __name__ == "__main__":
    unittest.main()
