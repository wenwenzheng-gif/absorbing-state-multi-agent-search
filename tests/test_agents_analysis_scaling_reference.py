#!/usr/bin/env python3
"""Tests for the ``exact``/``delayed`` reference switch in ``analysis.scaling``.

The protocol has no communication in round 1 (evidence is only sent after
local experiments), so the crossing that matches the measurement should be
computed with the survival-term sum starting at ``t=2`` -- the paper's
eq. (20), exposed here as the ``delayed`` column/reference. These tests are
synthetic (same fixture shape as ``test_agents_analysis_fixes.py``) and do
not depend on anything under ``runs/``.
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


class ReferenceColumnsTests(unittest.TestCase):
    """``cell_of`` always carries both references and their relative errors."""

    def test_exact_and_delayed_differ_and_rel_err_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(
                root, "cell", agents=80, families=8, variants=4,
                task_size=4, branching=3, d_c=18.0,
            )
            cell = scaling.cell_of(directory)
        self.assertIsNotNone(cell)
        self.assertNotAlmostEqual(cell["exact"], cell["delayed"])
        self.assertAlmostEqual(cell["rel_err_exact"], cell["d_c"] / cell["exact"] - 1.0)
        self.assertAlmostEqual(cell["rel_err_delayed"], cell["d_c"] / cell["delayed"] - 1.0)

    def test_rel_err_is_none_when_the_reference_is_not_a_clean_crossing(self) -> None:
        # T = M means the survival terms run out of room and exact/delayed can
        # come back nan; rel_err must degrade to None rather than nan leaking
        # into the CSV.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(
                root, "wide", agents=80, families=4, variants=1,
                task_size=8, branching=3, d_c=3.0,
            )
            cell = scaling.cell_of(directory)
        self.assertTrue(math.isnan(cell["exact"]) or math.isnan(cell["delayed"]))
        if math.isnan(cell["exact"]):
            self.assertIsNone(cell["rel_err_exact"])
        if math.isnan(cell["delayed"]):
            self.assertIsNone(cell["rel_err_delayed"])


class CellsCsvColumnsTests(unittest.TestCase):
    """``report`` writes the new columns into ``cells.csv``."""

    def test_cells_csv_has_rel_err_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(
                root, "cell", agents=80, families=8, variants=4,
                task_size=4, branching=3, d_c=18.0,
            )
            scaling.report([directory], root / "out", depletion_headroom=1.0)
            rows = read_csv(root / "out" / "cells.csv")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("rel_err_exact", row)
        self.assertIn("rel_err_delayed", row)
        self.assertAlmostEqual(float(row["rel_err_exact"]), 18.0 / float(row["exact"]) - 1.0)
        self.assertAlmostEqual(float(row["rel_err_delayed"]), 18.0 / float(row["delayed"]) - 1.0)


class CellsOutTests(unittest.TestCase):
    """``--cells-out`` writes a second copy of ``cells.csv`` elsewhere."""

    def test_cells_out_is_written_and_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(
                root, "cell", agents=80, families=8, variants=4,
                task_size=4, branching=3, d_c=18.0,
            )
            tracked = root / "tracked" / "scaling_cells.csv"
            summary = scaling.report(
                [directory], root / "out", depletion_headroom=1.0, cells_out=tracked,
            )
            self.assertTrue(tracked.is_file())
            self.assertEqual(summary["cells_out"], str(tracked))
            main_rows = read_csv(root / "out" / "cells.csv")
            copy_rows = read_csv(tracked)
        self.assertEqual(main_rows, copy_rows)

    def test_cells_out_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(
                root, "cell", agents=80, families=8, variants=4,
                task_size=4, branching=3, d_c=18.0,
            )
            summary = scaling.report([directory], root / "out", depletion_headroom=1.0)
        self.assertIsNone(summary["cells_out"])


class ReferenceSwitchRenderTests(unittest.TestCase):
    """``--reference`` changes which analytic column is the x-axis."""

    def _two_cells(self, root: Path) -> list[Path]:
        return [
            write_run(root, "a", agents=80, families=8, variants=4,
                      task_size=4, branching=3, d_c=18.0),
            write_run(root, "b", agents=80, families=8, variants=4,
                      task_size=5, branching=3, d_c=13.0),
        ]

    def test_default_reference_is_delayed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = self._two_cells(root)
            summary = scaling.report(runs, root / "out", depletion_headroom=1.0)
            self.assertEqual(summary["reference"], "delayed")
            self.assertTrue((root / "out" / "measured_vs_delayed.png").is_file())
            self.assertTrue((root / "out" / "measured_vs_analytic.png").is_file())
            self.assertTrue((root / "out" / "measured_vs_analytic.pdf").is_file())

    def test_exact_reference_writes_measured_vs_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = self._two_cells(root)
            summary = scaling.report(
                runs, root / "out", depletion_headroom=1.0, reference="exact",
            )
            self.assertEqual(summary["reference"], "exact")
            self.assertTrue((root / "out" / "measured_vs_exact.png").is_file())
            self.assertFalse((root / "out" / "measured_vs_delayed.png").is_file())

    def test_two_panel_figure_x_values_differ_between_panels(self) -> None:
        # Not a pixel check: verify the underlying data fed to each panel
        # (exact vs delayed) actually differs, which is what makes a
        # two-panel comparison meaningful.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = self._two_cells(root)
            cells = scaling.collect(runs)
        exact_values = [cell["exact"] for cell in cells]
        delayed_values = [cell["delayed"] for cell in cells]
        self.assertNotEqual(exact_values, delayed_values)


class CliOptionTests(unittest.TestCase):
    """The CLI exposes ``--reference`` and ``--cells-out``."""

    def test_main_accepts_reference_and_cells_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(
                root, "cell", agents=80, families=8, variants=4,
                task_size=4, branching=3, d_c=18.0,
            )
            tracked = root / "tracked.csv"
            code = scaling.main([
                "--run-dir", str(directory),
                "--out-dir", str(root / "out"),
                "--reference", "exact",
                "--cells-out", str(tracked),
            ])
            self.assertEqual(code, 0)
            self.assertTrue(tracked.is_file())
            self.assertTrue((root / "out" / "measured_vs_exact.png").is_file())


if __name__ == "__main__":
    unittest.main()
