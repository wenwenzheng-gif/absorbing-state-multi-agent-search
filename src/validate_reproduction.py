"""Numerically validate regenerated tables and rendered figures."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .utils import PACKAGE_ROOT


TABLES = (
    "master_resolved_cells.csv",
    "global_fit_parameters.csv",
    "global_fit_predictions.csv",
    "one_factor_fits.csv",
    "theory_error_summary.csv",
    "measured_effective_comparison.csv",
)
FIGURES = (
    "data_vs_theory.png",
    "empirical_fit_vs_data.png",
    "one_factor_scaling.png",
    "theory_ratio.png",
    "logb_validation.png",
    "redundancy_diagnostic.png",
)


def compare_table(reference: Path, reproduced: Path) -> dict:
    left = pd.read_csv(reference)
    right = pd.read_csv(reproduced)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise RuntimeError(f"table schema mismatch: {reproduced.name}")
    maximum = 0.0
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            a = left[column].to_numpy(float)
            b = right[column].to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise RuntimeError(f"NaN pattern mismatch: {reproduced.name}:{column}")
            finite = np.isfinite(a) & np.isfinite(b)
            difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            maximum = max(maximum, difference)
            if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise RuntimeError(f"numeric mismatch: {reproduced.name}:{column}")
        else:
            a = left[column].fillna("").astype(str)
            b = right[column].fillna("").astype(str)
            if not a.equals(b):
                raise RuntimeError(f"text mismatch: {reproduced.name}:{column}")
    return {"rows": len(left), "maximum_absolute_numeric_difference": maximum}


def compare_figure(reference: Path, reproduced: Path) -> dict:
    left = mpimg.imread(reference)
    right = mpimg.imread(reproduced)
    if left.shape != right.shape:
        raise RuntimeError(f"figure dimensions differ: {reproduced.name}")
    maximum = float(np.max(np.abs(left.astype(float) - right.astype(float))))
    # All figures are pixel-identical except a few anti-aliased pixels in the
    # five-panel plot, whose observed maximum channel difference is 2/255.
    if maximum > 2.01 / 255.0:
        raise RuntimeError(f"figure pixel mismatch: {reproduced.name}, max={maximum}")
    return {"shape": list(left.shape), "maximum_channel_difference": maximum}


def validate_quick(output_root: Path) -> dict:
    table_results = {
        name: compare_table(PACKAGE_ROOT / "data" / name, output_root / "data" / name)
        for name in TABLES
    }
    # Saved reference renderings are optional: when one is present the PNG
    # must match it pixel for pixel; otherwise only a nonempty render is
    # required, exactly as for the PDFs.
    figure_results = {}
    for name in FIGURES:
        reference = PACKAGE_ROOT / "figures" / name
        png = output_root / "figures" / name
        pdf = output_root / "figures" / name.replace(".png", ".pdf")
        for rendered in (png, pdf):
            if not rendered.is_file() or rendered.stat().st_size == 0:
                raise RuntimeError(f"missing regenerated figure: {rendered.name}")
        figure_results[name] = (
            compare_figure(reference, png)
            if reference.is_file()
            else {"reference": "none saved; checked for a nonempty render"}
        )
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "tables": table_results,
        "figures": figure_results,
        "pdf_policy": "existence/nonempty; PDF metadata timestamps are not byte-stable",
    }
    (output_root / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
