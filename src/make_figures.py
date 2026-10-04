"""Regenerate the six final figures from already-fitted tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import fit_scaling


def generate(master: Path, fitted_data: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_scaling.FIGURES = output_dir
    frame = pd.read_csv(master)
    for column in ["supply_clean", "independence_clean", "theory_clean", "primary_global_fit"]:
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    predictions = pd.read_csv(fitted_data / "global_fit_predictions.csv")
    one_factor = pd.read_csv(fitted_data / "one_factor_fits.csv")
    metrics = json.loads((fitted_data / "global_fit_metrics.json").read_text(encoding="utf-8"))
    fit_scaling.make_figure1(frame)
    fit_scaling.make_figure2(predictions, metrics)
    fit_scaling.make_figure3(frame, one_factor)
    fit_scaling.make_figure4(frame)
    fit_scaling.make_figure5(frame, one_factor)
    fit_scaling.make_figure6(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--fitted-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.master, args.fitted_data, args.output_dir)


if __name__ == "__main__":
    main()

