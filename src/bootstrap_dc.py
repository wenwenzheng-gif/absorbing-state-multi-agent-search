"""Paired endpoint bootstrap for every frozen resolved cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_config import CELLS, Cell
from .scan_dc import cell_directory
from .utils import (
    adjacent_crossing,
    crossing_from_logs,
    lineage_arrays,
    pooled_log_R_rows,
    raw_index,
    read_rows,
)


def paired_bootstrap(
    low_rows: list[dict],
    high_rows: list[dict],
    d_low: int,
    d_high: int,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if [row["seed"] for row in low_rows] != [row["seed"] for row in high_rows]:
        raise ValueError("crossing endpoint seeds are not paired")
    low_num, low_den = lineage_arrays(low_rows)
    high_num, high_den = lineage_arrays(high_rows)
    episodes = len(low_rows)
    rng = np.random.default_rng(seed)
    thresholds = np.full(draws, np.nan)
    low_logs = np.full(draws, np.nan)
    high_logs = np.full(draws, np.nan)
    cursor = 0
    while cursor < draws:
        size = min(200, draws - cursor)
        sample = rng.integers(0, episodes, size=(size, episodes), dtype=np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.log(low_num[sample].sum(axis=1) / low_den[sample].sum(axis=1)).sum(axis=1)
            hl = np.log(high_num[sample].sum(axis=1) / high_den[sample].sum(axis=1)).sum(axis=1)
        threshold = d_low - ll / ((hl - ll) / (d_high - d_low))
        valid = np.isfinite(threshold) & (ll > 0.0) & (hl < 0.0)
        threshold[~valid] = np.nan
        ll[~valid] = np.nan
        hl[~valid] = np.nan
        thresholds[cursor : cursor + size] = threshold
        low_logs[cursor : cursor + size] = ll
        high_logs[cursor : cursor + size] = hl
        cursor += size
    return thresholds, low_logs, high_logs


def endpoint_rows(raw_root: Path, cell: Cell) -> tuple[int, int, list[dict], list[dict]]:
    bracket = adjacent_crossing(cell_directory(raw_root, cell), cell.endpoint_episodes)
    if bracket is None:
        raise RuntimeError(f"{cell.cell_uid}: no adjacent E{cell.endpoint_episodes} crossing")
    d_low, d_high = bracket
    index = raw_index(cell_directory(raw_root, cell))
    low_episodes, low_path = index[d_low]
    high_episodes, high_path = index[d_high]
    if min(low_episodes, high_episodes) < cell.endpoint_episodes:
        raise RuntimeError(f"{cell.cell_uid}: incomplete crossing endpoints")
    low_rows = read_rows(low_path)[: cell.endpoint_episodes]
    high_rows = read_rows(high_path)[: cell.endpoint_episodes]
    return d_low, d_high, low_rows, high_rows


def run_bootstrap(raw_root: Path, output_root: Path) -> pd.DataFrame:
    analysis = output_root / "analysis"
    bootstrap = output_root / "bootstrap"
    analysis.mkdir(parents=True, exist_ok=True)
    bootstrap.mkdir(parents=True, exist_ok=True)
    threshold_archives: dict[str, dict[str, np.ndarray]] = {
        "v2": {},
        "v3_width": {},
        "logb": {},
    }
    endpoint_archive: dict[str, np.ndarray] = {}
    summaries: list[dict] = []
    for cell in CELLS:
        d_low, d_high, low_rows, high_rows = endpoint_rows(raw_root, cell)
        log_low = pooled_log_R_rows(low_rows)
        log_high = pooled_log_R_rows(high_rows)
        dc = crossing_from_logs(d_low, d_high, log_low, log_high)
        low_ci = high_ci = math.nan
        valid_fraction = math.nan
        if cell.bootstrap_draws:
            if cell.bootstrap_seed is None:
                raise RuntimeError(f"{cell.cell_uid}: missing paired-bootstrap seed")
            thresholds, low_logs, high_logs = paired_bootstrap(
                low_rows,
                high_rows,
                d_low,
                d_high,
                cell.bootstrap_draws,
                cell.bootstrap_seed,
            )
            valid = thresholds[np.isfinite(thresholds) & (thresholds > 0.0)]
            if len(valid) < 0.999 * cell.bootstrap_draws:
                raise RuntimeError(f"{cell.cell_uid}: too few valid bootstrap draws")
            low_ci, high_ci = (float(value) for value in np.quantile(valid, [0.025, 0.975]))
            valid_fraction = len(valid) / cell.bootstrap_draws
            threshold_archives[cell.source_family][cell.bootstrap_key] = thresholds
            endpoint_archive[f"{cell.raw_slug}__log_R_low"] = low_logs
            endpoint_archive[f"{cell.raw_slug}__log_R_high"] = high_logs
        summaries.append(
            {
                "cell_uid": cell.cell_uid,
                "source_cell_id": cell.source_cell_id,
                "source_family": cell.source_family,
                "d_low": d_low,
                "d_high": d_high,
                "paired_episodes": cell.endpoint_episodes,
                "log_R_low": log_low,
                "log_R_high": log_high,
                "d_c_data": dc,
                "d_c_CI_low": low_ci,
                "d_c_CI_high": high_ci,
                "valid_draw_fraction": valid_fraction,
                "reference_dc": cell.reference_dc,
                "absolute_reference_difference": abs(dc - cell.reference_dc),
            }
        )
        print(f"{cell.cell_uid:34s} d_c={dc:.6f}", flush=True)
    for source, arrays in threshold_archives.items():
        np.savez_compressed(bootstrap / f"{source}_threshold_bootstrap_draws.npz", **arrays)
    np.savez_compressed(bootstrap / "endpoint_logR_bootstrap_draws.npz", **endpoint_archive)
    frame = pd.DataFrame(summaries)
    frame.to_csv(analysis / "bootstrap_summary.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paired nonparametric episode bootstrap at adjacent integer endpoints",
        "source_term_excluded": True,
        "cells": len(frame),
        "bootstrapped_cells": int(frame.d_c_CI_low.notna().sum()),
        "max_absolute_reference_difference": float(frame.absolute_reference_difference.max()),
    }
    (bootstrap / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_bootstrap(args.raw_root, args.output_root)


if __name__ == "__main__":
    main()

