"""Assemble a canonical master table and recompute exact analytic columns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_config import CELLS, INDEPENDENCE_LIMITS, SUPPLY_LIMITS
from .utils import PACKAGE_ROOT, delayed_boundary_dc, exact_analytic_dc


REFERENCE_MASTER = PACKAGE_ROOT / "data" / "master_resolved_cells.csv"


def _booleans(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def frozen_label(row: pd.Series) -> tuple[bool, bool, str]:
    supply = bool(
        float(row.budget_utilization) >= SUPPLY_LIMITS["min_budget_utilization"]
        and float(row.message_availability) >= SUPPLY_LIMITS["min_message_availability"]
        and float(row.candidate_capacity_ratio) >= SUPPLY_LIMITS["min_candidate_capacity_ratio"]
        and float(row.frontier_empty_fraction) <= SUPPLY_LIMITS["max_frontier_empty_fraction"]
        and float(row.frontier_excess_Jaccard) <= SUPPLY_LIMITS["max_excess_frontier_jaccard"]
    )
    independence = bool(
        float(row.duplicate_evidence_fraction)
        <= INDEPENDENCE_LIMITS["max_duplicate_evidence_fraction"]
        and float(row.evidence_excess_Jaccard)
        <= INDEPENDENCE_LIMITS["max_evidence_excess_jaccard"]
        and float(row.degree_density) <= INDEPENDENCE_LIMITS["max_degree_density"]
    )
    label = (
        "SUPPLY-DEPLETED"
        if not supply
        else "THEORY-CLEAN"
        if independence
        else "SUPPLY-CLEAN-BUT-REDUNDANT"
    )
    return supply, independence, label


def recompute_derived(frame: pd.DataFrame, verify_labels: bool = True) -> pd.DataFrame:
    frame = frame.copy()
    analytic = []
    delayed = []
    for row in frame.itertuples(index=False):
        analytic.append(exact_analytic_dc(int(row.M), int(row.K), int(row.T), int(row.b), int(row.m)))
        delayed.append(delayed_boundary_dc(int(row.M), int(row.K), int(row.T), int(row.b), int(row.m)))
    frame["d_c_analytic"] = analytic
    frame["data_over_theory"] = frame.d_c_data / frame.d_c_analytic
    frame["relative_error"] = frame.data_over_theory - 1.0
    frame["d_c_delayed_boundary"] = delayed
    frame["data_over_delayed_boundary"] = frame.d_c_data / frame.d_c_delayed_boundary
    frame["delayed_boundary_relative_error"] = frame.data_over_delayed_boundary - 1.0
    frame["effective_relative_error"] = np.where(
        np.isfinite(frame.d_c_effective),
        (frame.d_c_effective - frame.d_c_data) / frame.d_c_data,
        np.nan,
    )
    recomputed = frame.apply(frozen_label, axis=1, result_type="expand")
    recomputed.columns = ["_supply", "_independence", "_label"]
    if verify_labels:
        original = _booleans(frame, ["supply_clean", "independence_clean", "theory_clean"])
        if not (original.supply_clean.to_numpy() == recomputed._supply.to_numpy()).all():
            raise RuntimeError("stored supply labels do not match frozen thresholds")
        if not (original.independence_clean.to_numpy() == recomputed._independence.to_numpy()).all():
            raise RuntimeError("stored independence labels do not match frozen thresholds")
        if not (frame.final_classification.astype(str).to_numpy() == recomputed._label.to_numpy()).all():
            raise RuntimeError("stored three-regime labels do not match frozen thresholds")
    frame["supply_clean"] = recomputed._supply.astype(bool)
    frame["independence_clean"] = recomputed._independence.astype(bool)
    frame["theory_clean"] = frame.supply_clean & frame.independence_clean
    frame["final_classification"] = recomputed._label
    if not ((frame.d_plus.astype(int) - frame.d_minus.astype(int)) == 1).all():
        raise RuntimeError("master table contains a non-adjacent crossing")
    return frame


def assemble_quick(output_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(REFERENCE_MASTER)
    stored_analytic = frame.d_c_analytic.to_numpy(float)
    frame = recompute_derived(frame, verify_labels=True)
    if not np.allclose(frame.d_c_analytic, stored_analytic, rtol=0.0, atol=2e-12):
        raise RuntimeError("saved exact-theory column is inconsistent")
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "master_resolved_cells.csv", index=False)
    return frame


def assemble_full(full_root: Path, output_root: Path) -> pd.DataFrame:
    classification = pd.read_csv(full_root / "analysis" / "classification_frozen.csv").set_index("cell_uid")
    bootstrap = pd.read_csv(full_root / "analysis" / "bootstrap_summary.csv").set_index("cell_uid")
    reference = pd.read_csv(REFERENCE_MASTER).set_index("cell_uid")
    records: list[dict] = []
    for cell in CELLS:
        diagnostic = classification.loc[cell.cell_uid]
        estimate = bootstrap.loc[cell.cell_uid]
        static = reference.loc[cell.cell_uid]
        dc = float(estimate.d_c_data)
        effective = float(static.d_c_effective) if pd.notna(static.d_c_effective) else math.nan
        record = {
            "cell_uid": cell.cell_uid,
            "source_cell_id": cell.source_cell_id,
            "experiment_family": cell.experiment_family,
            "source_design": cell.source_design,
            "N": cell.N,
            "M": cell.M,
            "K": cell.K,
            "T": cell.T,
            "b": cell.b,
            "m": cell.m,
            "n0": cell.n0,
            "d_minus": int(estimate.d_low),
            "d_plus": int(estimate.d_high),
            "endpoint_episodes": cell.endpoint_episodes,
            "d_c_data": dc,
            "d_c_CI_low": float(estimate.d_c_CI_low) if pd.notna(estimate.d_c_CI_low) else math.nan,
            "d_c_CI_high": float(estimate.d_c_CI_high) if pd.notna(estimate.d_c_CI_high) else math.nan,
            "d_c_analytic": cell.analytic_dc,
            "data_over_theory": dc / cell.analytic_dc,
            "relative_error": dc / cell.analytic_dc - 1.0,
            "d_c_delayed_boundary": delayed_boundary_dc(cell.M, cell.K, cell.T, cell.b, cell.m),
            "d_c_effective": effective,
            "budget_utilization": float(diagnostic.min_budget_utilization),
            "message_availability": float(diagnostic.min_message_availability),
            "candidate_capacity_ratio": float(diagnostic.min_candidate_capacity_ratio),
            "frontier_empty_fraction": float(diagnostic.max_frontier_empty_fraction),
            "duplicate_evidence_fraction": float(diagnostic.max_duplicate_evidence_fraction),
            "frontier_excess_Jaccard": float(diagnostic.max_excess_frontier_jaccard),
            "evidence_excess_Jaccard": float(diagnostic.max_evidence_excess_jaccard),
            "degree_density": float(diagnostic.degree_density),
            "supply_clean": bool(diagnostic.supply_clean),
            "independence_clean": bool(diagnostic.independence_clean),
            "theory_clean": bool(diagnostic.theory_clean),
            "final_classification": str(diagnostic.classification),
            "supply_reasons": diagnostic.supply_reasons if isinstance(diagnostic.supply_reasons, str) else "",
            "independence_reasons": diagnostic.independence_reasons if isinstance(diagnostic.independence_reasons, str) else "",
            "role": cell.role,
            "held_out": cell.held_out,
            "width_stable_theory_clean": cell.width_stable_theory_clean,
            "primary_global_fit": cell.primary_global_fit,
            "primary_one_factor_scans": cell.primary_one_factor_scans,
            "bootstrap_source": cell.bootstrap_source,
            "bootstrap_key": cell.bootstrap_key,
            "source_archive": str(static.source_archive),
            "source_result_file": "reproduced/full/analysis/classification_frozen.csv",
            "trajectory_schema_version": "canonical_budget_diagnostics_v1",
        }
        delayed = float(record["d_c_delayed_boundary"])
        record["data_over_delayed_boundary"] = dc / delayed if math.isfinite(delayed) else math.nan
        record["delayed_boundary_relative_error"] = (
            dc / delayed - 1.0 if math.isfinite(delayed) else math.nan
        )
        record["effective_relative_error"] = (
            (effective - dc) / dc if math.isfinite(effective) else math.nan
        )
        records.append(record)
    frame = pd.DataFrame(records)
    reference_order = list(reference.index)
    frame["_order"] = frame.cell_uid.map({uid: index for index, uid in enumerate(reference_order)})
    frame = frame.sort_values("_order").drop(columns="_order")
    frame = frame[list(pd.read_csv(REFERENCE_MASTER, nrows=0).columns)]
    frame = recompute_derived(frame, verify_labels=True)
    if float(np.max(np.abs(frame.d_c_data - frame.cell_uid.map(reference.d_c_data)))) > 1e-10:
        raise RuntimeError("full reproduction differs from deterministic reference d_c values")
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "master_resolved_cells.csv", index=False)
    return frame


def write_manifest(frame: pd.DataFrame, output_root: Path, mode: str) -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "master_rows": int(len(frame)),
        "classification_counts": frame.final_classification.value_counts().to_dict(),
        "exact_finite_T_theory_recomputed": True,
        "analytic_formula": "-(T-1) ln(b) / sum_{t=1}^{T-1} ln(1-m t/(M K))",
    }
    (output_root / "assembly_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "quick":
        frame = assemble_quick(args.output_root)
    else:
        if args.full_root is None:
            raise ValueError("--full-root is required in full mode")
        frame = assemble_full(args.full_root, args.output_root)
    write_manifest(frame, args.output_root, args.mode)
    print(f"assembled {len(frame)} cells into {args.output_root}")


if __name__ == "__main__":
    main()

