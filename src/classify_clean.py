"""Freeze supply/independence labels without consulting theory error."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_config import CELLS, INDEPENDENCE_LIMITS, SUPPLY_LIMITS, Cell
from .scan_dc import cell_directory
from .utils import adjacent_crossing, crossing_from_logs, pooled_log_R_rows, raw_index, read_rows, sha256_file


ROUND_KEYS = (
    "active_agent_fraction",
    "frontier_empty_fraction",
    "budget_utilization",
    "message_availability",
    "outgoing_message_availability",
    "candidate_capacity_ratio",
    "frontier_post_jaccard",
    "frontier_post_null_jaccard",
    "frontier_post_excess_jaccard",
    "frontier_receive_excess_jaccard",
    "action_jaccard",
    "evidence_post_jaccard",
    "evidence_post_null_jaccard",
    "evidence_post_excess_jaccard",
    "duplicate_evidence_fraction",
    "already_known_evidence_fraction",
    "effective_branching",
)


def finite_mean(rows: list[dict], round_index: int, key: str) -> float:
    values = np.asarray(
        [row["transitions"][round_index].get(key, math.nan) for row in rows],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def pooled_ratio(rows: list[dict], round_index: int, numerator: str, denominator: str) -> float:
    num = sum(float(row["transitions"][round_index].get(numerator, 0.0)) for row in rows)
    den = sum(float(row["transitions"][round_index].get(denominator, 0.0)) for row in rows)
    return num / den if den else 0.0


def _minimum(records: list[dict], key: str) -> float:
    values = [float(record[key]) for record in records if math.isfinite(float(record[key]))]
    return min(values) if values else math.nan


def _maximum(records: list[dict], key: str) -> float:
    values = [float(record[key]) for record in records if math.isfinite(float(record[key]))]
    return max(values) if values else math.nan


def endpoint_data(raw_root: Path, cell: Cell) -> tuple[int, int, list[dict], list[dict], Path, Path]:
    directory = cell_directory(raw_root, cell)
    bracket = adjacent_crossing(directory, cell.endpoint_episodes)
    if bracket is None:
        raise RuntimeError(f"{cell.cell_uid}: unresolved crossing")
    d_low, d_high = bracket
    index = raw_index(directory)
    low_path = index[d_low][1]
    high_path = index[d_high][1]
    low_rows = read_rows(low_path)[: cell.endpoint_episodes]
    high_rows = read_rows(high_path)[: cell.endpoint_episodes]
    return d_low, d_high, low_rows, high_rows, low_path, high_path


def freeze_classification(raw_root: Path, output_root: Path) -> pd.DataFrame:
    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    endpoints: list[dict] = []
    round_records: list[dict] = []
    novelty: list[dict] = []
    source_files: list[dict] = []
    for cell in CELLS:
        d_low, d_high, low_rows, high_rows, low_path, high_path = endpoint_data(raw_root, cell)
        log_low = pooled_log_R_rows(low_rows)
        log_high = pooled_log_R_rows(high_rows)
        dc = crossing_from_logs(d_low, d_high, log_low, log_high)
        cell_endpoints: list[dict] = []
        for degree, rows, path in (
            (d_low, low_rows, low_path),
            (d_high, high_rows, high_path),
        ):
            source_files.append(
                {
                    "cell_uid": cell.cell_uid,
                    "degree": degree,
                    "episodes": len(rows),
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
            endpoint_rounds: list[dict] = []
            for round_index in range(cell.rounds):
                record = {
                    "cell_uid": cell.cell_uid,
                    "degree": degree,
                    "episodes": len(rows),
                    "round": round_index + 1,
                }
                for key in ROUND_KEYS:
                    record[key] = finite_mean(rows, round_index, key)
                record["merge_loss_fraction"] = pooled_ratio(
                    rows, round_index, "merge_loss", "sampled_edges"
                )
                record["effective_branching_ratio"] = record["effective_branching"] / cell.b
                round_records.append(record)
                endpoint_rounds.append(record)
                novelty.append(
                    {
                        "cell_uid": cell.cell_uid,
                        "degree": degree,
                        "round": round_index + 1,
                        "received_messages": sum(
                            row["transitions"][round_index]["recv_messages"] for row in rows
                        ),
                        "duplicate_fraction": pooled_ratio(
                            rows, round_index, "recv_duplicate", "recv_messages"
                        ),
                        "already_explicit_fraction": pooled_ratio(
                            rows, round_index, "recv_already_explicit", "recv_messages"
                        ),
                        "already_implied_fraction": pooled_ratio(
                            rows, round_index, "recv_already_implied", "recv_messages"
                        ),
                        "informative_fraction": pooled_ratio(
                            rows, round_index, "recv_informative", "recv_messages"
                        ),
                    }
                )
            endpoint = {
                "cell_uid": cell.cell_uid,
                "degree": degree,
                "episodes": len(rows),
                "min_budget_utilization": _minimum(endpoint_rounds, "budget_utilization"),
                "min_message_availability": _minimum(
                    endpoint_rounds, "outgoing_message_availability"
                ),
                "min_candidate_capacity_ratio": _minimum(
                    endpoint_rounds, "candidate_capacity_ratio"
                ),
                "min_active_fraction": _minimum(endpoint_rounds, "active_agent_fraction"),
                "max_frontier_empty_fraction": _maximum(
                    endpoint_rounds, "frontier_empty_fraction"
                ),
                "max_excess_frontier_jaccard": _maximum(
                    endpoint_rounds, "frontier_post_excess_jaccard"
                ),
                "max_duplicate_evidence_fraction": _maximum(
                    endpoint_rounds, "duplicate_evidence_fraction"
                ),
                "max_evidence_excess_jaccard": _maximum(
                    endpoint_rounds, "evidence_post_excess_jaccard"
                ),
                "max_already_known_evidence_fraction": _maximum(
                    endpoint_rounds, "already_known_evidence_fraction"
                ),
                "min_effective_branching_ratio": _minimum(
                    endpoint_rounds, "effective_branching_ratio"
                ),
                "max_merge_loss_fraction": _maximum(endpoint_rounds, "merge_loss_fraction"),
                "neighbor_set_jaccard": float(np.mean([row["neighbor_set_jaccard"] for row in rows])),
                "neighbor_set_excess_jaccard": float(
                    np.mean([row["neighbor_set_excess_jaccard"] for row in rows])
                ),
            }
            endpoints.append(endpoint)
            cell_endpoints.append(endpoint)
        metrics = {
            "min_budget_utilization": _minimum(cell_endpoints, "min_budget_utilization"),
            "min_message_availability": _minimum(cell_endpoints, "min_message_availability"),
            "min_candidate_capacity_ratio": _minimum(
                cell_endpoints, "min_candidate_capacity_ratio"
            ),
            "min_active_fraction": _minimum(cell_endpoints, "min_active_fraction"),
            "max_frontier_empty_fraction": _maximum(
                cell_endpoints, "max_frontier_empty_fraction"
            ),
            "max_excess_frontier_jaccard": _maximum(
                cell_endpoints, "max_excess_frontier_jaccard"
            ),
            "max_duplicate_evidence_fraction": _maximum(
                cell_endpoints, "max_duplicate_evidence_fraction"
            ),
            "max_evidence_excess_jaccard": _maximum(
                cell_endpoints, "max_evidence_excess_jaccard"
            ),
            "max_already_known_evidence_fraction": _maximum(
                cell_endpoints, "max_already_known_evidence_fraction"
            ),
            "min_effective_branching_ratio": _minimum(
                cell_endpoints, "min_effective_branching_ratio"
            ),
            "max_merge_loss_fraction": _maximum(cell_endpoints, "max_merge_loss_fraction"),
            "degree_density": dc / (cell.N - 1),
        }
        supply_reasons: list[str] = []
        if metrics["min_budget_utilization"] < SUPPLY_LIMITS["min_budget_utilization"]:
            supply_reasons.append("budget utilization")
        if metrics["min_message_availability"] < SUPPLY_LIMITS["min_message_availability"]:
            supply_reasons.append("message availability")
        if metrics["min_candidate_capacity_ratio"] < SUPPLY_LIMITS["min_candidate_capacity_ratio"]:
            supply_reasons.append("candidate capacity")
        if metrics["max_frontier_empty_fraction"] > SUPPLY_LIMITS["max_frontier_empty_fraction"]:
            supply_reasons.append("frontier empty")
        if metrics["max_excess_frontier_jaccard"] > SUPPLY_LIMITS["max_excess_frontier_jaccard"]:
            supply_reasons.append("frontier overlap")
        independence_reasons: list[str] = []
        if (
            metrics["max_duplicate_evidence_fraction"]
            > INDEPENDENCE_LIMITS["max_duplicate_evidence_fraction"]
        ):
            independence_reasons.append("duplicate evidence")
        if (
            metrics["max_evidence_excess_jaccard"]
            > INDEPENDENCE_LIMITS["max_evidence_excess_jaccard"]
        ):
            independence_reasons.append("evidence overlap")
        if metrics["degree_density"] > INDEPENDENCE_LIMITS["max_degree_density"]:
            independence_reasons.append("graph density")
        supply_clean = not supply_reasons
        independence_clean = not independence_reasons
        classification = (
            "SUPPLY-DEPLETED"
            if not supply_clean
            else "THEORY-CLEAN"
            if independence_clean
            else "SUPPLY-CLEAN-BUT-REDUNDANT"
        )
        cells.append(
            {
                "cell_uid": cell.cell_uid,
                "source_cell_id": cell.source_cell_id,
                "source_family": cell.source_family,
                "N": cell.N,
                "M": cell.M,
                "K": cell.K,
                "T": cell.T,
                "b": cell.b,
                "m": cell.m,
                "n0": cell.n0,
                "d_low": d_low,
                "d_high": d_high,
                "log_R_low": log_low,
                "log_R_high": log_high,
                "d_c_diagnostic_only": dc,
                "endpoint_episodes": cell.endpoint_episodes,
                **metrics,
                "supply_clean": supply_clean,
                "independence_clean": independence_clean,
                "theory_clean": supply_clean and independence_clean,
                "classification": classification,
                "supply_reasons": "; ".join(supply_reasons),
                "independence_reasons": "; ".join(independence_reasons),
            }
        )
    frame = pd.DataFrame(cells)
    frame.to_csv(analysis / "classification_frozen.csv", index=False)
    pd.DataFrame(endpoints).to_csv(analysis / "crossing_endpoint_diagnostics.csv", index=False)
    pd.DataFrame(round_records).to_csv(analysis / "round_diagnostics.csv", index=False)
    pd.DataFrame(novelty).to_csv(analysis / "message_novelty.csv", index=False)
    pd.DataFrame(source_files).to_csv(analysis / "source_files.csv", index=False)
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification_uses_theory_error": False,
        "theory_columns_present_at_freeze": False,
        "supply_limits": SUPPLY_LIMITS,
        "independence_limits": INDEPENDENCE_LIMITS,
        "counts": dict(Counter(frame.classification)),
        "cells": len(frame),
    }
    (analysis / "classification_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    frame = freeze_classification(args.raw_root, args.output_root)
    print(frame[["cell_uid", "classification"]].to_string(index=False))


if __name__ == "__main__":
    main()

