#!/usr/bin/env python3
"""Reuse the frozen supply / independence labels on the agent runs.

The long paper classifies every cell as THEORY-CLEAN, SUPPLY-CLEAN-BUT-
REDUNDANT or SUPPLY-DEPLETED, and that classification is not invented here:
``src/classify_clean.py`` implements it for the frozen synthetic pipeline and
``configs/clean_criteria.json`` holds the thresholds, which are imported
unchanged through ``src.experiment_config``.  Repeating them here would risk
them drifting apart, so this module imports and reports the versions it used.

    supply:        budget utilisation >= 0.85, outgoing message availability
                   >= 0.85, candidate capacity ratio >= 0.9, frontier-empty
                   fraction <= 0.25  (and frontier overlap <= 0.05)
    independence:  duplicate-evidence fraction <= 0.2, degree density <= 0.4
                   (and evidence overlap <= 0.3)
    label:         SUPPLY-DEPLETED if any supply criterion fails, else
                   THEORY-CLEAN if every independence criterion holds, else
                   SUPPLY-CLEAN-BUT-REDUNDANT

Two of the eight frozen diagnostics -- the two Jaccard overlaps -- are not
recorded by the agent runner, so they cannot be evaluated and are reported as
``unavailable`` rather than silently passed.  The label is therefore taken on
the six that exist; a cell called THEORY-CLEAN here would only ever become
less clean if the overlaps were added, never more.

Aggregation follows the frozen code exactly: a diagnostic is averaged over
episodes within a round, then the worst round is taken, and a cell is scored
at the two degrees that bracket its crossing.  Every degree is also scored
separately, because the regime is a property of where you sit on the degree
axis, not only of the cell.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from ...experiment_config import INDEPENDENCE_LIMITS, SUPPLY_LIMITS
from .run_cell import (
    cell_parameters,
    episodes_by_degree,
    load_episodes,
    measured_crossing,
    pooled_ratio,
    round_mean,
)
from .summarize import write_csv

THEORY_CLEAN = "THEORY-CLEAN"
REDUNDANT = "SUPPLY-CLEAN-BUT-REDUNDANT"
DEPLETED = "SUPPLY-DEPLETED"

# The two frozen diagnostics the agent runner does not log.  Named here so the
# report can say which criteria were actually applied.
UNAVAILABLE = ("max_excess_frontier_jaccard", "max_evidence_excess_jaccard")

DEGREE_COLUMNS = (
    "run", "d", "episodes", "min_budget_utilization", "min_message_availability",
    "min_candidate_capacity_ratio", "min_active_fraction", "max_frontier_empty_fraction",
    "max_duplicate_evidence_fraction", "max_already_known_evidence_fraction",
    "degree_density", "supply_clean", "independence_clean", "classification",
    "supply_reasons", "independence_reasons",
)
CELL_COLUMNS = (
    "run", "kind", "N", "M", "K", "V", "T", "b", "m", "d_c", "scored_at",
    "d_minus", "d_plus",
    "classification", "supply_reasons", "independence_reasons",
    "min_budget_utilization", "min_candidate_capacity_ratio",
    "max_frontier_empty_fraction", "max_duplicate_evidence_fraction", "degree_density",
    "geometric_T_le_M_over_2", "agrees_with_geometric_filter",
)


def degree_metrics(episodes: Sequence[dict], rounds: int) -> dict:
    """Worst-round value of each available diagnostic, over one degree."""
    per_round = []
    for index in range(rounds):
        per_round.append(
            {
                "budget_utilization": round_mean(episodes, index, "budget_utilization"),
                "message_availability": round_mean(
                    episodes, index, "outgoing_message_availability"
                ),
                "candidate_capacity_ratio": round_mean(
                    episodes, index, "candidate_capacity_ratio"
                ),
                "active_fraction": round_mean(episodes, index, "active_agent_fraction"),
                "frontier_empty_fraction": round_mean(
                    episodes, index, "frontier_empty_fraction"
                ),
                "duplicate_evidence_fraction": pooled_ratio(
                    episodes, index, "recv_duplicate", "recv_messages"
                ),
                "already_known_evidence_fraction": pooled_ratio(
                    episodes, index, "recv_already_explicit", "recv_messages"
                ),
            }
        )

    def worst(key: str, kind: str) -> float | None:
        values = [row[key] for row in per_round if row[key] is not None]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            return None
        return min(values) if kind == "min" else max(values)

    return {
        "min_budget_utilization": worst("budget_utilization", "min"),
        "min_message_availability": worst("message_availability", "min"),
        "min_candidate_capacity_ratio": worst("candidate_capacity_ratio", "min"),
        "min_active_fraction": worst("active_fraction", "min"),
        "max_frontier_empty_fraction": worst("frontier_empty_fraction", "max"),
        "max_duplicate_evidence_fraction": worst("duplicate_evidence_fraction", "max"),
        "max_already_known_evidence_fraction": worst(
            "already_known_evidence_fraction", "max"
        ),
    }


def classify(metrics: dict, degree_density: float | None) -> dict:
    """The frozen decision, on the criteria this run can actually support."""
    supply: list[str] = []
    value = metrics.get("min_budget_utilization")
    if value is not None and value < SUPPLY_LIMITS["min_budget_utilization"]:
        supply.append("budget utilization")
    value = metrics.get("min_message_availability")
    if value is not None and value < SUPPLY_LIMITS["min_message_availability"]:
        supply.append("message availability")
    value = metrics.get("min_candidate_capacity_ratio")
    if value is not None and value < SUPPLY_LIMITS["min_candidate_capacity_ratio"]:
        supply.append("candidate capacity")
    value = metrics.get("max_frontier_empty_fraction")
    if value is not None and value > SUPPLY_LIMITS["max_frontier_empty_fraction"]:
        supply.append("frontier empty")

    independence: list[str] = []
    value = metrics.get("max_duplicate_evidence_fraction")
    if value is not None and value > INDEPENDENCE_LIMITS["max_duplicate_evidence_fraction"]:
        independence.append("duplicate evidence")
    if degree_density is not None and degree_density > INDEPENDENCE_LIMITS["max_degree_density"]:
        independence.append("graph density")

    classification = (
        DEPLETED if supply else THEORY_CLEAN if not independence else REDUNDANT
    )
    return {
        "supply_clean": not supply,
        "independence_clean": not independence,
        "classification": classification,
        "supply_reasons": "; ".join(supply),
        "independence_reasons": "; ".join(independence),
    }


def _bracket(run_dir: Path, degrees: Sequence[int]) -> tuple[float | None, list[int], str]:
    """The two degrees that bracket the crossing, or the whole degree axis.

    A run whose crossing is unresolved still has a regime -- that is often
    exactly why it is unresolved -- so it is scored over every degree it ran
    instead of being dropped, and ``scored_at`` says which happened.
    """
    crossing = measured_crossing(run_dir)
    d_c = crossing["d_c"]
    if d_c is None:
        return None, list(degrees), "all degrees (crossing unresolved)"
    path = Path(run_dir) / "analysis" / "critical_degree.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    entries = (report.get("crossing") or {}).get("crossings") or []
    if entries:
        first = entries[0]
        return d_c, [int(first["d_minus"]), int(first["d_plus"])], "crossing endpoints"
    nearest = min(degrees, key=lambda d: abs(d - d_c))
    return d_c, [nearest], "nearest degree to the crossing"


def report(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    cell = cell_parameters(run_dir)
    if cell is None:
        raise FileNotFoundError(f"no manifest in {run_dir}")
    grouped = episodes_by_degree(load_episodes(run_dir))
    rounds = max(
        (len(e.get("transitions") or []) for group in grouped.values() for e in group),
        default=0,
    )
    d_c, bracket, scored_at = _bracket(run_dir, sorted(grouped))
    density = (d_c / (cell["N"] - 1)) if d_c is not None and cell["N"] > 1 else None

    degree_rows = []
    for degree, group in sorted(grouped.items()):
        metrics = degree_metrics(group, rounds)
        verdict = classify(metrics, degree / (cell["N"] - 1) if cell["N"] > 1 else None)
        degree_rows.append(
            {
                "run": run_dir.name,
                "d": degree,
                "episodes": len(group),
                **metrics,
                "degree_density": degree / (cell["N"] - 1) if cell["N"] > 1 else None,
                **verdict,
            }
        )

    endpoint_episodes = [e for d in bracket for e in grouped.get(d, [])]
    cell_metrics = degree_metrics(endpoint_episodes, rounds)
    cell_verdict = classify(cell_metrics, density)

    geometric_clean = cell["T"] <= cell["M"] / 2.0
    cell_row = {
        "run": run_dir.name,
        **{key: cell[key] for key in ("kind", "N", "M", "K", "V", "T", "b", "m")},
        "d_c": d_c,
        "scored_at": scored_at,
        "d_minus": bracket[0] if bracket else None,
        "d_plus": bracket[1] if len(bracket) > 1 else None,
        **cell_verdict,
        **{key: cell_metrics.get(key) for key in (
            "min_budget_utilization", "min_candidate_capacity_ratio",
            "max_frontier_empty_fraction", "max_duplicate_evidence_fraction",
        )},
        "degree_density": density,
        "geometric_T_le_M_over_2": geometric_clean,
        "agrees_with_geometric_filter": (
            (cell_verdict["classification"] != DEPLETED) == geometric_clean
        ),
    }

    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "regimes_by_degree.csv", degree_rows, DEGREE_COLUMNS)
    write_csv(analysis_dir / "regimes_cell.csv", [cell_row], CELL_COLUMNS)
    summary = {
        "run_dir": str(run_dir),
        "thresholds": {
            "source": "configs/clean_criteria.json via src.experiment_config",
            "supply": SUPPLY_LIMITS,
            "independence": INDEPENDENCE_LIMITS,
            "unavailable_in_agent_logs": list(UNAVAILABLE),
        },
        "cell": cell_row,
        "by_degree": degree_rows,
        "note": (
            "Diagnostics are averaged over episodes within a round and then "
            "taken at the worst round, as in src/classify_clean.py. The cell "
            "label uses the two degrees bracketing the crossing; the per-degree "
            "table uses each degree's own density."
        ),
    }
    (analysis_dir / "regimes.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def cross_run(run_dirs: Sequence[Path], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    cells, skipped = [], []
    for run_dir in run_dirs:
        try:
            cells.append(report(Path(run_dir))["cell"])
        except (FileNotFoundError, ValueError, KeyError) as error:
            skipped.append({"run": Path(run_dir).name, "reason": str(error)})
    cells.sort(key=lambda c: c["run"])
    write_csv(out_dir / "regimes_cells.csv", cells, CELL_COLUMNS)

    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["classification"]] = counts.get(cell["classification"], 0) + 1
    # Agreement matrix against the purely geometric depletion filter used by
    # scaling.py, which keeps a cell when T <= M / 2.
    matrix: dict[str, dict[str, int]] = {}
    for cell in cells:
        row = matrix.setdefault(cell["classification"], {"T<=M/2": 0, "T>M/2": 0})
        row["T<=M/2" if cell["geometric_T_le_M_over_2"] else "T>M/2"] += 1
    resolved = [c for c in cells if c["agrees_with_geometric_filter"] is not None]
    summary = {
        "cells": len(cells),
        "skipped": skipped,
        "counts": counts,
        "agreement_matrix_vs_T_le_M_over_2": matrix,
        "agreement_rate": (
            sum(1 for c in resolved if c["agrees_with_geometric_filter"]) / len(resolved)
            if resolved else None
        ),
        "disagreements": [
            {"run": c["run"], "classification": c["classification"],
             "T": c["T"], "M": c["M"], "supply_reasons": c["supply_reasons"]}
            for c in resolved if not c["agrees_with_geometric_filter"]
        ],
        "thresholds": {"supply": SUPPLY_LIMITS, "independence": INDEPENDENCE_LIMITS,
                       "unavailable_in_agent_logs": list(UNAVAILABLE)},
    }
    (out_dir / "regimes_cross_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, help="cross-run mode: one table over all cells")
    args = parser.parse_args(argv)
    if args.out_dir is not None:
        payload = cross_run(args.run_dir, args.out_dir)
    else:
        payload = [report(run_dir) for run_dir in args.run_dir]
        payload = payload[0] if len(payload) == 1 else payload
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
