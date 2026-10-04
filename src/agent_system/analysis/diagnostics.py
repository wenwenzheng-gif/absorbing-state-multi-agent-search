#!/usr/bin/env python3
"""Extinction and budget: what the population is doing where the crossing is.

Two mechanisms can make a measured ``d_c`` mean something other than the
balance between branching and pruning, and both are already logged.

*Extinction.* An agent whose frontier empties stops contributing candidates
whatever the degree.  The lineage ratio keeps falling, but for a reason the
theory does not contain.  ``frontier_empty_fraction`` is that, per round.

*Budget redistribution.* The runner hands the system-level budget ``N m`` to
whoever can still use it, so once some agents go idle the active ones run more
than ``m`` experiments each.  The theory's ``m`` is then not the ``m`` that
acted.  ``agents_exceeding_nominal_m`` counts the agents above the nominal
allowance and ``B_used / N_experiment_active_start`` is the allowance that was
actually realised per active agent.

Both are reported per degree and per round, and summarised at the degrees
bracketing the crossing, which is the only place the number matters for
``d_c``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .run_cell import (
    cell_parameters,
    episodes_by_degree,
    load_episodes,
    matplotlib_pyplot,
    measured_crossing,
    pooled_ratio,
    round_mean,
    save_figure,
)
from .summarize import write_csv

ROUND_COLUMNS = (
    "run", "d", "round", "episodes", "frontier_empty_fraction",
    "empty_after_receive_fraction", "active_agent_fraction", "budget_utilization",
    "agents_exceeding_nominal_m", "fraction_exceeding_nominal_m",
    "budget_redistributed_above_m", "realised_m_per_active_agent",
    "max_experiments_per_agent", "candidate_capacity_ratio",
)
CELL_COLUMNS = (
    "run", "kind", "N", "M", "K", "V", "T", "b", "m", "d_c", "degrees_used",
    "max_frontier_empty_fraction", "final_frontier_empty_fraction",
    "min_active_agent_fraction", "min_budget_utilization",
    "max_agents_exceeding_nominal_m", "max_fraction_exceeding_nominal_m",
    "max_realised_m_per_active_agent", "max_experiments_per_agent",
)


def _rows(run_dir: Path, grouped: dict[int, list[dict]], rounds: int, agents: int) -> list[dict]:
    rows = []
    for degree, group in sorted(grouped.items()):
        for index in range(rounds):
            exceeding = round_mean(group, index, "agents_exceeding_nominal_m")
            rows.append(
                {
                    "run": Path(run_dir).name,
                    "d": degree,
                    "round": index + 1,
                    "episodes": len(group),
                    "frontier_empty_fraction": round_mean(group, index, "frontier_empty_fraction"),
                    "empty_after_receive_fraction": round_mean(
                        group, index, "empty_after_receive_fraction"
                    ),
                    "active_agent_fraction": round_mean(group, index, "active_agent_fraction"),
                    "budget_utilization": round_mean(group, index, "budget_utilization"),
                    "agents_exceeding_nominal_m": exceeding,
                    "fraction_exceeding_nominal_m": (
                        exceeding / agents if exceeding is not None and agents else None
                    ),
                    "budget_redistributed_above_m": round_mean(
                        group, index, "budget_redistributed_above_m"
                    ),
                    # Pooled rather than averaged: an episode with no active
                    # agent would otherwise contribute a division by zero.
                    "realised_m_per_active_agent": pooled_ratio(
                        group, index, "B_used", "N_experiment_active_start"
                    ),
                    "max_experiments_per_agent": round_mean(
                        group, index, "max_experiments_per_agent"
                    ),
                    "candidate_capacity_ratio": round_mean(
                        group, index, "candidate_capacity_ratio"
                    ),
                }
            )
    return rows


def _at_crossing(rows: Sequence[dict], degrees: Sequence[int]) -> dict:
    subset = [row for row in rows if row["d"] in set(degrees)]
    if not subset:
        return {}

    def worst(key: str, kind: str) -> float | None:
        values = [row[key] for row in subset if row[key] is not None]
        return (min(values) if kind == "min" else max(values)) if values else None

    last_round = max(row["round"] for row in subset)
    final = [row["frontier_empty_fraction"] for row in subset if row["round"] == last_round]
    final = [value for value in final if value is not None]
    return {
        "max_frontier_empty_fraction": worst("frontier_empty_fraction", "max"),
        "final_frontier_empty_fraction": sum(final) / len(final) if final else None,
        "min_active_agent_fraction": worst("active_agent_fraction", "min"),
        "min_budget_utilization": worst("budget_utilization", "min"),
        "max_agents_exceeding_nominal_m": worst("agents_exceeding_nominal_m", "max"),
        "max_fraction_exceeding_nominal_m": worst("fraction_exceeding_nominal_m", "max"),
        "max_realised_m_per_active_agent": worst("realised_m_per_active_agent", "max"),
        "max_experiments_per_agent": worst("max_experiments_per_agent", "max"),
    }


def _render(rows: Sequence[dict], figures_dir: Path, title: str, d_c: float | None) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not rows:
        return []
    rounds = sorted({row["round"] for row in rows})
    figure, axes = pyplot.subplots(1, 2, figsize=(7.4, 3.1), dpi=160)
    left, right = axes
    for index, panel, key, label in (
        (0, left, "frontier_empty_fraction", "fraction of agents with an empty frontier"),
        (1, right, "fraction_exceeding_nominal_m", "fraction of agents above nominal m"),
    ):
        for number in rounds:
            series = [row for row in rows if row["round"] == number and row[key] is not None]
            if not series:
                continue
            panel.plot([r["d"] for r in series], [r[key] for r in series],
                       linewidth=1.0, marker="o", markersize=2.2, label=f"round {number}")
        if d_c is not None:
            panel.axvline(d_c, color="grey", linewidth=0.8, linestyle=":")
        panel.set_xlabel("degree d")
        panel.set_ylabel(label, fontsize=7)
        panel.legend(fontsize=6, frameon=False, ncol=2)
    left.set_title("extinction", fontsize=9)
    right.set_title("budget redistribution", fontsize=9)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    written = save_figure(figure, figures_dir, "extinction_and_budget")
    pyplot.close(figure)
    return written


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
    rows = _rows(run_dir, grouped, rounds, cell["N"])
    crossing = measured_crossing(run_dir)
    d_c = crossing["d_c"]
    if d_c is None:
        degrees = sorted(grouped)
    else:
        degrees = sorted(grouped, key=lambda d: abs(d - d_c))[:2]
    cell_row = {
        "run": run_dir.name,
        **{key: cell[key] for key in ("kind", "N", "M", "K", "V", "T", "b", "m")},
        "d_c": d_c,
        "degrees_used": " ".join(str(d) for d in sorted(degrees)),
        **_at_crossing(rows, degrees),
    }
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "extinction_and_budget_by_round.csv", rows, ROUND_COLUMNS)
    write_csv(analysis_dir / "extinction_and_budget_cell.csv", [cell_row], CELL_COLUMNS)
    figures = _render(rows, run_dir / "figures", run_dir.name, d_c)
    summary = {
        "run_dir": str(run_dir),
        "cell": cell,
        "at_crossing": cell_row,
        "figures": figures,
        "note": (
            "Summaries are taken at the two degrees nearest the crossing. "
            "realised_m_per_active_agent is B_used / N_experiment_active_start "
            "pooled over episodes, so it exceeds the nominal m exactly when the "
            "budget freed by idle agents was redistributed."
        ),
    }
    (analysis_dir / "extinction_and_budget.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def cross_run(run_dirs: Sequence[Path], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    cells, skipped = [], []
    for run_dir in run_dirs:
        try:
            cells.append(report(Path(run_dir))["at_crossing"])
        except (FileNotFoundError, ValueError, KeyError) as error:
            skipped.append({"run": Path(run_dir).name, "reason": str(error)})
    cells.sort(key=lambda c: c["run"])
    write_csv(out_dir / "extinction_and_budget_cells.csv", cells, CELL_COLUMNS)
    summary = {
        "cells": len(cells),
        "skipped": skipped,
        "cells_with_any_extinction": sum(
            1 for c in cells
            if (c.get("max_frontier_empty_fraction") or 0.0) > 0.0
        ),
        "cells_over_budget": sum(
            1 for c in cells if (c.get("max_agents_exceeding_nominal_m") or 0.0) > 0.0
        ),
        "worst_extinction": max(
            (c for c in cells if c.get("max_frontier_empty_fraction") is not None),
            key=lambda c: c["max_frontier_empty_fraction"], default=None,
        ),
        "worst_over_budget": max(
            (c for c in cells if c.get("max_agents_exceeding_nominal_m") is not None),
            key=lambda c: c["max_agents_exceeding_nominal_m"], default=None,
        ),
    }
    (out_dir / "extinction_and_budget_cross_run.json").write_text(
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
