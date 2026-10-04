#!/usr/bin/env python3
"""Figures for an agent run, each with the CSV it was drawn from.

The CSV files are always written; the PNG/PDF pair is written only when
matplotlib is importable, so analysis never silently depends on a plotting
backend being present.

``--compare`` overlays several runs on one axis.  It is the only way to see
the quantity the paired runs exist to measure: two policies on the same
tasks, degrees and budget differ only in how fast their lineage ratio falls
with ``d``, and that slope ratio is the effective communication degree.

``--convention`` picks the lineage numerator, with the same names and the
same default as ``analysis.critical_degree``, and it is honoured by every
table that has a choice: the pooled curve, the per-round response and the
slope reported by ``--compare``.  The ``post_own`` panel keeps its own
convention, since that is what its name says.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from ..evaluation import CRITERIA
from ..storage import iter_jsonl
from .critical_degree import (
    CONVENTIONS,
    DEFAULT_CONVENTION,
    pooled_log_ratio,
    pooled_step_log_ratio,
)
from .summarize import episode_row, write_csv

FIGURES = (
    "log_R_vs_degree",
    "log_R_post_own_vs_degree",
    "success_vs_degree",
    "frontier_and_budget",
    "evidence_redundancy",
    "cost_vs_success",
    "heldout_error",
    "per_round_response",
)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        return pyplot
    except ImportError:
        return None


def _series(rows: Sequence[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        xs.append(float(row["d"]))
        ys.append(value)
    return xs, ys


def build_tables(
    run_dir: Path, convention: str = DEFAULT_CONVENTION
) -> dict[str, list[dict]]:
    episodes = list(iter_jsonl(run_dir / "episodes.jsonl"))
    rows = [episode_row(payload) for payload in episodes]
    by_degree: dict[int, list[dict]] = {}
    raw_by_degree: dict[int, list[dict]] = {}
    for row, payload in zip(rows, episodes):
        by_degree.setdefault(int(row["d"]), []).append(row)
        raw_by_degree.setdefault(int(payload["d"]), []).append(payload)

    lineage, lineage_post = [], []
    for degree in sorted(raw_by_degree):
        entry = pooled_log_ratio(raw_by_degree[degree], convention)
        lineage.append({"d": degree, "log_R": entry["value"], "status": entry["status"],
                        "convention": convention})
        other = pooled_log_ratio(raw_by_degree[degree], "post_own")
        lineage_post.append({"d": degree, "log_R": other["value"], "status": other["status"]})

    success, supply, redundancy, cost, heldout = [], [], [], [], []
    for degree in sorted(by_degree):
        group = [row for row in by_degree[degree] if row["status"] == "ok"]
        if not group:
            continue
        entry: dict[str, object] = {"d": degree, "episodes": len(group)}
        for name in CRITERIA:
            achieved = [bool(row.get(f"any_{name}")) for row in group]
            entry[f"P_{name}"] = sum(achieved) / len(achieved)
        success.append(entry)
        supply.append(
            {
                "d": degree,
                "mean_final_frontier": _mean(group, "final_mean_frontier"),
                "mean_budget_utilization": _mean(group, "budget_utilization_mean"),
                "mean_candidate_capacity": _mean(group, "candidate_capacity_total"),
                "mean_candidate_invalidated": _mean(group, "candidate_invalidated_total"),
            }
        )
        redundancy.append(
            {
                "d": degree,
                "mean_recv_messages": _mean(group, "recv_messages_total"),
                "mean_recv_informative": _mean(group, "recv_informative_total"),
                "mean_recv_duplicate": _mean(group, "recv_duplicate_total"),
                "informative_fraction": _ratio(group, "recv_informative_total", "recv_messages_total"),
                "duplicate_per_informative": _ratio(
                    group, "recv_duplicate_total", "recv_informative_total"
                ),
            }
        )
        # One arbitrary episode's token count is not the degree's cost: the
        # mean is what scales with d, and the sum is what the run actually
        # spent there.
        usage = [
            float(payload.get("llm_usage", {}).get("llm_total_tokens") or 0)
            for payload in raw_by_degree[degree]
            if payload.get("llm_usage")
        ]
        cost.append(
            {
                "d": degree,
                "mean_wall_seconds": _mean(group, "wall_seconds"),
                "llm_episodes": len(usage),
                "mean_llm_total_tokens": sum(usage) / len(usage) if usage else None,
                "sum_llm_total_tokens": sum(usage) if usage else None,
                "P_predictive_success": entry["P_predictive_success"],
            }
        )
        heldout.append({"d": degree, "mean_heldout_error_min": _mean(group, "heldout_error_min")})
    per_round = []
    # Over every episode, not the first of each degree: truncated episodes
    # (stop_on_success) make the groups ragged, and reading the round count
    # off one episode either lost rounds or indexed past the short ones.
    rounds = max(
        (
            len(payload.get("transitions") or [])
            for group in raw_by_degree.values()
            for payload in group
        ),
        default=0,
    )
    for index in range(rounds):
        for degree in sorted(raw_by_degree):
            entry = pooled_step_log_ratio(raw_by_degree[degree], convention, index)
            per_round.append(
                {"round": index + 1, "d": degree, "log_r": entry["value"],
                 "status": entry["status"], "convention": convention,
                 "episodes": entry.get("episodes"), "ragged": bool(entry.get("ragged"))}
            )

    return {
        "log_R_vs_degree": lineage,
        "log_R_post_own_vs_degree": lineage_post,
        "per_round_response": per_round,
        "success_vs_degree": success,
        "frontier_and_budget": supply,
        "evidence_redundancy": redundancy,
        "cost_vs_success": cost,
        "heldout_error": heldout,
    }


COMPARISONS = (
    ("log_R_vs_degree", "log_R", "pooled log R", "pre_own"),
    ("log_R_post_own_vs_degree", "log_R", "pooled log R (post_own)", "post_own"),
    ("success_vs_degree", "P_verified_predictive_success", "P(verified predictive)", None),
    ("evidence_redundancy", "duplicate_per_informative", "duplicate / informative", None),
)


def _comparisons(convention: str = DEFAULT_CONVENTION):
    """``COMPARISONS`` with the first panel relabelled for the convention in use."""
    return tuple(
        (figure, key, f"pooled log R ({convention})", convention)
        if figure == "log_R_vs_degree"
        else (figure, key, ylabel, other)
        for figure, key, ylabel, other in COMPARISONS
    )


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of ``y`` on ``x``; ``None`` below two distinct x."""
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _mean(rows: Sequence[dict], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return sum(values) / len(values) if values else None


def _ratio(rows: Sequence[dict], top: str, bottom: str) -> float | None:
    numerator = _mean(rows, top)
    denominator = _mean(rows, bottom)
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def render(run_dir: Path, convention: str = DEFAULT_CONVENTION) -> dict:
    run_dir = Path(run_dir)
    tables = build_tables(run_dir, convention)
    figures_dir = run_dir / "figures"
    data_dir = run_dir / "analysis"
    written = []
    for name, rows in tables.items():
        if rows:
            written.append(str(write_csv(data_dir / f"{name}.csv", rows)))

    pyplot = _matplotlib()
    if pyplot is None:
        return {
            "figures": [],
            "data": written,
            "status": "csv_only",
            "reason": "matplotlib is not installed; install requirements.txt to render figures",
        }

    figures_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("log_R_vs_degree", "log_R", "degree d", f"pooled log R ({convention})"),
        ("success_vs_degree", "P_predictive_success", "degree d", "P(predictive success)"),
        ("frontier_and_budget", "mean_final_frontier", "degree d", "mean final frontier"),
        ("evidence_redundancy", "informative_fraction", "degree d", "informative share"),
        ("cost_vs_success", "P_predictive_success", "degree d", "P(predictive success)"),
        ("heldout_error", "mean_heldout_error_min", "degree d", "best hold-out error"),
    ]
    rendered = []
    rendered += _render_per_round(
        pyplot, tables.get("per_round_response") or [], figures_dir, convention
    )
    for name, key, xlabel, ylabel in plots:
        rows = tables[name]
        xs, ys = _series(rows, key)
        if not xs:
            continue
        figure, axes = pyplot.subplots(figsize=(4.2, 3.2), dpi=160)
        axes.plot(xs, ys, marker="o", linewidth=1.3)
        if name == "log_R_vs_degree":
            axes.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(name.replace("_", " "))
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            target = figures_dir / f"{name}.{suffix}"
            figure.savefig(target)
            rendered.append(str(target))
        pyplot.close(figure)
    return {"figures": rendered, "data": written, "status": "ok", "convention": convention}


def _render_per_round(
    pyplot, rows, figures_dir, convention: str = DEFAULT_CONVENTION
) -> list[str]:
    """One line per round of ``log r_t`` against degree.

    The theory gives every round the same form, ``ln b + d ln(1 - p_t)``, so a
    round that is flat in ``d``, or that stops responding past a small ``d``,
    is visible here and nowhere else.
    """
    if not rows:
        return []
    by_round: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        value = row.get("log_r")
        if value is None or not math.isfinite(float(value)):
            continue
        by_round.setdefault(int(row["round"]), []).append((float(row["d"]), float(value)))
    if not by_round:
        return []
    figure, axes = pyplot.subplots(figsize=(4.6, 3.4), dpi=160)
    for number in sorted(by_round):
        points = sorted(by_round[number])
        axes.plot(
            [x for x, _ in points], [y for _, y in points],
            marker="o", markersize=3, linewidth=1.1, label=f"round {number}",
        )
    axes.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axes.set_xlabel("degree d")
    axes.set_ylabel(f"pooled log r per round ({convention})")
    axes.set_title("per-round response to degree")
    axes.legend(fontsize=6.5, frameon=False, ncol=2)
    figure.tight_layout()
    written = []
    for suffix in ("png", "pdf"):
        target = figures_dir / f"per_round_response.{suffix}"
        figure.savefig(target)
        written.append(str(target))
    pyplot.close(figure)
    return written


def compare(
    run_dirs: Sequence[Path],
    out_dir: Path,
    labels: Sequence[str] | None = None,
    convention: str = DEFAULT_CONVENTION,
) -> dict:
    """Overlay several runs and report each one's slope of log R in ``d``.

    Comparing runs that were not paired on task seeds, degrees and budget is
    meaningless, so the caller is responsible for that; what this adds is the
    one number the pairing buys, ``d(log R)/dd``, whose ratio between two
    policies is their ratio of effective communication degree.
    """
    out_dir = Path(out_dir)
    names = list(labels) if labels else [Path(d).name for d in run_dirs]
    if len(names) != len(run_dirs):
        raise ValueError("--labels must give one label per run directory")
    tables = {name: build_tables(Path(d), convention) for name, d in zip(names, run_dirs)}
    comparisons = _comparisons(convention)

    slopes: dict[str, dict[str, float | None]] = {}
    rows: list[dict] = []
    for figure, key, _ylabel, panel in comparisons:
        for name in names:
            xs, ys = _series(tables[name][figure], key)
            if panel is not None:
                slopes.setdefault(name, {})[panel] = _slope(xs, ys)
            for x, y in zip(xs, ys):
                rows.append({"figure": figure, "run": name, "d": x, "value": y})
    written = [str(write_csv(out_dir / "comparison.csv", rows))]

    pyplot = _matplotlib()
    if pyplot is None:
        return {
            "figures": [],
            "data": written,
            "slopes": slopes,
            "convention": convention,
            "status": "csv_only",
            "reason": "matplotlib is not installed; install requirements.txt to render figures",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for figure, key, ylabel, panel in comparisons:
        drawn = False
        fig, axes = pyplot.subplots(figsize=(4.6, 3.4), dpi=160)
        for name in names:
            xs, ys = _series(tables[name][figure], key)
            if not xs:
                continue
            label = name
            if panel is not None:
                slope = slopes.get(name, {}).get(panel)
                if slope is not None:
                    label = f"{name} (slope {slope:+.3f})"
            axes.plot(xs, ys, marker="o", linewidth=1.3, markersize=4, label=label)
            drawn = True
        if not drawn:
            pyplot.close(fig)
            continue
        if figure.startswith("log_R"):
            axes.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        axes.set_xlabel("degree d")
        axes.set_ylabel(ylabel)
        axes.set_title(figure.replace("_", " "))
        axes.legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            target = out_dir / f"compare_{figure}.{suffix}"
            fig.savefig(target)
            rendered.append(str(target))
        pyplot.close(fig)
    return {"figures": rendered, "data": written, "slopes": slopes,
            "convention": convention, "status": "ok"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--compare", type=Path, nargs="+", help="two or more run directories to overlay"
    )
    parser.add_argument("--labels", nargs="+", help="one legend label per compared run")
    parser.add_argument("--out-dir", type=Path, help="where --compare writes; default figures/compare")
    parser.add_argument(
        "--convention", choices=sorted(CONVENTIONS), default=DEFAULT_CONVENTION,
        help="lineage numerator convention, as in analysis.critical_degree",
    )
    args = parser.parse_args(argv)
    if bool(args.run_dir) == bool(args.compare):
        parser.error("give exactly one of --run-dir or --compare")
    if args.compare:
        out_dir = args.out_dir or Path("figures/compare")
        report = compare(args.compare, out_dir, args.labels, args.convention)
    else:
        report = render(args.run_dir, args.convention)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
