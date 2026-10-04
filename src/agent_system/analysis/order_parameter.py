#!/usr/bin/env python3
"""The order parameter itself: ``A_t`` against the round, one line per degree.

The paper defines ``A_t`` -- active incorrect candidates summed over all
agents -- as the order parameter, and states the transition as a change of
sign in its growth: below ``d_c`` it grows across rounds, above ``d_c`` it
decays.  Every other analysis module in this package measures the growth rate
instead (the pooled lineage ratio), which is the sharper estimator but hides
the picture the claim is actually about.  This module draws that picture.

Two things are reported that the ratio scan cannot give directly:

* the trajectory ``A_0 .. A_T`` per degree, with a task-cluster band, on a log
  axis, so growth and decay separate visually at a single degree;
* the literal endpoint ratio ``log(A_T / A_0) / (T - 1)``, whose zero is a
  third critical-degree convention.  It differs from the ``pre_own`` and
  ``post_own`` crossings on purpose: it counts every candidate alive at the
  end, including lineages founded after round 1, where the lineage ratio
  follows only the incorrect population that already existed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from .run_cell import (
    cell_parameters,
    cluster_bootstrap,
    episodes_by_degree,
    load_episodes,
    matplotlib_pyplot,
    measured_crossing,
    save_figure,
)
from .summarize import write_csv

TRAJECTORY_COLUMNS = (
    "d", "t", "episodes", "mean_A_t", "ci_low", "ci_high", "mean_A_t_over_A_0",
)
ENDPOINT_COLUMNS = (
    "d", "episodes", "mean_A_0", "mean_A_T", "ratio_A_T_over_A_0",
    "log_ratio_per_round", "ci_low", "ci_high", "growing",
)


def trajectory(episode: dict) -> list[float]:
    """``[A_0, A_1, ... A_T]`` for one episode.

    ``A_0`` is the Mode-B initial incorrect pool and ``A_t`` is ``A_post`` of
    round ``t``, i.e. the population after that round's branch, experiment and
    pruning -- the same quantity the recursion in the paper advances.
    """
    values = [float(episode.get("A_initial_incorrect") or 0.0)]
    for transition in episode.get("transitions") or []:
        values.append(float(transition.get("A_post") or 0.0))
    return values


def _mean_at(episodes: Sequence[dict], index: int) -> float | None:
    values = [t[index] for t in (trajectory(e) for e in episodes) if index < len(t)]
    return sum(values) / len(values) if values else None


def _log_ratio_per_round(episodes: Sequence[dict]) -> float | None:
    """``log(A_T / A_0) / (T - 1)`` pooled by averaging both endpoints first."""
    trajectories = [trajectory(e) for e in episodes]
    trajectories = [t for t in trajectories if len(t) >= 2]
    if not trajectories:
        return None
    rounds = len(trajectories[0]) - 1
    first = sum(t[0] for t in trajectories) / len(trajectories)
    last = sum(t[-1] for t in trajectories) / len(trajectories)
    if first <= 0.0 or last <= 0.0 or rounds <= 0:
        return None
    return math.log(last / first) / rounds


def _crossing(rows: Sequence[dict]) -> dict:
    """Where ``log(A_T/A_0)`` changes sign, by linear interpolation in ``d``."""
    points = [
        (int(row["d"]), float(row["log_ratio_per_round"]))
        for row in rows
        if row.get("log_ratio_per_round") is not None
    ]
    for (d_low, low), (d_high, high) in zip(points, points[1:]):
        if low > 0.0 > high:
            return {
                "status": "crossing",
                "d_minus": d_low,
                "d_plus": d_high,
                "d_c": d_low + (d_high - d_low) * low / (low - high),
            }
    if not points:
        return {"status": "not_estimable", "d_c": None}
    if all(value > 0.0 for _, value in points):
        return {"status": "no_crossing_all_growing", "d_c": None}
    if all(value < 0.0 for _, value in points):
        return {"status": "no_crossing_all_decaying", "d_c": None}
    return {"status": "no_monotone_crossing", "d_c": None}


def tables(run_dir: Path, *, draws: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    episodes = load_episodes(run_dir)
    grouped = episodes_by_degree(episodes)
    if not grouped:
        raise ValueError(f"no complete episodes in {run_dir}")
    rounds = max(len(trajectory(e)) for group in grouped.values() for e in group)

    trajectory_rows: list[dict] = []
    for degree, group in sorted(grouped.items()):
        base = _mean_at(group, 0)
        for index in range(rounds):
            mean = _mean_at(group, index)
            if mean is None:
                continue
            band = cluster_bootstrap(
                {degree: group},
                lambda sample, i=index: _mean_at(sample[degree], i),
                draws=draws,
                seed=seed + 1009 * index,
            )
            trajectory_rows.append(
                {
                    "d": degree,
                    "t": index,
                    "episodes": len(group),
                    "mean_A_t": mean,
                    "ci_low": band["ci"][0],
                    "ci_high": band["ci"][1],
                    "mean_A_t_over_A_0": (mean / base) if base else None,
                }
            )

    endpoint_rows: list[dict] = []
    for degree, group in sorted(grouped.items()):
        first, last = _mean_at(group, 0), _mean_at(group, rounds - 1)
        value = _log_ratio_per_round(group)
        band = cluster_bootstrap(
            {degree: group},
            lambda sample: _log_ratio_per_round(sample[degree]),
            draws=draws,
            seed=seed,
        )
        endpoint_rows.append(
            {
                "d": degree,
                "episodes": len(group),
                "mean_A_0": first,
                "mean_A_T": last,
                "ratio_A_T_over_A_0": (last / first) if first else None,
                "log_ratio_per_round": value,
                "ci_low": band["ci"][0],
                "ci_high": band["ci"][1],
                "growing": None if value is None else value > 0.0,
            }
        )
    return trajectory_rows, endpoint_rows, {"rounds": rounds - 1, "degrees": len(grouped)}


def _render(
    trajectory_rows: Sequence[dict],
    endpoint_rows: Sequence[dict],
    figures_dir: Path,
    title: str,
) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not trajectory_rows:
        return []
    from matplotlib import cm, colors

    degrees = sorted({int(row["d"]) for row in trajectory_rows})
    norm = colors.Normalize(vmin=min(degrees), vmax=max(degrees) or 1)
    colormap = pyplot.get_cmap("viridis")
    written: list[str] = []

    figure, axes = pyplot.subplots(figsize=(5.0, 3.6), dpi=160)
    for degree in degrees:
        series = [row for row in trajectory_rows if int(row["d"]) == degree]
        xs = [row["t"] for row in series]
        ys = [row["mean_A_t"] for row in series]
        if not any(value and value > 0 for value in ys):
            continue
        colour = colormap(norm(degree))
        axes.plot(xs, ys, color=colour, linewidth=1.1, marker="o", markersize=2.5)
        lows = [row["ci_low"] for row in series]
        highs = [row["ci_high"] for row in series]
        if all(value is not None for value in lows + highs):
            axes.fill_between(xs, lows, highs, color=colour, alpha=0.15, linewidth=0)
    axes.set_yscale("log")
    axes.set_xlabel("round t")
    axes.set_ylabel("A_t (incorrect candidates, all agents)")
    axes.set_title(f"order parameter: {title}", fontsize=9)
    mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    figure.colorbar(mappable, ax=axes, label="degree d")
    figure.tight_layout()
    written += save_figure(figure, figures_dir, "order_parameter_A_t")
    pyplot.close(figure)

    figure, axes = pyplot.subplots(figsize=(4.6, 3.4), dpi=160)
    xs = [row["d"] for row in endpoint_rows if row["mean_A_T"]]
    ys = [row["mean_A_T"] for row in endpoint_rows if row["mean_A_T"]]
    axes.plot(xs, ys, marker="o", markersize=3, linewidth=1.1, label="A_T")
    base = [row["mean_A_0"] for row in endpoint_rows if row["mean_A_T"]]
    if base:
        axes.plot(xs, base, linestyle="--", linewidth=0.9, color="grey", label="A_0")
    axes.set_yscale("log")
    axes.set_xlabel("degree d")
    axes.set_ylabel("A_T")
    axes.set_title(f"final order parameter: {title}", fontsize=9)
    axes.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    written += save_figure(figure, figures_dir, "order_parameter_A_T_vs_degree")
    pyplot.close(figure)
    return written


def report(run_dir: Path, *, draws: int = 400, seed: int = 20260920) -> dict:
    run_dir = Path(run_dir)
    trajectory_rows, endpoint_rows, shape = tables(run_dir, draws=draws, seed=seed)
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "order_parameter_A_t.csv", trajectory_rows, TRAJECTORY_COLUMNS)
    write_csv(analysis_dir / "order_parameter_A_T.csv", endpoint_rows, ENDPOINT_COLUMNS)
    cell = cell_parameters(run_dir) or {}
    figures = _render(trajectory_rows, endpoint_rows, run_dir / "figures", run_dir.name)
    crossing = _crossing(endpoint_rows)
    lineage = measured_crossing(run_dir)
    growing = [row["d"] for row in endpoint_rows if row["growing"] is True]
    decaying = [row["d"] for row in endpoint_rows if row["growing"] is False]
    summary = {
        "run_dir": str(run_dir),
        "cell": cell,
        **shape,
        "degrees_growing": growing,
        "degrees_decaying": decaying,
        "separation_is_monotone": bool(growing and decaying and max(growing) < min(decaying)),
        "endpoint_crossing": crossing,
        "lineage_crossing": lineage,
        "figures": figures,
        "note": (
            "A_t is the per-episode mean of the incorrect population summed over "
            "agents; the band is a 95% task-cluster bootstrap interval. "
            "endpoint_crossing solves log(A_T/A_0) = 0 and is a third convention, "
            "not the same estimator as the pooled lineage ratio in "
            "critical_degree.json: it includes lineages founded after round 1."
        ),
    }
    (analysis_dir / "order_parameter.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.run_dir, draws=args.draws, seed=args.seed), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
