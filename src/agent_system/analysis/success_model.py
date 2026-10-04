#!/usr/bin/env python3
"""Test the paper's link between the order parameter and success probability.

The revised text derives, from one correct candidate against ``A_T`` incorrect
ones after ``T`` rounds,

    P_succ(d) ~= 1 / (1 + A_T(d)).

That is a prediction about levels, not just about where the crossing sits, so
it can be checked directly: every run already records ``A_T`` per episode and
whether any agent succeeded.  This module puts the two side by side per
degree and reports how far apart they are.

Two things it is careful about.

First, ``1/(1 + A_T)`` is convex in ``A_T``, so evaluating it at the mean of
``A_T`` is not the mean of the prediction and the gap reaches a factor of two
near the transition.  The prediction reported as ``P_predicted`` is therefore
the average over episodes of ``1/(1 + A_T)``; the old plug-in reading is kept
beside it as ``P_pred_of_mean_A`` so the two can be compared rather than
confused.

Second, "success" is four different events.  ``verified_predictive_success``
is the strictest and is close to impossible below the critical degree, so
comparing the closed form with it alone makes the model look worse than it is
for reasons that have nothing to do with the model.  Every criterion recorded
in the episodes gets its own observed column and its own error.

It is deliberately blunt about the comparison.  The relation treats the
correct candidate as one draw among ``1 + A_T``, which ignores that agents
choose what to verify rather than sampling uniformly, so a large gap is
informative rather than a bug.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from ..evaluation import CRITERIA
from ..storage import iter_jsonl
from .summarize import write_csv

PRIMARY = "verified_predictive_success"

BASE_COLUMNS = (
    "d", "episodes", "agents", "mean_A_T", "criterion",
    "P_predicted", "P_predicted_per_agent", "P_predicted_any_agent",
    "P_pred_of_mean_A", "P_pred_per_agent_of_mean_A", "P_pred_any_agent_of_mean_A",
    "P_observed", "abs_error", "abs_error_per_agent", "log10_gap",
)


def final_incorrect(episode: dict) -> float | None:
    """``A_T``: active incorrect candidates over all agents after the last round."""
    transitions = episode.get("transitions") or []
    if not transitions:
        return None
    value = transitions[-1].get("A_post")
    return None if value is None else float(value)


def criteria_present(episodes: Sequence[dict]) -> list[str]:
    """Success criteria actually recorded, in the order the evaluator lists."""
    seen = set()
    for episode in episodes:
        for key in (episode.get("evaluation") or {}):
            if key.startswith("any_"):
                seen.add(key[len("any_"):])
    ordered = [name for name in CRITERIA if name in seen]
    return ordered + sorted(seen - set(ordered))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def rows_by_degree(
    episodes: Sequence[dict],
    criterion: str = PRIMARY,
    criteria: Sequence[str] | None = None,
) -> list[dict]:
    names = list(criteria) if criteria is not None else criteria_present(episodes)
    if criterion not in names:
        names = [criterion, *names]
    by_degree: dict[int, list[dict]] = {}
    for episode in episodes:
        if episode.get("status") != "ok":
            continue
        by_degree.setdefault(int(episode["d"]), []).append(episode)

    rows = []
    for degree in sorted(by_degree):
        group = by_degree[degree]
        pairs = [(e, final_incorrect(e)) for e in group]
        pairs = [(e, value) for e, value in pairs if value is not None]
        if not pairs:
            continue
        finals = [value for _, value in pairs]
        mean_final = _mean(finals)
        # A_T is summed over all N agents, but the run succeeds if any single
        # agent holds the target, so the same argument applied per agent and
        # then taken over N independent agents gives a second prediction.
        agents = int(group[0].get("N") or 1)
        predicted = _mean([1.0 / (1.0 + value) for value in finals])
        per_agent = _mean([1.0 / (1.0 + value / agents) for value in finals])
        any_agent = _mean(
            [1.0 - (1.0 - 1.0 / (1.0 + value / agents)) ** agents for value in finals]
        )
        # The plug-in readings, kept so the Jensen gap stays visible.
        plug_in = 1.0 / (1.0 + mean_final)
        plug_in_per_agent = 1.0 / (1.0 + mean_final / agents)
        plug_in_any_agent = 1.0 - (1.0 - plug_in_per_agent) ** agents
        row = {
            "d": degree,
            "episodes": len(pairs),
            "agents": agents,
            "mean_A_T": mean_final,
            "criterion": criterion,
            "P_predicted": predicted,
            "P_predicted_per_agent": per_agent,
            "P_predicted_any_agent": any_agent,
            "P_pred_of_mean_A": plug_in,
            "P_pred_per_agent_of_mean_A": plug_in_per_agent,
            "P_pred_any_agent_of_mean_A": plug_in_any_agent,
        }
        for name in names:
            achieved = [
                bool((e.get("evaluation") or {}).get(f"any_{name}")) for e, _ in pairs
            ]
            observed = sum(achieved) / len(achieved)
            row[f"P_observed_{name}"] = observed
            row[f"abs_error_{name}"] = abs(observed - predicted)
            row[f"abs_error_per_agent_{name}"] = abs(observed - per_agent)
        observed = row[f"P_observed_{criterion}"]
        row["P_observed"] = observed
        row["abs_error"] = row[f"abs_error_{criterion}"]
        row["abs_error_per_agent"] = row[f"abs_error_per_agent_{criterion}"]
        # A ratio is unreadable when the prediction underflows, so the
        # log10 gap is reported instead and left blank at the ends.
        row["log10_gap"] = (
            math.log10(observed / predicted)
            if observed > 0.0 and predicted > 0.0
            else None
        )
        rows.append(row)
    return rows


def _columns(names: Sequence[str]) -> tuple[str, ...]:
    per_criterion = [
        column
        for name in names
        for column in (f"P_observed_{name}", f"abs_error_{name}", f"abs_error_per_agent_{name}")
    ]
    return (*BASE_COLUMNS, *per_criterion)


def _render(rows: list[dict], figures_dir: Path, criterion: str = PRIMARY) -> list[str]:
    """Observed success against both readings of the closed form."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError:
        return []
    if not rows:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    degrees = [row["d"] for row in rows]
    figure, axes = pyplot.subplots(figsize=(4.6, 3.4), dpi=160)
    axes.plot(degrees, [r["P_observed"] for r in rows], marker="o", markersize=4,
              linewidth=1.3, label=f"observed: {criterion}")
    axes.plot(degrees, [r["P_predicted"] for r in rows], marker="s", markersize=3,
              linewidth=1.1, label="mean 1/(1+A_T)")
    axes.plot(degrees, [r["P_predicted_per_agent"] for r in rows], marker="^",
              markersize=3, linewidth=1.1, label="mean 1/(1+A_T/N)")
    axes.plot(degrees, [r["P_pred_of_mean_A"] for r in rows], marker="x",
              markersize=3, linewidth=0.9, linestyle=":", label="1/(1+mean A_T)")
    axes.set_xlabel("degree d")
    axes.set_ylabel("P(success)")
    axes.set_ylim(-0.05, 1.05)
    axes.set_title("success against the closed form")
    axes.legend(fontsize=6.5, frameon=False)
    figure.tight_layout()
    written = []
    for suffix in ("png", "pdf"):
        target = figures_dir / f"success_model.{suffix}"
        figure.savefig(target)
        written.append(str(target))
    pyplot.close(figure)
    return written


def report(
    run_dir: Path, criterion: str = PRIMARY, criteria: Sequence[str] | None = None
) -> dict:
    run_dir = Path(run_dir)
    episodes = list(iter_jsonl(run_dir / "episodes.jsonl"))
    if not episodes:
        raise FileNotFoundError(f"no episodes recorded in {run_dir}")
    names = list(criteria) if criteria is not None else criteria_present(episodes)
    if criterion not in names:
        names = [criterion, *names]
    rows = rows_by_degree(episodes, criterion, names)
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "success_model.csv", rows, _columns(names))
    figures = _render(rows, run_dir / "figures", criterion)
    errors = [row["abs_error"] for row in rows]
    errors_per_agent = [row["abs_error_per_agent"] for row in rows]
    gaps = [row["log10_gap"] for row in rows if row["log10_gap"] is not None]
    by_criterion = {
        name: {
            "mean_absolute_error": (
                _mean([row[f"abs_error_{name}"] for row in rows]) if rows else None
            ),
            "max_absolute_error": (
                max(row[f"abs_error_{name}"] for row in rows) if rows else None
            ),
            "mean_absolute_error_per_agent": (
                _mean([row[f"abs_error_per_agent_{name}"] for row in rows]) if rows else None
            ),
        }
        for name in names
    }
    plug_in_errors = (
        [abs(row["P_observed"] - row["P_pred_of_mean_A"]) for row in rows] if rows else []
    )
    summary = {
        "run_dir": str(run_dir),
        "criterion": criterion,
        "criteria": names,
        "degrees": len(rows),
        "mean_absolute_error": _mean(errors) if errors else None,
        "max_absolute_error": max(errors) if errors else None,
        "mean_absolute_error_per_agent": (
            _mean(errors_per_agent) if errors_per_agent else None
        ),
        "mean_absolute_error_of_mean_A": _mean(plug_in_errors) if plug_in_errors else None,
        "by_criterion": by_criterion,
        "median_log10_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
        "degrees_with_both_nonzero": len(gaps),
        "figures": figures,
        "curve": rows,
        "columns": {
            "P_predicted": "mean over episodes of 1/(1+A_T) -- the unbiased reading",
            "P_pred_of_mean_A": "1/(1+mean A_T) -- the plug-in reading, convex-biased",
            "P_predicted_per_agent": "mean over episodes of 1/(1+A_T/N)",
            "P_predicted_any_agent": "mean over episodes of 1-(1-1/(1+A_T/N))^N",
            "P_observed_<criterion>": "share of episodes at that degree achieving it",
            "abs_error_<criterion>": "|P_observed_<criterion> - P_predicted|",
            "P_observed": f"alias of P_observed_{criterion}",
        },
        "note": (
            "P_predicted averages 1/(1+A_T) over episodes rather than "
            "evaluating it at the mean A_T: the function is convex, so the "
            "plug-in reading kept as P_pred_of_mean_A is systematically low. "
            "The relation assumes the correct candidate is one draw among "
            "1 + A_T; agents choose what to verify instead. "
            "P_predicted_per_agent divides A_T by N first, since A_T is summed "
            "over agents while a run succeeds if any one agent holds the "
            "target. Every recorded success criterion is compared, because "
            "verified_predictive_success is near-impossible below d_c and is "
            "not the only reading of P_succ."
        ),
    }
    (analysis_dir / "success_model.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--criterion", choices=sorted(CRITERIA), default=PRIMARY)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.run_dir, args.criterion), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
