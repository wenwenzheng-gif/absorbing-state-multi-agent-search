#!/usr/bin/env python3
"""The crossover in ``P_succ(d)``: a logistic midpoint as a second ``d_c``.

The paper's success prediction is not only a level (that is what
``success_model.py`` checks) but a shape: ``P_succ`` is a smooth crossover at
finite ``T`` that approaches a step at ``d_c`` as ``T`` grows.  Two things
follow that nothing in the package measured yet.

First, the midpoint of that crossover is an estimate of ``d_c`` that never
touches the lineage ratio.  It is a weaker estimator -- one bit per episode
instead of a population count -- but it is independent, so agreement is
evidence and disagreement localises the problem.

Second, the width is a prediction in its own right: increasing ``T`` at fixed
``|V|, b, m`` should sharpen the crossover once it is measured in the scaled
variable ``d / d_c``, because the same ``d`` is applied for more rounds.  The
cross-run mode collapses several ``T`` onto one axis and reports whether the
relative width actually falls.

Fit: ``P(d) = 1 / (1 + exp(-(d - d_half) / w))`` by maximum likelihood over the
individual episode outcomes, with a task-cluster bootstrap for both
parameters.  ``w`` is the logistic scale; ``width_10_90 = w ln 81`` is the
degree interval over which the curve runs from 0.1 to 0.9, and is the number
quoted as "the width".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from ..evaluation import CRITERIA
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

DEFAULT_CRITERION = "verified_predictive_success"
LOG81 = math.log(81.0)

FIT_COLUMNS = (
    "run", "criterion", "status", "episodes", "degrees", "d_half", "d_half_ci_low",
    "d_half_ci_high", "w", "w_ci_low", "w_ci_high", "width_10_90",
    "relative_width_10_90", "d_c_lineage", "d_half_over_d_c",
)
CURVE_COLUMNS = ("run", "criterion", "d", "episodes", "successes", "P_observed", "P_logistic")


def _outcomes(grouped: dict[int, list[dict]], criterion: str) -> list[tuple[float, int]]:
    """``(degree, 0/1)`` per complete episode."""
    points = []
    for degree, group in grouped.items():
        for episode in group:
            achieved = bool((episode.get("evaluation") or {}).get(f"any_{criterion}"))
            points.append((float(degree), int(achieved)))
    return points


def logistic_fit(points: Sequence[tuple[float, int]]) -> dict:
    """Maximum-likelihood logistic in ``d``, reported as midpoint and scale.

    Fitted in the natural parameterisation ``logit = a + c d`` because that
    likelihood is concave, then converted; a separating sample (every failure
    below every success) sends ``c`` to infinity, so the slope is capped and
    the fit is reported as ``separated`` rather than pretending to a width.
    """
    if not points:
        return {"status": "no_episodes", "d_half": None, "w": None}
    labels = {label for _, label in points}
    if len(labels) < 2:
        status = "all_success" if labels == {1} else "all_failure"
        return {"status": status, "d_half": None, "w": None}

    def negative_log_likelihood(a: float, c: float) -> float:
        total = 0.0
        for degree, label in points:
            z = a + c * degree
            # log(1 + exp(z)) written so that a large |z| cannot overflow.
            softplus = z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))
            total += softplus - label * z
        return total

    # Newton steps on a two-parameter concave problem; a plain loop keeps the
    # dependency surface the same as the rest of the package.
    a, c = 0.0, 0.0
    for _ in range(200):
        g_a = g_c = h_aa = h_ac = h_cc = 0.0
        for degree, label in points:
            z = a + c * degree
            p = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
            residual = p - label
            weight = p * (1.0 - p)
            g_a += residual
            g_c += residual * degree
            h_aa += weight
            h_ac += weight * degree
            h_cc += weight * degree * degree
        determinant = h_aa * h_cc - h_ac * h_ac
        if abs(determinant) < 1e-12:
            break
        step_a = (h_cc * g_a - h_ac * g_c) / determinant
        step_c = (h_aa * g_c - h_ac * g_a) / determinant
        # Halve the step until the likelihood actually improves.
        current = negative_log_likelihood(a, c)
        scale = 1.0
        for _ in range(40):
            if negative_log_likelihood(a - scale * step_a, c - scale * step_c) <= current:
                break
            scale *= 0.5
        a, c = a - scale * step_a, c - scale * step_c
        if max(abs(scale * step_a), abs(scale * step_c)) < 1e-10:
            break
    if not math.isfinite(a) or not math.isfinite(c) or abs(c) < 1e-9:
        return {"status": "flat", "d_half": None, "w": None}
    if abs(c) > 50.0:
        return {"status": "separated", "d_half": -a / c, "w": 1.0 / abs(c)}
    return {
        "status": "ok" if c > 0 else "ok_decreasing",
        "d_half": -a / c,
        "w": 1.0 / c,
        "a": a,
        "c": c,
    }


def _fit_from_groups(grouped: dict[int, list[dict]], criterion: str, key: str) -> float | None:
    fit = logistic_fit(_outcomes(grouped, criterion))
    if fit["status"] not in {"ok", "ok_decreasing"}:
        return None
    return fit[key]


def fit_run(
    run_dir: Path, criterion: str, *, draws: int, seed: int
) -> tuple[dict, list[dict]]:
    episodes = load_episodes(run_dir)
    grouped = episodes_by_degree(episodes)
    points = _outcomes(grouped, criterion)
    fit = logistic_fit(points)
    lineage = measured_crossing(run_dir)

    half = cluster_bootstrap(
        grouped, lambda s: _fit_from_groups(s, criterion, "d_half"), draws=draws, seed=seed
    )
    width = cluster_bootstrap(
        grouped, lambda s: _fit_from_groups(s, criterion, "w"), draws=draws, seed=seed
    )
    row = {
        "run": Path(run_dir).name,
        "criterion": criterion,
        "status": fit["status"],
        "episodes": len(points),
        "degrees": len(grouped),
        "d_half": fit["d_half"],
        "d_half_ci_low": half["ci"][0],
        "d_half_ci_high": half["ci"][1],
        "w": fit["w"],
        "w_ci_low": width["ci"][0],
        "w_ci_high": width["ci"][1],
        "width_10_90": (fit["w"] * LOG81) if fit["w"] else None,
        "relative_width_10_90": (
            fit["w"] * LOG81 / fit["d_half"] if fit["w"] and fit["d_half"] else None
        ),
        "d_c_lineage": lineage["d_c"],
        "d_half_over_d_c": (
            fit["d_half"] / lineage["d_c"]
            if fit["d_half"] and lineage["d_c"]
            else None
        ),
    }
    curve = []
    for degree, group in sorted(grouped.items()):
        successes = sum(
            bool((e.get("evaluation") or {}).get(f"any_{criterion}")) for e in group
        )
        predicted = None
        if fit["status"] in {"ok", "ok_decreasing"}:
            z = fit["a"] + fit["c"] * degree
            predicted = 1.0 / (1.0 + math.exp(-z)) if -700 < z < 700 else float(z > 0)
        curve.append(
            {
                "run": Path(run_dir).name,
                "criterion": criterion,
                "d": degree,
                "episodes": len(group),
                "successes": successes,
                "P_observed": successes / len(group),
                "P_logistic": predicted,
            }
        )
    return row, curve


def _render_run(rows: Sequence[dict], curves: Sequence[dict], figures_dir: Path, title: str) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not curves:
        return []
    figure, axes = pyplot.subplots(figsize=(4.8, 3.4), dpi=160)
    for row in rows:
        series = [c for c in curves if c["criterion"] == row["criterion"]]
        if not series:
            continue
        line, = axes.plot(
            [c["d"] for c in series], [c["P_observed"] for c in series],
            marker="o", markersize=3, linewidth=0.9, label=row["criterion"],
        )
        if any(c["P_logistic"] is not None for c in series):
            axes.plot(
                [c["d"] for c in series], [c["P_logistic"] for c in series],
                linestyle="--", linewidth=1.0, color=line.get_color(),
            )
    reference = next((r["d_c_lineage"] for r in rows if r["d_c_lineage"]), None)
    if reference:
        axes.axvline(reference, color="grey", linewidth=0.8, linestyle=":")
    axes.set_xlabel("degree d")
    axes.set_ylabel("P(success)")
    axes.set_ylim(-0.05, 1.05)
    axes.set_title(f"success crossover: {title}", fontsize=9)
    axes.legend(fontsize=6.5, frameon=False)
    figure.tight_layout()
    written = save_figure(figure, figures_dir, "success_crossover")
    pyplot.close(figure)
    return written


def report(
    run_dir: Path,
    *,
    criteria: Sequence[str] = CRITERIA,
    draws: int = 400,
    seed: int = 20260920,
) -> dict:
    run_dir = Path(run_dir)
    rows, curves = [], []
    for criterion in criteria:
        row, curve = fit_run(run_dir, criterion, draws=draws, seed=seed)
        rows.append(row)
        curves.extend(curve)
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "success_crossover.csv", rows, FIT_COLUMNS)
    write_csv(analysis_dir / "success_crossover_curves.csv", curves, CURVE_COLUMNS)
    figures = _render_run(rows, curves, run_dir / "figures", run_dir.name)
    summary = {
        "run_dir": str(run_dir),
        "cell": cell_parameters(run_dir) or {},
        "fits": rows,
        "figures": figures,
        "note": (
            "d_half is the logistic midpoint of P_succ(d), an estimate of d_c "
            "independent of the lineage ratio. w is the logistic scale; "
            "width_10_90 = w ln 81. A criterion never achieved, or always "
            "achieved, is reported by status rather than fitted."
        ),
    }
    (analysis_dir / "success_crossover.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _render_cross(rows: Sequence[dict], curves: Sequence[dict], out_dir: Path) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not rows:
        return []
    from matplotlib import cm, colors

    horizons = sorted({row["T"] for row in rows})
    norm = colors.Normalize(vmin=min(horizons), vmax=max(horizons) or 1)
    colormap = pyplot.get_cmap("plasma")
    written: list[str] = []

    figure, axes = pyplot.subplots(figsize=(4.8, 3.4), dpi=160)
    for row in rows:
        series = [c for c in curves if c["run"] == row["run"] and c["scaled_d"] is not None]
        if not series:
            continue
        axes.plot(
            [c["scaled_d"] for c in series], [c["P_observed"] for c in series],
            marker="o", markersize=2.5, linewidth=0.9, color=colormap(norm(row["T"])),
            label=f"T={row['T']}",
        )
    axes.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    axes.set_xlabel("d / d_half")
    axes.set_ylabel("P(success)")
    axes.set_ylim(-0.05, 1.05)
    axes.set_title("crossover collapsed on its own midpoint", fontsize=9)
    axes.legend(fontsize=6.5, frameon=False, ncol=2)
    figure.tight_layout()
    written += save_figure(figure, out_dir, "success_crossover_collapse")
    pyplot.close(figure)

    figure, axes = pyplot.subplots(figsize=(4.4, 3.2), dpi=160)
    usable = [row for row in rows if row["relative_width_10_90"] is not None]
    axes.plot([r["T"] for r in usable], [r["relative_width_10_90"] for r in usable],
              marker="o", markersize=4, linewidth=1.1, label="width_10_90 / d_half")
    absolute = [r for r in rows if r["width_10_90"] is not None]
    twin = axes.twinx()
    twin.plot([r["T"] for r in absolute], [r["width_10_90"] for r in absolute],
              marker="s", markersize=3, linewidth=1.0, color="tab:orange",
              label="width_10_90 (degrees)")
    axes.set_xlabel("search horizon T")
    axes.set_ylabel("relative width")
    twin.set_ylabel("absolute width (degrees)")
    axes.set_title("does the crossover sharpen with T?", fontsize=9)
    figure.legend(fontsize=6.5, frameon=False, loc="upper right")
    figure.tight_layout()
    written += save_figure(figure, out_dir, "success_crossover_width_vs_T")
    pyplot.close(figure)
    return written


def cross_run(
    run_dirs: Sequence[Path],
    out_dir: Path,
    *,
    criterion: str = DEFAULT_CRITERION,
    draws: int = 400,
    seed: int = 20260920,
) -> dict:
    """Collapse several runs of one dataset that differ only in ``T``."""
    out_dir = Path(out_dir)
    rows, curves = [], []
    for run_dir in run_dirs:
        cell = cell_parameters(run_dir)
        if cell is None:
            continue
        row, curve = fit_run(Path(run_dir), criterion, draws=draws, seed=seed)
        row.update({key: cell[key] for key in ("kind", "N", "M", "K", "V", "T", "b", "m")})
        rows.append(row)
        for entry in curve:
            entry["T"] = cell["T"]
            entry["scaled_d"] = (
                entry["d"] / row["d_half"] if row["d_half"] else None
            )
            entry["scaled_d_lineage"] = (
                entry["d"] / row["d_c_lineage"] if row["d_c_lineage"] else None
            )
        curves.extend(curve)
    rows.sort(key=lambda r: (r["kind"], r["T"]))
    write_csv(
        out_dir / "success_crossover_by_T.csv", rows,
        ("run", "kind", "N", "M", "K", "V", "T", "b", "m", *FIT_COLUMNS[2:]),
    )
    write_csv(
        out_dir / "success_crossover_curves_by_T.csv", curves,
        (*CURVE_COLUMNS, "T", "scaled_d", "scaled_d_lineage"),
    )
    usable = [r for r in rows if r["relative_width_10_90"] is not None]
    trend = None
    if len(usable) >= 2:
        ordered = sorted(usable, key=lambda r: r["T"])
        trend = {
            "T": [r["T"] for r in ordered],
            "relative_width_10_90": [r["relative_width_10_90"] for r in ordered],
            "absolute_width_10_90": [r["width_10_90"] for r in ordered],
            "relative_width_falls_monotonically": all(
                b <= a for a, b in zip(
                    [r["relative_width_10_90"] for r in ordered],
                    [r["relative_width_10_90"] for r in ordered][1:],
                )
            ),
            "relative_width_first_to_last": (
                ordered[-1]["relative_width_10_90"] / ordered[0]["relative_width_10_90"]
            ),
        }
    summary = {
        "runs": [r["run"] for r in rows],
        "criterion": criterion,
        "fits": rows,
        "sharpening": trend,
        "figures": _render_cross(rows, curves, out_dir),
        "note": (
            "Sharpening is judged on width_10_90 / d_half, the width in the "
            "scaled variable the T -> infinity step function is stated in; the "
            "absolute width is reported alongside because d_half itself falls "
            "with T, which would shrink an absolute width on its own."
        ),
    }
    (out_dir / "success_crossover_by_T.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, help="cross-run mode: collapse several T")
    parser.add_argument("--criterion", choices=sorted(CRITERIA), default=DEFAULT_CRITERION)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    if args.out_dir is not None:
        payload = cross_run(
            args.run_dir, args.out_dir, criterion=args.criterion,
            draws=args.draws, seed=args.seed,
        )
    else:
        payload = [
            report(run_dir, draws=args.draws, seed=args.seed) for run_dir in args.run_dir
        ]
        payload = payload[0] if len(payload) == 1 else payload
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
