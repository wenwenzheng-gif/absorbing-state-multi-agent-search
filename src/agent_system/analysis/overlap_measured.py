#!/usr/bin/env python3
"""Measure ``p_nbr,t`` and ``q_overlap,t`` per round instead of assuming them.

The closed form for ``d_c`` rests on one modelled quantity,
``q_overlap,t = t / |V|``: the chance that a neighbour's shared label happens
to be relevant to a given candidate after ``t`` rounds of depth.  Every round
of every run already contains a direct measurement of it.  The theory says

    ln r_t(d) = ln b_t + d ln(1 - m q_overlap,t q_model),

so a straight line through the low-degree part of the measured ``ln r_t(d)``
gives ``slope_t``, hence

    p_nbr,t = 1 - exp(slope_t),     q_meas,t = p_nbr,t / m,

and the amplification ``amp_t = q_meas,t / (t / |V|)`` says by how much the
model under- or over-states the real per-neighbour pruning.

The window is the low-degree end (``d <= 8`` by default) because that is where
the response is linear: past a few neighbours the same candidate is pruned by
several of them at once and the curve bends.  Fitting the whole range would
measure the bend, not the slope the theory names.

Four reference values for ``d_c`` are reported per cell:

``exact``        ``src/utils.exact_analytic_dc``, the paper's formula;
``delayed``      the same with the ``t = 1`` term dropped, because messages
                 arrive a round late and round 1 is never communicated;
``delayed/amp``  ``delayed`` with ``q_t`` replaced by ``amp * t / |V|`` using
                 the mean measured amplification over communicated rounds;
``reconstruction``  no model at all: ``sum_t ln b_t / (-sum_t slope_t)``, the
                 degree at which the measured per-round lines sum to zero.

The claim under test (from the audit of the 3% agreement on toy and tcas) is
that ``exact`` is close there only because two errors cancel: the
uncommunicated first round pushes ``d_c`` up by roughly a fifth, and pruning a
whole family's siblings off one positive label pushes it back down by about as
much.  A flat library (``K = 1``) has no siblings, so nothing cancels.  The
sibling effect predicts ``amp = (1 - f) + f (K - 1)`` with ``f`` the positive
test rate, which the cross-run mode checks against ``K in {1, 2, 4, 8}``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from ...utils import delayed_boundary_dc, exact_analytic_dc
from .critical_degree import pooled_step_log_ratio
from .run_cell import (
    cell_parameters,
    episodes_by_degree,
    linear_fit,
    load_episodes,
    matplotlib_pyplot,
    measured_crossing,
    pooled_ratio,
    save_figure,
)
from .summarize import write_csv

DEFAULT_WINDOW = 8
REFERENCES = ("exact", "delayed", "delayed_amp", "reconstruction")

ROUND_COLUMNS = (
    "run", "round", "degrees_in_window", "ln_b_t", "slope_t", "slope_stderr", "r2",
    "p_nbr_t", "q_measured_t", "q_theory_t", "amp_t", "f_positive_t",
    "amp_predicted_from_K", "communicated",
)
CELL_COLUMNS = (
    "run", "kind", "N", "M", "K", "V", "T", "b", "m", "d_c_measured",
    "ci_low", "ci_high", "exact", "delayed", "delayed_amp", "reconstruction",
    "amp_mean", "f_mean", "amp_predicted_from_K",
    *(f"rel_err_{name}" for name in REFERENCES),
)


def per_round_response(
    grouped: dict[int, list[dict]], rounds: int, *, window: int, convention: str = "pre_own"
) -> list[dict]:
    """``ln r_t(d)`` per round and degree, and the low-degree line through it."""
    rows = []
    for index in range(rounds):
        points = []
        for degree, group in sorted(grouped.items()):
            if degree > window:
                continue
            entry = pooled_step_log_ratio(group, convention, index)
            if entry["status"] == "ok" and entry["value"] is not None:
                points.append((float(degree), float(entry["value"])))
        fit = linear_fit(points)
        rows.append({"round": index + 1, "fit": fit, "degrees_in_window": len(points)})
    return rows


def measure(run_dir: Path, *, window: int = DEFAULT_WINDOW) -> dict:
    run_dir = Path(run_dir)
    cell = cell_parameters(run_dir)
    if cell is None:
        raise FileNotFoundError(f"no manifest to read cell parameters from in {run_dir}")
    episodes = load_episodes(run_dir)
    grouped = episodes_by_degree(episodes)
    complete = [e for group in grouped.values() for e in group]
    rounds = max((len(e.get("transitions") or []) for e in complete), default=0)
    if rounds == 0:
        raise ValueError(f"no transitions recorded in {run_dir}")

    pool, count, agents = float(cell["V"]), cell["m"], cell["N"]
    rows: list[dict] = []
    for entry in per_round_response(grouped, rounds, window=window):
        index = entry["round"] - 1
        fit = entry["fit"]
        positive = pooled_ratio(complete, index, "tests_pos", "tests")
        theory = (index + 1) / pool
        slope = fit["slope"] if fit else None
        q_measured = (1.0 - math.exp(slope)) / count if slope is not None else None
        predicted = (
            (1.0 - positive) + positive * (cell["K"] - 1.0)
            if positive is not None
            else None
        )
        rows.append(
            {
                "run": run_dir.name,
                "round": entry["round"],
                "degrees_in_window": entry["degrees_in_window"],
                "ln_b_t": fit["intercept"] if fit else None,
                "slope_t": slope,
                "slope_stderr": fit["stderr"] if fit else None,
                "r2": fit["r2"] if fit else None,
                "p_nbr_t": (1.0 - math.exp(slope)) if slope is not None else None,
                "q_measured_t": q_measured,
                "q_theory_t": theory,
                "amp_t": (q_measured / theory) if q_measured is not None else None,
                "f_positive_t": positive,
                "amp_predicted_from_K": predicted,
                # Round 1 receives nothing: messages are delivered a round late,
                # so its slope is structurally zero and it carries no q.
                "communicated": entry["round"] >= 2,
            }
        )

    communicated = [r for r in rows if r["communicated"] and r["amp_t"] is not None]
    amp_mean = (
        sum(r["amp_t"] for r in communicated) / len(communicated) if communicated else None
    )
    f_values = [r["f_positive_t"] for r in rows if r["f_positive_t"] is not None]
    f_mean = sum(f_values) / len(f_values) if f_values else None
    predicted_amp = (
        (1.0 - f_mean) + f_mean * (cell["K"] - 1.0) if f_mean is not None else None
    )

    exact = exact_analytic_dc(pool, 1, cell["T"], cell["b"], count)
    delayed = delayed_boundary_dc(pool, 1, cell["T"], cell["b"], count)
    delayed_amp = _delayed_with_amplification(cell, amp_mean)
    reconstruction = _reconstruction(rows)

    lineage = measured_crossing(run_dir)
    measured = lineage["d_c"]
    references = {
        "exact": exact,
        "delayed": delayed,
        "delayed_amp": delayed_amp,
        "reconstruction": reconstruction,
    }
    errors = {
        f"rel_err_{name}": (
            (measured - value) / value
            if measured is not None and value is not None and math.isfinite(value) and value
            else None
        )
        for name, value in references.items()
    }
    cell_row = {
        "run": run_dir.name,
        **{key: cell[key] for key in ("kind", "N", "M", "K", "V", "T", "b", "m")},
        "d_c_measured": measured,
        "ci_low": lineage["ci"][0],
        "ci_high": lineage["ci"][1],
        **references,
        "amp_mean": amp_mean,
        "f_mean": f_mean,
        "amp_predicted_from_K": predicted_amp,
        **errors,
    }
    return {"cell": cell, "rounds": rows, "summary": cell_row, "window": window,
            "crossing_status": lineage["status"], "agents": agents}


def _delayed_with_amplification(cell: dict, amp: float | None) -> float | None:
    """``delayed`` with ``q_t -> amp t / |V|``; ``None`` if that leaves no term."""
    if amp is None or not math.isfinite(amp) or amp <= 0.0:
        return None
    terms = []
    for t in range(2, cell["T"]):
        value = 1.0 - cell["m"] * amp * t / cell["V"]
        if not 0.0 < value < 1.0:
            return None
        terms.append(math.log(value))
    if not terms:
        return None
    return -(cell["T"] - 1) * math.log(cell["b"]) / sum(terms)


def _reconstruction(rows: Sequence[dict]) -> float | None:
    """``sum_t ln b_t / (-sum_t slope_t)``: the measured lines' own zero.

    This uses no ``q`` model and no analytic branching factor, but it does
    extrapolate the low-degree slopes out to ``d_c``, which is usually well
    outside the fitting window, so it is a reference, not a better estimator.
    """
    intercepts = [r["ln_b_t"] for r in rows if r["ln_b_t"] is not None]
    slopes = [r["slope_t"] for r in rows if r["slope_t"] is not None]
    if not intercepts or not slopes or sum(slopes) >= 0.0:
        return None
    return sum(intercepts) / (-sum(slopes))


def _render_run(rows: Sequence[dict], figures_dir: Path, title: str) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not rows:
        return []
    figure, axes = pyplot.subplots(1, 2, figsize=(7.2, 3.1), dpi=160)
    left, right = axes
    xs = [r["round"] for r in rows]
    left.plot(xs, [r["q_measured_t"] for r in rows], marker="o", markersize=3.5,
              linewidth=1.1, label="measured q_overlap,t")
    left.plot(xs, [r["q_theory_t"] for r in rows], marker="s", markersize=3,
              linewidth=1.0, linestyle="--", label="theory t/|V|")
    left.set_xlabel("round t")
    left.set_ylabel("q_overlap,t")
    left.set_title("per-round overlap", fontsize=9)
    left.legend(fontsize=6.5, frameon=False)
    right.plot(xs, [r["amp_t"] for r in rows], marker="o", markersize=3.5,
               linewidth=1.1, label="amp_t = q_meas / (t/|V|)")
    right.plot(xs, [r["amp_predicted_from_K"] for r in rows], marker="^", markersize=3,
               linewidth=1.0, linestyle="--", label="(1-f) + f(K-1)")
    right.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    right.set_xlabel("round t")
    right.set_ylabel("amplification")
    right.set_title("measured against the sibling model", fontsize=9)
    right.legend(fontsize=6.5, frameon=False)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    written = save_figure(figure, figures_dir, "overlap_measured")
    pyplot.close(figure)
    return written


def report(run_dir: Path, *, window: int = DEFAULT_WINDOW) -> dict:
    run_dir = Path(run_dir)
    measurement = measure(run_dir, window=window)
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "overlap_measured_by_round.csv", measurement["rounds"], ROUND_COLUMNS)
    write_csv(analysis_dir / "overlap_measured_cell.csv", [measurement["summary"]], CELL_COLUMNS)
    figures = _render_run(measurement["rounds"], run_dir / "figures", run_dir.name)
    summary = {
        "run_dir": str(run_dir),
        "window": f"d <= {window}",
        "crossing_status": measurement["crossing_status"],
        "cell": measurement["cell"],
        "per_round": measurement["rounds"],
        "references": measurement["summary"],
        "figures": figures,
        "note": (
            "slope_t is fitted over the low-degree window only, where ln r_t(d) "
            "is linear. amp_t is undefined for round 1, which receives no "
            "messages; amp_mean averages the communicated rounds."
        ),
    }
    (analysis_dir / "overlap_measured.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def _render_cross(cells: Sequence[dict], out_dir: Path) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not cells:
        return []
    written: list[str] = []
    figure, axes = pyplot.subplots(1, 2, figsize=(8.0, 3.6), dpi=160)
    scatter, sibling = axes
    markers = {"exact": "o", "delayed": "s", "delayed_amp": "^", "reconstruction": "d"}
    limit = 0.0
    for name, marker in markers.items():
        xs = [c[name] for c in cells if c[name] is not None and c["d_c_measured"]]
        ys = [c["d_c_measured"] for c in cells if c[name] is not None and c["d_c_measured"]]
        if not xs:
            continue
        limit = max(limit, max(xs), max(ys))
        scatter.plot(xs, ys, marker, markersize=4, linestyle="none", alpha=0.75, label=name)
    limit *= 1.08
    scatter.plot([0, limit], [0, limit], color="grey", linewidth=0.8, linestyle="--")
    scatter.set_xlim(0, limit)
    scatter.set_ylim(0, limit)
    scatter.set_xlabel("reference d_c")
    scatter.set_ylabel("measured d_c")
    scatter.set_title("four references against the measurement", fontsize=9)
    scatter.legend(fontsize=6.5, frameon=False)

    by_k: dict[float, list[dict]] = {}
    for cell in cells:
        if cell["amp_mean"] is not None:
            by_k.setdefault(round(float(cell["K"]), 2), []).append(cell)
    ks = sorted(by_k)
    sibling.plot(ks, [
        sum(c["amp_mean"] for c in by_k[k]) / len(by_k[k]) for k in ks
    ], marker="o", markersize=4, linewidth=1.1, label="measured amp")
    sibling.plot(ks, [
        sum(c["amp_predicted_from_K"] for c in by_k[k]) / len(by_k[k]) for k in ks
    ], marker="^", markersize=4, linewidth=1.0, linestyle="--", label="(1-f) + f(K-1)")
    sibling.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    sibling.set_xlabel("variants per family K")
    sibling.set_ylabel("amplification")
    sibling.set_title("does amp grow with K?", fontsize=9)
    sibling.legend(fontsize=6.5, frameon=False)
    figure.tight_layout()
    written += save_figure(figure, out_dir, "overlap_measured_cross_run")
    pyplot.close(figure)
    return written


def cross_run(run_dirs: Sequence[Path], out_dir: Path, *, window: int = DEFAULT_WINDOW) -> dict:
    out_dir = Path(out_dir)
    cells, rounds, skipped = [], [], []
    for run_dir in run_dirs:
        try:
            measurement = measure(Path(run_dir), window=window)
        except (FileNotFoundError, ValueError, KeyError) as error:
            skipped.append({"run": Path(run_dir).name, "reason": str(error)})
            continue
        rounds.extend(measurement["rounds"])
        if measurement["summary"]["d_c_measured"] is None:
            skipped.append({"run": Path(run_dir).name, "reason": measurement["crossing_status"]})
            continue
        cells.append(measurement["summary"])
    cells.sort(key=lambda c: (c["kind"], c["V"], c["T"], c["b"]))
    write_csv(out_dir / "overlap_measured_cells.csv", cells, CELL_COLUMNS)
    write_csv(out_dir / "overlap_measured_rounds.csv", rounds, ROUND_COLUMNS)

    accuracy = {}
    for name in REFERENCES:
        errors = [
            abs(c[f"rel_err_{name}"]) for c in cells if c.get(f"rel_err_{name}") is not None
        ]
        signed = [c[f"rel_err_{name}"] for c in cells if c.get(f"rel_err_{name}") is not None]
        accuracy[name] = {
            "cells": len(errors),
            "mean_abs_rel_error": sum(errors) / len(errors) if errors else None,
            "median_abs_rel_error": sorted(errors)[len(errors) // 2] if errors else None,
            "max_abs_rel_error": max(errors) if errors else None,
            "mean_signed_rel_error": sum(signed) / len(signed) if signed else None,
            "cells_within_8_percent": sum(1 for e in errors if e <= 0.08),
        }
    by_k: dict[float, list[dict]] = {}
    for cell in cells:
        if cell["amp_mean"] is not None:
            by_k.setdefault(round(float(cell["K"]), 2), []).append(cell)
    sibling = [
        {
            "K": k,
            "cells": len(group),
            "amp_measured": sum(c["amp_mean"] for c in group) / len(group),
            "amp_predicted": sum(c["amp_predicted_from_K"] for c in group) / len(group),
            "f_mean": sum(c["f_mean"] for c in group) / len(group),
        }
        for k, group in sorted(by_k.items())
    ]
    for entry in sibling:
        entry["ratio_measured_over_predicted"] = (
            entry["amp_measured"] / entry["amp_predicted"] if entry["amp_predicted"] else None
        )
    write_csv(
        out_dir / "overlap_amplification_by_K.csv", sibling,
        ("K", "cells", "f_mean", "amp_measured", "amp_predicted", "ratio_measured_over_predicted"),
    )
    summary = {
        "cells": len(cells),
        "skipped": skipped,
        "window": f"d <= {window}",
        "reference_accuracy": accuracy,
        "amplification_by_K": sibling,
        "amp_increases_with_K": all(
            b["amp_measured"] >= a["amp_measured"] for a, b in zip(sibling, sibling[1:])
        ) if len(sibling) >= 2 else None,
        "figures": _render_cross(cells, out_dir),
        "note": (
            "rel_err = (measured - reference) / reference. The cancellation "
            "claim predicts exact to be good where K > 1 and high where K = 1, "
            "and delayed/amp to be good everywhere."
        ),
    }
    (out_dir / "overlap_measured_cross_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, help="cross-run mode: one table over all cells")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="largest degree kept in the linear slope fit")
    args = parser.parse_args(argv)
    if args.out_dir is not None:
        payload = cross_run(args.run_dir, args.out_dir, window=args.window)
    else:
        payload = [report(run_dir, window=args.window) for run_dir in args.run_dir]
        payload = payload[0] if len(payload) == 1 else payload
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
