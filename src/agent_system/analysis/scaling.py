#!/usr/bin/env python3
"""Fit the critical degree across runs and compare it with the analytic value.

One run gives one critical degree; the scaling law is a statement about many.
This collects the crossing from a set of run directories, reads each cell's
parameters back from its manifest, and fits

    d_c = C * M^a_M * K^a_K * (ln b)^a_b * T^a_T

The paper's revised text collapses ``M`` and ``K`` into ``|V| = M K`` and
expects ``|V|^1 (ln b)^1 T^-1``; both forms are fitted, so the collapse can be
judged rather than assumed.

Three things are deliberately *not* pooled blindly:

* ``N``.  The critical degree cannot exceed ``N - 1``, and a cell measured at
  ``N = 40`` is not interchangeable with the same ``(M, K, T, b, m)`` at
  ``N = 80``.  ``N`` is part of the deduplication key and of every row, and
  the default report fits each ``N`` separately as well as pooling.
* Environments without a real ``M x K`` product.  In tcas the components are
  parameter-value assignments and the parameters have unequal numbers of
  values, so ``|V|`` is their sum and there is no ``K``; such cells enter the
  collapsed ``|V|`` fit and the measured-against-analytic scatter, and are
  held out of the ``M``-and-``K``-separate fit.
* Cells whose crossing is not a clean single sign change.  They are listed
  with their reason instead of contributing a number the scan did not
  actually resolve.

Cells whose depth approaches the number of families are excluded by default.
There the candidate pool runs out -- a depth-``t`` hypothesis in an ``M``
family space leaves ``M - t`` families to branch into -- and the exponent in
``T`` then reflects that rather than the pruning dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ...utils import delayed_boundary_dc, exact_analytic_dc
from .summarize import write_csv

CLEAN_CROSSING_STATUSES = ("single_crossing",)

CELL_COLUMNS = (
    "run", "kind", "N", "M", "K", "V", "T", "b", "m", "mk_structure",
    "crossing_status", "d_c", "ci_low", "ci_high",
    "d_c_fit", "d_c_fit_se", "fit_status", "convention",
    "exact", "delayed", "rel_err_exact", "rel_err_delayed", "free_families_at_depth",
)

REFERENCE_LABELS = {
    "exact": "analytic d_c (exact finite-T)",
    "delayed": "analytic d_c (delayed, sum from t=2)",
}


def _rel_err(measured: float | None, reference: float | None) -> float | None:
    """``measured / reference - 1``, or ``None`` when either side is unusable."""
    if measured is None or reference is None:
        return None
    if isinstance(reference, float) and math.isnan(reference):
        return None
    if reference == 0:
        return None
    return measured / reference - 1.0


def cell_of(run_dir: Path, *, include_unresolved: bool = False) -> dict | None:
    """Parameters and measured crossing for one run, or ``None`` if unresolved.

    With ``include_unresolved`` a run whose scan ran but produced no clean
    crossing is still returned, with ``d_c`` ``None`` and the scan's own
    ``crossing_status``, so the caller can list it rather than lose it.
    """
    run_dir = Path(run_dir)
    manifest = run_dir / "manifest.json"
    crossing = run_dir / "analysis" / "critical_degree.json"
    if not manifest.is_file() or not crossing.is_file():
        return None
    config = json.loads(manifest.read_text(encoding="utf-8")).get("config")
    if not config:
        return None
    report = json.loads(crossing.read_text(encoding="utf-8"))
    measured = report["crossing"]["d_c"]
    status = report["crossing"].get("status", "unknown")
    if measured is None and not include_unresolved:
        return None
    env = config["environment"]
    families, variants = int(env["families"]), int(env["variants"])
    mk_structure = True
    if env["kind"] == "tcas":
        # The components are parameter-value assignments, and the parameters
        # do not all have the same number of values, so |V| is their sum and
        # the padded encoding width is not it.  The per-family mean below is
        # good enough for |V| and meaningless as a K, hence mk_structure.
        from ..environments.tcas_dataset import assignment_count, space

        sizes = [size for _, size in space(env.get("tcas_space", "v1_460800"))]
        families, variants = len(sizes), assignment_count(env.get("tcas_space", "v1_460800")) / len(sizes)
        mk_structure = len(set(sizes)) == 1
    task_size, branching = int(env["task_size"]), int(config["branching"])
    pool = families * variants
    bootstrap = report.get("bootstrap") or {}
    interval = bootstrap.get("d_c_ci") or [None, None]
    fit = report.get("fit") or {}
    fit_bootstrap = bootstrap.get("fit") or {}
    exact = exact_analytic_dc(pool, 1, task_size, branching, config["experiments_per_agent"])
    delayed = delayed_boundary_dc(pool, 1, task_size, branching, config["experiments_per_agent"])
    return {
        "run": run_dir.name,
        "kind": env["kind"],
        "N": int(config["agents"]),
        "M": families,
        "K": variants,
        "V": pool,
        "T": task_size,
        "b": branching,
        "m": int(config["experiments_per_agent"]),
        "mk_structure": mk_structure,
        "crossing_status": status,
        "d_c": measured,
        "ci_low": interval[0],
        "ci_high": interval[1],
        "d_c_fit": fit.get("d_c_fit"),
        "d_c_fit_se": fit_bootstrap.get("d_c_fit_se"),
        "fit_status": fit.get("status"),
        "convention": report.get("convention"),
        "exact": exact,
        "delayed": delayed,
        "rel_err_exact": _rel_err(measured, exact),
        "rel_err_delayed": _rel_err(measured, delayed),
        "free_families_at_depth": families - (task_size - 1),
    }


def collect(run_dirs: Sequence[Path], *, include_unresolved: bool = False) -> list[dict]:
    cells = [cell_of(Path(d), include_unresolved=include_unresolved) for d in run_dirs]
    return [cell for cell in cells if cell is not None]


def _fit(cells: Sequence[dict], columns, names) -> dict | None:
    if len(cells) <= len(columns):
        return None
    design = np.array([[1.0] + [column(cell) for column in columns] for cell in cells])
    target = np.array([math.log(cell["d_c"]) for cell in cells])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    spread = ((target - target.mean()) ** 2).sum()
    return {
        "n_cells": len(cells),
        "agents": sorted({int(cell["N"]) for cell in cells}),
        "constant": math.exp(coefficients[0]),
        "exponents": {name: float(value) for name, value in zip(names, coefficients[1:])},
        "R2_log": float(1.0 - (residual ** 2).sum() / spread) if spread else None,
    }


def fits(cells: Sequence[dict]) -> dict:
    """Both fit forms.  Cells without a real ``M x K`` product are held out
    of the separate fit and counted, never silently regressed on a made-up
    ``K``."""
    collapsed = _fit(
        cells,
        (lambda c: math.log(c["V"]), lambda c: math.log(math.log(c["b"])), lambda c: math.log(c["T"])),
        ("|V|", "ln_b", "T"),
    )
    factored = [cell for cell in cells if cell.get("mk_structure", True)]
    separate = _fit(
        factored,
        (
            lambda c: math.log(c["M"]),
            lambda c: math.log(c["K"]),
            lambda c: math.log(math.log(c["b"])),
            lambda c: math.log(c["T"]),
        ),
        ("M", "K", "ln_b", "T"),
    )
    if separate is not None:
        separate["cells_without_MK_structure_excluded"] = len(cells) - len(factored)
    return {"collapsed_into_V": collapsed, "M_and_K_separate": separate}


def fits_by_agents(cells: Sequence[dict]) -> dict:
    """One fit per ``N``, because ``d_c`` is capped at ``N - 1``."""
    groups: dict[int, list[dict]] = {}
    for cell in cells:
        groups.setdefault(int(cell["N"]), []).append(cell)
    return {str(agents): fits(group) for agents, group in sorted(groups.items())}


def _render(cells: Sequence[dict], out_dir: Path, reference: str = "delayed") -> list[str]:
    """Two-panel exact/delayed scatter plus a single panel for ``reference``.

    Column names in ``cells`` are unchanged (``exact``, ``delayed``); this
    only picks which one sits on the x-axis of the single-panel figure. Both
    panels of the combined figure share the y-axis and the same limits so
    they can be compared directly.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError:
        return []
    if not cells:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = sorted({cell["kind"] for cell in cells})
    limit = max(
        max(c["exact"] for c in cells),
        max(c["delayed"] for c in cells),
        max(c["d_c"] for c in cells),
    ) * 1.08

    def _plot(axes, ref_key: str, *, legend: bool) -> None:
        for kind in kinds:
            group = [cell for cell in cells if cell["kind"] == kind]
            axes.errorbar(
                [cell[ref_key] for cell in group],
                [cell["d_c"] for cell in group],
                yerr=[
                    [
                        cell["d_c"] - cell["ci_low"] if cell["ci_low"] is not None else 0.0
                        for cell in group
                    ],
                    [
                        cell["ci_high"] - cell["d_c"] if cell["ci_high"] is not None else 0.0
                        for cell in group
                    ],
                ],
                fmt="o", markersize=4, linewidth=0.8, capsize=2, label=kind,
            )
        axes.plot([0, limit], [0, limit], color="grey", linewidth=0.8, linestyle="--")
        axes.set_xlim(0, limit)
        axes.set_ylim(0, limit)
        axes.set_xlabel(REFERENCE_LABELS[ref_key])
        if legend:
            axes.legend(fontsize=7, frameon=False)

    written = []

    figure, (left, right) = pyplot.subplots(1, 2, figsize=(8.6, 4.0), dpi=160, sharey=True)
    _plot(left, "exact", legend=False)
    _plot(right, "delayed", legend=True)
    left.set_ylabel("measured d_c")
    left.set_title("exact")
    right.set_title("delayed")
    figure.suptitle("measured against analytic")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        target = out_dir / f"measured_vs_analytic.{suffix}"
        figure.savefig(target)
        written.append(str(target))
    pyplot.close(figure)

    figure2, axes2 = pyplot.subplots(figsize=(4.4, 4.0), dpi=160)
    _plot(axes2, reference, legend=True)
    axes2.set_ylabel("measured d_c")
    axes2.set_title("measured against analytic")
    figure2.tight_layout()
    for suffix in ("png", "pdf"):
        target = out_dir / f"measured_vs_{reference}.{suffix}"
        figure2.savefig(target)
        written.append(str(target))
    pyplot.close(figure2)

    return written


def report(
    run_dirs: Sequence[Path],
    out_dir: Path,
    depletion_headroom: float = 2.0,
    *,
    agents: int | None = None,
    crossing_statuses: Sequence[str] = CLEAN_CROSSING_STATUSES,
    reference: str = "delayed",
    cells_out: Path | None = None,
) -> dict:
    """Collect, filter, fit and draw.

    ``depletion_headroom`` keeps cells with ``T <= M / headroom``; at 1.0
    nothing is excluded.  ``agents`` restricts every fit to one ``N``.
    Everything that is dropped -- unresolved crossing, duplicate cell, wrong
    ``N`` -- is written to ``dropped_cells.csv`` with its reason.
    """
    out_dir = Path(out_dir)
    found = collect(run_dirs, include_unresolved=True)
    if not found:
        raise FileNotFoundError(
            "no run directory had both a manifest and a resolved crossing; "
            "run analysis.critical_degree first"
        )
    allowed = set(crossing_statuses)
    dropped: list[dict] = []
    resolved: list[dict] = []
    for cell in sorted(found, key=lambda c: c["run"]):
        if cell["d_c"] is None:
            dropped.append({**cell, "drop_reason": f"no crossing ({cell['crossing_status']})"})
        elif cell["crossing_status"] not in allowed:
            dropped.append(
                {**cell, "drop_reason": f"crossing status {cell['crossing_status']} is not clean"}
            )
        else:
            resolved.append(cell)

    seen: dict[tuple, str] = {}
    unique = []
    for cell in resolved:
        key = (cell["kind"], cell["N"], cell["M"], cell["K"], cell["T"], cell["b"], cell["m"])
        if key in seen:
            dropped.append({**cell, "drop_reason": f"duplicate cell, kept {seen[key]}"})
            continue
        seen[key] = cell["run"]
        unique.append(cell)

    if agents is not None:
        kept = []
        for cell in unique:
            if int(cell["N"]) == int(agents):
                kept.append(cell)
            else:
                dropped.append({**cell, "drop_reason": f"N={cell['N']} filtered out by --agents"})
        unique = kept

    clean = [c for c in unique if c["T"] <= c["M"] / depletion_headroom]
    write_csv(out_dir / "cells.csv", unique, CELL_COLUMNS)
    if cells_out is not None:
        write_csv(Path(cells_out), unique, CELL_COLUMNS)
    write_csv(
        out_dir / "dropped_cells.csv", dropped, ("run", "kind", "N", "M", "K", "V", "T", "b", "m",
                                                 "crossing_status", "d_c", "drop_reason"),
    )
    counts: dict[str, int] = {}
    for cell in unique:
        counts[str(cell["N"])] = counts.get(str(cell["N"]), 0) + 1
    summary = {
        "cells": len(unique),
        "cells_after_depletion_filter": len(clean),
        "cells_by_agents": counts,
        "cells_without_MK_structure": sum(1 for c in unique if not c["mk_structure"]),
        "dropped_cells": len(dropped),
        "dropped": [
            {"run": c["run"], "N": c["N"], "reason": c["drop_reason"]} for c in dropped
        ],
        "depletion_headroom": depletion_headroom,
        "agents_filter": agents,
        "crossing_statuses_accepted": list(crossing_statuses),
        "fits_by_agents_all_cells": fits_by_agents(unique),
        "fits_by_agents_non_depleted": fits_by_agents(clean),
        "fits_all_cells": fits(unique),
        "fits_non_depleted": fits(clean),
        "figures": _render(clean or unique, out_dir, reference=reference),
        "reference": reference,
        "cells_out": str(cells_out) if cells_out is not None else None,
        "note": (
            "Theory expects |V|^1 (ln b)^1 T^-1. Cells are deduplicated by "
            "(kind, N, M, K, T, b, m); the depletion filter keeps T <= M/headroom; "
            "only crossings with an accepted status are fitted and everything "
            "dropped is listed in dropped_cells.csv. fits_all_cells and "
            "fits_non_depleted pool every N present (listed as 'agents' inside "
            "each fit); fits_by_agents_* fit one N at a time, which is the "
            "comparison to read since d_c is capped at N-1. Environments "
            "without a real M x K product (tcas) are in the collapsed |V| fit "
            "and the scatter, never in M_and_K_separate."
        ),
    }
    (out_dir / "scaling.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depletion-headroom", type=float, default=2.0)
    parser.add_argument(
        "--agents", type=int, default=None, help="fit only cells with this N"
    )
    parser.add_argument(
        "--crossing-status", nargs="+", default=list(CLEAN_CROSSING_STATUSES),
        help="crossing statuses a cell may have to enter the fits",
    )
    parser.add_argument(
        "--reference", choices=("exact", "delayed"), default="delayed",
        help="which analytic reference is the x-axis of measured_vs_<reference>.{png,pdf} "
             "(the two-panel measured_vs_analytic.{png,pdf} always shows both)",
    )
    parser.add_argument(
        "--cells-out", type=Path, default=None,
        help="also write a copy of cells.csv to this path",
    )
    args = parser.parse_args(argv)
    print(json.dumps(
        report(
            args.run_dir,
            args.out_dir,
            args.depletion_headroom,
            agents=args.agents,
            crossing_statuses=args.crossing_status,
            reference=args.reference,
            cells_out=args.cells_out,
        ),
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
