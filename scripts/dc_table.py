#!/usr/bin/env python3
"""Build the 3x3 (policy x cell) d_c table for one config tag.

The main table's d_c is the paper's comparison: empirical **pre_own**,
summed over **all rounds t>=1**, against theory **delayed_dc**.

* ``pre_own`` is the frozen ``src/utils.py`` lineage-ratio convention: the
  A-to-A (incorrect-to-incorrect) count is read right after an agent
  receives its neighbours' branches, *before* that agent runs its own
  experiments and prunes -- so one round's ratio isolates ``b (1 -
  p_nbr)^d``, the quantity the paper's derivation
  actually tracks (self-pruning excluded). ``post_own`` (the same count
  read after the agent's own experiment/pruning step) additionally
  multiplies in the agent's own self-pruning factor ``(1 - p_own)``, which
  that derivation does not include, so it is not the convention comparable
  to theory -- it is kept only as a diagnostic of how much self-pruning
  contributes.
* the sum is over **all rounds t>=1** (``critical_degree``'s default,
  ``--first-round 1``). Round 1 has no neighbour messages yet under
  one-round-delayed communication, so under ``pre_own`` (own pruning
  excluded) round 1's own ratio is ``A_initial_incorrect * b /
  A_initial_incorrect = b`` regardless of degree -- empirically its log
  ratio measures 0.64-0.69 (~ln 2) at every degree -- so it contributes a
  fixed ``ln b`` to the cumulative sum with **no d-dependence**, exactly
  the ``(T-1) ln b`` numerator of ``references.delayed_dc``. Summing
  ``pre_own`` from t=1 therefore gives ``(T-1) ln b + d *
  sum_{t=2}^{T-1} ln(1 - p_t)``, whose zero is exactly ``delayed_dc``
  (which sums its *denominator* survival terms from t=2 but keeps the full
  ``(T-1) ln b`` numerator) -- so the two are the correct pair to compare,
  not an approximation of it. The previous version of this table summed
  the empirical side from t=2 (``--first-round 2``) while still comparing
  to ``delayed_dc``, which silently drops one ``ln b`` from the empirical
  side that the theory's numerator still has, and is superseded by this
  one.
* the empirical value is the interpolation estimate (``critical_degree``'s
  published two-point crossing estimator) with its task-cluster bootstrap
  95% CI, plus the measured degrees bracketing the crossing
  (``bracket=[d_minus, d_plus]``, straight from ``critical_degree``'s
  ``crossing.bracket``, or derived from the ``pre_own`` log R curve --
  adjacent measured degrees where log R changes sign -- if the JSON does
  not carry it).

The t>=2 pre_own value (what the previous version of this table used as
its main empirical column), the post_own all-rounds diagnostic, and
``exact_dc`` (theory summed from t=1) are kept as secondary CSV columns
(``emp_pre_own_t2``, ``emp_post_own_all_rounds``, ``theory_exact``) but
dropped from the markdown main table, to avoid mixing several different
definitions of d_c in one cell.

Reads, for each of the 9 cell x policy cells:

* the cell's environment parameters (T, b, m), used to compute theory d_c
  via ``src.agent_system.analysis.references`` (``exact_dc`` /
  ``delayed_dc``, ``q_model=1``), with a per-cell fixed ``|V|``: synthetic
  ``|V|=M*K=16``, tcas ``|V|=46`` (component-level evidence -- NOT
  ``M*K=120``), physics ``|V|=16`` (support library). The config is read
  from the run dir's ``manifest.json`` (its ``config`` field) when that run
  dir exists, since a pooled/merged tag (``scripts/merge_runs.py``) has no
  ``configs/agents/<tag>/`` folder of its own; otherwise it falls back to
  ``configs/agents/<tag>/<cell>_<policy>.json`` (an ordinary, single-seed
  tag always has one, and any one of the 3 policies' configs is enough --
  they share the same environment block);
* the empirical d_c from ``<run_dir>/analysis/critical_degree.json``
  (``--first-round 1``, the default all-rounds sum, written by every
  ``analyze.sh`` run regardless of which ``--convention`` is passed) for
  the main column, and from ``<run_dir>/analysis/critical_degree_t2.json``
  (``--first-round 2``) for the ``emp_pre_own_t2`` secondary column.
  ``critical_degree`` reports every convention (pre_own/post_own/literal)
  side by side in the same JSON's ``estimates`` list regardless of which
  one was passed as ``--convention`` on the command line, so this reads
  whichever convention row it needs out of either file without caring
  which one was primary;
* communication stats (reports sent, false-claim rate, silence rate) from
  ``<run_dir>/episodes.jsonl`` transitions, for C1/I1 cells: this repo's
  runner (``src/agent_system/runner.py``) does record these per-round
  counters (``outgoing_reports``, ``outgoing_false_reports``,
  ``outgoing_messages``, ``comm_calls``), so this is computed whenever an
  episode has been recorded, not left as TODO. Per cell: ``reports`` is the
  total ``outgoing_reports`` across all transitions; ``false`` is
  ``outgoing_false_reports / outgoing_reports``; ``silent`` is
  ``1 - outgoing_messages / comm_calls`` -- computed from ``comm_calls``
  (successful communication-phase calls) rather than from the
  ``silent_agents`` counter, so it is correct regardless of how
  ``silent_agents`` itself is defined. Only truly-absent counters (no run,
  or an episode format that predates free communication) are shown as TODO.

A missing config or run dir produces empty/TODO cells, not a crash: this is
expected to run correctly with zero completed runs.

Writes ``<run-root>/dc_table.csv`` and ``<run-root>/dc_table.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agent_system.analysis.references import delayed_dc, exact_dc  # noqa: E402

CELLS = ["synthetic", "tcas", "physics"]
CELL_LABELS = {"synthetic": "synthetic", "tcas": "tcas", "physics": "physics"}
POLICIES = ["C0", "C1", "I1"]

# The main table's empirical convention: pre_own, counted before the
# agent's own experiments/pruning (see the module docstring). post_own is
# kept only as a self-pruning diagnostic secondary column.
CONVENTION_MAIN = "pre_own"
CONVENTION_DIAGNOSTIC = "post_own"

EPISODE_ID_RE = re.compile(r"^d(\d+)_t(\d+)_r(\d+)$")

# Fixed |V| per cell, mirroring the reference configs
# (the reference configs): synthetic and physics use M*K/the
# support library size (both 16); tcas uses the component-level evidence
# count (46), NOT the full M*K=120 assignment count.
CELL_V: dict[str, int] = {"synthetic": 16, "tcas": 46, "physics": 16}

# Per-round free-communication counters read from episodes.jsonl
# transitions (src/agent_system/runner.py records all of these whenever
# communication_mode == "free"; they stay all-zero for communication_mode
# == "fixed" / C0).
COMM_FIELDS = (
    "outgoing_reports",
    "outgoing_false_reports",
    "outgoing_messages",
    "comm_calls",
    "recv_false_reports",
    "recv_conflicting_reports",
)


def load_config(config_dir: Path, run_root: Path, tag: str, cell: str) -> dict[str, Any] | None:
    """The cell's config: from any one policy's run-dir manifest.json if a
    run dir exists (pooled/merged tags have no configs/agents/<tag>/ folder
    of their own), else from configs/agents/<tag>/<cell>_<policy>.json.
    M/K/T/b/m are shared by all three policies of a cell, so any one
    existing source is enough for the theory numbers.
    """
    for policy in POLICIES:
        manifest_path = run_root / f"{tag}_{cell}_{policy}" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config")
            if config is not None:
                return config
    for policy in POLICIES:
        path = config_dir / f"{cell}_{policy}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def theory_for_cell(cell: str, config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {"note": "no config found"}
    env = config["environment"]
    M, K, T = env["families"], env["variants"], env["task_size"]
    b, m = config["branching"], config["experiments_per_agent"]
    V = CELL_V[cell]
    return {
        "M": M, "K": K, "T": T, "b": b, "m": m, "V": V,
        "exact": exact_dc(V, T, b, m),
        "delayed": delayed_dc(V, T, b, m),
    }


def load_empirical(
    run_dir: Path, convention: str, filename: str = "critical_degree.json"
) -> dict[str, Any] | None:
    path = run_dir / "analysis" / filename
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    estimates = report.get("estimates", [])
    row = next(
        (r for r in estimates if r.get("convention") == convention and r.get("estimator") == "interpolation"),
        None,
    )
    if row is None:
        return None
    return {
        "d_c": row.get("d_c"),
        "ci_low": row.get("ci_low"),
        "ci_high": row.get("ci_high"),
        "status": row.get("status"),
    }


def _bracket_from_curve_rows(curve_rows: list[dict[str, Any]] | None) -> list[int] | None:
    """Adjacent measured degrees where ``log_R`` changes sign, the same rule
    ``critical_degree.find_crossings`` uses: both degrees must be
    consecutive in the (sorted) curve *and* have status ``"ok"`` with a
    finite value -- a non-estimable degree in between still blocks the
    pair either side of it.
    """
    if not curve_rows:
        return None
    by_degree = {row["d"]: row for row in curve_rows}
    degrees = sorted(by_degree)
    for left, right in zip(degrees, degrees[1:]):
        left_row, right_row = by_degree[left], by_degree[right]
        if left_row.get("status") != "ok" or right_row.get("status") != "ok":
            continue
        log_left, log_right = left_row.get("log_R"), right_row.get("log_R")
        if log_left is None or log_right is None:
            continue
        if log_left > 0.0 > log_right:
            return [left, right]
    return None


def load_bracket(
    run_dir: Path, convention: str, filename: str = "critical_degree.json"
) -> list[int] | None:
    """The measured degrees ``[d_minus, d_plus]`` bracketing the crossing.

    Read straight from ``critical_degree``'s own ``crossing.bracket`` --
    either the top-level one (when ``convention`` is the report's primary
    convention) or the matching entry under ``alternate_conventions`` (the
    convention every ``analyze.sh`` call reports side by side regardless of
    which one was passed as ``--convention``). Falls back to deriving it
    from that same convention's log R curve (``_bracket_from_curve_rows``)
    only if the JSON has no crossing recorded for it at all -- which, since
    the derivation replicates the exact rule ``find_crossings`` itself
    uses, only happens when there genuinely is no adjacent sign change to
    find.
    """
    path = run_dir / "analysis" / filename
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("convention") == convention:
        bracket = (report.get("crossing") or {}).get("bracket")
        curve_rows = report.get("curve")
    else:
        alternate = (report.get("alternate_conventions") or {}).get(convention) or {}
        bracket = (alternate.get("crossing") or {}).get("bracket")
        curve_rows = alternate.get("curve")
    if bracket:
        return list(bracket)
    return _bracket_from_curve_rows(curve_rows)


def load_communication(run_dir: Path) -> dict[str, Any] | None:
    """Sum free-communication counters across every transition, if recorded."""
    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.exists():
        return None
    totals = {field: 0 for field in COMM_FIELDS}
    seen_any_field = False
    n_episodes = 0
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_episodes += 1
        payload = json.loads(line)
        for transition in payload.get("transitions") or []:
            for field in COMM_FIELDS:
                if field in transition:
                    seen_any_field = True
                    totals[field] += float(transition.get(field) or 0)
    if not seen_any_field:
        return None
    reports_sent = totals["outgoing_reports"]
    false_claims = totals["outgoing_false_reports"]
    outgoing_messages = totals["outgoing_messages"]
    comm_calls = totals["comm_calls"]
    return {
        "episodes": n_episodes,
        "outgoing_reports_total": reports_sent,
        "outgoing_false_reports_total": false_claims,
        # false = outgoing_false_reports / outgoing_reports.
        "false_claim_rate": (false_claims / reports_sent) if reports_sent else None,
        "outgoing_messages_total": outgoing_messages,
        "comm_calls_total": comm_calls,
        # silent = 1 - outgoing_messages / comm_calls, computed from
        # comm_calls (successful communication-phase calls) so it is
        # correct regardless of how the runner's own silent_agents counter
        # is defined.
        "silent_rate": (1 - outgoing_messages / comm_calls) if comm_calls else None,
        "recv_false_reports_total": totals["recv_false_reports"],
        "recv_conflicting_reports_total": totals["recv_conflicting_reports"],
    }


def load_failures(run_dir: Path) -> dict[str, Any]:
    """Non-``"ok"`` episodes recorded in ``episodes.jsonl``, if any.

    Reads straight from ``episodes.jsonl`` (rather than a merged cell's
    ``manifest.json`` ``failed_episodes``, which only exists for pooled
    tags), so this works uniformly for an ordinary single-seed run and for
    a ``merge_runs.py --allow-failed`` pooled one. Analysis
    (``critical_degree.pooled_log_ratio`` / ``pooled_step_log_ratio``)
    already filters to ``status == "ok"`` episodes, so these are simply
    excluded there -- this only surfaces them for the table.
    """
    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.exists():
        return {"n_failed": 0, "failed_degrees": [], "episodes": []}
    failed: list[dict[str, Any]] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        status = record.get("status")
        if status is None or status == "ok":
            continue
        degree = record.get("d")
        task_seed = record.get("task_seed")
        episode_id = record.get("episode_id") or record.get("episode")
        if (degree is None or task_seed is None) and episode_id:
            match = EPISODE_ID_RE.match(episode_id)
            if match:
                if degree is None:
                    degree = int(match.group(1))
                if task_seed is None:
                    task_seed = int(match.group(2))
        failed.append(
            {
                "episode_id": episode_id,
                "d": degree,
                "task_seed": task_seed,
                "status": status,
                "error": (record.get("error") or "")[:200],
            }
        )
    failed_degrees = sorted({entry["d"] for entry in failed if entry["d"] is not None})
    return {"n_failed": len(failed), "failed_degrees": failed_degrees, "episodes": failed}


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def build_table(tag: str, config_dir: Path, run_root: Path) -> dict[str, Any]:
    theory = {
        cell: theory_for_cell(cell, load_config(config_dir, run_root, tag, cell)) for cell in CELLS
    }
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    comm: dict[tuple[str, str], dict[str, Any] | None] = {}
    for cell in CELLS:
        for policy in POLICIES:
            run_dir = run_root / f"{tag}_{cell}_{policy}"
            cells[(cell, policy)] = {
                # Main: pre_own, summed over all rounds t>=1 (the default,
                # unsuffixed critical_degree.json), matches theory
                # delayed_dc -- see the module docstring for why the two
                # numerators line up.
                "empirical": load_empirical(run_dir, CONVENTION_MAIN, "critical_degree.json"),
                "bracket": load_bracket(run_dir, CONVENTION_MAIN, "critical_degree.json"),
                # Secondary: pre_own, sum from round 2 (delayed) -- the
                # previous version's main column, kept for comparison.
                "empirical_pre_own_t2": load_empirical(
                    run_dir, CONVENTION_MAIN, "critical_degree_t2.json"
                ),
                # Secondary: post_own, sum from round 1 (all-rounds) --
                # self-pruning diagnostic against the new main column,
                # not comparable to theory.
                "empirical_post_own_all_rounds": load_empirical(
                    run_dir, CONVENTION_DIAGNOSTIC, "critical_degree.json"
                ),
                "run_dir": str(run_dir),
                "has_run": run_dir.exists(),
                "failures": load_failures(run_dir),
            }
            comm[(cell, policy)] = load_communication(run_dir) if policy in ("C1", "I1") else None
    return {"tag": tag, "convention": CONVENTION_MAIN, "theory": theory, "cells": cells, "comm": comm}


def write_csv(path: Path, table: dict[str, Any]) -> None:
    fieldnames = ["policy"]
    for cell in CELLS:
        label = CELL_LABELS[cell]
        # Main: pre_own, all rounds t>=1, matches theory_delayed (see the
        # module docstring for why that pairing is exact, not approximate).
        fieldnames += [
            f"{label}_empirical_dc", f"{label}_ci_low", f"{label}_ci_high", f"{label}_status",
            f"{label}_bracket",
        ]
        fieldnames += [f"{label}_theory_delayed"]
        # Secondary: pre_own t>=2 (the previous version's main column),
        # post_own all-rounds (self-pruning diagnostic), and exact_dc, kept
        # for comparison only -- not in the markdown main table.
        fieldnames += [
            f"{label}_emp_pre_own_t2", f"{label}_emp_post_own_all_rounds", f"{label}_theory_exact",
        ]
        fieldnames += [f"{label}_reports_sent", f"{label}_false_claim_rate", f"{label}_silent_rate"]
        fieldnames += [f"{label}_n_failed", f"{label}_failed_degrees"]

    rows = []
    for policy in POLICIES:
        row = {"policy": policy}
        for cell in CELLS:
            label = CELL_LABELS[cell]
            emp = table["cells"][(cell, policy)]["empirical"]
            row[f"{label}_empirical_dc"] = fmt((emp or {}).get("d_c"))
            row[f"{label}_ci_low"] = fmt((emp or {}).get("ci_low"))
            row[f"{label}_ci_high"] = fmt((emp or {}).get("ci_high"))
            row[f"{label}_status"] = (emp or {}).get("status", "")
            bracket = table["cells"][(cell, policy)]["bracket"]
            row[f"{label}_bracket"] = ";".join(str(d) for d in bracket) if bracket else ""
            th = table["theory"][cell]
            row[f"{label}_theory_delayed"] = fmt(th.get("delayed"))
            emp_pre_t2 = table["cells"][(cell, policy)]["empirical_pre_own_t2"]
            row[f"{label}_emp_pre_own_t2"] = fmt((emp_pre_t2 or {}).get("d_c"))
            emp_post_all = table["cells"][(cell, policy)]["empirical_post_own_all_rounds"]
            row[f"{label}_emp_post_own_all_rounds"] = fmt((emp_post_all or {}).get("d_c"))
            row[f"{label}_theory_exact"] = fmt(th.get("exact"))
            c = table["comm"][(cell, policy)]
            not_applicable = "n/a" if policy == "C0" else "TODO (not recorded)"
            row[f"{label}_reports_sent"] = fmt((c or {}).get("outgoing_reports_total")) if c else not_applicable
            row[f"{label}_false_claim_rate"] = fmt((c or {}).get("false_claim_rate"), 3) if c else not_applicable
            row[f"{label}_silent_rate"] = fmt((c or {}).get("silent_rate"), 3) if c else not_applicable
            failures = table["cells"][(cell, policy)]["failures"]
            row[f"{label}_n_failed"] = failures["n_failed"]
            row[f"{label}_failed_degrees"] = (
                ";".join(str(d) for d in failures["failed_degrees"]) if failures["failed_degrees"] else ""
            )
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, table: dict[str, Any]) -> None:
    tag = table["tag"]
    convention = table["convention"]
    lines = [
        f"# d_c table -- {tag}",
        "",
        f"d_c: empirical = `critical_degree` interpolation estimator, convention "
        f"`{convention}` (pre_own: before the agent's own experiments/pruning -- the "
        "frozen `src/utils.py` lineage-ratio convention: the A-to-A count is read right "
        "after an agent receives its neighbours' branches, *before* that agent's own "
        "experiments and pruning -- so one round's ratio isolates `b (1-p_nbr)^d`, "
        "self-pruning excluded per the paper's own derivation), "
        "cumulative pooled log R summed over **all rounds t>=1** (`critical_degree`'s "
        "default, `--first-round 1`); round 1 contributes a fixed `ln b` with **no "
        "d-dependence** (no neighbour messages have arrived yet under one-round-delayed "
        "communication, and under `pre_own` the agent's own pruning is excluded, so round "
        "1's ratio is just `b`) -- that `ln b` is exactly the `(T-1) ln b` term in "
        "`delayed_dc`'s numerator, so summing `pre_own` from t=1 is the correct empirical "
        "counterpart to `delayed_dc`, not merely a superset of it; "
        "theory = `delayed_dc` (`src/agent_system/analysis/references.py`, denominator "
        "survival terms summed from t=2, since that factor genuinely depends on "
        "communication that hasn't arrived yet in round 1, while the `(T-1) ln b` "
        "numerator is kept in full -- both likewise excluding self-pruning), "
        "b=2, q_model=1, m and T from the config; |V| is fixed per cell: "
        f"synthetic={CELL_V['synthetic']} (M*K), tcas={CELL_V['tcas']} (component-level "
        f"evidence, NOT M*K=120), physics={CELL_V['physics']} (support library). Each "
        "emp cell also reports `bracket=[d_minus, d_plus]`, the two measured degrees "
        "the interpolation crossed between. "
        "Secondary columns in `dc_table.csv` only (not in this table): `emp_pre_own_t2` "
        "(pre_own summed from t=2 instead of t=1 -- the previous version of this table's "
        "main column, kept for comparison), `emp_post_own_all_rounds` (same t>=1 sum but "
        "convention `post_own`, which additionally folds in the agent's own self-pruning "
        "factor `(1-p_own)` -- a diagnostic of how much self-pruning contributes against "
        "the new main column, not a value comparable to theory), and `theory_exact` "
        "(`exact_dc`, sum from t=1). A cell with "
        "non-`\"ok\"` episodes (e.g. `policy_failed`, "
        "merged in via `merge_runs.py --allow-failed`) shows `failed: <n> (d=<degrees>)` "
        "below its emp/theory line; those episodes are excluded from the empirical d_c "
        "(`critical_degree.pooled_log_ratio` only pools `status == \"ok\"` episodes) -- a "
        "degree with fewer surviving (ok) episodes than the cell's other degrees is still "
        "pooled and estimated, just over fewer tasks, so its point is noisier but not "
        "dropped.",
        "",
        "| Policy | " + " | ".join(CELL_LABELS[c] for c in CELLS) + " |",
        "|---" * (1 + len(CELLS)) + "|",
    ]
    all_failed: list[tuple[str, str, dict[str, Any]]] = []
    for policy in POLICIES:
        cells_text = []
        for cell in CELLS:
            label = CELL_LABELS[cell]
            emp = table["cells"][(cell, policy)]["empirical"]
            th = table["theory"][cell]
            if emp is not None:
                if emp.get("d_c") is not None:
                    emp_text = f"emp={fmt(emp['d_c'])}"
                else:
                    # The point-estimate curve itself found no crossing
                    # (e.g. status "no_crossing_all_positive"): d_c is None
                    # even though some bootstrap draws' resampled curves did
                    # cross, so a CI can still exist below.
                    emp_text = f"emp=(no crossing: {emp.get('status')})"
                if emp.get("ci_low") is not None:
                    emp_text += f" [{fmt(emp['ci_low'])}, {fmt(emp['ci_high'])}]"
                bracket = table["cells"][(cell, policy)]["bracket"]
                if bracket:
                    emp_text += f" bracket=[{bracket[0]}, {bracket[1]}]"
            elif table["cells"][(cell, policy)]["has_run"]:
                emp_text = "emp=(not analyzed / no crossing)"
            else:
                emp_text = "emp=(no run)"
            theory_text = f"theory={fmt(th.get('delayed'))}"
            cell_text = f"{emp_text}<br>{theory_text}"
            failures = table["cells"][(cell, policy)]["failures"]
            if failures["n_failed"]:
                degrees_text = ",".join(str(d) for d in failures["failed_degrees"])
                cell_text += f"<br>failed: {failures['n_failed']} (d=[{degrees_text}])"
                for entry in failures["episodes"]:
                    all_failed.append((cell, policy, entry))
            cells_text.append(cell_text)
        lines.append(f"| {policy} | " + " | ".join(cells_text) + " |")

    lines += ["", "### Failed episodes", ""]
    if all_failed:
        for cell, policy, entry in all_failed:
            lines.append(
                f"- {cell}/{policy} `{entry['episode_id']}` (d={entry['d']}, "
                f"task_seed={entry['task_seed']}): status={entry['status']} -- {entry['error']}"
            )
    else:
        lines.append("(none)")

    lines += [
        "",
        "## Communication (C1 / I1 only)",
        "",
        "Computed from `episodes.jsonl` transitions, which this repo's runner "
        "(`src/agent_system/runner.py`) does record for free-communication "
        "cells (C1, I1, communication_mode free). `reports` = total "
        "`outgoing_reports`; `false` = `outgoing_false_reports / "
        "outgoing_reports`; `silent` = `1 - outgoing_messages / comm_calls`, "
        "computed from `comm_calls` (successful communication-phase calls) "
        "rather than from the runner's own `silent_agents` counter, so it "
        "stays correct regardless of how `silent_agents` is defined. Shown "
        "as TODO only when a cell has no run, or its episodes predate these "
        "counters.",
        "",
        "| Policy | " + " | ".join(CELL_LABELS[c] for c in CELLS) + " |",
        "|---" * (1 + len(CELLS)) + "|",
    ]
    for policy in ("C1", "I1"):
        cells_text = []
        for cell in CELLS:
            c = table["comm"][(cell, policy)]
            if c is None:
                cells_text.append("TODO (not recorded)")
            else:
                cells_text.append(
                    f"reports={int(c['outgoing_reports_total'])}, "
                    f"false={fmt(c['false_claim_rate'], 3)}, "
                    f"silent={fmt(c['silent_rate'], 3)}"
                )
        lines.append(f"| {policy} | " + " | ".join(cells_text) + " |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args(argv)

    config_dir = args.config_dir or (REPO_ROOT / "configs" / "agents" / args.tag)
    run_root = args.run_root or (REPO_ROOT / "runs" / "agents" / args.tag)

    table = build_table(args.tag, config_dir, run_root)
    csv_path = run_root / "dc_table.csv"
    md_path = run_root / "dc_table.md"
    write_csv(csv_path, table)
    write_markdown(md_path, table)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
