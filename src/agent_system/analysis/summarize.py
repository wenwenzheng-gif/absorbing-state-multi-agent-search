#!/usr/bin/env python3
"""Rebuild tables from a run directory, independently of the runner.

Nothing here reads the frozen registry or the reference master table: every
number comes from ``episodes.jsonl`` and ``events.jsonl`` of one run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Iterable, Sequence

from ..evaluation import CRITERIA
from ..storage import iter_jsonl

EPISODE_COLUMNS = (
    "episode_id",
    "status",
    "d",
    "task_seed",
    "seed",
    "policy",
    "environment",
    "N",
    "M",
    "K",
    "task_size",
    "rounds",
    "b",
    "m",
    "n0",
    "agents_reached_truth",
    "A_initial_incorrect",
    "B_used_total",
    "budget_utilization_mean",
    "tests_pos_total",
    "tests_neg_total",
    "recv_messages_total",
    "recv_informative_total",
    "recv_duplicate_total",
    "final_frontier_nonempty",
    "final_mean_frontier",
    "candidate_capacity_total",
    "candidate_invalidated_total",
    "merge_loss_total",
    "effective_branching_mean",
    "wall_seconds",
    *(f"agents_{name}" for name in CRITERIA),
    *(f"any_{name}" for name in CRITERIA),
    "mean_complete_candidates",
    "heldout_error_min",
    "first_symbolic_recovered_round",
    "first_predictive_success_round",
)


def _sum(rows: Sequence[dict], key: str) -> float:
    return sum(float(row.get(key) or 0) for row in rows)


def episode_row(payload: dict) -> dict:
    transitions = payload.get("transitions") or []
    evaluation = payload.get("evaluation") or {}
    first = evaluation.get("first_rounds") or {}
    row = {name: payload.get(name) for name in EPISODE_COLUMNS if name in payload}
    row["episode_id"] = payload.get("episode_id")
    row["status"] = payload.get("status")
    row["B_used_total"] = _sum(transitions, "B_used")
    row["budget_utilization_mean"] = (
        statistics.mean([float(t.get("budget_utilization") or 0) for t in transitions])
        if transitions
        else 0.0
    )
    row["tests_pos_total"] = _sum(transitions, "tests_pos")
    row["tests_neg_total"] = _sum(transitions, "tests_neg")
    row["recv_messages_total"] = _sum(transitions, "recv_messages")
    row["recv_informative_total"] = _sum(transitions, "recv_informative")
    row["recv_duplicate_total"] = _sum(transitions, "recv_duplicate")
    row["candidate_capacity_total"] = _sum(transitions, "total_candidate_capacity")
    row["candidate_invalidated_total"] = _sum(transitions, "candidate_invalidated_without_test")
    row["merge_loss_total"] = _sum(transitions, "merge_loss")
    row["effective_branching_mean"] = (
        statistics.mean([float(t.get("effective_branching") or 0) for t in transitions])
        if transitions
        else 0.0
    )
    if transitions:
        row["final_frontier_nonempty"] = transitions[-1].get("frontier_nonempty")
        row["final_mean_frontier"] = transitions[-1].get("mean_frontier_size")
    for name in CRITERIA:
        row[f"agents_{name}"] = evaluation.get(f"agents_{name}")
        row[f"any_{name}"] = evaluation.get(f"any_{name}")
    row["mean_complete_candidates"] = evaluation.get("mean_complete_candidates")
    row["heldout_error_min"] = evaluation.get("heldout_error_min")
    row["first_symbolic_recovered_round"] = first.get("symbolic_recovered")
    row["first_predictive_success_round"] = first.get("predictive_success")
    return {name: row.get(name) for name in EPISODE_COLUMNS}


def degree_rows(rows: Sequence[dict]) -> list[dict]:
    by_degree: dict[int, list[dict]] = {}
    for row in rows:
        by_degree.setdefault(int(row["d"]), []).append(row)
    summary = []
    for degree in sorted(by_degree):
        group = by_degree[degree]
        complete = [row for row in group if row["status"] == "ok"]
        entry: dict[str, object] = {
            "d": degree,
            "episodes_planned": len(group),
            "episodes_complete": len(complete),
            "completion_rate": len(complete) / len(group) if group else 0.0,
        }
        for status in sorted({row["status"] for row in group}):
            entry[f"n_{status}"] = sum(1 for row in group if row["status"] == status)
        for name in CRITERIA:
            achieved = [bool(row.get(f"any_{name}")) for row in complete]
            entry[f"P_{name}"] = sum(achieved) / len(achieved) if achieved else None
        for key in (
            "B_used_total",
            "budget_utilization_mean",
            "recv_informative_total",
            "recv_duplicate_total",
            "candidate_capacity_total",
            "candidate_invalidated_total",
            "effective_branching_mean",
            "mean_complete_candidates",
            "final_mean_frontier",
            "wall_seconds",
        ):
            values = [float(row[key]) for row in complete if row.get(key) is not None]
            entry[f"mean_{key}"] = statistics.mean(values) if values else None
        summary.append(entry)
    return summary


def write_csv(path: Path, rows: Sequence[dict], columns: Iterable[str] | None = None) -> Path:
    fields = list(columns) if columns is not None else sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def summarize(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    episodes = list(iter_jsonl(run_dir / "episodes.jsonl"))
    if not episodes:
        raise FileNotFoundError(f"no episodes recorded in {run_dir}")
    rows = [episode_row(payload) for payload in episodes]
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "episodes.csv", rows, EPISODE_COLUMNS)
    per_degree = degree_rows(rows)
    write_csv(analysis_dir / "by_degree.csv", per_degree)

    events = list(iter_jsonl(run_dir / "events.jsonl"))
    usage = [payload.get("llm_usage") for payload in episodes if payload.get("llm_usage")]
    report = {
        "run_dir": str(run_dir),
        "episodes": len(rows),
        "complete": sum(1 for row in rows if row["status"] == "ok"),
        "statuses": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
        "degrees": [entry["d"] for entry in per_degree],
        "experiment_events": len(events),
        "llm_usage_final": usage[-1] if usage else {},
        "outputs": [
            str(analysis_dir / "episodes.csv"),
            str(analysis_dir / "by_degree.csv"),
        ],
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.run_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
