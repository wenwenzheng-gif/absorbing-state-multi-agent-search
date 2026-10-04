"""Run the frozen d scans and refine adjacent crossing endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .experiment_config import CELLS, CELL_BY_UID, Cell
from .scaling_model import run_episode
from .utils import adjacent_crossing, raw_index, read_rows, sha256_file, write_rows


def cell_directory(raw_root: Path, cell: Cell) -> Path:
    return raw_root / cell.raw_slug


def raw_path(raw_root: Path, cell: Cell, degree: int, episodes: int) -> Path:
    return cell_directory(raw_root, cell) / f"d{degree}_E{episodes}.jsonl.gz"


def validate_rows(rows: list[dict], cell: Cell, degree: int, episodes: int) -> None:
    if len(rows) != episodes:
        raise RuntimeError(f"{cell.cell_uid} d={degree}: expected {episodes} rows")
    expected_seeds = list(range(cell.seed_start, cell.seed_start + episodes))
    if [int(row["seed"]) for row in rows] != expected_seeds:
        raise RuntimeError(f"{cell.cell_uid} d={degree}: noncanonical seed sequence")
    for row in rows:
        if row["status"] != "ok" or row["mode"] != "width_only":
            raise RuntimeError(f"{cell.cell_uid} d={degree}: invalid trajectory status")
        observed = (
            int(row["N"]), int(row["M"]), int(row["K"]),
            int(row["task_size"]), int(row["b"]), int(row["m"]), int(row["n0"]),
        )
        expected = (cell.N, cell.M, cell.K, cell.T, cell.b, cell.m, cell.n0)
        if observed != expected or int(row["d"]) != degree:
            raise RuntimeError(f"{cell.cell_uid} d={degree}: parameter mismatch")
        if int(row["rounds"]) != cell.rounds:
            raise RuntimeError(f"{cell.cell_uid} d={degree}: round-count mismatch")
        if int(row["A_initial_incorrect"]) != cell.N * (cell.n0 - 1):
            raise RuntimeError(f"{cell.cell_uid} d={degree}: Mode-B initialization mismatch")
        if any(row["audit"].values()):
            raise RuntimeError(f"{cell.cell_uid} d={degree}: simulator audit failure")


def run_task(task: tuple[str, int, int, str]) -> dict:
    cell_uid, degree, episodes, raw_root_text = task
    cell = CELL_BY_UID[cell_uid]
    raw_root = Path(raw_root_text)
    destination = raw_path(raw_root, cell, degree, episodes)
    started = time.monotonic()
    if destination.exists():
        rows = read_rows(destination)
        validate_rows(rows, cell, degree, episodes)
        status = "cached"
    else:
        prior_rows: list[dict] = []
        prior_episodes = 0
        for candidate_episodes, candidate_path in raw_index(cell_directory(raw_root, cell)).values():
            if candidate_path.name.startswith(f"d{degree}_") and candidate_episodes < episodes:
                if candidate_episodes > prior_episodes:
                    prior_episodes = candidate_episodes
                    prior_rows = read_rows(candidate_path)
        if prior_rows:
            validate_rows(prior_rows, cell, degree, prior_episodes)
        rows = list(prior_rows)
        for seed in range(cell.seed_start + prior_episodes, cell.seed_start + episodes):
            result = run_episode(cell.parameters, degree, seed, "width_only")
            result["cell_uid"] = cell.cell_uid
            result["source_cell_id"] = cell.source_cell_id
            rows.append(result)
        validate_rows(rows, cell, degree, episodes)
        write_rows(destination, rows)
        status = f"extended_E{prior_episodes}" if prior_episodes else "ran"
    return {
        "cell_uid": cell.cell_uid,
        "degree": degree,
        "episodes": episodes,
        "status": status,
        "elapsed_s": time.monotonic() - started,
        "path": str(destination),
        "sha256": sha256_file(destination),
    }


def execute(
    tasks: list[tuple[str, int, int]], raw_root: Path, workers: int, stage: str
) -> list[dict]:
    unique = sorted(set(tasks))
    if not unique:
        return []
    raw_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results: list[dict] = []
    expanded = [(uid, degree, episodes, str(raw_root)) for uid, degree, episodes in unique]
    worker_count = max(1, min(workers, len(expanded)))
    print(f"{stage}: {len(expanded)} degree files; workers={worker_count}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(run_task, task) for task in expanded]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"{stage} {index}/{len(expanded)} {result['cell_uid']} "
                f"d={result['degree']} E={result['episodes']} {result['status']} "
                f"{result['elapsed_s']:.1f}s",
                flush=True,
            )
    results.sort(key=lambda row: (row["cell_uid"], row["degree"], row["episodes"]))
    manifest_dir = raw_root.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "results": results,
        "wall_s": time.monotonic() - started,
    }
    (manifest_dir / f"{stage}_{stamp}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results


def selected_cells(cell_uids: str = "") -> tuple[Cell, ...]:
    if not cell_uids:
        return CELLS
    requested = [value.strip() for value in cell_uids.split(",") if value.strip()]
    return tuple(CELL_BY_UID[value] for value in requested)


def verify_crossings(cells: tuple[Cell, ...], raw_root: Path, endpoint: bool) -> None:
    for cell in cells:
        minimum = cell.endpoint_episodes if endpoint else cell.pilot_episodes
        bracket = adjacent_crossing(cell_directory(raw_root, cell), minimum)
        expected = (cell.reference_d_minus, cell.reference_d_plus)
        if bracket != expected:
            raise RuntimeError(
                f"{cell.cell_uid}: crossing {bracket} at E{minimum}, expected {expected}"
            )


def run_full_scan(raw_root: Path, workers: int, cells: tuple[Cell, ...] = CELLS) -> None:
    pilot_tasks = [
        (cell.cell_uid, degree, cell.pilot_episodes)
        for cell in cells
        for degree in cell.pilot_degrees
    ]
    execute(pilot_tasks, raw_root, workers, "pilot_E128")
    verify_crossings(cells, raw_root, endpoint=False)
    endpoint_tasks = [
        (cell.cell_uid, degree, cell.endpoint_episodes)
        for cell in cells
        for degree in (cell.reference_d_minus, cell.reference_d_plus)
        if cell.endpoint_episodes > cell.pilot_episodes
    ]
    execute(endpoint_tasks, raw_root, workers, "adjacent_endpoints")
    verify_crossings(cells, raw_root, endpoint=True)


def print_status(raw_root: Path, cells: tuple[Cell, ...]) -> None:
    for cell in cells:
        pilot = adjacent_crossing(cell_directory(raw_root, cell), cell.pilot_episodes)
        endpoint = adjacent_crossing(cell_directory(raw_root, cell), cell.endpoint_episodes)
        print(
            f"{cell.cell_uid:34s} pilot={pilot!s:10s} "
            f"endpoint=E{cell.endpoint_episodes}:{endpoint}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "refine", "full", "status"), nargs="?", default="full")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--cells", default="", help="comma-separated cell_uid values")
    args = parser.parse_args()
    cells = selected_cells(args.cells)
    if args.stage == "pilot":
        tasks = [(cell.cell_uid, d, cell.pilot_episodes) for cell in cells for d in cell.pilot_degrees]
        execute(tasks, args.raw_root, args.workers, "pilot_E128")
        verify_crossings(cells, args.raw_root, endpoint=False)
    elif args.stage == "refine":
        tasks = [
            (cell.cell_uid, d, cell.endpoint_episodes)
            for cell in cells
            for d in (cell.reference_d_minus, cell.reference_d_plus)
            if cell.endpoint_episodes > cell.pilot_episodes
        ]
        execute(tasks, args.raw_root, args.workers, "adjacent_endpoints")
        verify_crossings(cells, args.raw_root, endpoint=True)
    elif args.stage == "full":
        run_full_scan(args.raw_root, args.workers, cells)
    print_status(args.raw_root, cells)


if __name__ == "__main__":
    main()

