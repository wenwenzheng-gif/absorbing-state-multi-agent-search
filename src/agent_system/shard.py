#!/usr/bin/env python3
"""Run one config as several processes, then merge the shards.

``--split seed`` (default, unchanged behaviour): one process per task seed,
each covering every degree.  ``--split degree``: one process per (degree,
task_seed) pair, letting every degree of a single-seed cell run at once.

Episodes are independent across task seeds *and* across degrees: the physics
prediction cache is keyed by seed, the RNG streams are derived from the seed,
and ``driver.py::episode_plan`` derives the episode seed from
``(task_seed, repeat)`` only, independent of degree.  Sharding therefore
changes wall time only, not any recorded number.  The merged directory has
the same layout as a single run, with episodes ordered exactly as
``episode_plan`` would order them (degree-major, then seed), so
``analysis.summarize`` and ``analysis.critical_degree`` read it unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from ..utils import PACKAGE_ROOT
from .cli import read_config
from .schemas import RunConfig
from .storage import utc_now

MERGED_FILES = ("episodes.jsonl", "events.jsonl", "requests.jsonl", "responses.jsonl")


def _apply_concurrency(body: dict, payload: dict, concurrency: int | None) -> None:
    if concurrency is None:
        return
    body["policy"] = dict(payload["policy"])
    body["policy"]["llm"] = dict(payload["policy"]["llm"])
    body["policy"]["llm"]["concurrency"] = concurrency


def shard_configs(
    config: RunConfig,
    shard_root: Path,
    split: str = "seed",
    concurrency: int | None = None,
) -> list[tuple[int, Path, str]]:
    """Write one shard config per task seed (or per degree, task seed pair).

    ``split="seed"`` (default) writes one config per task seed, each keeping
    every degree -- byte-identical to the pre-``--split`` behaviour.
    ``split="degree"`` writes one config per (degree, task_seed) pair, so a
    single-seed, multi-degree cell can run every degree at once.  Either way
    the returned list is already in the order the merged files must be
    concatenated: seed order for ``"seed"``, degree-major then seed for
    ``"degree"`` (matching ``driver.py::episode_plan``).
    """
    if split not in ("seed", "degree"):
        raise ValueError(f"unknown split {split!r}; expected 'seed' or 'degree'")
    shard_root.mkdir(parents=True, exist_ok=True)
    payload = config.as_dict()
    shards = []
    # Any config list that runs parallel to task_seeds has to be sliced with
    # it; grn_targets names one target gene per seed, and a shard that kept
    # all of them would be rejected by RunConfig.validate.
    parallel = ("grn_targets",)
    if split == "seed":
        for position, seed in enumerate(config.task_seeds):
            run_id = f"{config.name}_s{seed}"
            body = dict(payload)
            body["task_seeds"] = [seed]
            body["environment"] = dict(payload["environment"])
            for name in parallel:
                values = body["environment"].get(name)
                if values:
                    body["environment"][name] = [list(values)[position]]
            body["output_root"] = str(shard_root)
            _apply_concurrency(body, payload, concurrency)
            path = shard_root / f"{run_id}.json"
            path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            shards.append((seed, path, run_id))
        return shards
    # split == "degree": one shard per (degree, task_seed) pair, written
    # degree-major so the shard order already matches episode_plan's order.
    for degree in config.degrees:
        for position, seed in enumerate(config.task_seeds):
            run_id = f"{config.name}_d{degree}_s{seed}"
            body = dict(payload)
            body["degrees"] = [degree]
            body["task_seeds"] = [seed]
            body["environment"] = dict(payload["environment"])
            for name in parallel:
                values = body["environment"].get(name)
                if values:
                    body["environment"][name] = [list(values)[position]]
            body["output_root"] = str(shard_root)
            _apply_concurrency(body, payload, concurrency)
            path = shard_root / f"{run_id}.json"
            path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            shards.append((seed, path, run_id))
    return shards


def _run_shard(args: tuple[str, str, str, str | None]) -> dict:
    config_path, shard_root, run_id, env_file = args
    command = [
        sys.executable,
        "-m",
        "src.agent_system.cli",
        "run",
        "--config",
        config_path,
        "--run-id",
        run_id,
        "--output-root",
        shard_root,
    ]
    if env_file:
        command += ["--env-file", env_file]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(PACKAGE_ROOT), capture_output=True, text=True
    )
    return {
        "run_id": run_id,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "stderr_tail": completed.stderr.strip()[-2000:],
    }


def merge(shard_root: Path, shards: list[tuple[int, Path, str]], merged: Path) -> dict:
    """Concatenate the shard records into one run directory."""
    merged.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in MERGED_FILES:
        sources = [shard_root / run_id / name for _seed, _path, run_id in shards]
        sources = [source for source in sources if source.exists()]
        if not sources:
            continue
        with (merged / name).open("w", encoding="utf-8") as sink:
            for source in sources:
                with source.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            sink.write(line)
                            counts[name] = counts.get(name, 0) + 1
    statuses: dict[str, int] = {}
    episodes = merged / "episodes.jsonl"
    if not episodes.exists():
        return {"records": counts, "statuses": {}, "note": "no shard produced an episode"}
    for line in episodes.read_text(encoding="utf-8").splitlines():
        status = json.loads(line).get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return {"records": counts, "statuses": statuses}


def aggregate_shard_summaries(
    shard_root: Path, shards: list[tuple[int, Path, str]]
) -> dict:
    """Sum each shard's ``llm_usage`` and collect its ``wall_seconds``."""
    llm_usage: dict[str, float] = {}
    per_shard_wall_seconds: dict[str, float] = {}
    for _seed, _path, run_id in shards:
        summary_path = shard_root / run_id / "run_summary.json"
        if not summary_path.exists():
            continue
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        wall_seconds = shard_summary.get("wall_seconds")
        if isinstance(wall_seconds, (int, float)):
            per_shard_wall_seconds[run_id] = wall_seconds
        for key, value in (shard_summary.get("llm_usage") or {}).items():
            if isinstance(value, (int, float)):
                llm_usage[key] = llm_usage.get(key, 0) + value
    return {"llm_usage": llm_usage, "per_shard_wall_seconds": per_shard_wall_seconds}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("seed", "degree"),
        default="seed",
        help=(
            "'seed' (default): one process per task seed, each covering "
            "every degree -- byte-identical to the original behaviour. "
            "'degree': one process per (degree, task_seed) pair, so a "
            "single-seed, multi-degree cell runs every degree at once."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="override policy.llm.concurrency in every shard config",
    )
    args = parser.parse_args(argv)

    config = read_config(args.config)
    run_id = args.run_id or config.name
    root = Path(args.output_root or config.output_root)
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    merged = root / run_id
    shard_root = merged / "shards"

    shards = shard_configs(
        config, shard_root, split=args.split, concurrency=args.concurrency
    )
    work = [
        (str(path), str(shard_root), shard_run_id, str(args.env_file) if args.env_file else None)
        for _seed, path, shard_run_id in shards
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(_run_shard, work))
    failed = [result for result in results if result["returncode"] != 0]
    merge_report = merge(shard_root, shards, merged)
    usage_report = aggregate_shard_summaries(shard_root, shards)
    summary = {
        "run_id": run_id,
        "finished_at": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "shards": len(shards),
        "failed_shards": failed,
        **merge_report,
        **usage_report,
    }
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "config": config.as_dict(),
        "sharded_by": "degree_task_seed" if args.split == "degree" else "task_seed",
        "shards": [shard_run_id for _seed, _path, shard_run_id in shards],
    }
    if args.concurrency is not None:
        manifest["concurrency_override"] = args.concurrency
    (merged / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (merged / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
