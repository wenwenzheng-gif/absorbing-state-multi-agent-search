#!/usr/bin/env python3
"""Cell parameters and cluster resampling shared by the newer analysis modules.

``scaling.py`` already recovers a cell's parameters from its manifest, but it
does so only for runs that resolved a crossing -- it returns ``None`` otherwise
and folds the measured ``d_c`` into the same record.  Several of the modules
added alongside this one need the parameters of *every* run, crossing or not,
so the manifest reader is repeated here in a form that does not depend on an
analysis product already existing.  Nothing here writes anything.

The task-cluster bootstrap is also shared.  Every run holds a handful of task
seeds and many episodes per seed, so the unit of independence is the task, not
the episode; resampling episodes would understate the spread by roughly the
number of repeats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Callable, Sequence

from ..storage import iter_jsonl

TCAS_DEFAULT_SPACE = "v1_460800"


def load_episodes(run_dir: Path) -> list[dict]:
    """Every recorded episode, complete or not; callers filter on ``status``."""
    episodes = list(iter_jsonl(Path(run_dir) / "episodes.jsonl"))
    if not episodes:
        raise FileNotFoundError(f"no episodes recorded in {run_dir}")
    return episodes


def episodes_by_degree(episodes: Sequence[dict], *, complete_only: bool = True) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for episode in episodes:
        if complete_only and episode.get("status") != "ok":
            continue
        grouped.setdefault(int(episode["d"]), []).append(episode)
    return grouped


def cell_parameters(run_dir: Path) -> dict | None:
    """``(kind, N, M, K, |V|, T, b, m)`` for one run, read back from its manifest.

    The tcas branch follows ``scaling.cell_of``: the components are
    parameter-value assignments and the parameters do not all carry the same
    number of values, so ``|V|`` is their sum and ``K`` is only an average.
    Anything that treats ``K`` as a per-family count must say so.
    """
    run_dir = Path(run_dir)
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    config = json.loads(manifest.read_text(encoding="utf-8")).get("config")
    if not config:
        return None
    env = config["environment"]
    families, variants = int(env["families"]), float(env["variants"])
    uniform_families = True
    if env["kind"] == "tcas":
        from ..environments.tcas_dataset import assignment_count, space

        sizes = [size for _, size in space(env.get("tcas_space", TCAS_DEFAULT_SPACE))]
        families = len(sizes)
        variants = assignment_count(env.get("tcas_space", TCAS_DEFAULT_SPACE)) / len(sizes)
        uniform_families = len(set(sizes)) == 1
    task_size = int(env["task_size"])
    return {
        "run": run_dir.name,
        "kind": env["kind"],
        "policy": (config.get("policy") or {}).get("kind"),
        "N": int(config["agents"]),
        "M": families,
        "K": variants,
        "V": families * variants,
        "T": task_size,
        "rounds": task_size - 1,
        "b": int(config["branching"]),
        "m": int(config["experiments_per_agent"]),
        "uniform_families": uniform_families,
        "tcas_evidence": env.get("tcas_evidence"),
    }


def measured_crossing(run_dir: Path) -> dict:
    """The ``R(d) = 1`` estimate written by ``critical_degree.py``, if present."""
    path = Path(run_dir) / "analysis" / "critical_degree.json"
    if not path.is_file():
        return {"d_c": None, "status": "absent", "ci": [None, None]}
    report = json.loads(path.read_text(encoding="utf-8"))
    crossing = report.get("crossing") or {}
    bootstrap = report.get("bootstrap") or {}
    return {
        "d_c": crossing.get("d_c"),
        "status": crossing.get("status"),
        "ci": bootstrap.get("d_c_ci") or [None, None],
    }


def task_seeds(episodes: Sequence[dict]) -> list:
    return sorted({e.get("task_seed") for e in episodes if e.get("task_seed") is not None})


def cluster_bootstrap(
    groups: dict[object, list[dict]],
    statistic: Callable[[dict[object, list[dict]]], float | None],
    *,
    draws: int,
    seed: int,
) -> dict:
    """Resample whole tasks, keeping every group paired inside one draw.

    ``groups`` is keyed by whatever the caller wants held together -- a degree,
    or a ``(degree, round)`` pair.  A draw that the statistic cannot evaluate is
    counted but not used, and the share of usable draws is reported so a wide
    interval built from three draws is not mistaken for a narrow one.
    """
    tasks = task_seeds([e for group in groups.values() for e in group])
    if not tasks or draws <= 0:
        return {"draws": 0, "valid_draws": 0, "ci": [None, None], "median": None}
    by_task: dict[object, dict[object, list[dict]]] = {}
    for key, group in groups.items():
        for episode in group:
            by_task.setdefault(episode.get("task_seed"), {}).setdefault(key, []).append(episode)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(draws):
        picked = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        resampled: dict[object, list[dict]] = {key: [] for key in groups}
        for task in picked:
            for key, group in by_task.get(task, {}).items():
                resampled[key].extend(group)
        value = statistic(resampled)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    values.sort()

    def quantile(q: float) -> float | None:
        if not values:
            return None
        position = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[position]

    return {
        "draws": draws,
        "valid_draws": len(values),
        "valid_fraction": len(values) / draws,
        "median": quantile(0.5),
        "ci": [quantile(0.025), quantile(0.975)],
        "clusters": len(tasks),
    }


def pooled_ratio(episodes: Sequence[dict], round_index: int, numerator: str, denominator: str) -> float | None:
    """Pool a per-round ratio over episodes, as ``classify_clean`` does."""
    top = bottom = 0.0
    for episode in episodes:
        transitions = episode.get("transitions") or []
        if round_index >= len(transitions):
            continue
        top += float(transitions[round_index].get(numerator) or 0.0)
        bottom += float(transitions[round_index].get(denominator) or 0.0)
    return top / bottom if bottom > 0.0 else None


def round_mean(episodes: Sequence[dict], round_index: int, key: str) -> float | None:
    values = []
    for episode in episodes:
        transitions = episode.get("transitions") or []
        if round_index >= len(transitions):
            continue
        value = transitions[round_index].get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def matplotlib_pyplot():
    """The plotting backend, or ``None`` -- CSVs are written either way."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        return pyplot
    except ImportError:
        return None


def save_figure(figure, out_dir: Path, stem: str) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        target = out_dir / f"{stem}.{suffix}"
        figure.savefig(target)
        written.append(str(target))
    return written


def linear_fit(points: Sequence[tuple[float, float]]) -> dict | None:
    """Ordinary least squares with the standard error of the slope.

    Returned rather than imported from numpy so that a two-point window, which
    has no residual degrees of freedom, comes back with a finite slope and a
    ``None`` standard error instead of a warning.
    """
    points = [(float(x), float(y)) for x, y in points if math.isfinite(y)]
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residual = sum((y - intercept - slope * x) ** 2 for x, y in points)
    stderr = math.sqrt(residual / (n - 2) / sxx) if n > 2 and residual > 0.0 else None
    total = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "slope": slope,
        "intercept": intercept,
        "stderr": stderr,
        "points": n,
        "r2": (1.0 - residual / total) if total > 0.0 else None,
    }
