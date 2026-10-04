"""Shared numerical and file utilities for the reproducibility pipeline."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_PATTERN = re.compile(r"^d(?P<degree>\d+)_E(?P<episodes>\d+)\.jsonl\.gz$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_analytic_dc(M: int, K: int, T: int, b: int, m: int) -> float:
    """Exact finite-T mean-field crossing for the frozen model."""
    terms = [1.0 - m * t / (M * K) for t in range(1, T)]
    if not terms or any(value <= 0.0 or value >= 1.0 for value in terms):
        return math.nan
    return -(T - 1) * math.log(b) / sum(math.log(value) for value in terms)


def dilute_analytic_dc(M: int, K: int, T: int, b: int, m: int) -> float:
    """Leading dilute approximation, valid when mT/(MK) is small."""
    return 2.0 * M * K * math.log(b) / (m * T)


def delayed_boundary_dc(M: int, K: int, T: int, b: int, m: int) -> float:
    """Diagnostic boundary obtained by omitting the first pruning factor."""
    terms = [1.0 - m * t / (M * K) for t in range(2, T)]
    if not terms or any(value <= 0.0 or value >= 1.0 for value in terms):
        return math.nan
    return -(T - 1) * math.log(b) / sum(math.log(value) for value in terms)


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
    temporary.replace(path)


def lineage_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return episode-by-round arrays for the A-to-A lineage observable.

    The numerator is ``old_pre`` only: newly generated source lineages are
    deliberately excluded.  The denominator is the preceding A population,
    with the Mode-B initial incorrect pool used at the first round.
    """
    if not rows:
        raise ValueError("at least one trajectory is required")
    rounds = int(rows[0]["rounds"])
    numerator = np.asarray(
        [[row["transitions"][t]["old_pre"] for t in range(rounds)] for row in rows],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [
            [
                row["A_initial_incorrect"]
                if t == 0
                else row["transitions"][t - 1]["A_post"]
                for t in range(rounds)
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return numerator, denominator


def pooled_log_R_rows(rows: list[dict]) -> float:
    numerator, denominator = lineage_arrays(rows)
    numerator_sum = numerator.sum(axis=0)
    denominator_sum = denominator.sum(axis=0)
    if np.any(numerator_sum <= 0) or np.any(denominator_sum <= 0):
        return -math.inf
    return float(np.log(numerator_sum / denominator_sum).sum())


def pooled_log_R(path: Path) -> float:
    return pooled_log_R_rows(read_rows(path))


def crossing_from_logs(d_low: int, d_high: int, log_low: float, log_high: float) -> float:
    if d_high != d_low + 1:
        raise ValueError("critical-degree interpolation requires adjacent integers")
    if not (log_low > 0.0 and log_high < 0.0):
        raise ValueError("crossing endpoints must have positive/negative signs")
    return d_low - log_low / (log_high - log_low)


def raw_index(directory: Path) -> dict[int, tuple[int, Path]]:
    result: dict[int, tuple[int, Path]] = {}
    if not directory.exists():
        return result
    for path in directory.glob("d*_E*.jsonl.gz"):
        match = RAW_PATTERN.match(path.name)
        if not match:
            continue
        degree = int(match.group("degree"))
        episodes = int(match.group("episodes"))
        if degree not in result or episodes > result[degree][0]:
            result[degree] = episodes, path
    return result


def adjacent_crossing(
    directory: Path, minimum_episodes: int
) -> tuple[int, int] | None:
    logs = {
        degree: pooled_log_R(path)
        for degree, (episodes, path) in raw_index(directory).items()
        if episodes >= minimum_episodes
    }
    for left in sorted(logs):
        right = left + 1
        if right in logs and logs[left] > 0.0 and logs[right] < 0.0:
            return left, right
    return None

