"""Load and validate the frozen canonical parameter registry."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .utils import PACKAGE_ROOT, dilute_analytic_dc, exact_analytic_dc


CONFIG_ROOT = PACKAGE_ROOT / "configs"
REGISTRY_PATH = CONFIG_ROOT / "final_parameter_registry.csv"
CRITERIA_PATH = CONFIG_ROOT / "clean_criteria.json"


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


@dataclass(frozen=True)
class Cell:
    cell_uid: str
    source_family: str
    source_cell_id: str
    raw_slug: str
    N: int
    M: int
    K: int
    T: int
    rounds: int
    b: int
    m: int
    n0: int
    seed_start: int
    pilot_episodes: int
    endpoint_episodes: int
    bootstrap_draws: int
    bootstrap_seed: int | None
    pilot_degrees: tuple[int, ...]
    reference_d_minus: int
    reference_d_plus: int
    reference_dc: float
    experiment_family: str
    source_design: str
    role: str
    held_out: bool
    width_stable_theory_clean: bool
    primary_global_fit: bool
    primary_one_factor_scans: str
    bootstrap_source: str
    bootstrap_key: str
    trajectory_schema_version: str

    @property
    def parameters(self) -> dict[str, int]:
        return {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "task_size": self.T,
            "rounds": self.rounds,
            "b": self.b,
            "m": self.m,
            "n0": self.n0,
            "w": 0,
        }

    @property
    def analytic_dc(self) -> float:
        return exact_analytic_dc(self.M, self.K, self.T, self.b, self.m)

    @property
    def dilute_dc(self) -> float:
        return dilute_analytic_dc(self.M, self.K, self.T, self.b, self.m)


def _load_cells() -> tuple[Cell, ...]:
    rows: list[Cell] = []
    with REGISTRY_PATH.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            bootstrap_seed = record["bootstrap_seed"].strip()
            rows.append(
                Cell(
                    cell_uid=record["cell_uid"],
                    source_family=record["source_family"],
                    source_cell_id=record["source_cell_id"],
                    raw_slug=record["raw_slug"],
                    N=int(record["N"]),
                    M=int(record["M"]),
                    K=int(record["K"]),
                    T=int(record["T"]),
                    rounds=int(record["rounds"]),
                    b=int(record["b"]),
                    m=int(record["m"]),
                    n0=int(record["n0"]),
                    seed_start=int(record["seed_start"]),
                    pilot_episodes=int(record["pilot_episodes"]),
                    endpoint_episodes=int(record["endpoint_episodes"]),
                    bootstrap_draws=int(record["bootstrap_draws"]),
                    bootstrap_seed=int(float(bootstrap_seed)) if bootstrap_seed else None,
                    pilot_degrees=tuple(int(value) for value in record["pilot_degrees"].split(";") if value),
                    reference_d_minus=int(record["reference_d_minus"]),
                    reference_d_plus=int(record["reference_d_plus"]),
                    reference_dc=float(record["reference_dc"]),
                    experiment_family=record["experiment_family"],
                    source_design=record["source_design"],
                    role=record["role"],
                    held_out=_as_bool(record["held_out"]),
                    width_stable_theory_clean=_as_bool(record["width_stable_theory_clean"]),
                    primary_global_fit=_as_bool(record["primary_global_fit"]),
                    primary_one_factor_scans=record["primary_one_factor_scans"],
                    bootstrap_source=record["bootstrap_source"],
                    bootstrap_key=record["bootstrap_key"],
                    trajectory_schema_version=record["trajectory_schema_version"],
                )
            )
    return tuple(rows)


CELLS = _load_cells()
CELL_BY_UID = {cell.cell_uid: cell for cell in CELLS}
with CRITERIA_PATH.open(encoding="utf-8") as stream:
    CLEAN_CRITERIA = json.load(stream)

SUPPLY_LIMITS = {
    name: float(spec["threshold"])
    for name, spec in CLEAN_CRITERIA["supply_clean"].items()
}
INDEPENDENCE_LIMITS = {
    name: float(spec["threshold"])
    for name, spec in CLEAN_CRITERIA["independence_clean"].items()
}


def validate() -> None:
    if len(CELLS) != 53 or len(CELL_BY_UID) != 53:
        raise RuntimeError("the frozen registry must contain 53 unique cells")
    for cell in CELLS:
        if cell.rounds != cell.T - 1:
            raise RuntimeError(f"{cell.cell_uid}: rounds must equal T-1")
        if cell.reference_d_plus != cell.reference_d_minus + 1:
            raise RuntimeError(f"{cell.cell_uid}: non-adjacent reference crossing")
        if not {cell.reference_d_minus, cell.reference_d_plus}.issubset(cell.pilot_degrees):
            raise RuntimeError(f"{cell.cell_uid}: endpoints absent from pilot degrees")
        if not math.isfinite(cell.analytic_dc):
            raise RuntimeError(f"{cell.cell_uid}: invalid analytic regime")
        if cell.n0 - 1 > cell.M * cell.K - cell.T:
            raise RuntimeError(f"{cell.cell_uid}: Mode-B initial width is infeasible")


validate()

