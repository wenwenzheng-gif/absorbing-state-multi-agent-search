"""Frozen data domains for the physics task.

The public grid library, each agent's private initial observations, the
numerical diagnostic set and the evaluator-only hold-out set are drawn from
disjoint deterministic random domains, so changing one never perturbs another.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ...scaling_model import RNG, derived_rng
from ..schemas import EnvironmentConfig, ExperimentCondition

PHYSICS_INITIAL_DOMAIN = 0x4F1BBCDCBFA53E0B
PHYSICS_HELDOUT_DOMAIN = 0x2545F4914F6CDD1D
PHYSICS_DIAGNOSTIC_DOMAIN = 0x7FB5D329728EA185

HELDOUT_RANGE = {"x0": (-2.5, 2.5), "v0": (-2.5, 2.5), "t0": (0.0, 2.5)}
INITIAL_RANGE = {"x0": (-2.0, 2.0), "v0": (-2.0, 2.0), "t0": (0.0, 2.0)}
MIN_LIBRARY_DISTANCE = 0.1


def uniform(rng: RNG, low: float, high: float) -> float:
    return low + (high - low) * (rng.next() / float(1 << 64))


def build_experiment_library(config: EnvironmentConfig) -> tuple[ExperimentCondition, ...]:
    """Deterministic public grid, ordered x0 -> v0 -> t0."""
    conditions: list[ExperimentCondition] = []
    for x0 in config.library_x0:
        for v0 in config.library_v0:
            for t0 in config.library_t0:
                index = len(conditions)
                conditions.append(
                    ExperimentCondition(
                        experiment_id=f"e_{index:03d}",
                        description=f"start at x={x0:g}, v={v0:g} from t={t0:g}",
                        settings={"x0": float(x0), "v0": float(v0), "t0": float(t0)},
                    )
                )
    return tuple(conditions)


def _far_from_library(
    settings: dict[str, float], library: tuple[ExperimentCondition, ...]
) -> bool:
    for condition in library:
        other = condition.settings
        distance = math.dist(
            (settings["x0"], settings["v0"], settings["t0"]),
            (other["x0"], other["v0"], other["t0"]),
        )
        if distance < MIN_LIBRARY_DISTANCE:
            return False
    return True


def build_heldout_conditions(
    config: EnvironmentConfig, seed: int, library: tuple[ExperimentCondition, ...]
) -> tuple[ExperimentCondition, ...]:
    """Continuously sampled conditions that avoid every published grid point."""
    rng = derived_rng(seed, PHYSICS_HELDOUT_DOMAIN)
    conditions: list[ExperimentCondition] = []
    attempts = 0
    while len(conditions) < config.heldout_size:
        attempts += 1
        if attempts > 100 * config.heldout_size:
            raise RuntimeError("could not sample a hold-out set away from the grid")
        settings = {
            "x0": uniform(rng, *HELDOUT_RANGE["x0"]),
            "v0": uniform(rng, *HELDOUT_RANGE["v0"]),
            "t0": uniform(rng, *HELDOUT_RANGE["t0"]),
        }
        if not _far_from_library(settings, library):
            continue
        index = len(conditions)
        conditions.append(
            ExperimentCondition(
                experiment_id=f"holdout_{index:03d}",
                description="evaluator-only hold-out condition",
                settings=settings,
            )
        )
    return tuple(conditions)


def build_initial_conditions(
    config: EnvironmentConfig, seed: int, agent_index: int
) -> tuple[ExperimentCondition, ...]:
    """Private seed observations for one agent."""
    rng = derived_rng(seed, PHYSICS_INITIAL_DOMAIN, agent_index)
    conditions: list[ExperimentCondition] = []
    for index in range(config.initial_observations):
        conditions.append(
            ExperimentCondition(
                experiment_id=f"init_{agent_index:04d}_{index:02d}",
                description="private initial observation",
                settings={
                    "x0": uniform(rng, *INITIAL_RANGE["x0"]),
                    "v0": uniform(rng, *INITIAL_RANGE["v0"]),
                    "t0": uniform(rng, *INITIAL_RANGE["t0"]),
                },
            )
        )
    return tuple(conditions)


def build_diagnostic_conditions(
    config: EnvironmentConfig, seed: int, count: int = 16
) -> tuple[ExperimentCondition, ...]:
    """Conditions used for data-quality checks only; never shown to a policy."""
    rng = derived_rng(seed, PHYSICS_DIAGNOSTIC_DOMAIN)
    return tuple(
        ExperimentCondition(
            experiment_id=f"diag_{index:03d}",
            description="numerical diagnostic condition",
            settings={
                "x0": uniform(rng, *HELDOUT_RANGE["x0"]),
                "v0": uniform(rng, *HELDOUT_RANGE["v0"]),
                "t0": uniform(rng, *HELDOUT_RANGE["t0"]),
            },
        )
        for index in range(count)
    )


@dataclass(frozen=True)
class PhysicsData:
    library: tuple[ExperimentCondition, ...]
    heldout: tuple[ExperimentCondition, ...]
    diagnostic: tuple[ExperimentCondition, ...]
    initial: tuple[tuple[ExperimentCondition, ...], ...]


def build_physics_data(config: EnvironmentConfig, seed: int, agents: int) -> PhysicsData:
    library = build_experiment_library(config)
    return PhysicsData(
        library=library,
        heldout=build_heldout_conditions(config, seed, library),
        diagnostic=build_diagnostic_conditions(config, seed),
        initial=tuple(
            build_initial_conditions(config, seed, index) for index in range(agents)
        ),
    )
