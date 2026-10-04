"""Hidden second-order mechanism ``dx/dt = v``, ``dv/dt = sum of terms``.

An experiment returns two separate channels: the exact atomic label of the
tested component, and the trajectory of the true system under the requested
condition.  The label is read from the hidden mechanism, not inferred from the
trajectory, and does not depend on the condition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..schemas import (
    ComponentSpec,
    EnvironmentConfig,
    EnvironmentFailure,
    ExperimentCondition,
    Pair,
    PrivateObservation,
)
from .base import Environment, TaskInstance
from .component_library import LIBRARY_VERSION, support_term_table, term_table
from .physics_dataset import PhysicsData, build_physics_data


class PhysicsEnvironment(Environment):
    kind = "physics"
    notes = (
        "The hidden system obeys dx/dt = v and dv/dt = sum of the true additive terms "
        "drawn from the public library. An experiment runs the true system from a "
        "chosen initial condition and also returns the exact label of one component."
    )

    def __init__(
        self,
        task: TaskInstance,
        config: EnvironmentConfig,
        agents: int,
        data: PhysicsData | None = None,
        prediction_cache: dict | None = None,
        truth_cache: dict | None = None,
        mode: str = "factored_v1",
    ) -> None:
        super().__init__(task)
        self.config = config
        self.agents = agents
        self.mode = mode
        # support_v1 flattens the library so the target is any subset of terms,
        # which is the SINDy support-recovery reading; factored_v1 keeps the
        # (family, variant) structure where one variant per family is active.
        self.terms = (
            support_term_table(task.encoding.families)
            if mode == "support_v1"
            else term_table(task.encoding.families, task.encoding.variants)
        )
        self.data = data if data is not None else build_physics_data(config, task.seed, agents)
        self._library = tuple(term.spec for term in sorted(
            self.terms.values(), key=lambda item: item.pair
        ))
        self._cache: dict[tuple[tuple[Pair, ...], str], np.ndarray] = (
            prediction_cache if prediction_cache is not None else {}
        )
        self._truth_cache: dict[str, np.ndarray] = (
            truth_cache if truth_cache is not None else {}
        )
        self._conditions = {
            condition.experiment_id: condition
            for group in (self.data.library, self.data.heldout, self.data.diagnostic)
            for condition in group
        }
        for group in self.data.initial:
            for condition in group:
                self._conditions[condition.experiment_id] = condition
        self.failures = {"integration": 0, "nonfinite": 0}

    # -- public surface --------------------------------------------------
    def component_library(self) -> tuple[ComponentSpec, ...]:
        return self._library

    def experiment_library(self) -> tuple[ExperimentCondition, ...]:
        return self.data.library

    def initial_observations(self, agent_index: int) -> tuple[PrivateObservation, ...]:
        return tuple(
            self._observation(condition, kind="initial")
            for condition in self.data.initial[agent_index]
        )

    def supports_prediction(self) -> bool:
        return True

    # -- numerics ---------------------------------------------------------
    def _integrate(
        self, components: Sequence[Pair], condition: ExperimentCondition
    ) -> np.ndarray:
        active = [self.terms[pair].evaluate for pair in components]
        settings = condition.settings
        t0 = float(settings["t0"])
        t1 = t0 + self.config.observation_window
        times = np.linspace(t0, t1, self.config.observation_samples)

        def rhs(t: float, y: np.ndarray) -> list[float]:
            x, v = float(y[0]), float(y[1])
            return [v, sum(term(t, x, v) for term in active)]

        solution = solve_ivp(
            rhs,
            (t0, t1),
            [float(settings["x0"]), float(settings["v0"])],
            method=self.config.integrator,
            t_eval=times,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )
        if not solution.success or solution.y.shape[1] != len(times):
            self.failures["integration"] += 1
            raise EnvironmentFailure(
                f"integration failed for {tuple(components)} at {condition.experiment_id}"
            )
        states = np.vstack([times, solution.y[0], solution.y[1]]).T
        if not np.all(np.isfinite(states)):
            self.failures["nonfinite"] += 1
            raise EnvironmentFailure(
                f"non-finite trajectory for {tuple(components)} at {condition.experiment_id}"
            )
        return states

    def predict(
        self, components: Sequence[Pair], condition: ExperimentCondition
    ) -> np.ndarray | None:
        key = (tuple(sorted(components)), condition.experiment_id)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._integrate(key[0], condition)
            self._cache[key] = cached
        return cached

    def true_trajectory(self, condition: ExperimentCondition) -> np.ndarray:
        cached = self._truth_cache.get(condition.experiment_id)
        if cached is None:
            cached = self._integrate(self.task.truth_pairs, condition)
            self._truth_cache[condition.experiment_id] = cached
        return cached

    # -- observations ------------------------------------------------------
    def _observation(
        self, condition: ExperimentCondition, *, kind: str, observation_id: str | None = None
    ) -> PrivateObservation:
        states = np.round(self.true_trajectory(condition), self.config.observation_decimals)
        return PrivateObservation(
            observation_id=observation_id or f"obs_{condition.experiment_id}",
            kind=kind,  # type: ignore[arg-type]
            experiment_id=condition.experiment_id,
            settings=dict(condition.settings),
            times=tuple(float(value) for value in states[:, 0]),
            states=tuple((float(row[1]), float(row[2])) for row in states),
        )

    def observe(
        self, experiment_id: str | None, *, owner: int, event_id: str
    ) -> PrivateObservation | None:
        if experiment_id is None:
            raise EnvironmentFailure("the physics environment requires an experiment_id")
        condition = self._conditions.get(experiment_id)
        if condition is None or condition not in self.data.library:
            raise EnvironmentFailure(f"unknown public experiment condition {experiment_id!r}")
        return self._observation(condition, kind="experiment", observation_id=f"obs_{event_id}")

    # -- evaluator-only ----------------------------------------------------
    def heldout_conditions(self) -> tuple[ExperimentCondition, ...]:
        return self.data.heldout

    def heldout_error(self, components: Sequence[Pair]) -> float:
        """Normalised mean squared trajectory distance on the hold-out set."""
        total = 0.0
        for condition in self.data.heldout:
            truth = self.true_trajectory(condition)
            try:
                predicted = self.predict(components, condition)
            except EnvironmentFailure:
                return float("inf")
            if predicted is None:
                return float("nan")
            difference = predicted[:, 1:] - truth[:, 1:]
            total += float(np.mean(difference * difference) * 2.0)
        return total / len(self.data.heldout)

    def diagnostics(self) -> dict[str, object]:
        payload = super().diagnostics()
        payload.update(
            {
                "library_version": LIBRARY_VERSION,
                "experiment_library_size": len(self.data.library),
                "heldout_size": len(self.data.heldout),
                "prediction_cache_entries": len(self._cache),
            }
        )
        return payload
