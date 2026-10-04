"""Versioned public library of atomic equation terms for the physics task.

Every family contributes one additive term to ``dv/dt``.  All parameters are
public; only which ``(family, variant)`` pairs belong to the hidden mechanism
is withheld.  Parameter values are frozen and must not be retuned against
experiment outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from ..schemas import ComponentSpec, Pair

LIBRARY_VERSION = "physics_components_v1"


@dataclass(frozen=True)
class TermSpec:
    spec: ComponentSpec
    evaluate: Callable[[float, float, float], float]

    @property
    def pair(self) -> Pair:
        return self.spec.pair


def _restoring(alpha: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: -alpha * x


def _cubic(beta: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: -beta * x * x * x


def _linear_damping(gamma: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: -gamma * v


def _quadratic_drag(q: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: -q * abs(v) * v


def _saturating_damping(gamma: float, lam: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: -gamma * v / (1.0 + lam * x * x)


def _sine_drive(amplitude: float, omega: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: amplitude * math.sin(omega * t)


def _cosine_drive(amplitude: float, omega: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: amplitude * math.cos(omega * t)


def _decay_drive(amplitude: float, tau: float) -> Callable[[float, float, float], float]:
    return lambda t, x, v: amplitude * math.exp(-t / tau)


_FAMILIES: tuple[tuple[str, str, tuple[tuple[Mapping[str, float], Callable], ...]], ...] = (
    (
        "linear_restoring",
        "-alpha * x",
        tuple(({"alpha": a}, _restoring(a)) for a in (0.5, 0.9, 1.3, 1.7)),
    ),
    (
        "cubic_restoring",
        "-beta * x**3",
        tuple(({"beta": b}, _cubic(b)) for b in (0.05, 0.10, 0.25, 0.40)),
    ),
    (
        "linear_damping",
        "-gamma * v",
        tuple(({"gamma": g}, _linear_damping(g)) for g in (0.10, 0.20, 0.40, 0.80)),
    ),
    (
        "quadratic_drag",
        "-q * abs(v) * v",
        tuple(({"q": q}, _quadratic_drag(q)) for q in (0.03, 0.08, 0.15, 0.30)),
    ),
    (
        "saturating_damping",
        "-gamma * v / (1 + lambda * x**2)",
        tuple(
            ({"gamma": g, "lambda": l}, _saturating_damping(g, l))
            for g, l in ((0.2, 0.1), (0.4, 0.2), (0.6, 0.4), (0.8, 0.8))
        ),
    ),
    (
        "sine_drive",
        "A * sin(omega * t)",
        tuple(
            ({"A": a, "omega": w}, _sine_drive(a, w))
            for a, w in ((0.15, 0.7), (0.3, 1.1), (0.3, 1.7), (0.5, 2.3))
        ),
    ),
    (
        "cosine_drive",
        "A * cos(omega * t)",
        tuple(
            ({"A": a, "omega": w}, _cosine_drive(a, w))
            for a, w in ((0.15, 0.7), (0.3, 1.1), (0.3, 1.7), (0.5, 2.3))
        ),
    ),
    (
        "exponential_drive",
        "A * exp(-t / tau)",
        tuple(
            ({"A": a, "tau": s}, _decay_drive(a, s))
            for a, s in ((0.1, 0.5), (0.2, 1.0), (0.3, 2.0), (0.5, 4.0))
        ),
    ),
)

FAMILY_COUNT = len(_FAMILIES)
VARIANT_COUNT = len(_FAMILIES[0][2])


def build_library(families: int = FAMILY_COUNT, variants: int = VARIANT_COUNT) -> tuple[TermSpec, ...]:
    if families > FAMILY_COUNT or variants > VARIANT_COUNT:
        raise ValueError(
            f"{LIBRARY_VERSION} provides at most {FAMILY_COUNT} families x {VARIANT_COUNT} variants"
        )
    terms: list[TermSpec] = []
    for family_id in range(families):
        name, expression, entries = _FAMILIES[family_id]
        for variant_id in range(variants):
            parameters, evaluate = entries[variant_id]
            rendered = expression
            for key, value in parameters.items():
                rendered = rendered.replace(key, f"{value:g}")
            terms.append(
                TermSpec(
                    spec=ComponentSpec(
                        family_id=family_id,
                        variant_id=variant_id,
                        name=f"{name}_v{variant_id}",
                        expression=rendered,
                        parameters=dict(parameters),
                    ),
                    evaluate=evaluate,
                )
            )
    return tuple(terms)


def term_table(families: int = FAMILY_COUNT, variants: int = VARIANT_COUNT) -> dict[Pair, TermSpec]:
    return {term.pair: term for term in build_library(families, variants)}


SUPPORT_LIBRARY_VERSION = "physics_terms_support_v1"


def build_support_library(candidates: int | None = None) -> tuple[TermSpec, ...]:
    """The same terms, flattened into one pool with no family structure.

    SINDy-style support recovery asks which terms of a fixed library are
    active, with no constraint that at most one variant of a family can be.
    Flattening the factored table gives ``FAMILY_COUNT * VARIANT_COUNT``
    candidate terms, each its own family with a single variant, so the shared
    encoding represents a subset rather than an assignment.
    """
    flat = build_library()
    if candidates is None:
        candidates = len(flat)
    if not 1 <= candidates <= len(flat):
        raise ValueError(
            f"{SUPPORT_LIBRARY_VERSION} provides at most {len(flat)} candidate terms"
        )
    return tuple(
        TermSpec(
            spec=ComponentSpec(
                family_id=index,
                variant_id=0,
                name=term.spec.name,
                expression=term.spec.expression,
                parameters=dict(term.spec.parameters),
            ),
            evaluate=term.evaluate,
        )
        for index, term in enumerate(flat[:candidates])
    )


def support_term_table(candidates: int | None = None) -> dict[Pair, TermSpec]:
    return {term.pair: term for term in build_support_library(candidates)}
