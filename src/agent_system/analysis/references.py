"""Analytic critical-degree references generalised to an imperfect pruner.

The frozen formulas in :mod:`src.utils` assume every piece of evidence an
agent holds is acted on, i.e. ``q_model = 1``.  The paper writes the lineage
reproduction ratio with the elimination probability kept explicit,

``d_c = -(T-1) ln b / sum_{t=1}^{T-1} ln(1 - m q_t q_model)``,  ``q_t = t/|V|``,

so this module carries the same expressions with ``q_model`` as a free
parameter.  ``src/utils.py`` is frozen and is not touched: at ``q_model = 1``
these functions reproduce it exactly, which :mod:`tests.test_agents_knobs_q`
checks over a grid.

Only ``|V| = M*K`` enters, never ``M`` and ``K`` separately, because the
per-round survival factor of the mean-field model depends on the number of
atomic assignments alone.
"""

from __future__ import annotations

import math

__all__ = ["survival_terms", "exact_dc", "delayed_dc", "dilute_dc", "amplified_dc"]


def survival_terms(V: int, T: int, m: int, q_model: float = 1.0, *, first: int = 1):
    """``1 - m t q_model / |V|`` for ``t = first .. T-1``.

    ``first = 1`` is the paper's exact sum; ``first = 2`` drops the round that
    carries no communication, which is what ``delayed_dc`` needs.
    """
    if V <= 0:
        raise ValueError("|V| must be positive")
    if not 0.0 < q_model <= 1.0:
        raise ValueError("q_model must lie in (0, 1]")
    return [1.0 - m * t * q_model / V for t in range(first, T)]


def _crossing(terms: list[float], T: int, b: int) -> float:
    # The frozen helpers return nan rather than raising when a survival factor
    # leaves (0, 1); mirror that so a sweep can plot the gap instead of dying.
    if not terms or any(value <= 0.0 or value >= 1.0 for value in terms):
        return math.nan
    return -(T - 1) * math.log(b) / sum(math.log(value) for value in terms)


def exact_dc(V: int, T: int, b: int, m: int, q_model: float = 1.0) -> float:
    """Exact finite-T mean-field crossing, ``src.utils.exact_analytic_dc`` at q=1."""
    return _crossing(survival_terms(V, T, m, q_model, first=1), T, b)


def delayed_dc(V: int, T: int, b: int, m: int, q_model: float = 1.0) -> float:
    """Crossing with the first round's pruning omitted (messages arrive late).

    ``src.utils.delayed_boundary_dc`` at ``q_model = 1``.
    """
    return _crossing(survival_terms(V, T, m, q_model, first=2), T, b)


def dilute_dc(V: int, T: int, b: int, m: int, q_model: float = 1.0) -> float:
    """Leading dilute approximation, ``src.utils.dilute_analytic_dc`` at q=1."""
    if not 0.0 < q_model <= 1.0:
        raise ValueError("q_model must lie in (0, 1]")
    return 2.0 * V * math.log(b) / (m * T * q_model)


def amplified_dc(
    V: int, T: int, b: int, m: int, q_model: float = 1.0, *, amplification: float = 1.0
) -> float:
    """``exact_dc`` with the per-round pruning rate scaled by ``amplification``.

    Measured runs prune more than ``m q_t`` per round because a positive label
    also kills the siblings of the component it fixes.  Folding that into an
    effective rate is the same algebra as folding in ``q_model``, so the two
    multiply; keeping them apart makes it explicit which one is protocol and
    which one is agent fallibility.
    """
    if amplification <= 0.0:
        raise ValueError("amplification must be positive")
    if not 0.0 < q_model <= 1.0:
        raise ValueError("q_model must lie in (0, 1]")
    if V <= 0:
        raise ValueError("|V| must be positive")
    rate = q_model * amplification
    return _crossing([1.0 - m * t * rate / V for t in range(1, T)], T, b)
