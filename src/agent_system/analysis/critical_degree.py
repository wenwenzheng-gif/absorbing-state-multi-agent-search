#!/usr/bin/env python3
"""Lineage ratio, degree scan and crossing estimate for agent runs.

Differences from the frozen ``scan_dc.py`` are deliberate:

* a degree whose denominator pool is empty is reported as *not estimable*
  (0/0), separately from a degree with a positive pool and no descendants;
* no epsilon is substituted for a zero or non-finite endpoint, and such an
  endpoint is never interpolated through;
* the bootstrap resamples whole tasks (clusters), keeps adjacent degrees
  paired inside a draw, and reports the share of draws that produced a
  crossing at all.

Three numerator conventions are reported side by side.  ``pre_own`` is the
frozen convention of ``src/utils.py``: the A-to-A count is read after the
branch step, so one ratio measures ``b (1 - p_nbr)^d``.  ``post_own`` reads
the same count after the agent's own experiment and pruning, so one ratio
measures ``b (1 - p_own)(1 - p_nbr)^d`` and matches the reproduction factor
written in the paper.  ``literal`` is the paper's own ``A_{t+1} / A_t``: the
whole incorrect population of consecutive rounds, newly generated source
lineages included, which is the quantity the text writes down but not the
one the frozen code measures.  The published synthetic fit uses ``pre_own``.

Two estimators of the critical degree are reported for every convention.
``interpolation`` is the published one: a two-point linear interpolation of
``log R`` at the first adjacent sign change.  ``fit`` is the estimator the
paper's wording implies -- least squares of ``log R(d) = a + s d`` over a
window of degrees, with ``d_c = -a/s``.  ``log R(d)`` is not in fact linear
over the whole measured range -- it is steep near the crossing and flattens
or plateaus at high d -- so a global window lets points far from the
crossing pull the line and disagree with the interpolation by a lot.  The
default window is therefore ``local``: the measured degrees immediately
around the interpolated crossing (see ``fit_window_degrees_local``), not
every measured degree.  The old global window (``estimable`` / ``all`` / a
``|log R|`` bound) is still available via ``--fit-window`` for comparison.
Both estimators come with a task-cluster bootstrap interval.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from ..storage import iter_jsonl
from .summarize import write_csv

STATUS_OK = "ok"
STATUS_EXTINCT = "extinct_lineage"
STATUS_NOT_ESTIMABLE = "not_estimable"
STATUS_PARTIAL = "partial_rounds"

CONVENTIONS = {"pre_own": "old_pre", "post_own": "old_post", "literal": "A_post"}
DEFAULT_CONVENTION = "pre_own"

# The fit window.  "local" (the default of ``fit_log_ratio`` / ``scan`` /
# the CLI) keeps only the measured degrees immediately around the
# interpolated crossing -- see ``fit_window_degrees_local`` -- because
# log R(d) is not log-linear over the whole range and a window far from the
# crossing biases d_c_fit.  "estimable" keeps every degree whose pooled
# log R is a real measurement and stops at the first degree where the
# frontier has gone essentially extinct, because past that point log R is
# censored at -inf rather than measured and a least-squares line would be
# dragged by it.  "all" keeps every estimable degree wherever it sits, and a
# float keeps the estimable degrees whose |log R| is at most that bound.
# "estimable" / "all" / a bound are the old global windows, unchanged, and
# are still available via ``--fit-window`` for comparison with "local".
#
# ``DEFAULT_GLOBAL_FIT_WINDOW`` is the *unchanged* default of the low-level
# ``fit_window_degrees`` helper itself (it has no notion of "local" -- that
# is built on top of it in ``fit_window_degrees_local``), kept separate from
# ``DEFAULT_FIT_WINDOW`` so existing direct callers of
# ``fit_window_degrees(curve)`` keep exactly their old behaviour.
DEFAULT_GLOBAL_FIT_WINDOW = "estimable"
DEFAULT_FIT_WINDOW = "local"
DEFAULT_FIT_LOCAL_K = 3
DEFAULT_FIT_LOCAL_BOUND = 1.0
DEFAULT_LOW_D_MAX = 8
FIT_STATUS_OK = "ok"


def lineage_terms(episode: dict, convention: str = DEFAULT_CONVENTION) -> list[tuple[float, float]]:
    """Per-round ``(numerator, denominator)`` for the A-to-A lineage ratio."""
    try:
        field = CONVENTIONS[convention]
    except KeyError:
        raise ValueError(f"unknown lineage convention {convention!r}") from None
    transitions = episode.get("transitions") or []
    terms = []
    for index, transition in enumerate(transitions):
        numerator = float(transition.get(field) or 0)
        if index == 0:
            denominator = float(episode.get("A_initial_incorrect") or 0)
        else:
            denominator = float(transitions[index - 1].get("A_post") or 0)
        terms.append((numerator, denominator))
    return terms


def convention_available(episodes: Sequence[dict], convention: str) -> bool:
    """Whether the field this convention reads is actually recorded."""
    field = CONVENTIONS[convention]
    for episode in episodes:
        for transition in episode.get("transitions") or []:
            if field in transition:
                return True
    return False


def pooled_step_log_ratio(
    episodes: Sequence[dict], convention: str = DEFAULT_CONVENTION, round_index: int = -1
) -> dict:
    """Pooled ``log r_t`` for one round, rather than the product over rounds.

    The paper reports a cumulative and a per-step critical degree separately,
    and they differ for a real reason: with one-round-delayed messages the
    first round is never communicated, so it contributes a fixed ``log b`` to
    the cumulative product that no amount of communication can remove.  The
    step reported here is the last round, the deepest one and the only one in
    which every agent has had a full round of neighbour evidence.
    """
    usable = [ep for ep in episodes if ep.get("status") == "ok" and ep.get("transitions")]
    if not usable:
        return {"value": None, "status": STATUS_NOT_ESTIMABLE, "episodes": 0, "round": None}
    lengths = [len(ep["transitions"]) for ep in usable]
    # Episodes can be ragged -- ``stop_on_success`` truncates some of them --
    # so the round count is the longest episode and each round is pooled over
    # the episodes that actually reached it.  Reading it off ``usable[0]``
    # raised IndexError as soon as a later episode was longer.
    rounds = max(lengths)
    ragged = len(set(lengths)) > 1
    index = round_index if round_index >= 0 else rounds + round_index
    if not 0 <= index < rounds:
        return {"value": None, "status": STATUS_NOT_ESTIMABLE, "episodes": len(usable),
                "round": None, "ragged": ragged}
    numerator = denominator = 0.0
    contributing = 0
    for episode in usable:
        terms = lineage_terms(episode, convention)
        if index >= len(terms):
            continue
        contributing += 1
        numerator += terms[index][0]
        denominator += terms[index][1]
    payload = {"episodes": contributing, "episodes_total": len(usable),
               "ragged": ragged, "round": index + 1, "rounds": rounds,
               "numerator": numerator, "denominator": denominator}
    if not contributing:
        return {"value": None, "status": STATUS_NOT_ESTIMABLE,
                "reason": f"no episode reached round {index + 1}", **payload}
    if denominator <= 0.0:
        return {"value": None, "status": STATUS_NOT_ESTIMABLE,
                "reason": "no incorrect lineage at risk in this round (0/0)", **payload}
    if numerator <= 0.0:
        return {"value": -math.inf, "status": STATUS_EXTINCT, **payload}
    return {"value": math.log(numerator / denominator), "status": STATUS_OK, **payload}


def pooled_log_ratio(
    episodes: Sequence[dict],
    convention: str = DEFAULT_CONVENTION,
    allow_partial: bool = False,
    first_round: int = 1,
) -> dict:
    """Cross-episode pooled log R with an explicit estimability status.

    One dead round used to void the whole degree with no record of which
    round died.  Every round that could not be estimated is now listed in
    ``failed_rounds`` with its reason and its two counts.  Dropping those
    rounds from the product is *not* the default: the cumulative log R is a
    sum over a fixed number of rounds, so a degree fitted on fewer rounds is
    systematically larger and is not comparable with its neighbours.  With
    ``allow_partial`` the surviving rounds are summed anyway and the status
    becomes ``partial_rounds``, which no estimator here consumes -- it is a
    diagnostic, reported so the shape of the loss is visible.

    ``first_round`` (1-based, default 1) drops every transition before it
    from the cumulative sum -- the ``failed_rounds`` check, the extinction
    check and ``value`` all only look at rounds ``>= first_round``.  With
    ``first_round == 1`` this is exactly the old behaviour.  ``first_round
    == 2`` is the paper's *delayed* d_c: round 1 has no neighbour messages
    yet (one-round-delayed communication), so it carries no d-dependence and
    is excluded, matching ``references.delayed_dc``.
    """
    if first_round < 1:
        raise ValueError("first_round must be >= 1")
    usable = [ep for ep in episodes if ep.get("status") == "ok" and ep.get("transitions")]
    if not usable:
        return {"value": None, "status": STATUS_NOT_ESTIMABLE, "episodes": 0, "rounds": 0,
                "rounds_used": 0, "failed_rounds": [], "ragged": False,
                "episodes_per_round": [], "first_round": first_round}
    # Ragged episodes are normal once ``stop_on_success`` can truncate one:
    # the round count is the longest episode, every round is pooled over the
    # episodes that reached it, and the per-round counts are reported so a
    # round backed by two episodes is not read as one backed by sixteen.
    lengths = [len(ep["transitions"]) for ep in usable]
    rounds = max(lengths)
    ragged = len(set(lengths)) > 1
    numerators = [0.0] * rounds
    denominators = [0.0] * rounds
    counts = [0] * rounds
    for episode in usable:
        for index, (numerator, denominator) in enumerate(lineage_terms(episode, convention)):
            numerators[index] += numerator
            denominators[index] += denominator
            counts[index] += 1
    included = range(min(first_round - 1, rounds), rounds)
    failed_rounds = [
        {
            "round": index + 1,
            "reason": "no incorrect lineage at risk (0/0)",
            "numerator": numerators[index],
            "denominator": denominators[index],
            "episodes": counts[index],
        }
        for index in included
        if denominators[index] <= 0.0
    ]
    live = [index for index in included if denominators[index] > 0.0]
    common = {
        "episodes": len(usable),
        "rounds": rounds,
        "rounds_used": len(live),
        "failed_rounds": failed_rounds,
        "numerators": numerators,
        "denominators": denominators,
        "episodes_per_round": counts,
        "ragged": ragged,
        "first_round": first_round,
    }
    if failed_rounds and (not allow_partial or not live):
        listed = ", ".join(str(entry["round"]) for entry in failed_rounds)
        return {
            "value": None,
            "status": STATUS_NOT_ESTIMABLE,
            "reason": f"round(s) {listed} had no incorrect lineage at risk (0/0)",
            **common,
        }
    if any(numerators[index] <= 0.0 for index in live):
        return {"value": -math.inf, "status": STATUS_EXTINCT, **common}
    value = sum(math.log(numerators[index] / denominators[index]) for index in live)
    return {
        "value": value,
        "status": STATUS_PARTIAL if failed_rounds else STATUS_OK,
        **common,
    }


def crossing_from_logs(d_low: float, d_high: float, log_low: float, log_high: float) -> float:
    """Linear interpolation of ``log R`` between two *measured* degrees.

    The endpoints used to have to be adjacent integers, which silently voided
    every scan on a step-2 grid -- all the topology runs, every LLM run --
    with ``no_adjacent_crossing``.  They now only have to be consecutive
    points of whatever grid was measured; at step 1 the arithmetic is
    unchanged, so dense scans keep their published values.
    """
    if d_high <= d_low:
        raise ValueError("crossing interpolation needs an increasing bracket")
    if not (log_low > 0.0 and log_high < 0.0):
        raise ValueError("crossing endpoints must have positive/negative signs")
    return d_low - log_low * (d_high - d_low) / (log_high - log_low)


def _grid_step(degrees: Sequence[int]) -> int | None:
    """The resolution of the degree grid: the commonest spacing, or None."""
    gaps = [b - a for a, b in zip(degrees, degrees[1:])]
    if not gaps:
        return None
    return int(max(set(gaps), key=lambda gap: (gaps.count(gap), -gap)))


def find_crossings(curve: dict[int, dict]) -> dict:
    """Locate sign changes, distinguishing every inconclusive case.

    ``d_c`` is the first crossing.  When there is more than one it is not a
    defensible point estimate, which is why ``status`` says so and every
    crossing is listed: downstream fits are expected to filter on the status
    rather than to read ``d_c`` unconditionally.

    A crossing is read between two degrees that are *consecutive in the
    measured grid*, not between two adjacent integers, so a scan over
    ``{0, 4, 8, ...}`` resolves instead of returning ``no_adjacent_crossing``.
    The price is resolution rather than validity, and it is reported: each
    crossing carries the ``bracket`` it was interpolated in, and the result
    carries the ``grid_step`` of the scan.  A degree that is present but not
    measurable still blocks the pair it sits between, exactly as before, so
    an extinct or non-estimable degree is never interpolated across.
    """
    finite = {
        degree: entry["value"]
        for degree, entry in curve.items()
        if entry["status"] == STATUS_OK and entry["value"] is not None
    }
    degrees = sorted(curve)
    crossings = []
    for left, right in zip(degrees, degrees[1:]):
        if left not in finite or right not in finite:
            continue
        if finite[left] > 0.0 > finite[right]:
            crossings.append(
                {
                    "d_minus": left,
                    "d_plus": right,
                    "log_R_minus": finite[left],
                    "log_R_plus": finite[right],
                    "d_c": crossing_from_logs(left, right, finite[left], finite[right]),
                    "bracket": [left, right],
                    "bracket_width": right - left,
                }
            )
    grid_step = _grid_step(degrees)
    if not crossings:
        below = [d for d in degrees if d in finite and finite[d] < 0.0]
        above = [d for d in degrees if d in finite and finite[d] > 0.0]
        if not finite:
            status = "not_estimable"
        elif not below:
            status = "no_crossing_all_positive"
        elif not above:
            status = "no_crossing_all_negative"
        elif min(degrees) in finite and finite[min(degrees)] < 0.0:
            status = "boundary_crossing_below_range"
        else:
            # A sign change exists but the two degrees that straddle it are
            # not consecutive measurements: something in between is censored.
            status = "no_adjacent_crossing"
        return {"status": status, "crossings": [], "d_c": None, "n_crossings": 0,
                "d_c_all": [], "grid_step": grid_step, "bracket": None}
    status = "single_crossing" if len(crossings) == 1 else "multiple_crossings"
    return {
        "status": status,
        "crossings": crossings,
        "d_c": crossings[0]["d_c"],
        "n_crossings": len(crossings),
        "d_c_all": [entry["d_c"] for entry in crossings],
        "grid_step": grid_step,
        "bracket": crossings[0]["bracket"],
    }


CLEAN_CROSSING_STATUS = "single_crossing"


def fit_window_degrees(curve: dict[int, dict], window=DEFAULT_GLOBAL_FIT_WINDOW) -> list[int]:
    """Degrees that enter the log-linear fit, under the documented window.

    A degree enters only if its pooled log R is a real measurement: status
    ``ok`` and a finite value.  An extinct or non-estimable degree is not a
    large negative log R, it is a censored one, so the scan is truncated at
    the first such degree that follows a measured one -- otherwise every
    degree past the extinction point would silently vote for a steeper line.
    """
    degrees = sorted(curve)
    usable, seen_measured = [], False
    for degree in degrees:
        entry = curve[degree]
        value = entry.get("value")
        measured = (
            entry.get("status") == STATUS_OK
            and value is not None
            and math.isfinite(float(value))
        )
        if measured:
            usable.append(degree)
            seen_measured = True
        elif seen_measured and window != "all":
            break
    if window in ("estimable", "all"):
        return usable
    bound = float(window)
    return [d for d in usable if abs(float(curve[d]["value"])) <= bound]


def fit_window_degrees_local(
    curve: dict[int, dict],
    crossing: dict,
    k: int = DEFAULT_FIT_LOCAL_K,
    bound: float = DEFAULT_FIT_LOCAL_BOUND,
) -> tuple[list[int], str]:
    """Measured degrees immediately around the interpolated crossing.

    ``log R(d)`` is steep near the crossing and flattens or plateaus at high
    d, so a global least-squares window lets distant points bias ``d_c_fit``
    away from the interpolation.  This takes the interpolation's bracket
    ``[d_lo, d_hi]`` and extends it symmetrically to ``k`` more *measured*
    degrees on each side (skipping degrees that are extinct or otherwise not
    estimable, so the count is of real measurements, not grid positions).  It
    then drops points whose ``|log R|`` exceeds ``bound`` -- points that far
    from zero are deep in the flat/bent part of the curve -- but only if at
    least 4 points would remain, since a starved fit is worse than a slightly
    wider window.

    Returns ``(degrees, mode)``: ``mode`` is ``"local"`` normally, or
    ``"estimable_fallback"`` when there is no single crossing to anchor on,
    in which case the old global ``"estimable"`` window is used instead.
    """
    if crossing.get("d_c") is None or not crossing.get("bracket"):
        return fit_window_degrees(curve, "estimable"), "estimable_fallback"
    measured = fit_window_degrees(curve, "all")
    d_lo, d_hi = crossing["bracket"]
    if d_lo not in measured or d_hi not in measured:
        return fit_window_degrees(curve, "estimable"), "estimable_fallback"
    i_lo, i_hi = measured.index(d_lo), measured.index(d_hi)
    lo = max(0, i_lo - k)
    hi = min(len(measured) - 1, i_hi + k)
    window_degrees = measured[lo : hi + 1]
    filtered = [d for d in window_degrees if abs(float(curve[d]["value"])) <= bound]
    if len(filtered) >= 4:
        window_degrees = filtered
    return window_degrees, "local"


def _weighted_line_fit(curve: dict[int, dict], degrees: list[int], weighted: bool) -> dict:
    """Shared weighted-least-squares core for both fit windows."""
    x = np.array([float(d) for d in degrees])
    y = np.array([float(curve[d]["value"]) for d in degrees])
    w = (
        np.array([float(curve[d].get("episodes") or 1) for d in degrees])
        if weighted
        else np.ones_like(x)
    )
    design = np.vstack([np.ones_like(x), x]).T
    root_w = np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])
    residual = y - design @ coefficients
    spread = float((w * (y - np.average(y, weights=w)) ** 2).sum())
    dof = len(degrees) - 2
    return {
        "intercept": intercept,
        "slope": slope,
        "R2": float(1.0 - (w * residual ** 2).sum() / spread) if spread else None,
        "residual_sd": float(math.sqrt((w * residual ** 2).sum() / dof)) if dof else None,
    }


def fit_low_d_slope(
    curve: dict[int, dict], max_degree: int = DEFAULT_LOW_D_MAX, weighted: bool = False
) -> dict:
    """Slope of ``log R = a + s d`` restricted to measured degrees ``d <= max_degree``.

    Downstream analyses read this as the measured per-neighbour pruning
    rate, so it is reported unconditionally alongside whichever fit window
    was requested, not only when that window happens to sit at low d.
    """
    degrees = [d for d in fit_window_degrees(curve, "all") if d <= max_degree]
    if len(degrees) < 3:
        return {"slope_low_d": None, "n_points_low_d": len(degrees), "degrees_low_d": degrees}
    fit = _weighted_line_fit(curve, degrees, weighted)
    return {
        "slope_low_d": fit["slope"],
        "n_points_low_d": len(degrees),
        "degrees_low_d": degrees,
    }


def fit_log_ratio(
    curve: dict[int, dict],
    window=DEFAULT_FIT_WINDOW,
    weighted: bool = False,
    local_k: int = DEFAULT_FIT_LOCAL_K,
    local_bound: float = DEFAULT_FIT_LOCAL_BOUND,
    crossing: dict | None = None,
) -> dict:
    """Least squares ``log R(d) = a + s d`` over the window; ``d_c = -a/s``.

    This is the estimator the paper's wording ("empirically fitting R(d)")
    describes.  It differs from the interpolation in what it is sensitive
    to: the interpolation uses only the two degrees that straddle zero and
    ignores everything else, while this one uses a window of degrees and so
    is stable when a single degree is noisy but biased when the curve is not
    in fact log-linear over that window.  ``window="local"`` (the default)
    keeps that window small and centred on the crossing for exactly that
    reason; ``"estimable"``, ``"all"`` and a ``|log R|`` bound are the old
    global windows.  ``weighted`` weights each degree by the number of
    episodes behind it.  ``crossing`` lets a caller that already computed
    ``find_crossings(curve)`` pass it in instead of recomputing it here (the
    bootstrap does this so each resample's local window is anchored on that
    resample's own crossing).
    """
    if window == "local":
        crossing = crossing if crossing is not None else find_crossings(curve)
        degrees, fit_window_mode = fit_window_degrees_local(curve, crossing, local_k, local_bound)
    else:
        degrees = fit_window_degrees(curve, window)
        fit_window_mode = str(window)
    payload: dict = {
        "window": str(window),
        "fit_window_mode": fit_window_mode,
        "local_k": local_k if window == "local" else None,
        "local_bound": local_bound if window == "local" else None,
        "weighted": weighted,
        "degrees": degrees,
        "n_points": len(degrees),
        "d_c_fit": None,
        "intercept": None,
        "slope": None,
        "R2": None,
        "residual_sd": None,
    }
    payload.update(fit_low_d_slope(curve, DEFAULT_LOW_D_MAX, weighted))
    if len(degrees) < 3:
        payload["status"] = "too_few_points"
        return payload
    line = _weighted_line_fit(curve, degrees, weighted)
    intercept, slope = line["intercept"], line["slope"]
    payload.update(line)
    payload["d_min"] = degrees[0]
    payload["d_max"] = degrees[-1]
    if slope >= 0.0:
        payload["status"] = "no_decay_in_degree"
        return payload
    value = -intercept / slope
    if not math.isfinite(value) or value <= 0.0:
        payload["status"] = "root_not_positive"
        return payload
    payload["d_c_fit"] = value
    payload["extrapolated"] = not (degrees[0] <= value <= degrees[-1])
    payload["status"] = FIT_STATUS_OK
    return payload


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, not the nearest order statistic."""
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), 100.0 * q))


def _spread(values: Sequence[float]) -> dict:
    return {
        "median": _quantile(values, 0.5),
        "ci": [_quantile(values, 0.025), _quantile(values, 0.975)] if len(values) else None,
        "se": float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else None,
    }


def cluster_bootstrap(
    episodes_by_degree: dict[int, list[dict]],
    draws: int,
    seed: int,
    convention: str = DEFAULT_CONVENTION,
    ratio=None,
    fit_window=DEFAULT_FIT_WINDOW,
    fit_local_k: int = DEFAULT_FIT_LOCAL_K,
    fit_local_bound: float = DEFAULT_FIT_LOCAL_BOUND,
) -> dict:
    """Resample whole tasks; keep degrees paired inside each draw.

    Both estimators are resampled together on the same draws, so their
    intervals are comparable and their difference per draw is meaningful.
    When ``fit_window == "local"`` the local window is recomputed *inside*
    each draw from that draw's own interpolated crossing (the ``result``
    already computed for the interpolation estimator is reused, so a
    resample whose curve moved does not get anchored on the point estimate's
    window) rather than held fixed across draws.  This is the more faithful
    option -- it propagates the crossing's own resampling uncertainty into
    the fit's interval -- and it was not unstable in practice on these
    curves, so the "hold the window fixed" alternative was not needed.
    """
    tasks = sorted(
        {
            episode.get("task_seed")
            for group in episodes_by_degree.values()
            for episode in group
            if episode.get("task_seed") is not None
        }
    )
    empty_fit = {"valid_draws": 0, "valid_fraction": None, "d_c_fit_median": None,
                 "d_c_fit_ci": None, "d_c_fit_se": None, "slope_median": None, "statuses": {},
                 "window_modes": {}}
    if not tasks or draws <= 0:
        return {"clusters": len(tasks), "draws": 0, "valid_draws": 0,
                "valid_fraction": None, "d_c_ci": None, "fit": empty_fit}
    rng = random.Random(seed)
    values: list[float] = []
    fitted: list[float] = []
    slopes: list[float] = []
    statuses: dict[str, int] = {}
    fit_statuses: dict[str, int] = {}
    fit_window_modes: dict[str, int] = {}
    for _ in range(draws):
        picked = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        curve = {}
        for degree, group in episodes_by_degree.items():
            by_task: dict[object, list[dict]] = {}
            for episode in group:
                by_task.setdefault(episode.get("task_seed"), []).append(episode)
            resampled = [ep for task in picked for ep in by_task.get(task, [])]
            curve[degree] = (ratio or pooled_log_ratio)(resampled, convention)
        result = find_crossings(curve)
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        if result["d_c"] is not None:
            values.append(result["d_c"])
        fit = fit_log_ratio(
            curve, fit_window, local_k=fit_local_k, local_bound=fit_local_bound, crossing=result
        )
        fit_statuses[fit["status"]] = fit_statuses.get(fit["status"], 0) + 1
        fit_window_modes[fit["fit_window_mode"]] = fit_window_modes.get(fit["fit_window_mode"], 0) + 1
        if fit["d_c_fit"] is not None:
            fitted.append(fit["d_c_fit"])
        if fit["slope"] is not None:
            slopes.append(fit["slope"])
    interpolated = _spread(values)
    fit_spread = _spread(fitted)
    return {
        # The two numbers that say whether the interval means anything at
        # all come first: how many independent tasks there were, and on what
        # share of the draws the estimator resolved.
        "clusters": len(tasks),
        "valid_fraction": len(values) / draws,
        "draws": draws,
        "valid_draws": len(values),
        "statuses": statuses,
        "d_c_median": interpolated["median"],
        "d_c_ci": interpolated["ci"],
        "d_c_se": interpolated["se"],
        "fit": {
            "valid_fraction": len(fitted) / draws,
            "valid_draws": len(fitted),
            "statuses": fit_statuses,
            "window_modes": fit_window_modes,
            "d_c_fit_median": fit_spread["median"],
            "d_c_fit_ci": fit_spread["ci"],
            "d_c_fit_se": fit_spread["se"],
            "slope_median": _quantile(slopes, 0.5),
        },
    }


CURVE_COLUMNS = (
    "d", "log_R", "status", "episodes", "rounds_used", "ragged",
    "episodes_per_round", "in_fit_window", "reason",
)
ESTIMATE_COLUMNS = (
    "convention", "estimator", "d_c", "se", "ci_low", "ci_high", "status",
    "clusters", "draws", "valid_fraction", "n_fit_points", "slope", "intercept", "R2",
    "fit_window_mode", "slope_low_d",
)


def _curve_rows(curve: dict[int, dict], in_window: Sequence[int]) -> list[dict]:
    window = set(in_window)
    return [
        {
            "d": degree,
            "log_R": entry["value"],
            "status": entry["status"],
            "episodes": entry["episodes"],
            "rounds_used": entry.get("rounds_used"),
            # Ragged: the episodes behind this degree do not all have the
            # same number of rounds, so the per-round pools differ in size.
            "ragged": bool(entry.get("ragged")),
            "episodes_per_round": ";".join(
                str(count) for count in (entry.get("episodes_per_round") or [])
            ),
            "in_fit_window": degree in window,
            "reason": entry.get("reason", ""),
        }
        for degree, entry in sorted(curve.items())
    ]


def _degree_scan(
    by_degree: dict[int, list[dict]],
    convention: str,
    draws: int,
    seed: int,
    ratio=pooled_log_ratio,
    fit_window=DEFAULT_FIT_WINDOW,
    fit_local_k: int = DEFAULT_FIT_LOCAL_K,
    fit_local_bound: float = DEFAULT_FIT_LOCAL_BOUND,
) -> tuple[list[dict], dict, dict, dict]:
    curve = {
        degree: ratio(group, convention)
        for degree, group in sorted(by_degree.items())
    }
    crossing = find_crossings(curve)
    fit = fit_log_ratio(
        curve, fit_window, local_k=fit_local_k, local_bound=fit_local_bound, crossing=crossing
    )
    rows = _curve_rows(curve, fit["degrees"])
    bootstrap = cluster_bootstrap(
        by_degree, draws, seed, convention, ratio, fit_window, fit_local_k, fit_local_bound
    )
    return rows, crossing, bootstrap, fit


def _estimate_rows(convention: str, crossing: dict, bootstrap: dict, fit: dict) -> list[dict]:
    fit_boot = bootstrap.get("fit") or {}
    interval = bootstrap.get("d_c_ci") or [None, None]
    fit_interval = fit_boot.get("d_c_fit_ci") or [None, None]
    return [
        {
            "convention": convention,
            "estimator": "interpolation",
            "d_c": crossing["d_c"],
            "se": bootstrap.get("d_c_se"),
            "ci_low": interval[0],
            "ci_high": interval[1],
            "status": crossing["status"],
            "clusters": bootstrap.get("clusters"),
            "draws": bootstrap.get("draws"),
            "valid_fraction": bootstrap.get("valid_fraction"),
            "n_fit_points": None,
            "slope": None,
            "intercept": None,
            "R2": None,
            "fit_window_mode": None,
            "slope_low_d": None,
        },
        {
            "convention": convention,
            "estimator": "fit",
            "d_c": fit["d_c_fit"],
            "se": fit_boot.get("d_c_fit_se"),
            "ci_low": fit_interval[0],
            "ci_high": fit_interval[1],
            "status": fit["status"],
            "clusters": bootstrap.get("clusters"),
            "draws": bootstrap.get("draws"),
            "valid_fraction": fit_boot.get("valid_fraction"),
            "n_fit_points": fit["n_points"],
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "R2": fit["R2"],
            "fit_window_mode": fit.get("fit_window_mode"),
            "slope_low_d": fit.get("slope_low_d"),
        },
    ]


def scan(
    run_dir: Path,
    *,
    draws: int = 1000,
    seed: int = 20260918,
    convention: str = DEFAULT_CONVENTION,
    fit_window=DEFAULT_FIT_WINDOW,
    fit_local_k: int = DEFAULT_FIT_LOCAL_K,
    fit_local_bound: float = DEFAULT_FIT_LOCAL_BOUND,
    first_round: int = 1,
) -> dict:
    """Run the full scan; ``first_round`` (1-based, default 1) drops every

    transition before it from the cumulative pooled log R -- the curve, its
    crossing, the bootstrap and the fit all use it, since they all build on
    ``pooled_log_ratio``.  ``first_round == 1`` reproduces the old behaviour
    byte-for-byte, including output filenames.  Any other value is the
    *delayed* d_c (round 1 carries no neighbour messages yet, so it has no
    d-dependence and is excluded -- matches ``references.delayed_dc``) and
    is written to filenames suffixed ``_t<first_round>`` so the default
    (``first_round=1``) outputs are never touched.  Per-round/step outputs
    (``log_r_step_by_degree.csv``, ``log_r_per_round_by_degree.csv``) do not
    depend on ``first_round`` -- each round is already isolated -- but are
    still written under the suffixed name for a non-default run, so nothing
    at the default path is rewritten.
    """
    run_dir = Path(run_dir)
    episodes = list(iter_jsonl(run_dir / "episodes.jsonl"))
    if not episodes:
        raise FileNotFoundError(f"no episodes recorded in {run_dir}")
    by_degree: dict[int, list[dict]] = {}
    for episode in episodes:
        by_degree.setdefault(int(episode["d"]), []).append(episode)
    suffix = "" if first_round == 1 else f"_t{first_round}"
    ratio = functools.partial(pooled_log_ratio, first_round=first_round)
    rows, result, bootstrap, fit = _degree_scan(
        by_degree, convention, draws, seed, ratio,
        fit_window=fit_window, fit_local_k=fit_local_k, fit_local_bound=fit_local_bound,
    )
    step_rows, step_result, step_bootstrap, step_fit = _degree_scan(
        by_degree, convention, draws, seed, pooled_step_log_ratio,
        fit_window, fit_local_k, fit_local_bound,
    )
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / f"log_R_by_degree{suffix}.csv", rows, CURVE_COLUMNS)
    write_csv(analysis_dir / f"log_r_step_by_degree{suffix}.csv", step_rows, CURVE_COLUMNS)

    # Per-round response to degree.  The theory gives every round the same
    # form, ln b + d ln(1 - p_t), so a round that does not respond to d, or
    # that stops responding past a small d, is a departure worth seeing.
    rounds = max(
        (
            len(episode.get("transitions") or [])
            for group in by_degree.values()
            for episode in group
        ),
        default=0,
    )
    per_round = []
    for index in range(rounds):
        for degree, group in sorted(by_degree.items()):
            entry = pooled_step_log_ratio(group, convention, index)
            per_round.append(
                {
                    "round": index + 1,
                    "d": degree,
                    "log_r": entry["value"],
                    "status": entry["status"],
                    "episodes": entry["episodes"],
                }
            )
    if per_round:
        write_csv(
            analysis_dir / f"log_r_per_round_by_degree{suffix}.csv",
            per_round,
            ("round", "d", "log_r", "status", "episodes"),
        )

    estimates = _estimate_rows(convention, result, bootstrap, fit)
    alternates = {}
    skipped = []
    for name in CONVENTIONS:
        if name == convention:
            continue
        if not convention_available(episodes, name):
            # "literal" needs A_post on every transition; an older log that
            # predates the field is reported as absent, not as a zero.
            skipped.append({"convention": name, "reason": f"{CONVENTIONS[name]} not recorded"})
            continue
        other_rows, other_result, other_bootstrap, other_fit = _degree_scan(
            by_degree, name, draws, seed, ratio,
            fit_window=fit_window, fit_local_k=fit_local_k, fit_local_bound=fit_local_bound,
        )
        write_csv(analysis_dir / f"log_R_by_degree_{name}{suffix}.csv", other_rows, CURVE_COLUMNS)
        alternates[name] = {
            "curve": other_rows,
            "crossing": other_result,
            "bootstrap": other_bootstrap,
            "fit": other_fit,
        }
        estimates += _estimate_rows(name, other_result, other_bootstrap, other_fit)
    write_csv(analysis_dir / f"d_c_estimates{suffix}.csv", estimates, ESTIMATE_COLUMNS)

    report = {
        "run_dir": str(run_dir),
        "convention": convention,
        "fit_window": str(fit_window),
        "fit_local_k": fit_local_k,
        "fit_local_bound": fit_local_bound,
        "bootstrap_clusters": bootstrap.get("clusters"),
        "bootstrap_valid_fraction": bootstrap.get("valid_fraction"),
        "curve": rows,
        "crossing": result,
        "bootstrap": bootstrap,
        "fit": fit,
        "estimates": estimates,
        "stepwise": {
            "definition": "last round only, so the uncommunicated first round is excluded",
            "curve": step_rows,
            "crossing": step_result,
            "bootstrap": step_bootstrap,
            "fit": step_fit,
        },
        "alternate_conventions": alternates,
        "conventions_unavailable": skipped,
        "note": (
            "log R < 1 over a finite horizon means the existing incorrect lineage decays "
            "under this pooled convention; it is not a proof of an absorbing transition. "
            "pre_own is the frozen reference convention and measures b (1-p_nbr)^d per "
            "round; post_own additionally includes the agent's own pruning; literal is "
            "the paper's A_{t+1}/A_t over consecutive rounds, source lineages included. "
            "crossing.d_c is the published two-point interpolation and is only a point "
            "estimate when crossing.status is single_crossing; fit.d_c_fit is -a/s from "
            "least squares of log R = a + s d over the fit window, which defaults to "
            "'local' -- the measured degrees around the interpolated crossing, since "
            "log R(d) bends/plateaus at high d and a global window is biased by that; "
            "fit.fit_window_mode says whether the local window was used or the run had "
            "no crossing to anchor on ('estimable_fallback'). fit.slope_low_d is the "
            "log-linear slope over measured degrees d <= 8 regardless of the fit "
            "window, and is the field downstream analyses read as the measured "
            "per-neighbour pruning rate."
        ),
    }
    if first_round != 1:
        # Kept out of the default (first_round=1) report so its JSON stays
        # byte-identical to before this parameter existed.
        report["first_round"] = first_round
    (analysis_dir / f"critical_degree{suffix}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report


def _fit_window_argument(text: str):
    if text in ("local", "estimable", "all"):
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--fit-window takes 'local', 'estimable', 'all' or a |log R| bound"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--convention", choices=sorted(CONVENTIONS), default=DEFAULT_CONVENTION
    )
    parser.add_argument(
        "--fit-window",
        type=_fit_window_argument,
        default=DEFAULT_FIT_WINDOW,
        help=(
            "'local' (default: measured degrees around the interpolated crossing), "
            "'estimable', 'all', or a |log R| bound for the old global log-linear fit"
        ),
    )
    parser.add_argument(
        "--fit-local-k",
        type=int,
        default=DEFAULT_FIT_LOCAL_K,
        help="measured degrees to keep on each side of the crossing bracket for --fit-window local",
    )
    parser.add_argument(
        "--fit-local-bound",
        type=float,
        default=DEFAULT_FIT_LOCAL_BOUND,
        help="drop local-window points with |log R| above this bound, if >=4 points remain",
    )
    parser.add_argument(
        "--first-round",
        type=int,
        default=1,
        help=(
            "1-based round to start the cumulative pooled log R from (default 1: "
            "unchanged behaviour). 2 is the paper's delayed d_c -- round 1 carries "
            "no neighbour messages yet -- and matches references.delayed_dc; outputs "
            "for any value other than 1 are written under a '_t<N>' suffix so the "
            "default (first_round=1) outputs are never touched."
        ),
    )
    args = parser.parse_args(argv)
    report = scan(
        args.run_dir,
        draws=args.draws,
        seed=args.seed,
        convention=args.convention,
        first_round=args.first_round,
        fit_window=args.fit_window,
        fit_local_k=args.fit_local_k,
        fit_local_bound=args.fit_local_bound,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
