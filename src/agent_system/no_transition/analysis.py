"""Boundary-safe estimates and paired diagnostics; never infer a finite dc.

Wilson intervals describe independent task-seed replicates within a degree.
All resampling for degree effects keeps a task's entire degree vector together.
The logistic likelihood ratio is descriptive because rows are paired. Tests
use an exact paired endpoint test and (when estimable) cluster-robust slopes.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .config import threshold
from .environments import public_schedule, reference_batch


def wilson(successes: int, total: int, confidence: float = .95) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, zz = successes / total, z * z
    center = (p + zz / (2 * total)) / (1 + zz / total)
    half = z * math.sqrt(p * (1 - p) / total + zz / (4 * total * total)) / (1 + zz / total)
    return [0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == total else min(1.0, center + half)]


def bh_adjust(values: list[float | None]) -> list[float | None]:
    valid = sorted((p, i) for i, p in enumerate(values) if p is not None)
    answer = [None] * len(values)
    running = 1.0
    for rank in range(len(valid), 0, -1):
        p, index = valid[rank - 1]
        running = min(running, p * len(valid) / rank)
        answer[index] = running
    return answer


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(episodes: list[dict], config: dict) -> list[dict]:
    groups = defaultdict(list)
    conditions = {c["name"]: c for c in config["conditions"]}
    for e in episodes:
        groups[e["condition"], e["agent_type"], e["degree"]].append(e)
    rows = []
    for (condition, agent_type, degree), es in sorted(groups.items()):
        successes, n = sum(e["success"] for e in es), len(es)
        attempts = sum(e["api_attempts"] for e in es)
        tests = sum(e["total_tests"] for e in es)
        actions = sum(e["action_count"] if "action_count" in e else len(e["actions"]) for e in es)
        rows.append({"condition": condition, "task": es[0]["task"], "agent_type": agent_type,
                     "degree": degree, "episodes": n, "successes": successes,
                     "p_success": successes / n, "ci_low": wilson(successes, n)[0],
                     "ci_high": wilson(successes, n)[1],
                     "mean_rounds_used": mean(e["rounds_used"] for e in es),
                     "mean_unique_candidates_tested": mean(e["unique_candidates_tested"] for e in es),
                     "coverage_fraction": mean(e["unique_candidates_tested"] / e["candidates"] for e in es) if es[0]["candidates"] else None,
                     "duplicate_fraction": sum(e["duplicate_tests"] for e in es) / tests if tests else 0.0,
                     "invalid_action_rate": sum(e["invalid_actions"] for e in es) / actions if actions else 0.0,
                     "api_failure_rate": sum(e["api_failed_attempts"] for e in es) / attempts if attempts else None,
                     "api_error_episode_fraction": mean(bool(e["api_errors"]) for e in es),
                     "mean_maximum_depth": mean(e["maximum_depth"] for e in es),
                     "illegal_future_attempts": sum(e["illegal_future_attempts"] for e in es),
                     **threshold(conditions[condition])})
    return rows


def paired_endpoint(es: list[dict], degrees: list[int], rng, resamples: int) -> dict:
    low, high = min(degrees), max(degrees)
    by_seed = defaultdict(dict)
    for e in es:
        by_seed[e["task_seed"]][e["degree"]] = int(e["success"])
    paired = [v for v in by_seed.values() if low in v and high in v]
    if not paired:
        return {"status": "no_complete_pairs", "p_value": None}
    delta = np.array([v[high] - v[low] for v in paired], dtype=float)
    boot = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    plus, minus = int(sum(delta > 0)), int(sum(delta < 0))
    discordant = plus + minus
    exact_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(plus, minus) + 1)) / 2 ** discordant) if discordant else 1.0
    # Bonferroni marginal Wilson intervals give a conservative paired difference interval.
    a, b = wilson(sum(v[low] for v in paired), len(paired), .975), wilson(sum(v[high] for v in paired), len(paired), .975)
    return {"status": "estimated", "paired_seeds": len(paired), "delta_p": float(delta.mean()),
            "ci95_conservative_wilson": [max(-1., b[0] - a[1]), min(1., b[1] - a[0])],
            "ci95_paired_bootstrap": np.quantile(boot, [.025, .975]).tolist(),
            "bootstrap_degenerate": bool(np.ptp(boot) == 0),
            "p_value": exact_p, "test": "exact paired discordance (McNemar)",
            "discordant_pairs": discordant, "unpaired_seeds": len(by_seed) - len(paired)}


def logistic_fit(x: np.ndarray, y: np.ndarray) -> dict:
    if len(set(y.tolist())) == 1:
        return {"status": "not_identifiable_constant_outcome", "coefficient": None}
    if np.ptp(x) == 0:
        return {"status": "not_identifiable_constant_degree", "coefficient": None}
    positive, negative = x[y == 1], x[y == 0]
    if min(positive) >= max(negative) or max(positive) <= min(negative):
        return {"status": "not_identifiable_separation", "coefficient": None}
    scale = float(np.ptp(x))
    design = np.column_stack([np.ones(len(x)), x / scale])
    p0 = float(y.mean())
    beta = np.array([math.log(p0 / (1 - p0)), 0.])
    for _ in range(100):
        eta = design @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -700, 700)))
        h = design.T @ ((p * (1 - p))[:, None] * design)
        try:
            change = np.linalg.solve(h, design.T @ (y - p))
        except np.linalg.LinAlgError:
            return {"status": "not_identifiable_singular", "coefficient": None}
        # Newton step halving ensures monotone log-likelihood ascent.
        old_ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
        step = 1.0
        while step > 1e-8:
            new_eta = design @ (beta + step * change)
            if np.sum(y * new_eta - np.logaddexp(0, new_eta)) >= old_ll - 1e-10:
                break
            step *= .5
        beta += step * change
        if np.max(np.abs(step * change)) < 1e-8:
            break
    else:
        return {"status": "not_identifiable_nonconvergence", "coefficient": None}
    eta = design @ beta
    ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
    null_ll = float(np.sum(y * math.log(p0) + (1 - y) * math.log1p(-p0)))
    return {"status": "estimated", "coefficient": float(beta[1] / scale),
            "intercept": float(beta[0]), "constant_probability": p0,
            "log_likelihood_constant": null_ll, "log_likelihood_degree": ll,
            "likelihood_ratio_descriptive": max(0., 2 * (ll - null_ll)),
            "aic_constant": 2 - 2 * null_ll, "aic_degree": 4 - 2 * ll}


def degree_effect(es: list[dict], rng, resamples: int) -> dict:
    x = np.array([e["degree"] for e in es], dtype=float)
    y = np.array([e["success"] for e in es], dtype=float)
    result = logistic_fit(x, y)
    result["p_value"] = None
    if result["status"] != "estimated":
        return result
    seeds = np.array([e["task_seed"] for e in es])
    unique = np.unique(seeds)
    groups = [np.flatnonzero(seeds == s) for s in unique]
    coefs = []
    for _ in range(resamples):
        sampled = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        fit = logistic_fit(x[sampled], y[sampled])
        if fit["status"] == "estimated":
            coefs.append(fit["coefficient"])
    quantiles = np.quantile(coefs, [.025, .975]).tolist() if coefs else None
    degenerate = bool(coefs and np.ptp(coefs) < 1e-12)
    reliable = len(coefs) >= .95 * resamples and not degenerate and len(unique) >= 6
    result.update(bootstrap_requested=resamples, bootstrap_estimable=len(coefs), seed_clusters=len(unique),
                  ci95_cluster_bootstrap=quantiles if reliable else None,
                  conditional_estimable_bootstrap_quantiles=quantiles,
                  bootstrap_interval_status=("estimated" if reliable else "unreliable: nonidentifiable resamples, degenerate distribution, or fewer than six clusters"),
                  bootstrap_degenerate=degenerate)
    # Cluster sandwich uncertainty preserves repeated observations of each seed.
    design = np.column_stack([np.ones(len(x)), x])
    p = 1 / (1 + np.exp(-(result["intercept"] + result["coefficient"] * x)))
    bread = np.linalg.inv(design.T @ ((p * (1 - p))[:, None] * design))
    scores = [design[g].T @ (y[g] - p[g]) for g in groups]
    # Sum influence outer-products to avoid cancellation in sandwich products
    # when every task has exactly the same outcome at all degrees.
    influences = [bread @ score for score in scores]
    covariance = sum(np.outer(v, v) for v in influences)
    if len(groups) > 1:
        covariance *= len(groups) / (len(groups) - 1)
    se = math.sqrt(max(0., float(covariance[1, 1])))
    result["cluster_robust_se"] = se
    if len(groups) >= 6 and se > 1e-12 and not result["bootstrap_degenerate"]:
        result["p_value"] = math.erfc(abs(result["coefficient"] / se) / math.sqrt(2))
        result["test"] = "asymptotic cluster-robust Wald; small cluster counts are exploratory"
    else:
        result["test"] = "no calibrated slope test: too few clusters or degenerate paired score"
    return result


def diagnostics(episodes: list[dict], config: dict) -> list[dict]:
    groups = defaultdict(list)
    for e in episodes:
        groups[e["condition"], e["agent_type"]].append(e)
    rng = np.random.default_rng(config["analysis_seed"])
    rows = []
    for (condition, kind), es in sorted(groups.items()):
        rows.append({"condition": condition, "agent_type": kind,
                     "endpoint": paired_endpoint(es, config["degrees"], rng, config["bootstrap_samples"]),
                     "regression": degree_effect(es, rng, config["bootstrap_samples"])})
    tests = [r[key] for r in rows for key in ("endpoint", "regression")]
    for test, adjusted in zip(tests, bh_adjust([t["p_value"] for t in tests])):
        test["p_bh_fdr"] = adjusted
    return rows


def verify_invariants(episodes: list[dict], config: dict) -> dict:
    checks = []
    paired = defaultdict(list)
    for e in episodes:
        assert all(len(v) == e["degree"] for v in e["graph"])
        assert e["num_agents"] == config["agents"]
        paired[e["condition"], e["agent_type"], e["task_seed"]].append(e)
        if e["task"] == "chain":
            depth = [0] + e["depth_after_each_round"]
            assert all(0 <= b - a <= 1 for a, b in zip(depth, depth[1:]))
            assert e["maximum_depth"] <= e["rounds_used"]
            if e["horizon"] < e["chain_length"]:
                assert not e["success"], "structurally impossible chain completion"
            if e["agent_type"] == "deterministic" and e["horizon"] >= e["chain_length"]:
                assert e["success"]
        else:
            valid = [a["candidate_tested"] for a in e["actions"] if not a["invalid"]]
            assert e["success"] == (e["hidden_target"] in valid)
            assert e["unique_candidates_tested"] == len(set(valid))
            assert e["duplicate_tests"] == len(valid) - len(set(valid))
            if e["agent_type"] == "deterministic":
                assert e["unique_candidates_tested"] == min(e["budget"], e["candidates"])
                assert e["duplicate_tests"] == max(0, e["budget"] - e["candidates"])
                if e["identity"]["condition"].get("coverage_passes", 1) > 1:
                    order = public_schedule(e["identity"]["condition"], e["task_seed"])
                    assert valid == order[:e["budget"]]
                    assert all(set(order[i:i + e["candidates"]]) == set(range(e["candidates"]))
                               for i in range(0, len(order), e["candidates"]))
    for group in paired.values():
        assert len({(e["hidden_target"], e["hidden_chain_fingerprint"], e["budget"], e["horizon"], e["graph_seed"]) for e in group}) == 1
        if group[0]["agent_type"] == "deterministic":
            assert len({e["success"] for e in group}) == 1
    checks.extend(["graph degree", "paired tasks/resources", "needle success iff target tested",
                   "coverage/duplicate accounting", "chain one level per round",
                   "chain T<C impossible", "deterministic sufficient horizon", "paired reference invariance"])
    return {"status": "passed", "episodes_checked": len(episodes), "checks": checks,
            "note": "Structural assertions only; empirical GPT agreement is not a pass/fail gate."}


def load_episodes(run_dir: Path) -> list[dict]:
    archived = run_dir / "episodes.jsonl.gz"
    if archived.exists():
        with gzip.open(archived, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    return [json.loads(p.read_text()) for p in sorted((run_dir / "episodes").glob("*/episode_result.json"))]


def needle_behavior(episodes: list[dict]) -> dict:
    """Trace-only diagnosis. Never changes an outcome, policy or sampling plan."""
    rows = []
    for e in episodes:
        if e["agent_type"] != "llm" or e["task"] != "needle":
            continue
        order = public_schedule(e["identity"]["condition"], e["task_seed"])
        batch = e["budget"] // (e["num_agents"] * e["horizon"])
        deviations = []
        for r in range(1, e["rounds_used"] + 1):
            for agent in range(e["num_agents"]):
                actual = [a["candidate_tested"] for a in e["actions"]
                          if a["agent_id"] == agent and a["round"] == r]
                scheduled = reference_batch(order, e["num_agents"], agent, r, batch)
                if sorted(actual, key=str) != sorted(scheduled, key=str):
                    deviations.append({"round": r, "agent_id": agent, "scheduled": scheduled,
                                       "actual": actual, "target_was_in_omitted_schedule":
                                       e["hidden_target"] in scheduled and e["hidden_target"] not in actual})
        tested = {a["candidate_tested"] for a in e["actions"] if not a["invalid"]}
        rows.append({"episode_id": e["episode_id"], "condition": e["condition"], "seed": e["task_seed"],
                     "degree": e["degree"], "success": e["success"], "hidden_target": e["hidden_target"],
                     "unique_coverage": len(tested), "total_tests": e["total_tests"],
                     "duplicate_tests": e["duplicate_tests"], "changed_batches": deviations,
                     "miss_explained_by_omitted_schedule": not e["success"] and
                         any(d["target_was_in_omitted_schedule"] for d in deviations)})
    high = [r for r in rows if r["condition"] == "Needle-High"]
    changed = [d for r in high for d in r["changed_batches"]]
    return {"episodes": rows, "high_resource": {
        "episodes": len(high), "successes": sum(r["success"] for r in high),
        "changed_batches": len(changed),
        "all_changed_batches_before_neighbor_evidence": bool(changed) and all(d["round"] == 1 for d in changed),
        "failed_episodes": sum(not r["success"] for r in high),
        "failed_episodes_with_target_omitted_from_schedule": sum(r["miss_explained_by_omitted_schedule"] for r in high),
        "interpretation": "Legal departures from public work assignments can reduce unique coverage. First-round departures occur before any neighbor evidence and cannot be attributed to connectivity from these samples. No outcomes were excluded or rerun."}}


def projection(episodes: list[dict], config: dict, full: dict, execution: dict) -> dict:
    llm = [e for e in episodes if e["agent_type"] == "llm"]
    det = [e for e in episodes if e["agent_type"] == "deterministic"]
    price = config["pricing_usd_per_million"]
    def cost(e):
        return ((e["prompt_tokens"] - e["cached_prompt_tokens"]) * price["input"]
                + e["cached_prompt_tokens"] * price["cached_input"]
                + e["completion_tokens"] * price["output"]) / 1e6
    actual_calls = sum(e["api_attempts"] for e in llm)
    projected_calls, projected_seconds, projected_cost = 0., 0., 0.
    wave_seconds, context_cost = 0., 0.
    for c in full["conditions"]:
        sample = [e for e in llm if e["condition"] == c["name"]]
        if not sample:
            continue
        count = len(full["llm_seeds"]) * len(full["degrees"])
        ratio = full["agents"] / config["agents"]
        projected_calls += count * mean(e["api_attempts"] for e in sample) * ratio
        projected_seconds += count * mean(e["wall_seconds"] for e in sample) * ratio
        projected_cost += count * mean(cost(e) for e in sample) * ratio
        wave_ratio = (math.ceil(full["agents"] / full["llm"]["concurrency"])
                      / math.ceil(config["agents"] / config["llm"]["concurrency"]))
        wave_seconds += count * mean(e["wall_seconds"] for e in sample) * wave_ratio
        # Extrapolate observed prompt growth per neighbor, condition by condition.
        valid = [e for e in sample if e["successful_api_calls"]]
        if valid:
            xs = np.array([e["degree"] for e in valid], dtype=float)
            ys = np.array([e["prompt_tokens"] / e["successful_api_calls"] for e in valid])
            slope, intercept = np.polyfit(xs, ys, 1) if np.ptp(xs) else (0., float(ys.mean()))
            slope = max(0., float(slope))
            output_per_call = mean(e["completion_tokens"] / e["successful_api_calls"] for e in valid)
            calls_per_cell = len(full["llm_seeds"]) * mean(e["successful_api_calls"] for e in valid) * ratio
            for degree in full["degrees"]:
                context_cost += calls_per_cell * (max(0., intercept + slope * degree) * price["input"]
                                                + output_per_call * price["output"]) / 1e6
    runtime = execution.get("llm_wall_seconds", sum(e["wall_seconds"] for e in llm))
    timed = sum(e["api_timed_attempts"] for e in llm)
    return {"deterministic_episodes": len(det), "llm_episodes": len(llm),
            "deterministic_seconds": execution.get("deterministic_wall_seconds", sum(e["wall_seconds"] for e in det)),
            "llm_seconds": runtime, "pilot_seconds": execution.get("execution_wall_seconds"),
            "api_calls": actual_calls, "successful_api_calls": sum(e["successful_api_calls"] for e in llm),
            "failed_attempts": sum(e["api_failed_attempts"] for e in llm),
            "retries": sum(e["retry_count"] for e in llm),
            "mean_http_latency_seconds": sum(e["api_service_seconds"] for e in llm) / timed if timed else None,
            "episodes_per_minute": len(llm) * 60 / runtime if runtime else None,
            "calls_per_episode": actual_calls / len(llm) if llm else None,
            "prompt_tokens": sum(e["prompt_tokens"] for e in llm),
            "completion_tokens": sum(e["completion_tokens"] for e in llm),
            "estimated_pilot_cost_usd": sum(cost(e) for e in llm),
            "full_llm_episodes": len(full["conditions"]) * len(full["degrees"]) * len(full["llm_seeds"]),
            "estimated_full_calls": projected_calls if llm else None,
            "estimated_full_seconds": projected_seconds if llm else None,
            "estimated_full_cost_usd": projected_cost if llm else None,
            "estimated_full_seconds_round_waves": wave_seconds if llm else None,
            "estimated_full_cost_usd_degree_context": float(context_cost) if llm else None,
            "full_executed": config["mode"] == "full",
            "execution_status": execution.get("status", "in_progress"),
            "blocker": execution.get("blocker"),
            "projection_assumptions": "Baseline scales measured per-condition throughput by agents and episodes at unchanged concurrency. Round-waves alternative uses ceil(N/concurrency) for barriers. Degree-context cost extrapolates a per-condition linear fit of observed input tokens/call against degree, at uncached input rates. Larger messages, rate limits, provider load and changed stopping rounds can alter all estimates; none is a billing statement."}
