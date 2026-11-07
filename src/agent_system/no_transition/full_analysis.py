"""Streaming full-run analysis: preserve raw traces without retaining them in RAM."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from ..analysis.summarize import write_csv
from .analysis import (aggregate, bh_adjust, diagnostics, mean, paired_endpoint,
                       projection, verify_invariants, wilson)
from .config import ROOT, expected_episode_count, fingerprint, resolved_llm_settings
from .environments import public_schedule, reference_batch
from .preservation import guard_full_paths, verify_full_design, verify_protected_pilots
from .report import write_json


def iter_episodes(path: Path):
    archive = path / "episodes.jsonl.gz"
    if archive.exists():
        with gzip.open(archive, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        for p in sorted((path / "episodes").glob("*/episode_result.json")):
            yield json.loads(p.read_text())


def compact(e: dict) -> dict:
    return {**{k: v for k, v in e.items() if k not in
               {"actions", "messages", "graph", "identity", "rounds"}}, "action_count": len(e["actions"])}


def deterministic_gate(run_dir: Path, config: dict) -> dict:
    rows, seen, paired = [], set(), defaultdict(set)
    expected = {(c["name"], s, d) for c in config["conditions"]
                for s in config["deterministic_seeds"] for d in config["degrees"]}
    for e in iter_episodes(run_dir):
        if e["agent_type"] != "deterministic":
            continue
        key = (e["condition"], e["task_seed"], e["degree"])
        assert key in expected and key not in seen
        assert e["identity"]["config_digest"] == fingerprint(config)
        verify_invariants([e], config)
        assert e["invalid_actions"] == 0 and e["api_attempts"] == 0
        if e["task"] == "needle":
            assert e["total_tests"] == e["budget"]
            if e["condition"] == "Needle-Low":
                assert e["unique_candidates_tested"] / e["candidates"] == .04
        paired[e["condition"], e["task_seed"]].add((e["success"], e["hidden_target"], e["hidden_chain_fingerprint"]))
        seen.add(key)
        rows.append(compact(e))
    if seen != expected:
        raise ValueError(f"deterministic gate incomplete: {len(seen)}/{len(expected)}; GPT prohibited")
    assert all(len(x) == 1 for x in paired.values())
    summary = aggregate(rows, config)
    return {"status": "passed", "episodes": len(rows), "paired_tasks_verified": len(paired),
            "low_baseline": .04,
            "low_baseline_within_wilson_interval": all(r["ci_low"] <= .04 <= r["ci_high"] for r in summary if r["condition"] == "Needle-Low"),
            "note": "Low empirical sampling noise is reported; exact .04 coverage and paired invariance are structural gates. High success and Chain 0/1 are enforced."}


def needle_record(e: dict) -> dict:
    condition = e["identity"]["condition"]
    order = public_schedule(condition, e["task_seed"])
    matched = target_batches = followed_target_batches = target_slot_followed = 0
    neighbor_retests = negative_retests = positive_retests = eligible_tests = 0
    departures = []
    actions_by_batch, received = defaultdict(list), defaultdict(dict)
    for a in e["actions"]:
        actions_by_batch[a["round"], a["agent_id"]].append(a)
    for message in e["messages"]:
        record = message["record"]
        if record["candidate_result"] in {"TARGET", "NOT TARGET"}:
            received[message["received_round"], message["receiver"]][record["candidate_tested"]] = record["candidate_result"]
    for r in range(1, e["horizon"] + 1):
        for a in range(e["num_agents"]):
            scheduled = reference_batch(order, e["num_agents"], a, r, condition["batch_size"])
            actual = [x["candidate_tested"] for x in actions_by_batch[r, a] if not x["invalid"]]
            common = sum((Counter(scheduled) & Counter(actual)).values())
            matched += common
            if common != len(scheduled):
                departures.append({"round": r, "agent_id": a, "scheduled": scheduled,
                                   "actual_valid": actual, "replaced": len(scheduled) - common})
            if e["hidden_target"] in scheduled:
                target_batches += 1
                followed_target_batches += common == len(scheduled)
                target_slot_followed += e["hidden_target"] in actual
            evidence = received[r, a]
            if evidence:
                eligible_tests += len(actual)
            neighbor_retests += sum(x in evidence for x in actual)
            negative_retests += sum(evidence.get(x) == "NOT TARGET" for x in actual)
            positive_retests += sum(evidence.get(x) == "TARGET" for x in actual)
    return {"episode_id": e["episode_id"], "condition": e["condition"], "agent_type": e["agent_type"],
            "degree": e["degree"], "seed": e["task_seed"], "success": e["success"],
            "nominal_slots": e["budget"], "total_tests": e["total_tests"],
            "unique_candidates_tested": e["unique_candidates_tested"],
            "coverage_fraction": e["unique_candidates_tested"] / e["candidates"],
            "duplicate_tests": e["duplicate_tests"],
            "duplicate_fraction": e["duplicate_tests"] / e["total_tests"] if e["total_tests"] else 0.,
            "scheduled_candidates_followed": matched, "scheduled_candidates_replaced": e["budget"] - matched,
            "schedule_departure_fraction": (e["budget"] - matched) / e["budget"],
            "target_in_nominal_schedule": e["hidden_target"] in order[:e["budget"]],
            "target_containing_scheduled_batches": target_batches,
            "fully_followed_target_batches": followed_target_batches,
            "target_candidate_followed_in_its_batch": target_slot_followed,
            "target_actually_tested": e["target_found"], "round_of_target_discovery": e["round_of_success"],
            "first_round_coverage": e["rounds"][0]["unique_coverage"] / e["candidates"],
            "second_pass_rescued_target": e["condition"] == "Needle-High" and e["round_of_success"] == 2,
            "neighbor_reported_retests": neighbor_retests,
            "neighbor_known_negative_retests": negative_retests,
            "neighbor_known_target_retests": positive_retests,
            "valid_tests_with_neighbor_evidence": eligible_tests,
            "neighbor_informed_duplicate_fraction": neighbor_retests / eligible_tests if eligible_tests else None,
            "departures": departures}


def mechanism_cells(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for e in records:
        groups[e["condition"], e["agent_type"], e["degree"]].append(e)
    answer = []
    for (condition, kind, degree), es in sorted(groups.items()):
        target_batches = sum(e["target_containing_scheduled_batches"] for e in es)
        eligible = sum(e["valid_tests_with_neighbor_evidence"] for e in es)
        successes = sum(e["success"] for e in es)
        answer.append({"condition": condition, "agent_type": kind, "degree": degree, "episodes": len(es),
            **{"mean_" + key: mean(e[key] for e in es) for key in
               ("unique_candidates_tested", "coverage_fraction", "duplicate_tests", "duplicate_fraction", "schedule_departure_fraction")},
            "target_batch_followed_fraction": sum(e["fully_followed_target_batches"] for e in es) / target_batches if target_batches else None,
            "target_tested_episode_fraction": successes / len(es),
            "neighbor_reported_retests": sum(e["neighbor_reported_retests"] for e in es),
            "neighbor_known_negative_retests": sum(e["neighbor_known_negative_retests"] for e in es),
            "neighbor_known_target_retests": sum(e["neighbor_known_target_retests"] for e in es),
            "neighbor_informed_duplicate_fraction": sum(e["neighbor_reported_retests"] for e in es) / eligible if eligible else None,
            "round1_success_fraction_of_all_episodes": mean(e["round_of_target_discovery"] == 1 for e in es),
            "round2_success_fraction_of_all_episodes": mean(e["round_of_target_discovery"] == 2 for e in es),
            "round1_fraction_of_successes": sum(e["round_of_target_discovery"] == 1 for e in es) / successes if successes else None,
            "round2_fraction_of_successes": sum(e["round_of_target_discovery"] == 2 for e in es) / successes if successes else None,
            "second_pass_rescue_fraction": mean(e["second_pass_rescued_target"] for e in es)})
    return answer


def crossover_classification(rows: list[dict], config: dict) -> str:
    ps = [r["p_success"] for r in sorted(rows, key=lambda r: r["degree"])]
    low, high = config["crossover_thresholds"]["low_max"], config["crossover_thresholds"]["high_min"]
    if not ps:
        return "NO-CLEAR-PATTERN"
    if max(ps) <= low:
        return "LOW-SATURATED"
    if min(ps) >= high:
        return "HIGH-SATURATED"
    if any(a <= low and b >= high for i, a in enumerate(ps) for b in ps[i+1:]):
        return "INTERIOR-CROSSOVER"
    return "NO-CLEAR-PATTERN"


def crossover_diagnostics(episodes: list[dict], rows: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for e in episodes:
        groups[e["condition"], e["agent_type"]].append(e)
    rng = np.random.default_rng(config["analysis_seed"] + 1)
    pairs, patterns = [], []
    for (condition, kind), es in sorted(groups.items()):
        for low, high in itertools.combinations(config["degrees"], 2):
            pair = paired_endpoint([e for e in es if e["degree"] in {low, high}], [low, high], rng, config["bootstrap_samples"])
            by_seed = defaultdict(dict)
            for e in es:
                if e["degree"] in {low, high}:
                    by_seed[e["task_seed"]][e["degree"]] = e["success"]
            complete = [v for v in by_seed.values() if low in v and high in v]
            pair.update(condition=condition, agent_type=kind, low_degree=low, high_degree=high,
                        increasing_pair_fraction=mean(not v[low] and v[high] for v in complete))
            pairs.append(pair)
    for pair, q in zip(pairs, bh_adjust([p["p_value"] for p in pairs])):
        pair["exploratory_pair_bh_q"] = q
    for condition, kind in sorted(groups):
        cell = [r for r in rows if r["condition"] == condition and r["agent_type"] == kind]
        probabilities = {r["degree"]: r["p_success"] for r in cell}
        credible = [p for p in pairs if p["condition"] == condition and p["agent_type"] == kind
                    and probabilities.get(p["low_degree"], 1) <= .2 and probabilities.get(p["high_degree"], 0) >= .8
                    and p["increasing_pair_fraction"] >= .75 and p["exploratory_pair_bh_q"] is not None
                    and p["exploratory_pair_bh_q"] <= .05]
        patterns.append({"condition": condition, "agent_type": kind,
            "classification": crossover_classification(cell, config), "credible_paired_crossover": bool(credible),
            "credible_brackets": [[p["low_degree"], p["high_degree"]] for p in credible],
            "observed_range": max(probabilities.values()) - min(probabilities.values()), "finite_dc_estimated": False})
    return patterns, pairs


def scientific_conclusion(patterns: list[dict], complete: bool, structural_ok: bool, reference_ok: bool) -> str:
    if not structural_ok or not reference_ok:
        return "INVALID DUE TO IMPLEMENTATION/PROTOCOL FAILURE"
    if not complete:
        return "INCONCLUSIVE"
    if any(p["credible_paired_crossover"] for p in patterns):
        return "EVIDENCE OF DEGREE-DEPENDENT TRANSITION"
    expected = {"Needle-Low": "LOW-SATURATED", "Needle-High": "HIGH-SATURATED",
                "Chain-Short-Horizon": "LOW-SATURATED", "Chain-Sufficient-Horizon": "HIGH-SATURATED"}
    if len(patterns) == 8 and all(p["classification"] == expected[p["condition"]] and p["observed_range"] <= .2 for p in patterns):
        return "CONSISTENT WITH NO-TRANSITION THEORY"
    return "INCONCLUSIVE"


def historical_comparison() -> tuple[list[dict], dict]:
    folder = ROOT / "data/needle_chain/historical_hierarchical"
    manifest = json.loads((folder / "source_manifest.json").read_text())
    groups = defaultdict(list)
    with gzip.open(folder / "episodes.jsonl.gz", "rt") as f:
        for line in f:
            e = json.loads(line)
            groups[e["d"]].append(e)
    rows = []
    for d, es in sorted(groups.items()):
        extinct = sum(e["transitions"][-1]["A_post"] == 0 for e in es)
        ci = wilson(extinct, len(es))
        rows.append({"degree": d, "agents": 40, "rho": d / 39, "episodes": len(es),
                     "extinct": extinct, "p_extinction": extinct / len(es), "ci_low": ci[0], "ci_high": ci[1],
                     "mean_final_incorrect": mean(e["transitions"][-1]["A_post"] for e in es),
                     "actual_task_success_available": False})
    return rows, manifest


def analyze_full(run_dir: Path, output: Path, config: dict, execution: dict) -> dict:
    guard_full_paths(run_dir, output)
    protection, design = verify_protected_pilots(), verify_full_design()
    output.mkdir(parents=True, exist_ok=True)
    expected = {(c["name"], k, s, d) for c in config["conditions"] for k in ("deterministic", "llm")
                for s in config["deterministic_seeds" if k == "deterministic" else "llm_seeds"] for d in config["degrees"]}
    seen, compact_rows, mechanisms, paired = set(), [], [], defaultdict(set)
    source_shas, models = set(), set()
    temp = output / "episodes.jsonl.gz.tmp"
    with temp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as archive:
        for e in iter_episodes(run_dir):
            key = (e["condition"], e["agent_type"], e["task_seed"], e["degree"])
            assert key in expected and key not in seen
            assert e["identity"]["config_digest"] == fingerprint(config)
            verify_invariants([e], config)
            paired[e["condition"], e["agent_type"], e["task_seed"]].add((e["hidden_target"], e["hidden_chain_fingerprint"], e["budget"], e["horizon"], e["graph_seed"]))
            seen.add(key)
            source_shas.add(e["git_commit_sha"])
            models.update(e["response_models"])
            compact_rows.append(compact(e))
            if e["task"] == "needle":
                mechanisms.append(needle_record(e))
            archive.write((json.dumps(e, sort_keys=True, allow_nan=False) + "\n").encode())
    assert all(len(values) == 1 for values in paired.values())
    assert models <= {resolved_llm_settings(config)["model"]}, "unexpected response-model alias requires verification"
    temp.replace(output / "episodes.jsonl.gz")
    complete = seen == expected
    rows = aggregate(compact_rows, config)
    diag = diagnostics(compact_rows, config)
    patterns, pairs = crossover_diagnostics(compact_rows, rows, config)
    reference = (deterministic_gate(run_dir, config) if sum(e['agent_type']=='deterministic' for e in compact_rows)
                 == expected_episode_count(config, 'deterministic') else {'status':'incomplete'})
    mechanism_rows = mechanism_cells(mechanisms)
    runtime = projection(compact_rows, config, {**config, "conditions": [], "llm_seeds": []}, execution)
    runtime = {k: v for k, v in runtime.items() if not k.startswith("estimated_full") and k not in {"full_llm_episodes", "projection_assumptions"}}
    runtime["cached_prompt_tokens"] = sum(e["cached_prompt_tokens"] for e in compact_rows)
    runtime['estimated_cost_usd'] = runtime.pop('estimated_pilot_cost_usd')
    runtime['execution_seconds'] = runtime.pop('pilot_seconds')
    conclusion = scientific_conclusion(patterns, complete, True, reference["status"] == "passed")
    historical, historical_manifest = historical_comparison()
    audit = {"status": "passed" if complete else "incomplete", "episodes": len(seen), "unique_episode_ids": len(seen),
             "expected_episodes": len(expected), "missing_episodes": sorted(expected - seen),
             "response_models": sorted(models), "source_shas": sorted(source_shas),
             "protected_pilots": protection, "frozen_design": design,
             "episode_archive_sha256": hashlib.sha256((output / "episodes.jsonl.gz").read_bytes()).hexdigest(),
             "api_errors": sum(e["api_errors"] for e in compact_rows), "retries": sum(e["retry_count"] for e in compact_rows)}
    freeze = json.loads((ROOT / "docs/needle_chain/full_v2_freeze.json").read_text())
    freeze["FULL_V2_PREREGISTRATION_SHA"] = execution.get("source_sha") or next(iter(source_shas))
    if complete:
        from .full_audit import audit_calls
        audit['provider_calls'] = audit_calls(run_dir, output, config, compact_rows)
        assert source_shas == {freeze['FULL_V2_PREREGISTRATION_SHA']}
        write_json(output/'validation.json', {'status':'passed', 'episodes_checked':len(seen),
            'structural_invariants': 'passed', 'paired_tasks':'passed', 'deterministic_gate':reference,
            'protected_pilots':protection, 'frozen_design':design, 'provider_calls':audit['provider_calls'],
            'deviations':[], 'visual_inspection':'pending', 'final_local_tests':'pending'})
    for name, value in (("config", config), ("provider_settings", resolved_llm_settings(config)),
        ("diagnostics", diag), ("crossover", patterns), ("exploratory_degree_pairs", pairs),
        ("runtime", runtime), ("execution", execution), ("deterministic_gate", reference),
        ("data_audit", audit), ("full_v2_freeze", freeze), ("historical_provenance", historical_manifest),
        ("completion_status", {"complete": complete, "expected_episodes": len(expected), "completed_episodes": len(seen)}),
        ("conclusion", {"classification": conclusion, "no_transition_proven": False,
                        "scope": "Finite preregistered degree grid, resources and policies only"})):
        write_json(output / (name + ".json"), value)
    write_csv(output / "by_degree.csv", rows)
    write_csv(output / "needle_mechanisms.csv", mechanism_rows)
    write_csv(output / "needle_episodes.csv", [{k: v for k, v in r.items() if k != "departures"} for r in mechanisms])
    with gzip.open(output / "needle_departures.jsonl.gz", "wt") as f:
        for r in mechanisms:
            if r["departures"]:
                f.write(json.dumps({"episode_id": r["episode_id"], "departures": r["departures"]}, sort_keys=True) + "\n")
    write_csv(output / "historical_hierarchical.csv", historical)
    from .full_plots import plot_full
    plot_full(rows, compact_rows, mechanism_rows, historical, output)
    provenance_path = output / "hosted_provenance.json"
    if not provenance_path.exists() and (run_dir / "hosted_provenance.json").exists():
        write_json(provenance_path, json.loads((run_dir / "hosted_provenance.json").read_text()))
    from .full_report import render_full
    (output / "full_report.md").write_text(render_full(config, rows, mechanism_rows, compact_rows, diag,
        patterns, historical, runtime, audit, conclusion, freeze["FULL_V2_PREREGISTRATION_SHA"]))
    return {"complete": complete, "episodes": len(seen), "expected_episodes": len(expected),
            "rows": rows, "runtime": runtime, "diagnostics": diag, "classification": conclusion}
