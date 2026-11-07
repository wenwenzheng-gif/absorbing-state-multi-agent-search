"""Separate v2 artifacts and a read-only comparison to the frozen v1 pilot."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ..analysis.summarize import write_csv
from .analysis import aggregate, diagnostics, load_episodes, mean, projection, verify_invariants
from .config import ROOT, expected_episode_count, episode_seeds, resolved_llm_settings
from .environments import public_schedule, reference_batch
from .plots import plot_all, STYLES
from .preservation import guard_v2_paths, verify_frozen_v1
from .report import _archive, write_json


def coverage_records(episodes: list[dict], version: str) -> list[dict]:
    rows = []
    for e in episodes:
        if e["condition"] != "Needle-High":
            continue
        condition = e["identity"]["condition"]
        order = public_schedule(condition, e["task_seed"])
        matched, changed, batches = 0, [], 0
        for round_number in range(1, e["horizon"] + 1):
            for agent in range(e["num_agents"]):
                scheduled = reference_batch(order, e["num_agents"], agent, round_number, condition["batch_size"])
                actions = [a for a in e["actions"] if a["round"] == round_number and a["agent_id"] == agent]
                actual = [a["candidate_tested"] for a in actions if not a["invalid"]]
                count = sum((Counter(scheduled) & Counter(actual)).values())
                matched += count
                batches += 1
                if count != len(scheduled):
                    changed.append({"round": round_number, "agent_id": agent, "scheduled": scheduled,
                                    "actual_valid": actual, "matched": count,
                                    "replaced": len(scheduled) - count})
        target_actions = [a for a in e["actions"] if a["candidate_result"] == "TARGET"]
        rows.append({"version": version, "agent_type": e["agent_type"], "episode_id": e["episode_id"],
                     "degree": e["degree"], "seed": e["task_seed"], "success": e["success"],
                     "nominal_capacity": e["budget"], "total_tests": e["total_tests"],
                     "unique_candidates_tested": e["unique_candidates_tested"],
                     "coverage_fraction": e["unique_candidates_tested"] / e["candidates"],
                     "p_coverage": e["unique_candidates_tested"] / e["candidates"],
                     "duplicate_count": e["duplicate_tests"],
                     "duplicate_fraction": e["duplicate_tests"] / e["total_tests"] if e["total_tests"] else 0.,
                     "scheduled_candidates_followed": matched,
                     "scheduled_candidates_replaced": e["budget"] - matched,
                     "schedule_departure_fraction": (e["budget"] - matched) / e["budget"],
                     "changed_batches": changed, "total_batches": batches,
                     "target_in_nominal_schedule": e["hidden_target"] in order[:e["budget"]],
                     "nominal_target_occurrences": order[:e["budget"]].count(e["hidden_target"]),
                     "target_actually_tested": bool(target_actions),
                     "round_target_tested": min((a["round"] for a in target_actions), default=None),
                     "invalid_actions": e["invalid_actions"], "api_errors": e["api_errors"]})
    return rows


def comparison_table(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[r["version"], r["agent_type"], r["degree"]].append(r)
    return [{"version": version, "agent_type": kind, "degree": degree, "episodes": len(es),
             "p_success": mean(e["success"] for e in es),
             "mean_unique_candidates": mean(e["unique_candidates_tested"] for e in es),
             "mean_unique_coverage": mean(e["coverage_fraction"] for e in es),
             "mean_duplicate_fraction": mean(e["duplicate_fraction"] for e in es),
             "mean_schedule_departure_fraction": mean(e["schedule_departure_fraction"] for e in es)}
            for (version, kind, degree), es in sorted(groups.items())]


def observed_findings(episodes: list[dict]) -> dict:
    high = [e for e in episodes if e["condition"] == "Needle-High" and e["agent_type"] == "llm"]
    records = coverage_records(high, "v2")
    by_id = {e["episode_id"]: e for e in high}
    changes = [(r, batch) for r in records for batch in r["changed_batches"]]
    known_negative_retests = 0
    for r, batch in changes:
        e = by_id[r["episode_id"]]
        negatives = {m["record"]["candidate_tested"] for m in e["messages"]
                     if m["receiver"] == batch["agent_id"] and m["received_round"] <= batch["round"]
                     and m["record"]["candidate_result"] == "NOT TARGET"}
        known_negative_retests += sum(x in negatives for x in batch["actual_valid"])
    return {"llm_high_episodes": len(high), "successes": sum(e["success"] for e in high),
            "first_round_full_coverage": sum(e["rounds"][0]["unique_coverage"] == 48 for e in high),
            "first_round_target_discovery": sum(e["round_of_success"] == 1 for e in high),
            "first_round_changed_batches": sum(b["round"] == 1 for _, b in changes),
            "changed_batches": len(changes),
            "changed_batch_degrees": sorted({r["degree"] for r, _ in changes}),
            "changed_batch_rounds": sorted({b["round"] for _, b in changes}),
            "changed_batch_slots": sum(len(b["actual_valid"]) for _, b in changes),
            "neighbor_known_negative_tests_in_changed_batches": known_negative_retests,
            "observed_second_pass_target_rescues": sum(e["round_of_success"] == 2 for e in high),
            "causal_limit": "Resource slack, schedule and guidance changed jointly; this pilot does not isolate the effect of extra slots or prove their necessity."}


def readiness(rows: list[dict], config: dict, complete: bool, structural_status: str) -> dict:
    high = [r for r in rows if r["condition"] == "Needle-High" and r["agent_type"] == "llm"]
    det = [r for r in rows if r["agent_type"] == "deterministic"]
    deterministic_ok = len(det) == 4 * len(config["degrees"]) and all(
        (r["ci_low"] <= .04 <= r["ci_high"] if r["condition"] == "Needle-Low" else
         r["p_success"] == (0. if r["condition"] == "Chain-Short-Horizon" else 1.)) for r in det)
    contrast = max((r["p_success"] for r in high), default=0) - min((r["p_success"] for r in high), default=0)
    poor_coverage = any(r["coverage_fraction"] < .9 for r in high)
    high_success = len(high) == len(config["degrees"]) and all(r["p_success"] >= .8 for r in high)
    reasons = []
    if structural_status != "passed" or (complete and not deterministic_ok) or poor_coverage:
        status = "NOT READY"
        reasons.append("Structural/reference checks failed or GPT high-resource mean coverage was below .9 at a degree.")
    elif not complete or not high_success or contrast >= .5:
        status = "INCONCLUSIVE"
        reasons.append("Incomplete data, insufficient high-success evidence, or a large degree contrast requires review.")
    else:
        status = "READY FOR FULL RUN"
        reasons.append("Structural and deterministic checks passed; coverage >=.9 and success >=.8 at every degree, with no large observed crossover.")
    return {"status": status, "reasons": reasons, "deterministic_theory_checks_passed": deterministic_ok,
            "observed_success_range": contrast, "large_crossover_observed": contrast >= .5,
            "absence_of_transition_established": False,
            "interpretation": "Operational readiness only; six paired seeds cannot establish equivalence or exclude an unobserved transition.",
            "full_run_authorized": False, "full_run_executed": False}


def comparison_figure(v1: list[dict], v2: list[dict], comparison: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, rows, title in zip(axes[:2], (v1, v2), ("A · v1: 48 slots = S", "B · v2: 96 slots = 2S")):
        for kind, (color, marker, label) in STYLES.items():
            sub = sorted([r for r in rows if r["condition"] == "Needle-High" and r["agent_type"] == kind], key=lambda r: r["degree"])
            if sub:
                ax.errorbar([r["degree"] for r in sub], [r["p_success"] for r in sub],
                    yerr=[[r["p_success"]-r["ci_low"] for r in sub], [r["ci_high"]-r["p_success"] for r in sub]],
                    color=color, marker=marker, capsize=4, linestyle="--" if kind == "llm" else "-",
                    markerfacecolor="white" if kind == "llm" else color,
                    label=f"{label}, n={sub[0]['episodes']}/degree")
        ax.set_title(title)
        ax.set_ylabel("Success probability (Wilson 95% CI)")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=8)
    for version, color, marker in (("v1", "#6A6674", "o"), ("v2", "#7A3E9D", "s")):
        sub = sorted([r for r in comparison if r["version"] == version and r["agent_type"] == "llm"], key=lambda r: r["degree"])
        if sub:
            axes[2].plot([r["degree"] for r in sub], [r["mean_unique_coverage"] for r in sub],
                         color=color, marker=marker, label=f"GPT {version}, n={sub[0]['episodes']}/degree")
    axes[2].set_title("C · Realized GPT coverage")
    axes[2].set_ylabel("Mean unique candidates / 48")
    axes[2].legend(loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=8)
    for ax in axes:
        ax.set_xlabel("Communication degree d")
        ax.set_xticks([0, 2, 5])
        ax.set_ylim(-.06, 1.06)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Needle-High: new resource regime, same atomic search structure", fontweight="bold")
    fig.supxlabel("Separate experiments; v2 adds a second coverage pass and stronger default-assignment guidance. No pooling.", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_4_v1_v2_comparison.{suffix}", bbox_inches="tight")
    plt.close(fig)


def render_v2(config, episodes, rows, comparison, diag, runtime, decision, preservation, complete, provenance):
    shas = sorted({e["git_commit_sha"] for e in episodes})
    observed = observed_findings(episodes)
    lines = ["# Needle / Chain pilot v2 report", "", f"Status: **{'COMPLETE' if complete else 'INCOMPLETE'}**. "
             f"Readiness: **{decision['status']}**. **No full experiment launched.**", "",
             f"V1 frozen commit: `{preservation['frozen_commit']}`. All {preservation['verified_files']} protected files remain byte-identical.",
             "V2 execution commit(s): " + ", ".join(f"`{s}`" for s in shas) + ".",
             "Branch: `experiment/needle-chain-no-transition-v1`; main is not merged or modified.", "",
             "## Preregistered change", "",
             "Needle-High changes from S=48, N=6, T=2, batch=4 (48 slots) to **batch=8 (96 slots = 2S)**. "
             "The first 48 public positions are a complete permutation; the next 48 are a second independently shuffled pass. "
             "The schedule is target-independent and degree-independent. The stronger prompt recommends the schedule while preserving legal freedom of selection. "
             "No branching, pruning, relational feedback or centralized adaptive allocator is introduced.", "",
             "Needle-Low remains S=1200, T=2, batch=4; Chain remains C=8 with T=6 or T=10. Their prompts, resources and environment logic are unchanged. "
             "All conditions use N=6, d={0,2,5}. There are 40 deterministic seeds per cell; GPT has six paired High seeds and two paired sanity-check seeds per unchanged condition. "
             "The exact design was committed in `docs/needle_chain/preregistration_v2.md` before v2 execution.", "",
             "V1 remains valid historical evidence. V2 is a new condition motivated by its zero-slack budget and observed omissions. "
             "Resource slack, second-pass scheduling and guidance change jointly; the pilot cannot isolate their individual causal contributions. "
             "The versions are never pooled or treated as interchangeable replicates.", "",
             "## Exact configuration and model", "", "```json", json.dumps(config, indent=2), "```", "",
             "Effective provider settings:", "```json", json.dumps(resolved_llm_settings(config), indent=2), "```", "",
             "Returned model IDs: " + (", ".join(sorted({m for e in episodes for m in e["response_models"]})) or "none yet") + ".", ""]
    for kind, title in (("deterministic", "Deterministic validation"), ("llm", "GPT results")):
        lines += ["## " + title, "", "| Condition | d | Successes / episodes | P(success) | Wilson 95% CI | Coverage | Duplicate fraction |", "|---|---:|---:|---:|---|---:|---:|"]
        for r in rows:
            if r["agent_type"] == kind:
                coverage = "—" if r["coverage_fraction"] is None else f"{r['coverage_fraction']:.4f}"
                lines.append(f"| {r['condition']} | {r['degree']} | {r['successes']}/{r['episodes']} | {r['p_success']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {coverage} | {r['duplicate_fraction']:.4f} |")
        lines += [""]
    lines += ["## Observed Needle-High behavior", "", "```json", json.dumps(observed, indent=2), "```", "",
              f"Of {observed['llm_high_episodes']} GPT High episodes, {observed['first_round_full_coverage']} "
              f"covered all 48 IDs in round 1 and {observed['first_round_target_discovery']} found the target in that round. "
              f"There were {observed['first_round_changed_batches']} first-round assignment departures. "
              f"The second pass first discovered the target in {observed['observed_second_pass_target_rescues']} episodes.", "",
              "This distinguishes an observed high-resource outcome from a causal claim that the extra pass rescued v1-like omissions. "
              "The pilot jointly changes slack and guidance and cannot identify which change produced any improvement.", "",
              f"There were {observed['changed_batches']} changed High batches, at degrees {observed['changed_batch_degrees']} "
              f"and rounds {observed['changed_batch_rounds']}. Of {observed['changed_batch_slots']} actual selections in those batches, "
              f"{observed['neighbor_known_negative_tests_in_changed_batches']} retested IDs already reported NOT TARGET by neighbors. "
              "These observed actions do not reveal the model's motive. Post-success departures cannot change an already achieved success; "
              "they still show that optimal use of evidence is not guaranteed.", "",
              "Readiness uses the preregistered observed-rate thresholds. It is not a confidence-bound guarantee: even 6/6 successes "
              "has a Wilson 95% lower bound of about .61. Six paired seeds cannot establish no degree effect.", "",
              "## Needle-High v1 versus v2", "", "| Version | Agent | d | Episodes | P(success) | Mean unique IDs | Coverage | Duplicate fraction | Schedule departure fraction |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['version']} | {r['agent_type']} | {r['degree']} | {r['episodes']} | {r['p_success']:.4f} | {r['mean_unique_candidates']:.2f} | {r['mean_unique_coverage']:.4f} | {r['mean_duplicate_fraction']:.4f} | {r['mean_schedule_departure_fraction']:.4f} |")
    lines += ["", "Matched scheduled IDs use multiset overlap within each agent-round batch, so reordering does not count as a departure. "
              "Missing/invalid slots count as replaced. The deterministic v2 duplicate fraction is .5 because the second pass deliberately repeats the candidate space. "
              "Duplicate fraction therefore measures both planned redundancy and any extra inefficiency; it should not be compared to v1 as a pure error rate.", "",
              "`needle_high_episodes.csv` and `needle_high_behavior.json` retain every episode's nominal slots, valid tests, unique coverage, duplicates, followed/replaced IDs, "
              "nominal target inclusion, actual target test and discovery round. Changed batches are retained verbatim as candidate IDs. "
              "P_coverage=unique IDs/48 is descriptive. Since later choices may respond to TARGET feedback, it is not a calibrated conditional probability for the adaptive GPT policy.", "",
              "## Degree-effect diagnostics", "", "| Condition | Agent | Endpoint ΔP | Conservative 95% CI | Paired p / BH q | Logistic status |", "|---|---|---:|---|---|---|"]
    for d in diag:
        ep, reg = d["endpoint"], d["regression"]
        status = reg["status"]
        if d["condition"] == "Needle-High" and reg["status"] == "not_identifiable_constant_outcome":
            sub = [r for r in rows if r["condition"] == "Needle-High" and r["agent_type"] == d["agent_type"]]
            status = "constant high-success outcome; degree slope not identifiable" if sub and all(r["p_success"] == 1 for r in sub) else status
        if reg["coefficient"] is not None:
            status += f"; coefficient={reg['coefficient']:.6g}; bootstrap CI={reg.get('ci95_cluster_bootstrap')}"
        lines.append(f"| {d['condition']} | {d['agent_type']} | {ep.get('delta_p')} | {ep.get('ci95_conservative_wilson')} | {ep.get('p_value')} / {ep.get('p_bh_fdr')} | {status} |")
    lines += ["", "The endpoint compares paired seeds at d=5 and d=0. The exact discordance test, paired bootstrap and conservative Wilson-based difference interval are retained. "
              "Constant/separated outcomes do not receive a fabricated logistic slope. Where identifiable, `diagnostics.json` records likelihoods, AIC, cluster uncertainty and fit counts. "
              "Six paired seeds provide limited inference; degenerate bootstrap intervals are not evidence of equivalence. No finite critical degree is interpolated.", "",
              "## Validation and readiness", "", "```json", json.dumps(decision, indent=2), "```", "",
              f"Structural invariants passed for {len(episodes)} episodes. API-failed decisions={sum(e['api_errors'] for e in episodes)}, "
              f"invalid actions={sum(e['invalid_actions'] for e in episodes)}, behavioral parse failures={sum(e['behavior_errors'] for e in episodes)}.",
              "The environment checks atomic verification, exact target success, duplicate accounting, paired resources and one chain level per round. "
              "Reference High actions must equal the two complete public passes. Chain-short cannot complete and reference Chain-sufficient completes without communication. "
              "Full test and frozen-file audit evidence is saved in `validation.json` and `v1_preservation.json`.", "",
              "Failures are preserved without favorable reruns. A–F deviation categories remain implementation, finite-sample noise, LLM behavior, API/runtime, task definition, or empirical deviation, respectively. "
              "Inspect changed candidate batches and technical counters before attributing differences to degree. An operational READY result would authorize no additional execution.", "",
              "## Runtime and API usage", "", "```json", json.dumps(runtime, indent=2), "```", "",
              "Stage timings exclude test preflight; the pipeline timer also includes intermediate analysis. Cost is an estimate from measured tokens and the frozen published rates, not a billing receipt. "
              "The existing full.json is still the v1 design. A v2 full resource design is not registered or executed by this task.", "",
              "## Figures and reproducibility", "",
              "- [Needle success](figures/figure_1_needle.png)", "- [Chain success](figures/figure_2_chain.png)",
              "- [Coverage, duplication and depth](figures/figure_3_mechanisms.png)",
              "- [V1–v2 resource-regime comparison](figures/figure_4_v1_v2_comparison.png)",
              "- Vector PDF versions accompany all four figures.",
              "- `episodes.jsonl.gz` contains all processed v2 traces; raw requests/responses remain in ignored `runs/needle_chain/pilot_v2` and the hosted artifact.",
              "- `v1_v2_comparison.csv` is a derived comparison in the v2 directory; it does not modify any original v1 output.", "",
              "Reproduction commands are in `docs/needle_chain/README_v2.md`. The implementation and preregistration commit is recorded above; "
              "the final results commit is the commit containing this report. No full experiment was launched.", ""]
    if provenance:
        lines += ["Hosted execution: " + f"[run {provenance['run_id']}]({provenance['run_url']}). "
                  "The existing runner authentication was used in place; no secret was copied or changed.", ""]
    return "\n".join(lines)


def analyze_v2(run_dir: Path, output: Path, config: dict, execution: dict) -> dict:
    guard_v2_paths(run_dir, output)
    preservation = verify_frozen_v1()
    episodes = load_episodes(run_dir)
    if not episodes:
        raise ValueError("no v2 episodes to analyze")
    expected = {(c["name"], k, s, d) for c in config["conditions"] for k in ("deterministic", "llm")
                for s in episode_seeds(config, c, k) for d in config["degrees"]}
    keys = [(e["condition"], e["agent_type"], e["task_seed"], e["degree"]) for e in episodes]
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    assert all(e["identity"]["protocol"] == "needle_chain_v2" for e in episodes)
    complete = set(keys) == expected
    rows, diag = aggregate(episodes, config), diagnostics(episodes, config)
    invariants = verify_invariants(episodes, config)
    runtime = projection(episodes, config, {**config, "conditions": [], "llm_seeds": []}, execution)
    runtime = {k: v for k, v in runtime.items() if not k.startswith("estimated_full") and k not in {"full_llm_episodes", "projection_assumptions"}}
    runtime["full_projection_note"] = "No v2 full resource design is registered or executed. Existing full.json remains v1."
    old = load_episodes(ROOT / "results/needle_chain/pilot")
    old_config = json.loads((ROOT / "results/needle_chain/pilot/config.json").read_text())
    old_rows = aggregate(old, old_config)
    records = coverage_records(episodes, "v2")
    comparison = comparison_table(coverage_records(old, "v1") + records)
    decision = readiness(rows, config, complete, invariants["status"])
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("config", config), ("provider_settings", resolved_llm_settings(config)),
                        ("diagnostics", diag), ("invariants", invariants), ("runtime", runtime),
                        ("readiness", decision), ("needle_high_behavior", records),
                        ("observed_findings", observed_findings(episodes)),
                        ("v1_preservation", preservation), ("execution", execution),
                        ("completion_status", {"complete": complete, "completed_episodes": len(episodes),
                            "expected_episodes": expected_episode_count(config), "missing_episodes": sorted(expected - set(keys))})):
        write_json(output / (name + ".json"), value)
    write_csv(output / "by_degree.csv", rows)
    write_csv(output / "v1_v2_comparison.csv", comparison)
    write_csv(output / "needle_high_episodes.csv", [{k: v for k, v in r.items() if k != "changed_batches"} for r in records])
    _archive(output / "episodes.jsonl.gz", episodes)
    calls = [{"episode_id": p.parent.name, "ledger": p.stem, "record": json.loads(line)}
             for p in sorted((run_dir / "episodes").glob("*/*.jsonl"))
             if p.name in {"requests.jsonl", "responses.jsonl", "events.jsonl"}
             for line in p.read_text().splitlines() if line.strip()]
    if calls:
        _archive(run_dir / "model_calls.jsonl.gz", calls)
    plot_all(rows, episodes, output / "figures", "pilot v2")
    comparison_figure(old_rows, rows, comparison, output / "figures")
    provenance_path = run_dir / "hosted_provenance.json"
    if not provenance_path.exists():
        provenance_path = output / "hosted_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance:
        write_json(output / "hosted_provenance.json", provenance)
    (output / "pilot_report_v2.md").write_text(render_v2(config, episodes, rows, comparison, diag,
        runtime, decision, preservation, complete, provenance))
    return {"complete": complete, "episodes": len(episodes), "expected_episodes": len(expected),
            "rows": rows, "runtime": runtime, "diagnostics": diag, "readiness": decision}
