"""Complete pilot report and compact, reproducible archives."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..analysis.summarize import write_csv
from .analysis import aggregate, diagnostics, load_episodes, projection, verify_invariants
from .config import ROOT, load_config
from .plots import plot_all


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _archive(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as z:
        for row in rows:
            z.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def render_report(rows: list[dict], diag: list[dict], runtime: dict, invariants: dict,
                  config: dict, episodes: list[dict], complete: bool) -> str:
    cleanup = json.loads((ROOT / "docs/needle_chain/branch_cleanup.json").read_text())
    shas = sorted({e["git_commit_sha"] for e in episodes})
    statuses = "; ".join(f"`{b['branch']}`: {b['status']}" for b in cleanup["branches"])
    llm = [e for e in episodes if e["agent_type"] == "llm"]
    unexpected = [r for r in rows if r["agent_type"] == "llm" and r["condition"] in {"Needle-High", "Chain-Sufficient-Horizon"} and r["p_success"] < .9]
    crossover = any(max(r["p_success"] for r in rows if (r["condition"], r["agent_type"]) == key)
                    - min(r["p_success"] for r in rows if (r["condition"], r["agent_type"]) == key) >= .5
                    for key in {(r["condition"], r["agent_type"]) for r in rows})
    classification = ("inconclusive (incomplete pilot)" if not complete else
                      "partially consistent; deviations require diagnosis" if unexpected or crossover else
                      "consistent with theory at the tested resources; statistically inconclusive about absence of a transition")
    launch_status = "Full experiment **NOT launched**" if config["mode"] == "pilot" else "Explicitly authorized full configuration was executed"
    lines = [f"# Needle / Chain {config['mode']} report", "", f"Status: **{'COMPLETE' if complete else 'INCOMPLETE'}**. {launch_status}.", "",
             "## A. Repository state", "", f"Repository: `{cleanup['repository']}`.", "",
             f"Branch: `experiment/needle-chain-no-transition-v1`. Starting main SHA: `87cfc41eb3bcbef3f5f70029c73f17708d324cc2`.",
             f"Pilot execution commit SHA(s): {', '.join('`' + s + '`' for s in shas)}.",
             "The final results commit is the commit containing this report (avoids a self-referential SHA); `git log -1 --format=%H -- results/needle_chain/pilot/pilot_report.md` resolves it.", "",
             statuses + ". Local and origin absence verified before implementation. The separate `absorb-state-new` repository was not modified.", "",
             "## B. Implementation", "",
             "Atomic candidate verification has exactly one target and no child/pruning API. The reference policy traverses a seed-shuffled permutation in disjoint agent/round batches. LLM agents receive the same public work assignment but can choose any legal candidate. Assignments use a separate RNG domain from target selection. Needle always expends the fixed two-round budget, including after discovery, so coverage diagnostics are not biased by early stopping.", "",
             "Chain uses a shared hidden sequence of capability keys, per-agent key possession, and a locked round transaction. A submission never releases the next key. The commit barrier advances zero or one level and gives the next key to every valid submitter; truthful neighbor receipts can convey it on the next round. No level is enumerable, readable or executable early, even if its key is guessed. Chain stops on completion.", "",
             "Reused: `Router` and its seeded exact-degree graphs, SplitMix64 `derived_rng`, `DecisionContext`, `LLMConfig`, `LLMBackend`, `RunStore`, and CSV/PNG/PDF conventions. A task-specific scheduler is necessary because the historical runner and `Policy.propose` require literal branching. No historical environment or finite-critical-degree fitter is changed or called.", "",
             "Tests cover target atomicity and coverage, future-level read/guess rejection, concurrent submissions, barriers, sufficient/insufficient horizons, truthful one-hop communication, observation isolation, failed decisions, cached responses and interrupted episode replay, paired statistics and boundary serialization.", "",
             "## C. Exact pilot parameters", "", "```json", json.dumps(config, indent=2), "```", "",
             "Model identifier came from `configs/agents/gpt-5.4-nano-t1_seed1/synthetic_C0.json`. Runtime response model IDs: " + (", ".join(sorted({m for e in llm for m in e["response_models"]})) or "none; no completed live model response") + ".",
             "No scientific expectation or communication degree is included in decision prompts. The only degree-dependent input is naturally received neighbor evidence. API model sampling is stochastic; task/graph seeds are paired, and returned samples are archived. No seed-count reduction was applied.", ""]
    for kind, section in (("deterministic", "D. Deterministic results"), ("llm", "E. GPT-5.4 Nano results")):
        lines += ["## " + section, "", "| Condition | d | Episodes | Successes | P(success) | Wilson 95% CI |", "|---|---:|---:|---:|---:|---|"]
        for r in rows:
            if r["agent_type"] == kind:
                lines.append(f"| {r['condition']} | {r['degree']} | {r['episodes']} | {r['successes']} | {r['p_success']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
        lines.append("")
    lines += ["## F. Endpoint and degree-effect diagnostics", "",
              "| Condition | Agent | ΔP | Conservative 95% CI | Paired p / BH q | Logistic degree coefficient / status |", "|---|---|---:|---|---|---|"]
    for d in diag:
        ep, reg = d["endpoint"], d["regression"]
        ci = ep.get("ci95_conservative_wilson", [])
        coef = reg["status"] if reg["coefficient"] is None else f"{reg['coefficient']:.6f}; bootstrap CI {reg.get('ci95_cluster_bootstrap')}"
        lines.append(f"| {d['condition']} | {d['agent_type']} | {ep.get('delta_p')} | {ci} | {ep.get('p_value')} / {ep.get('p_bh_fdr')} | {coef} |")
    lines += ["", "The endpoint is computed on paired seeds. Conservative difference intervals use Bonferroni marginal Wilson bounds; paired-bootstrap intervals are also saved. A degenerate bootstrap interval (often [0,0] with two identical outcomes) does not establish equivalence. Constant all-zero/all-one outcomes and separation produce no logistic coefficient. When estimable, `diagnostics.json` records the constant-vs-degree likelihoods, AIC, coefficient, cluster bootstrap and cluster-robust uncertainty. The likelihood ratio is descriptive because rows are paired. BH/FDR covers all estimable endpoint and slope tests together. No finite dc is interpolated or fitted.", "",
              "## G. Needle mechanism", "", "| Condition | Agent | d | Mean unique IDs | Coverage / S | Duplicate fraction | Invalid action rate |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["task"] == "needle":
            lines.append(f"| {r['condition']} | {r['agent_type']} | {r['degree']} | {r['mean_unique_candidates_tested']:.2f} | {r['coverage_fraction']:.4f} | {r['duplicate_fraction']:.4f} | {r['invalid_action_rate']:.4f} |")
    lines += ["", "The reference visits B=min(N × batch × T,S) distinct IDs; its target is uniform and independent of ordering, so P(success)=B/S exactly over task draws. Here the predicted probabilities are .04 and 1.0. Observed success is empirical; it is never replaced by those values. Any sampling discrepancy must be judged against the binomial interval. Rejection reports only that same candidate's verdict; no hidden branching or cross-candidate pruning exists. The coordinated reference has no duplicates to reduce. Degree-related LLM efficiency changes, if present, can be read separately above.", "",
              "## H. Chain mechanism", "", "| Condition | Agent | d | Mean maximum depth | Invalid action rate | Future attempts |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["task"] == "chain":
            lines.append(f"| {r['condition']} | {r['agent_type']} | {r['degree']} | {r['mean_maximum_depth']:.2f} | {r['invalid_action_rate']:.4f} | {r['illegal_future_attempts']} |")
    lines += ["", "Per-episode depth sequences are in `episodes.jsonl.gz`; means for every round are in `depth_by_round.csv`. The plot extends an already completed chain at its absorbing depth for the remaining horizon, without claiming additional calls. Future nodes are capability-locked; advancing more than one level per round is rejected by construction and checked in the trace. T<C completion is impossible. Correct deterministic agents finish at round C even at d=0. Communication cannot bypass the barrier.", "",
              "## I. Figures and data", "", "- [Figure 1 — Needle](figures/figure_1_needle.png)", "- [Figure 2 — Chain](figures/figure_2_chain.png)", "- [Figure 3 — Mechanisms](figures/figure_3_mechanisms.png)",
              "- Each figure also has a vector PDF. Plot inputs: `by_degree.csv`, `depth_by_round.csv`.",
              "- `episodes.jsonl.gz`: all episode actions, messages, responses, seeds, graphs, outcomes and runtime metrics.",
              "- Raw request/response ledgers and `model_calls.jsonl.gz` remain under ignored `runs/needle_chain/pilot/`, following the repository's raw-transcript convention. They are retained locally, not published in this results directory.", "",
              "## J. Theory comparison and deviations", "", f"Preliminary classification: **{classification}**.", "",
              f"Observed large within-condition degree contrast (≥0.5): **{'YES — diagnose before scaling' if crossover else 'NO'}**. Whether a nontrivial transition is absent remains **INCONCLUSIVE** from this finite scan and small GPT sample.", "",
              "The theoretical labels are Needle dc=0; Chain T<C dc=\"infinity\"; Chain T≥C dc=0. These labels are metadata about task structure, not estimates derived from flat data. The paper's claim concerns the absence of a branching/pruning critical point. It does not prove every possible LLM policy has a perfectly flat finite-resource curve.", "",
              "The public disjoint schedule deliberately removes the need for communication in the reference and provides useful coordination to GPT. Thus this pilot probes these specified policies and resource regimes; it cannot exclude a crossover caused by other policies, other resources or untested degrees.", "",
              f"Structural invariant checks: **{invariants['status']}**, {invariants['episodes_checked']} episodes. API-failed decisions: {sum(e['api_errors'] for e in episodes)}. Invalid environment actions: {sum(e['invalid_actions'] for e in episodes)}. Invalid/refused/truncated JSON decisions: {sum(e['behavior_errors'] for e in episodes)}.",
              "Deviation categories: A implementation error and E task-definition violation require concrete invariant/test evidence; B finite-sample noise is plausible for rare Needle-Low outcomes; C LLM behavior covers malformed/illegal choices; D API/runtime failure is recorded independently and retained in the success denominator; F genuine empirical deviation remains a possible interpretation if a reproducible contrast survives diagnosis. No failed seed is excluded or rerun for a better outcome.", "",
              "## K. Measured runtime and future scale", "", "```json", json.dumps(runtime, indent=2), "```", "",
              "Cost uses measured prompt/completion/cache token counts and [published GPT-5.4 Nano rates](https://developers.openai.com/api/docs/models/gpt-5.4-nano). It is not a charge receipt. Full-run estimates scale measured per-condition calls and wall time by the larger population and number of episodes at unchanged concurrency. They do not include an unmeasured larger-context or rate-limit penalty.", "",
              "## L. Recommended next step", "",
              ("Review this pilot before authorizing the larger run. The pipeline is ready for review; increase statistical replication only in a subsequent task. Two GPT seeds cannot validate equivalence or rule out a transition." if complete and not unexpected and not crossover else "Resolve the documented incomplete work or deviations and review this pilot before any larger run."), "",
              "## Reproduction and provenance", "", "See `docs/needle_chain/README.md` for tests, resume, offline reanalysis and the separately gated full command. Preregistered design: `docs/needle_chain/preregistration.md`. Theory: [arXiv:2609.38327v2, §§2.2–3.1](https://arxiv.org/html/2609.38327v2#S3.SS1).", ""]
    return "\n".join(lines)


def analyze(run_dir: Path, output: Path, config: dict, execution: dict | None = None) -> dict:
    episodes = load_episodes(run_dir)
    if not episodes:
        raise ValueError("no episodes to analyze")
    output.mkdir(parents=True, exist_ok=True)
    rows = aggregate(episodes, config)
    diag = diagnostics(episodes, config)
    invariants = verify_invariants(episodes, config)
    full = load_config(ROOT / "configs/needle_chain/full.json")
    runtime = projection(episodes, config, full, execution or {})
    expected = len(config["conditions"]) * len(config["degrees"]) * (len(config["deterministic_seeds"]) + len(config["llm_seeds"]))
    complete = len(episodes) == expected
    write_csv(output / "by_degree.csv", rows)
    write_json(output / "config.json", config)
    write_json(output / "diagnostics.json", diag)
    write_json(output / "invariants.json", invariants)
    write_json(output / "runtime.json", runtime)
    if execution:
        write_json(output / "execution.json", execution)
    _archive(output / "episodes.jsonl.gz", episodes)
    calls = []
    for p in sorted((run_dir / "episodes").glob("*/*.jsonl")):
        if p.name not in {"requests.jsonl", "responses.jsonl", "events.jsonl"}:
            continue
        calls.extend({"episode_id": p.parent.name, "ledger": p.stem, "record": json.loads(line)}
                     for line in p.read_text().splitlines() if line.strip())
    if calls:
        _archive(run_dir / "model_calls.jsonl.gz", calls)
    plot_all(rows, episodes, output / "figures", config["mode"])
    (output / "pilot_report.md").write_text(render_report(rows, diag, runtime, invariants, config, episodes, complete))
    return {"complete": complete, "episodes": len(episodes), "expected_episodes": expected,
            "rows": rows, "runtime": runtime, "diagnostics": diag}
