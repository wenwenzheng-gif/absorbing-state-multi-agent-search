# Needle / Chain pilot report

Status: **COMPLETE**. Full experiment **NOT launched**.



## A. Repository state

Repository: `wenwenzheng-gif/absorbing-state-multi-agent-search`.

Branch: `experiment/needle-chain-no-transition-v1`. Starting main SHA: `87cfc41eb3bcbef3f5f70029c73f17708d324cc2`.
Pilot execution commit SHA(s): `175d28af2badca95c534d6e4b50b1c71db1487fb`, `82d6c2bb7eb79b77b069889b3d563c9122640069`.
The final results commit is the commit containing this report (avoids a self-referential SHA); `git log -1 --format=%H -- results/needle_chain/pilot/pilot_report.md` resolves it.

`experiment/discoverphysics-phase-mechanism-v2`: already absent; `experiment/discoverybench-critical-degree-v1`: already absent; `experiment/discoverybench-degree-scan`: already absent. Local and origin absence verified before implementation. The initial cleanup did not alter the separate `absorb-state-new` repository. The user's follow-up then authorized reuse of its existing hosted Actions environment on a dedicated experimental runner branch; neither main branch was changed.

## B. Implementation

Atomic candidate verification has exactly one target and no child/pruning API. The reference policy traverses a seed-shuffled permutation in disjoint agent/round batches. LLM agents receive the same public work assignment but can choose any legal candidate. Assignments use a separate RNG domain from target selection. Needle always expends the fixed two-round budget, including after discovery, so coverage diagnostics are not biased by early stopping.

Chain uses a shared hidden sequence of capability keys, per-agent key possession, and a locked round transaction. A submission never releases the next key. The commit barrier advances zero or one level and gives the next key to every valid submitter; truthful neighbor receipts can convey it on the next round. No level is enumerable, readable or executable early, even if its key is guessed. Chain stops on completion.

Reused: `Router` and its seeded exact-degree graphs, SplitMix64 `derived_rng`, `DecisionContext`, `LLMConfig`, `LLMBackend`, `RunStore`, and CSV/PNG/PDF conventions. A task-specific scheduler is necessary because the historical runner and `Policy.propose` require literal branching. No historical environment or finite-critical-degree fitter is changed or called.

Tests cover target atomicity and coverage, future-level read/guess rejection, concurrent submissions, barriers, sufficient/insufficient horizons, truthful one-hop communication, observation isolation, failed decisions, cached responses and interrupted episode replay, paired statistics and boundary serialization.

Live GPT response provenance is retained in the raw local ledgers and summarized below.

## C. Exact pilot parameters

```json
{
  "name": "needle-chain-no-transition-v1",
  "protocol_version": "needle_chain_v1",
  "mode": "pilot",
  "agents": 6,
  "degrees": [
    0,
    2,
    5
  ],
  "deterministic_seeds": [
    91000,
    91001,
    91002,
    91003,
    91004,
    91005,
    91006,
    91007,
    91008,
    91009,
    91010,
    91011,
    91012,
    91013,
    91014,
    91015,
    91016,
    91017,
    91018,
    91019,
    91020,
    91021,
    91022,
    91023,
    91024,
    91025,
    91026,
    91027,
    91028,
    91029,
    91030,
    91031,
    91032,
    91033,
    91034,
    91035,
    91036,
    91037,
    91038,
    91039
  ],
  "llm_seeds": [
    91000,
    91001
  ],
  "graph_seed_offset": 400000,
  "llm": {
    "model": "gpt-5.4-nano",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 1.0,
    "api_style": "openai",
    "reasoning_effort": "none",
    "base_url": "https://api.openai.com/v1",
    "concurrency": 6,
    "timeout_s": 60.0,
    "transport_retries": 2,
    "max_output_tokens": 256,
    "repair_attempts": 0,
    "per_agent_seed": false,
    "max_requests": null,
    "max_wall_seconds": null
  },
  "conditions": [
    {
      "name": "Needle-Low",
      "task": "needle",
      "candidates": 1200,
      "batch_size": 4,
      "horizon": 2
    },
    {
      "name": "Needle-High",
      "task": "needle",
      "candidates": 48,
      "batch_size": 4,
      "horizon": 2
    },
    {
      "name": "Chain-Short-Horizon",
      "task": "chain",
      "chain_length": 8,
      "batch_size": 1,
      "horizon": 6
    },
    {
      "name": "Chain-Sufficient-Horizon",
      "task": "chain",
      "chain_length": 8,
      "batch_size": 1,
      "horizon": 10
    }
  ],
  "bootstrap_samples": 1000,
  "analysis_seed": 77201,
  "pricing_usd_per_million": {
    "input": 0.2,
    "cached_input": 0.02,
    "output": 1.25
  },
  "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.4-nano"
}
```

The frozen resource config retains the repository's `gpt-5.4-nano` family alias. Before any live call, the user explicitly pinned `gpt-5.4-nano-2026-03-17`; `configs/needle_chain/provider.json` locks that exact request model while preserving deterministic provenance. Effective model settings:
```json
{
  "model": "gpt-5.4-nano-2026-03-17",
  "api_key_env": "OPENAI_API_KEY",
  "temperature": 1.0,
  "api_style": "openai",
  "reasoning_effort": "none",
  "base_url": "https://api.openai.com/v1",
  "concurrency": 6,
  "timeout_s": 60.0,
  "transport_retries": 2,
  "max_output_tokens": 256,
  "repair_attempts": 0,
  "per_agent_seed": false,
  "max_requests": null,
  "max_wall_seconds": null
}
```
Runtime response model IDs: gpt-5.4-nano-2026-03-17.
No scientific expectation or communication degree is included in decision prompts. The only degree-dependent input is naturally received neighbor evidence. API model sampling is stochastic; task/graph seeds are paired, and returned samples are archived. No seed-count reduction was applied.

## D. Deterministic results

| Condition | d | Episodes | Successes | P(success) | Wilson 95% CI |
|---|---:|---:|---:|---:|---|
| Chain-Short-Horizon | 0 | 40 | 0 | 0.000 | [0.000, 0.088] |
| Chain-Short-Horizon | 2 | 40 | 0 | 0.000 | [0.000, 0.088] |
| Chain-Short-Horizon | 5 | 40 | 0 | 0.000 | [0.000, 0.088] |
| Chain-Sufficient-Horizon | 0 | 40 | 40 | 1.000 | [0.912, 1.000] |
| Chain-Sufficient-Horizon | 2 | 40 | 40 | 1.000 | [0.912, 1.000] |
| Chain-Sufficient-Horizon | 5 | 40 | 40 | 1.000 | [0.912, 1.000] |
| Needle-High | 0 | 40 | 40 | 1.000 | [0.912, 1.000] |
| Needle-High | 2 | 40 | 40 | 1.000 | [0.912, 1.000] |
| Needle-High | 5 | 40 | 40 | 1.000 | [0.912, 1.000] |
| Needle-Low | 0 | 40 | 1 | 0.025 | [0.004, 0.129] |
| Needle-Low | 2 | 40 | 1 | 0.025 | [0.004, 0.129] |
| Needle-Low | 5 | 40 | 1 | 0.025 | [0.004, 0.129] |

## E. GPT-5.4 Nano results

| Condition | d | Episodes | Successes | P(success) | Wilson 95% CI |
|---|---:|---:|---:|---:|---|
| Chain-Short-Horizon | 0 | 2 | 0 | 0.000 | [0.000, 0.658] |
| Chain-Short-Horizon | 2 | 2 | 0 | 0.000 | [0.000, 0.658] |
| Chain-Short-Horizon | 5 | 2 | 0 | 0.000 | [0.000, 0.658] |
| Chain-Sufficient-Horizon | 0 | 2 | 2 | 1.000 | [0.342, 1.000] |
| Chain-Sufficient-Horizon | 2 | 2 | 2 | 1.000 | [0.342, 1.000] |
| Chain-Sufficient-Horizon | 5 | 2 | 2 | 1.000 | [0.342, 1.000] |
| Needle-High | 0 | 2 | 1 | 0.500 | [0.095, 0.905] |
| Needle-High | 2 | 2 | 1 | 0.500 | [0.095, 0.905] |
| Needle-High | 5 | 2 | 1 | 0.500 | [0.095, 0.905] |
| Needle-Low | 0 | 2 | 0 | 0.000 | [0.000, 0.658] |
| Needle-Low | 2 | 2 | 0 | 0.000 | [0.000, 0.658] |
| Needle-Low | 5 | 2 | 0 | 0.000 | [0.000, 0.658] |

## F. Endpoint and degree-effect diagnostics

| Condition | Agent | ΔP | Conservative 95% CI | Paired p / BH q | Logistic degree coefficient / status |
|---|---|---:|---|---|---|
| Chain-Short-Horizon | deterministic | 0.0 | [-0.11158268671908494, 0.11158268671908494] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Short-Horizon | llm | 0.0 | [-0.7152573451984471, 0.7152573451984471] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Sufficient-Horizon | deterministic | 0.0 | [-0.11158268671908489, 0.11158268671908489] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Sufficient-Horizon | llm | 0.0 | [-0.7152573451984472, 0.7152573451984472] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Needle-High | deterministic | 0.0 | [-0.11158268671908489, 0.11158268671908489] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Needle-High | llm | 0.0 | [-0.845728883980231, 0.845728883980231] | 1.0 / 1.0 | 0.000000; bootstrap CI None |
| Needle-Low | deterministic | 0.0 | [-0.1487147103448543, 0.1487147103448543] | 1.0 / 1.0 | -0.000000; bootstrap CI None |
| Needle-Low | llm | 0.0 | [-0.7152573451984471, 0.7152573451984471] | 1.0 / 1.0 | not_identifiable_constant_outcome |

The endpoint is computed on paired seeds. Conservative difference intervals use Bonferroni marginal Wilson bounds; paired-bootstrap intervals are also saved. A degenerate bootstrap interval (often [0,0] with two identical outcomes) does not establish equivalence. Constant all-zero/all-one outcomes and separation produce no logistic coefficient. When estimable, `diagnostics.json` records the constant-vs-degree likelihoods, AIC, coefficient, cluster bootstrap and cluster-robust uncertainty. A logistic bootstrap confidence interval is withheld if fewer than 95% of resamples are identifiable, its empirical distribution is degenerate, or there are fewer than six task clusters; conditional quantiles are retained separately and are not presented as a valid confidence interval. The likelihood ratio is descriptive because rows are paired. BH/FDR covers all estimable endpoint and slope tests together. No finite dc is interpolated or fitted.

## G. Needle mechanism

| Condition | Agent | d | Mean unique IDs | Coverage / S | Duplicate fraction | Invalid action rate |
|---|---|---:|---:|---:|---:|---:|
| Needle-High | deterministic | 0 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| Needle-High | deterministic | 2 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| Needle-High | deterministic | 5 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| Needle-High | llm | 0 | 37.50 | 0.7812 | 0.2188 | 0.0000 |
| Needle-High | llm | 2 | 44.00 | 0.9167 | 0.0833 | 0.0000 |
| Needle-High | llm | 5 | 42.50 | 0.8854 | 0.1146 | 0.0000 |
| Needle-Low | deterministic | 0 | 48.00 | 0.0400 | 0.0000 | 0.0000 |
| Needle-Low | deterministic | 2 | 48.00 | 0.0400 | 0.0000 | 0.0000 |
| Needle-Low | deterministic | 5 | 48.00 | 0.0400 | 0.0000 | 0.0000 |
| Needle-Low | llm | 0 | 47.50 | 0.0396 | 0.0104 | 0.0000 |
| Needle-Low | llm | 2 | 48.00 | 0.0400 | 0.0000 | 0.0000 |
| Needle-Low | llm | 5 | 48.00 | 0.0400 | 0.0000 | 0.0000 |

### Observed GPT Needle-High deviation

| Seed | d | Success | Unique IDs | Duplicate tests | Changed assignment batches | Missed target was in an omitted assignment |
|---:|---:|---|---:|---:|---:|---|
| 91000 | 0 | True | 44 | 4 | 1 | False |
| 91000 | 2 | True | 44 | 4 | 1 | False |
| 91000 | 5 | False | 45 | 3 | 1 | True |
| 91001 | 0 | False | 31 | 17 | 5 | True |
| 91001 | 2 | False | 44 | 4 | 1 | True |
| 91001 | 5 | True | 40 | 8 | 2 | False |

GPT departed from the disjoint public assignment in 11 batches. 3 of 3 failed high-resource episodes omitted the target-containing assignment. These were legal candidate selections, so invalid-action rate zero does not mean optimal search behavior. The nominal test-slot budget was sufficient, but realized unique coverage was lower. `needle_behavior.json` records the exact scheduled and actual IDs for every changed batch.

All changed high-resource batches were in round 1, before any neighbor evidence exists. Therefore the observed coverage differences across degrees cannot be attributed to communication in this pilot: first-round prompts are identical across degrees for a given agent/seed, and model samples are stochastic. All round-2 high-resource assignments were followed.

This supports category C (behavioral coordination inefficiency), with category B (finite-sample/model-sampling variability) limiting the degree comparison. It is not an API failure or an invalid environment action. Equal degree means can mask discordant paired outcomes: inspect the per-seed rows and paired intervals. No failed episode was rerun, no prompt was repaired, and no resource parameter was adjusted to force agreement.


The reference visits B=min(N × batch × T,S) distinct IDs; its target is uniform and independent of ordering, so P(success)=B/S exactly over task draws. Here the predicted probabilities are .04 and 1.0. Observed success is empirical; it is never replaced by those values. Any sampling discrepancy must be judged against the binomial interval. Rejection reports only that same candidate's verdict; no hidden branching or cross-candidate pruning exists. The coordinated reference has no duplicates to reduce. Degree-related LLM efficiency changes, if present, can be read separately above.

## H. Chain mechanism

| Condition | Agent | d | Mean maximum depth | Invalid action rate | Future attempts |
|---|---|---:|---:|---:|---:|
| Chain-Short-Horizon | deterministic | 0 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | deterministic | 2 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | deterministic | 5 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | llm | 0 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | llm | 2 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | llm | 5 | 6.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | deterministic | 0 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | deterministic | 2 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | deterministic | 5 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | llm | 0 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | llm | 2 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | llm | 5 | 8.00 | 0.0000 | 0 |

Per-episode depth sequences are in `episodes.jsonl.gz`; means for every round are in `depth_by_round.csv`. The plot extends an already completed chain at its absorbing depth for the remaining horizon, without claiming additional calls. Future nodes are capability-locked; advancing more than one level per round is rejected by construction and checked in the trace. T<C completion is impossible. Correct deterministic agents finish at round C even at d=0. Communication cannot bypass the barrier.

## I. Figures and data

- [Figure 1 — Needle](figures/figure_1_needle.png)
- [Figure 2 — Chain](figures/figure_2_chain.png)
- [Figure 3 — Mechanisms](figures/figure_3_mechanisms.png)
- Each figure also has a vector PDF. Plot inputs: `by_degree.csv`, `depth_by_round.csv`.
- `episodes.jsonl.gz`: all episode actions, messages, responses, seeds, graphs, outcomes and runtime metrics.
- Raw request/response ledgers and `model_calls.jsonl.gz` remain under ignored `runs/needle_chain/pilot/`, following the repository's raw-transcript convention. They are retained locally, not published in this results directory.

## J. Theory comparison and deviations

Preliminary classification: **partially consistent; deviations require diagnosis**.

Observed large within-condition degree contrast (≥0.5): **NO**. Whether a nontrivial transition is absent remains **INCONCLUSIVE** from this finite scan and small GPT sample.

The theoretical labels are Needle dc=0; Chain T<C dc="infinity"; Chain T≥C dc=0. These labels are metadata about task structure, not estimates derived from flat data. The paper's claim concerns the absence of a branching/pruning critical point. It does not prove every possible LLM policy has a perfectly flat finite-resource curve.

The public disjoint schedule deliberately removes the need for communication in the reference and provides useful coordination to GPT. Thus this pilot probes these specified policies and resource regimes; it cannot exclude a crossover caused by other policies, other resources or untested degrees.

Structural invariant checks: **passed**, 504 episodes. API-failed decisions: 0. Invalid environment actions: 0. Invalid/refused/truncated JSON decisions: 0.
Deviation categories: A implementation error and E task-definition violation require concrete invariant/test evidence; B finite-sample noise is plausible for rare Needle-Low outcomes; C LLM behavior includes legal but inefficient choices as well as malformed/illegal choices; D API/runtime failure is recorded independently and retained in the success denominator; F genuine empirical deviation remains a possible interpretation if a reproducible contrast survives diagnosis. No failed seed is excluded or rerun for a better outcome.

## K. Measured runtime and future scale

```json
{
  "deterministic_episodes": 480,
  "llm_episodes": 24,
  "deterministic_seconds": 7.63532292300718,
  "llm_seconds": 99.64534402199999,
  "pilot_seconds": 115.04609584209302,
  "api_calls": 648,
  "successful_api_calls": 648,
  "failed_attempts": 0,
  "retries": 0,
  "mean_http_latency_seconds": 0.7538000930787034,
  "episodes_per_minute": 14.451252229929304,
  "calls_per_episode": 27.0,
  "prompt_tokens": 502943,
  "completion_tokens": 21633,
  "estimated_pilot_cost_usd": 0.12762985000000002,
  "full_llm_episodes": 144,
  "estimated_full_calls": 12960.0,
  "estimated_full_seconds": 1991.1574717199994,
  "estimated_full_cost_usd": 2.5525970000000004,
  "estimated_full_seconds_round_waves": 2389.3889660639993,
  "estimated_full_cost_usd_degree_context": 4.547451736842106,
  "full_executed": false,
  "execution_status": "complete",
  "blocker": null,
  "projection_assumptions": "Baseline scales measured per-condition throughput by agents and episodes at unchanged concurrency. Round-waves alternative uses ceil(N/concurrency) for barriers. Degree-context cost extrapolates a per-condition linear fit of observed input tokens/call against degree, at uncached input rates. Larger messages, rate limits, provider load and changed stopping rounds can alter all estimates; none is a billing statement."
}
```

The execution timer starts after the automatic test preflight and includes stage execution and intermediate analysis; it excludes the final report regeneration. Deterministic/LLM stage timers are separate. Cost uses measured prompt/completion/cache token counts and [published GPT-5.4 Nano rates](https://developers.openai.com/api/docs/models/gpt-5.4-nano); it is not a charge receipt. The baseline runtime scales measured per-condition throughput by agents and episodes. The round-waves alternative accounts for ceil(N/concurrency) batches at each barrier. The degree-context cost additionally extrapolates observed input tokens per call against degree; this is an extrapolation from d≤5 to d≤19, not measured full-run usage. Provider load, rate limits, larger contexts and behavioral changes remain unmeasured. Null projections mean no live throughput has been measured.

## L. Recommended next step

The pipeline is operational, but the unchanged larger run is not recommended before reviewing GPT's Needle-High coordination failure. It missed the intended high-success regime despite adequate nominal slots. Any revised policy would require a new preregistered experiment, preserving this pilot. No further API calls or full experiment were launched.

## Reproduction and provenance

See `docs/needle_chain/README.md` for tests, resume, offline reanalysis and the separately gated full command. Preregistered design: `docs/needle_chain/preregistration.md`. Theory: [arXiv:2609.38327v2, §§2.2–3.1](https://arxiv.org/html/2609.38327v2#S3.SS1).

## Completed hosted pilot

The existing provider secret was consumed in place by the authorized prior Actions runner. [Run 42403291298](https://github.com/wenwenzheng-gif/absorb-state-new/actions/runs/42403291298) completed successfully; its job took 169 seconds including setup, tests and artifact upload. No secret was migrated. The temporary runner branch was removed after artifact import; all reviewed code and results remain on the canonical experimental branch.

Measured GPT execution: **99.65 seconds**, **648 API calls**, **14.45 episodes/minute**. Estimated pilot token cost: **$0.128**.

The unexecuted 144-episode full GPT configuration projects **33.2–39.8 minutes** and **12,960 calls** under the stated scaling assumptions. Token-cost estimates range from **$2.55** at pilot context sizes to **$4.55** with the degree-context extrapolation. These are scenario estimates, not confidence bounds.

Audit files: `hosted_provenance.json`, `data_audit.json`, `validation.json`; authentication discovery: `docs/needle_chain/authentication_audit.md`; post-pilot analysis changes: `docs/needle_chain/analysis_notes.md`.
