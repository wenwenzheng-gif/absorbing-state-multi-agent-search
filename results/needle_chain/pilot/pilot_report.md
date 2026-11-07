# Needle / Chain pilot report

Status: **INCOMPLETE**. Full experiment **NOT launched**.

**Execution blocker:** No authenticated OpenAI API setup is available: OPENAI_API_KEY is unset; no .env file was found in this checkout, and the requested GitHub repository has no Actions secrets. Supply the location of an existing environment file or authorized runner. No GPT calls were made. Measured GPT throughput and full-run runtime/cost projections are unavailable, not assumed.

## A. Repository state

Repository: `wenwenzheng-gif/absorbing-state-multi-agent-search`.

Branch: `experiment/needle-chain-no-transition-v1`. Starting main SHA: `87cfc41eb3bcbef3f5f70029c73f17708d324cc2`.
Pilot execution commit SHA(s): `175d28af2badca95c534d6e4b50b1c71db1487fb`.
The final results commit is the commit containing this report (avoids a self-referential SHA); `git log -1 --format=%H -- results/needle_chain/pilot/pilot_report.md` resolves it.

`experiment/discoverphysics-phase-mechanism-v2`: already absent; `experiment/discoverybench-critical-degree-v1`: already absent; `experiment/discoverybench-degree-scan`: already absent. Local and origin absence verified before implementation. No branches were deleted or code changed in the separate `absorb-state-new` repository.

## B. Implementation

Atomic candidate verification has exactly one target and no child/pruning API. The reference policy traverses a seed-shuffled permutation in disjoint agent/round batches. LLM agents receive the same public work assignment but can choose any legal candidate. Assignments use a separate RNG domain from target selection. Needle always expends the fixed two-round budget, including after discovery, so coverage diagnostics are not biased by early stopping.

Chain uses a shared hidden sequence of capability keys, per-agent key possession, and a locked round transaction. A submission never releases the next key. The commit barrier advances zero or one level and gives the next key to every valid submitter; truthful neighbor receipts can convey it on the next round. No level is enumerable, readable or executable early, even if its key is guessed. Chain stops on completion.

Reused: `Router` and its seeded exact-degree graphs, SplitMix64 `derived_rng`, `DecisionContext`, `LLMConfig`, `LLMBackend`, `RunStore`, and CSV/PNG/PDF conventions. A task-specific scheduler is necessary because the historical runner and `Policy.propose` require literal branching. No historical environment or finite-critical-degree fitter is changed or called.

Tests cover target atomicity and coverage, future-level read/guess rejection, concurrent submissions, barriers, sufficient/insufficient horizons, truthful one-hop communication, observation isolation, failed decisions, cached responses and interrupted episode replay, paired statistics and boundary serialization.

GPT integration has passed mocked backend tests but has not been validated with live API responses in this pilot.

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

Model identifier came from `configs/agents/gpt-5.4-nano-t1_seed1/synthetic_C0.json`. Runtime response model IDs: none; no completed live model response.
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

## F. Endpoint and degree-effect diagnostics

| Condition | Agent | ΔP | Conservative 95% CI | Paired p / BH q | Logistic degree coefficient / status |
|---|---|---:|---|---|---|
| Chain-Short-Horizon | deterministic | 0.0 | [-0.11158268671908494, 0.11158268671908494] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Sufficient-Horizon | deterministic | 0.0 | [-0.11158268671908489, 0.11158268671908489] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Needle-High | deterministic | 0.0 | [-0.11158268671908489, 0.11158268671908489] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Needle-Low | deterministic | 0.0 | [-0.1487147103448543, 0.1487147103448543] | 1.0 / 1.0 | -0.000000; bootstrap CI [-6.952003824913239e-16, 1.3650549177043175e-16] |

The endpoint is computed on paired seeds. Conservative difference intervals use Bonferroni marginal Wilson bounds; paired-bootstrap intervals are also saved. A degenerate bootstrap interval (often [0,0] with two identical outcomes) does not establish equivalence. Constant all-zero/all-one outcomes and separation produce no logistic coefficient. When estimable, `diagnostics.json` records the constant-vs-degree likelihoods, AIC, coefficient, cluster bootstrap and cluster-robust uncertainty. The likelihood ratio is descriptive because rows are paired. BH/FDR covers all estimable endpoint and slope tests together. No finite dc is interpolated or fitted.

## G. Needle mechanism

| Condition | Agent | d | Mean unique IDs | Coverage / S | Duplicate fraction | Invalid action rate |
|---|---|---:|---:|---:|---:|---:|
| Needle-High | deterministic | 0 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| Needle-High | deterministic | 2 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| Needle-High | deterministic | 5 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| Needle-Low | deterministic | 0 | 48.00 | 0.0400 | 0.0000 | 0.0000 |
| Needle-Low | deterministic | 2 | 48.00 | 0.0400 | 0.0000 | 0.0000 |
| Needle-Low | deterministic | 5 | 48.00 | 0.0400 | 0.0000 | 0.0000 |

The reference visits B=min(N × batch × T,S) distinct IDs; its target is uniform and independent of ordering, so P(success)=B/S exactly over task draws. Here the predicted probabilities are .04 and 1.0. Observed success is empirical; it is never replaced by those values. Any sampling discrepancy must be judged against the binomial interval. Rejection reports only that same candidate's verdict; no hidden branching or cross-candidate pruning exists. The coordinated reference has no duplicates to reduce. Degree-related LLM efficiency changes, if present, can be read separately above.

## H. Chain mechanism

| Condition | Agent | d | Mean maximum depth | Invalid action rate | Future attempts |
|---|---|---:|---:|---:|---:|
| Chain-Short-Horizon | deterministic | 0 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | deterministic | 2 | 6.00 | 0.0000 | 0 |
| Chain-Short-Horizon | deterministic | 5 | 6.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | deterministic | 0 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | deterministic | 2 | 8.00 | 0.0000 | 0 |
| Chain-Sufficient-Horizon | deterministic | 5 | 8.00 | 0.0000 | 0 |

Per-episode depth sequences are in `episodes.jsonl.gz`; means for every round are in `depth_by_round.csv`. The plot extends an already completed chain at its absorbing depth for the remaining horizon, without claiming additional calls. Future nodes are capability-locked; advancing more than one level per round is rejected by construction and checked in the trace. T<C completion is impossible. Correct deterministic agents finish at round C even at d=0. Communication cannot bypass the barrier.

## I. Figures and data

- [Figure 1 — Needle](figures/figure_1_needle.png)
- [Figure 2 — Chain](figures/figure_2_chain.png)
- [Figure 3 — Mechanisms](figures/figure_3_mechanisms.png)
- Each figure also has a vector PDF. Plot inputs: `by_degree.csv`, `depth_by_round.csv`.
- `episodes.jsonl.gz`: all episode actions, messages, responses, seeds, graphs, outcomes and runtime metrics.
- Raw request/response ledgers and `model_calls.jsonl.gz` remain under ignored `runs/needle_chain/pilot/`, following the repository's raw-transcript convention. They are retained locally, not published in this results directory.

## J. Theory comparison and deviations

Preliminary classification: **inconclusive (incomplete pilot)**.

Observed large within-condition degree contrast (≥0.5): **NO**. Whether a nontrivial transition is absent remains **INCONCLUSIVE** from this finite scan and small GPT sample.

The theoretical labels are Needle dc=0; Chain T<C dc="infinity"; Chain T≥C dc=0. These labels are metadata about task structure, not estimates derived from flat data. The paper's claim concerns the absence of a branching/pruning critical point. It does not prove every possible LLM policy has a perfectly flat finite-resource curve.

The public disjoint schedule deliberately removes the need for communication in the reference and provides useful coordination to GPT. Thus this pilot probes these specified policies and resource regimes; it cannot exclude a crossover caused by other policies, other resources or untested degrees.

Structural invariant checks: **passed**, 480 episodes. API-failed decisions: 0. Invalid environment actions: 0. Invalid/refused/truncated JSON decisions: 0.
Deviation categories: A implementation error and E task-definition violation require concrete invariant/test evidence; B finite-sample noise is plausible for rare Needle-Low outcomes; C LLM behavior covers malformed/illegal choices; D API/runtime failure is recorded independently and retained in the success denominator; F genuine empirical deviation remains a possible interpretation if a reproducible contrast survives diagnosis. No failed seed is excluded or rerun for a better outcome.

## K. Measured runtime and future scale

```json
{
  "deterministic_episodes": 480,
  "llm_episodes": 0,
  "deterministic_seconds": 7.374710917007178,
  "llm_seconds": 0,
  "pilot_seconds": 8.848555667093024,
  "api_calls": 0,
  "successful_api_calls": 0,
  "failed_attempts": 0,
  "retries": 0,
  "mean_http_latency_seconds": null,
  "episodes_per_minute": null,
  "calls_per_episode": null,
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "estimated_pilot_cost_usd": 0,
  "full_llm_episodes": 144,
  "estimated_full_calls": null,
  "estimated_full_seconds": null,
  "estimated_full_cost_usd": null,
  "full_executed": false,
  "execution_status": "interrupted",
  "blocker": "No authenticated OpenAI API setup is available: OPENAI_API_KEY is unset; no .env file was found in this checkout, and the requested GitHub repository has no Actions secrets. Supply the location of an existing environment file or authorized runner. No GPT calls were made.",
  "projection_assumptions": "Measured per-condition pilot throughput scaled by agents and episodes at unchanged concurrency. Larger messages, rate limits, provider load and changed stopping rounds can alter it. Token-based cost is an estimate, not a billing statement."
}
```

The execution timer starts after the automatic test preflight and includes stage execution and intermediate analysis; it excludes the final report regeneration. Deterministic/LLM stage timers are separate. Cost uses measured prompt/completion/cache token counts and [published GPT-5.4 Nano rates](https://developers.openai.com/api/docs/models/gpt-5.4-nano). It is not a charge receipt. Full-run estimates scale measured per-condition calls and wall time by the larger population and number of episodes at unchanged concurrency. They do not include an unmeasured larger-context or rate-limit penalty. Null projections mean no live throughput has been measured.

## L. Recommended next step

Provide the location of the existing authenticated API setup, then resume the same pilot command. The 480 completed deterministic episodes will be reused. All 24 GPT episodes and measured GPT runtime/cost projections remain outstanding. Do not launch the full run.

## Reproduction and provenance

See `docs/needle_chain/README.md` for tests, resume, offline reanalysis and the separately gated full command. Preregistered design: `docs/needle_chain/preregistration.md`. Theory: [arXiv:2609.38327v2, §§2.2–3.1](https://arxiv.org/html/2609.38327v2#S3.SS1).
