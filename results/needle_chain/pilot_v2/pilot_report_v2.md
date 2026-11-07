# Needle / Chain pilot v2 report

Status: **COMPLETE**. Readiness: **READY FOR FULL RUN**. **No full experiment launched.**

V1 frozen commit: `bfb1e3220a1be90e05c432bd33a363077bdb2543`. All 22 protected files remain byte-identical.
V2 execution commit(s): `bd94e66c7b65edd797f6751f0ab30d132187f351`.
Branch: `experiment/needle-chain-no-transition-v1`; main is not merged or modified.

## Preregistered change

Needle-High changes from S=48, N=6, T=2, batch=4 (48 slots) to **batch=8 (96 slots = 2S)**. The first 48 public positions are a complete permutation; the next 48 are a second independently shuffled pass. The schedule is target-independent and degree-independent. The stronger prompt recommends the schedule while preserving legal freedom of selection. No branching, pruning, relational feedback or centralized adaptive allocator is introduced.

Needle-Low remains S=1200, T=2, batch=4; Chain remains C=8 with T=6 or T=10. Their prompts, resources and environment logic are unchanged. All conditions use N=6, d={0,2,5}. There are 40 deterministic seeds per cell; GPT has six paired High seeds and two paired sanity-check seeds per unchanged condition. The exact design was committed in `docs/needle_chain/preregistration_v2.md` before v2 execution.

V1 remains valid historical evidence. V2 is a new condition motivated by its zero-slack budget and observed omissions. Resource slack, second-pass scheduling and guidance change jointly; the pilot cannot isolate their individual causal contributions. The versions are never pooled or treated as interchangeable replicates.

## Exact configuration and model

```json
{
  "name": "needle-chain-no-transition-v2",
  "protocol_version": "needle_chain_v2",
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
      "batch_size": 8,
      "horizon": 2,
      "coverage_passes": 2,
      "policy_variant": "needle_high_v2"
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
  "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.4-nano",
  "llm_seeds_by_condition": {
    "Needle-High": [
      91000,
      91001,
      91002,
      91003,
      91004,
      91005
    ]
  }
}
```

Effective provider settings:
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

Returned model IDs: gpt-5.4-nano-2026-03-17.

## Deterministic validation

| Condition | d | Successes / episodes | P(success) | Wilson 95% CI | Coverage | Duplicate fraction |
|---|---:|---:|---:|---|---:|---:|
| Chain-Short-Horizon | 0 | 0/40 | 0.0000 | [0.0000, 0.0876] | — | 0.0000 |
| Chain-Short-Horizon | 2 | 0/40 | 0.0000 | [0.0000, 0.0876] | — | 0.0000 |
| Chain-Short-Horizon | 5 | 0/40 | 0.0000 | [0.0000, 0.0876] | — | 0.0000 |
| Chain-Sufficient-Horizon | 0 | 40/40 | 1.0000 | [0.9124, 1.0000] | — | 0.0000 |
| Chain-Sufficient-Horizon | 2 | 40/40 | 1.0000 | [0.9124, 1.0000] | — | 0.0000 |
| Chain-Sufficient-Horizon | 5 | 40/40 | 1.0000 | [0.9124, 1.0000] | — | 0.0000 |
| Needle-High | 0 | 40/40 | 1.0000 | [0.9124, 1.0000] | 1.0000 | 0.5000 |
| Needle-High | 2 | 40/40 | 1.0000 | [0.9124, 1.0000] | 1.0000 | 0.5000 |
| Needle-High | 5 | 40/40 | 1.0000 | [0.9124, 1.0000] | 1.0000 | 0.5000 |
| Needle-Low | 0 | 1/40 | 0.0250 | [0.0044, 0.1288] | 0.0400 | 0.0000 |
| Needle-Low | 2 | 1/40 | 0.0250 | [0.0044, 0.1288] | 0.0400 | 0.0000 |
| Needle-Low | 5 | 1/40 | 0.0250 | [0.0044, 0.1288] | 0.0400 | 0.0000 |

## GPT results

| Condition | d | Successes / episodes | P(success) | Wilson 95% CI | Coverage | Duplicate fraction |
|---|---:|---:|---:|---|---:|---:|
| Chain-Short-Horizon | 0 | 0/2 | 0.0000 | [0.0000, 0.6576] | — | 0.0000 |
| Chain-Short-Horizon | 2 | 0/2 | 0.0000 | [0.0000, 0.6576] | — | 0.0000 |
| Chain-Short-Horizon | 5 | 0/2 | 0.0000 | [0.0000, 0.6576] | — | 0.0000 |
| Chain-Sufficient-Horizon | 0 | 2/2 | 1.0000 | [0.3424, 1.0000] | — | 0.0000 |
| Chain-Sufficient-Horizon | 2 | 2/2 | 1.0000 | [0.3424, 1.0000] | — | 0.0000 |
| Chain-Sufficient-Horizon | 5 | 2/2 | 1.0000 | [0.3424, 1.0000] | — | 0.0000 |
| Needle-High | 0 | 6/6 | 1.0000 | [0.6097, 1.0000] | 1.0000 | 0.5000 |
| Needle-High | 2 | 6/6 | 1.0000 | [0.6097, 1.0000] | 1.0000 | 0.5000 |
| Needle-High | 5 | 6/6 | 1.0000 | [0.6097, 1.0000] | 1.0000 | 0.5000 |
| Needle-Low | 0 | 0/2 | 0.0000 | [0.0000, 0.6576] | 0.0400 | 0.0000 |
| Needle-Low | 2 | 0/2 | 0.0000 | [0.0000, 0.6576] | 0.0396 | 0.0104 |
| Needle-Low | 5 | 0/2 | 0.0000 | [0.0000, 0.6576] | 0.0396 | 0.0104 |

## Observed Needle-High behavior

```json
{
  "llm_high_episodes": 18,
  "successes": 18,
  "first_round_full_coverage": 18,
  "first_round_target_discovery": 18,
  "first_round_changed_batches": 0,
  "changed_batches": 7,
  "changed_batch_degrees": [
    5
  ],
  "changed_batch_rounds": [
    2
  ],
  "changed_batch_slots": 56,
  "neighbor_known_negative_tests_in_changed_batches": 56,
  "observed_second_pass_target_rescues": 0,
  "causal_limit": "Resource slack, schedule and guidance changed jointly; this pilot does not isolate the effect of extra slots or prove their necessity."
}
```

Of 18 GPT High episodes, 18 covered all 48 IDs in round 1 and 18 found the target in that round. There were 0 first-round assignment departures. The second pass first discovered the target in 0 episodes.

This distinguishes an observed high-resource outcome from a causal claim that the extra pass rescued v1-like omissions. The pilot jointly changes slack and guidance and cannot identify which change produced any improvement.

There were 7 changed High batches, at degrees [5] and rounds [2]. Of 56 actual selections in those batches, 56 retested IDs already reported NOT TARGET by neighbors. These observed actions do not reveal the model's motive. Post-success departures cannot change an already achieved success; they still show that optimal use of evidence is not guaranteed.

Readiness uses the preregistered observed-rate thresholds. It is not a confidence-bound guarantee: even 6/6 successes has a Wilson 95% lower bound of about .61. Six paired seeds cannot establish no degree effect.

## Needle-High v1 versus v2

| Version | Agent | d | Episodes | P(success) | Mean unique IDs | Coverage | Duplicate fraction | Schedule departure fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | deterministic | 0 | 40 | 1.0000 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| v1 | deterministic | 2 | 40 | 1.0000 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| v1 | deterministic | 5 | 40 | 1.0000 | 48.00 | 1.0000 | 0.0000 | 0.0000 |
| v1 | llm | 0 | 2 | 0.5000 | 37.50 | 0.7812 | 0.2188 | 0.2292 |
| v1 | llm | 2 | 2 | 0.5000 | 44.00 | 0.9167 | 0.0833 | 0.0833 |
| v1 | llm | 5 | 2 | 0.5000 | 42.50 | 0.8854 | 0.1146 | 0.1146 |
| v2 | deterministic | 0 | 40 | 1.0000 | 48.00 | 1.0000 | 0.5000 | 0.0000 |
| v2 | deterministic | 2 | 40 | 1.0000 | 48.00 | 1.0000 | 0.5000 | 0.0000 |
| v2 | deterministic | 5 | 40 | 1.0000 | 48.00 | 1.0000 | 0.5000 | 0.0000 |
| v2 | llm | 0 | 6 | 1.0000 | 48.00 | 1.0000 | 0.5000 | 0.0000 |
| v2 | llm | 2 | 6 | 1.0000 | 48.00 | 1.0000 | 0.5000 | 0.0000 |
| v2 | llm | 5 | 6 | 1.0000 | 48.00 | 1.0000 | 0.5000 | 0.0972 |

Matched scheduled IDs use multiset overlap within each agent-round batch, so reordering does not count as a departure. Missing/invalid slots count as replaced. The deterministic v2 duplicate fraction is .5 because the second pass deliberately repeats the candidate space. Duplicate fraction therefore measures both planned redundancy and any extra inefficiency; it should not be compared to v1 as a pure error rate.

`needle_high_episodes.csv` and `needle_high_behavior.json` retain every episode's nominal slots, valid tests, unique coverage, duplicates, followed/replaced IDs, nominal target inclusion, actual target test and discovery round. Changed batches are retained verbatim as candidate IDs. P_coverage=unique IDs/48 is descriptive. Since later choices may respond to TARGET feedback, it is not a calibrated conditional probability for the adaptive GPT policy.

## Degree-effect diagnostics

| Condition | Agent | Endpoint ΔP | Conservative 95% CI | Paired p / BH q | Logistic status |
|---|---|---:|---|---|---|
| Chain-Short-Horizon | deterministic | 0.0 | [-0.11158268671908494, 0.11158268671908494] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Short-Horizon | llm | 0.0 | [-0.7152573451984471, 0.7152573451984471] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Sufficient-Horizon | deterministic | 0.0 | [-0.11158268671908489, 0.11158268671908489] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Chain-Sufficient-Horizon | llm | 0.0 | [-0.7152573451984472, 0.7152573451984472] | 1.0 / 1.0 | not_identifiable_constant_outcome |
| Needle-High | deterministic | 0.0 | [-0.11158268671908489, 0.11158268671908489] | 1.0 / 1.0 | constant high-success outcome; degree slope not identifiable |
| Needle-High | llm | 0.0 | [-0.4557273271830258, 0.4557273271830258] | 1.0 / 1.0 | constant high-success outcome; degree slope not identifiable |
| Needle-Low | deterministic | 0.0 | [-0.1487147103448543, 0.1487147103448543] | 1.0 / 1.0 | estimated; coefficient=-8.09069e-17; bootstrap CI=None |
| Needle-Low | llm | 0.0 | [-0.7152573451984471, 0.7152573451984471] | 1.0 / 1.0 | not_identifiable_constant_outcome |

The endpoint compares paired seeds at d=5 and d=0. The exact discordance test, paired bootstrap and conservative Wilson-based difference interval are retained. Constant/separated outcomes do not receive a fabricated logistic slope. Where identifiable, `diagnostics.json` records likelihoods, AIC, cluster uncertainty and fit counts. Six paired seeds provide limited inference; degenerate bootstrap intervals are not evidence of equivalence. No finite critical degree is interpolated.

## Validation and readiness

```json
{
  "status": "READY FOR FULL RUN",
  "reasons": [
    "Structural and deterministic checks passed; coverage >=.9 and success >=.8 at every degree, with no large observed crossover."
  ],
  "deterministic_theory_checks_passed": true,
  "observed_success_range": 0.0,
  "large_crossover_observed": false,
  "absence_of_transition_established": false,
  "interpretation": "Operational readiness only; six paired seeds cannot establish equivalence or exclude an unobserved transition.",
  "full_run_authorized": false,
  "full_run_executed": false
}
```

Structural invariants passed for 516 episodes. API-failed decisions=0, invalid actions=0, behavioral parse failures=0.
The environment checks atomic verification, exact target success, duplicate accounting, paired resources and one chain level per round. Reference High actions must equal the two complete public passes. Chain-short cannot complete and reference Chain-sufficient completes without communication. Full test and frozen-file audit evidence is saved in `validation.json` and `v1_preservation.json`.

Failures are preserved without favorable reruns. A–F deviation categories remain implementation, finite-sample noise, LLM behavior, API/runtime, task definition, or empirical deviation, respectively. Inspect changed candidate batches and technical counters before attributing differences to degree. An operational READY result would authorize no additional execution.

## Runtime and API usage

```json
{
  "deterministic_episodes": 480,
  "llm_episodes": 36,
  "deterministic_seconds": 3.0901075969999923,
  "llm_seconds": 124.762533031,
  "pilot_seconds": 134.99456628800002,
  "api_calls": 792,
  "successful_api_calls": 792,
  "failed_attempts": 0,
  "retries": 0,
  "mean_http_latency_seconds": 0.7624107134482321,
  "episodes_per_minute": 17.312889915943757,
  "calls_per_episode": 22.0,
  "prompt_tokens": 644868,
  "completion_tokens": 26383,
  "estimated_pilot_cost_usd": 0.16195235,
  "full_executed": false,
  "execution_status": "complete",
  "blocker": null,
  "full_projection_note": "No v2 full resource design is registered or executed. Existing full.json remains v1."
}
```

Stage timings exclude test preflight; the pipeline timer also includes intermediate analysis. Cost is an estimate from measured tokens and the frozen published rates, not a billing receipt. The existing full.json is still the v1 design. A v2 full resource design is not registered or executed by this task.

## Figures and reproducibility

- [Needle success](figures/figure_1_needle.png)
- [Chain success](figures/figure_2_chain.png)
- [Coverage, duplication and depth](figures/figure_3_mechanisms.png)
- [V1–v2 resource-regime comparison](figures/figure_4_v1_v2_comparison.png)
- Vector PDF versions accompany all four figures.
- `episodes.jsonl.gz` contains all processed v2 traces; raw requests/responses remain in ignored `runs/needle_chain/pilot_v2` and the hosted artifact.
- `v1_v2_comparison.csv` is a derived comparison in the v2 directory; it does not modify any original v1 output.

Reproduction commands are in `docs/needle_chain/README_v2.md`. The implementation and preregistration commit is recorded above; the final results commit is the commit containing this report. No full experiment was launched.

Hosted execution: [run 42415318874](https://github.com/wenwenzheng-gif/absorb-state-new/actions/runs/42415318874). The existing runner authentication was used in place; no secret was copied or changed.
