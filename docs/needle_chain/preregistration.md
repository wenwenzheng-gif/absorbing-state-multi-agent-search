# Needle / Chain: preregistration v1

Written before any pilot episode or live model call. The implementation commit
containing this file will be recorded in every episode. No pilot seeds or
degree-specific prompts will be changed after viewing outcomes.

## Scientific targets

Reference: [arXiv:2609.38327v2, §§2.2, 2.3, 3.1](https://arxiv.org/html/2609.38327v2#S3.SS1),
read 2026-11-07. The critical-degree definition concerns growth versus
contraction of incorrect candidates. Atomic search has no such production
mechanism. Sequential search has an irreducible depth constraint under a
one-level-per-round assumption. We will label the theoretical cases Needle
dc=0, Chain T<C dc="infinity", and Chain T≥C dc=0; these are not fitted empirical
thresholds. The experiment will measure outcomes rather than substitute those
labels for data.

## Frozen design

| Mode | Agents | Degrees | Deterministic seeds | GPT seeds |
|---|---:|---|---|---|
| Pilot | 6 | 0, 2, 5 | 91000–91039 | 91000, 91001 |
| Future full | 20 | 0, 4, 8, 12, 16, 19 | 91000–91199 | 91000–91005 |

| Condition | Pilot resources | Full resources | Prediction |
|---|---|---|---|
| Needle-Low | S=1200; 2 rounds; 4 tests/agent/round; B=48 | S=4000; B=160; same rounds/batch | low at every degree; reference P=.04 |
| Needle-High | S=48; B=48 | S=160; B=160 | high at every degree; reference P=1; GPT ideally ≥.9 |
| Chain-Short-Horizon | C=8, T=6 | same C,T | exactly zero at every degree |
| Chain-Sufficient-Horizon | C=8, T=10 | same C,T | reference exactly one; GPT ideally high at every degree |

Each task seed and its resource budget is identical across degrees and agent
types. The graph seed is task_seed+400000. The existing `Router` uses exact
undirected regular graphs and preserves degree. Target selection, candidate
ordering and chain key generation use distinct SplitMix64 domains. Agent
prompts do not contain the hidden target, task seed, graph seed, degree or
scientific hypothesis. Only received messages differ with connectivity.

Needle IDs are integers with no relational semantics. A public seed-shuffled
permutation supplies disjoint four-ID work assignments for each agent and
round. The reference uses those IDs; GPT may select any legal ID. A single
model call selects the four candidates before batch feedback. This is a
simultaneous batch policy, not a claim of equivalence to four adaptive calls.
Needle uses its complete fixed budget after success so measured coverage has
the same denominator in all episodes. Duplicate and invalid choices are
retained. Since the uniform target and permutation use independent streams,
the reference probability over targets is exactly min(B,S)/S. This control
already coordinates work at d=0: conclusions are limited to these policies.

Chain has one hidden shared pointer sequence and per-agent capability
possession. Every agent starts with the root pointer. Correctly submitting
the current pointer stages completion. Only the global commit unlocks the
next level, once, and supplies its pointer to all valid submitters. An agent
that missed a pointer can obtain it from a neighbor's truthful receipt on the
next round. No future key, even if guessed correctly, can be read or accepted
early. Each agent makes one separate decision for the current round; locked
levels are never batched. Completed episodes stop, with depth retained as an
absorbing state in plots.

Communication uses the existing `Router` message channel as an automatic
truthful one-hop, non-forwarding transport. The environment, not GPT, creates
message contents. The channel contains only the sender's own results and
receipts, never global evaluator metrics. No extra communication calls occur.

## Model and execution plan

Exact model: `gpt-5.4-nano`, resolved from the existing
`configs/agents/gpt-5.4-nano-t1_seed1/synthetic_C0.json`. Use the repository's
OpenAI-compatible `LLMBackend`, `api_style=openai`, temperature=1,
reasoning_effort=none, structured JSON, max_completion_tokens=256, concurrency=6,
timeout=60 seconds, at most two bounded-backoff transport retries. Do not send
unsupported sampling seeds; task reproducibility and raw-response replay are
separate from stochastic model sampling. Do not repair behavioral failures or
replace failed model actions with reference actions. No outcome-based retries.

Pilot sequence: tests; reference Needle; reference Chain; reference analysis
and structural invariants; GPT Needle; GPT Chain; aggregate statistics,
diagnostics, figures and report; commit and push; stop. The planned maximum is
720 successful decision requests (24 episodes); Chain completion can reduce
this. The expected correct-policy total is 648. Retry attempts are extra.
The target execution window is about 30 minutes or less, without a hard kill.
If initial API throughput requires reducing repetitions, record a change to
one paired seed across all four conditions before launching those comparisons;
do not alter task semantics or choose seeds by outcome. No reduction has been
made in this preregistration.

All responses are persisted before action validation. Restart resumes from
round checkpoints and cached calls. A process lock prevents duplicate active
drivers. Successfully persisted responses are never sampled again; an HTTP
call interrupted before a durable response exists is inherently ambiguous and
will be counted as such if retried. Raw prompts/results and terminal technical
failures are preserved. No existing result is removed at startup.

## Analysis and deviation policy

Report all episodes, including API/runtime failures, in the success denominator.
Report Wilson 95% intervals, rounds, unique coverage, duplicate fraction,
invalid-action rate, API-failure rate, depth trajectories, calls, retries,
tokens and actual execution timing. Derive runtime/cost projections from the
measured pilot and published token rates; state extrapolation assumptions.

Endpoint effects are paired by seed, with exact paired discordance tests,
paired-bootstrap intervals and conservative difference intervals from
Bonferroni marginal Wilson bounds. Logistic degree effects use task-cluster
bootstrap and cluster-robust uncertainty where estimable; likelihood/AIC
comparisons are descriptive because within-seed observations are dependent.
All-zero, all-one and separated outcomes produce explicit non-identifiability
statuses. BH/FDR covers available endpoint and regression tests together.
No finite-critical-degree interpolation, crossing search or sigmoid plot is
permitted in this analysis path.

Predictions are not pass/fail filters. Only structural correctness assertions
gate execution. A large contrast such as .05→.15→.95 triggers diagnosis; a
small efficiency improvement is not itself a phase transition. Classify
deviations as A implementation error, B finite-sample noise, C LLM behavioral
failure, D API/runtime failure, E task-definition violation or F genuine
empirical deviation, citing evidence and preserving uncertainty. Never delete
unexpected seeds, resample failures for better answers, modify prompts after
viewing degree outcomes, or tune resources per degree.

Two GPT task seeds cannot establish equivalence, prove the absence of a
transition, or rule out effects at unobserved degrees. This pilot tests the
implementation and provides preliminary evidence. The full configuration is
implemented but must not be executed during this task.

## User-authorized provider clarification before the first live call

After the deterministic pilot, and before any live GPT request, the user
specified the exact snapshot `gpt-5.4-nano-2026-03-17` and authorized reuse of
the existing hosted provider environment. `configs/needle_chain/provider.json`
pins this request model; the original resource config and all deterministic
episode identities are preserved. No prompts, seeds, resource budgets,
communication rules, task semantics or success criteria changed. All 24 GPT
episodes will use this same snapshot and the original two paired seeds.

The prior authenticated mechanism was verified from the `OPENAI_API_KEY`
secret reference, `ubuntu-latest` runner configuration and successful Actions
run `42325666972` in `wenwenzheng-gif/absorb-state-new`. Its log records the
same snapshot in provider responses. The existing secret remains in that
environment and is never printed, copied, committed or modified. A pilot-only
workflow checks out the canonical experimental commit and preserves results
as an artifact; neither repository's `main` changes.
