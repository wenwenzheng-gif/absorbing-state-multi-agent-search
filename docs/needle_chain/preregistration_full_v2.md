# Full v2 Needle / Chain preregistration

Status: frozen before any full-v2 live GPT request. The commit containing this document and `full_v2_freeze.json` is `FULL_V2_PREREGISTRATION_SHA`; execution metadata, each episode manifest and the final report record its exact value. It is pushed to `experiment/needle-chain-no-transition-v1` before hosted execution. This document does not authorize a main merge.

## Scope and preservation

Exactly 4,800 deterministic and 288 GPT episodes, no post-hoc extensions. The v1 commit `bfb1e3220a1be90e05c432bd33a363077bdb2543` and v2 commit `487e6571f9029b571a6611a38b19c820ec93071d` remain frozen. Their 51 processed/preregistration artifacts are checked by `protected_pilots_full_v2.json` before and after execution. Local raw pilot hashes are checked separately. New state is exclusively under `runs/needle_chain/full_v2` and `results/needle_chain/full_v2`.

## Scientific objective and predictions

The [theoretical reference](https://arxiv.org/abs/2609.38327), sections 3.1–3.2, predicts no positive finite communication threshold for atomic Needle search and a sequential horizon constraint for Chain. Primary estimand is verified task success P_success(d), not an estimated critical degree. Expected regimes: Needle-Low approximately 0.04 at all degrees; Needle-High near one; Chain T<C exactly zero; Chain T>=C near one. Needle d_c=0, Chain-Short d_c=infinity, Chain-Sufficient d_c=0 are theoretical structural labels, never finite-sample fitted thresholds. Unexpected outcomes are retained.

## Complete fixed design

N=20, undirected regular communication degrees **[0,4,8,12,16,19]**. All conditions and agent types use the same degree grid.

| Condition | S or C | T | Batch | Slots | Schedule |
| --- | --- | --- | --- | --- | --- |
| Needle-Low | S=4000 | 2 | 4 | 160 | First 160 positions of a public permutation; coverage=0.04 |
| Needle-High | S=160 | 2 | 8 | 320 | Two independently seed-derived complete permutations |
| Chain-Short-Horizon | C=8 | 6 | 1 | 120 | Existing sequential capability environment |
| Chain-Sufficient-Horizon | C=8 | 10 | 1 | 200 maximum | Stops on actual chain completion |

Deterministic task seeds are every integer **92000 through 92199 inclusive**, explicitly enumerated in the JSON config (200). GPT task seeds are **[92000,92001,92002,92003,92004,92005,92006,92007,92008,92009,92010,92011]** (12). Every seed is reused at all six degrees in each condition. Counts: 4×6×200=4800 and 4×6×12=288. No seed replacement, added degree, prompt tuning or extra replication after outcomes are seen.

Graph seed is task seed +400000. The existing Router generates the regular graph from N, degree and this graph seed using the existing controlled graph construction; its implementation is hash-frozen. Graph generation cannot consume a task RNG stream. Task RNG uses `derived_rng`: separate domain constants NEEDLE_TARGET=0x4E434E4545444C01, NEEDLE_ORDER=0x4E434E4545444C02, NEEDLE_REPEAT_ORDER=0x4E434E4545444C03 (plus pass index), CHAIN_KEYS=0x4E43434841494E01. High pass 1 is `candidate_order(160, seed)`; pass 2 is a fresh shuffle using the repeat domain and pass index 1. Each pass assigns exactly eight positions to each of 20 agents. Neither target nor degree is an input to schedule generation. Tests mutate target state and compare schedules; paired task identities and graph seeds are audited across degrees.

Needle has exactly one uniform target, atomic unrelated candidate IDs, no children, branching, similarity or pruning. Rejection reveals nothing about other candidates. LLM schemas permit any integer IDs and do not force schedule following. Low prompt and Chain environment/prompt remain unchanged; High uses the unchanged validated v2 guidance. Observations contain own history and truthful neighbor records delivered only in the next round; no hidden target or aggregate evaluator state is exposed. Needle executes both rounds even after success. Chain capabilities remain locked until the global barrier, including for guessed correct future keys. At most one level advances per round.

## Provider and retry policy

Exact response snapshot **gpt-5.4-nano-2026-03-17**, resolved through the existing `provider.json` lock despite the resource config's family alias. Existing OpenAI chat-completions provider path, base URL https://api.openai.com/v1, temperature=1.0, reasoning_effort=none, max_output_tokens=256, concurrency=6, timeout_s=60, transport_retries=2, repair_attempts=0, per_agent_seed=false. No request-count or experiment wall-clock cutoff. No model substitution. The existing secret reference `OPENAI_API_KEY` is injected only on the authorized hosted runner in `wenwenzheng-gif/absorb-state-new`; its value is never printed, copied, migrated, changed or committed. Checkout and all science/results belong to the canonical frozen branch/SHA. Temporary runner dispatch branches are cleaned up after artifact verification.

The existing transport policy retries retryable timeouts/connections and HTTP 408/409/429/5xx up to two retries with its existing bounded backoff. Successful provider responses are saved before decisions. Terminal errors become logged failed decisions; malformed/invalid actions are counted, not repaired. All planned episodes remain in denominators. No behavioral reruns for favorable outcomes. Ambiguous interrupted HTTP attempts are reported, and checkpoints preserve responses that were successfully saved. Same-source restart skips completed episodes and reuses saved decisions/responses; it only resumes unfinished work. One process owns the run directory lock; episode IDs and request cache keys are deterministic; storage writes are atomic.

## Execution order and gates

Run tests, then all 4800 deterministic episodes, then GPT Needle-Low, Needle-High, Chain-Short, Chain-Sufficient. Archive a checkpoint after the deterministic stage and every GPT condition; always upload a final checkpoint on failure. A same-run infrastructure retry restores the most recent artifact, using the same frozen source. The deterministic gate must pass before any GPT stage: exact counts/paired tasks, no reference invalid actions/API calls, Low exactly 160 unique of 4000 and degree-invariant outcomes, High actual success/complete unique coverage/two valid passes, Short failure, Sufficient success, and all Chain barrier invariants. The expected Low probability is .04; its finite-seed empirical Wilson interval is a reported sampling diagnostic, not a reason to replace seeds. The structural exact coverage and independence determine the .04 theoretical baseline.

Correct-policy GPT workload is 72×(40+40+120+160)=25920 calls; maximum unique decisions is 28800 before at most two transport retries per decision. V2 pilot measured 792 calls in 124.7625 s (about 6.35 calls/s). Initial linear estimate is about 68 minutes; round barriers, larger contexts and provider load can change it materially. Exceeding the projection is not a stopping rule. Cost uses frozen planning rates input $0.20/M, cached input $0.02/M, output $1.25/M, from the existing v2 pricing reference. Actual tokens/calls/runtime will be measured; estimated cost is not a billing statement.

## Outcomes and mechanism definitions

Needle success iff hidden target actually appears among valid tested candidates. Chain success iff final level C=8 actually completed before T. No nominal/potential coverage proxy. Every condition×agent×degree cell reports n, successes, P_success, 95% Wilson interval, mean rounds used, invalid actions/all action slots, and fraction of episodes with terminal API errors; transport failures/retries are separate.

For both Needle conditions: mean unique valid candidates, realized coverage/S, duplicate count and duplicate/valid-test fraction, multiset schedule-departure fraction (unmatched planned slots/nominal slots), fraction of target-containing scheduled batches fully followed, target-tested episode fraction, discovery round. Report valid tests whose candidate was present in received neighbor evidence; denominator for neighbor-informed duplicate fraction is all valid tests made with nonempty received candidate evidence. Separate neighbor-known negatives and known targets. This describes logged behavior, not motive. At d=0 this fraction is undefined, not zero. Full High reference runs naturally have 50% duplicates because the second full pass is deliberate. High reports round-1 and round-2 fractions among successes and all episodes; a round-2 first discovery is a second-pass rescue of an earlier omission. Saturation before feedback limits communication-causal interpretation.

For Chain: maximum depth, mean depth after each global round (completion is absorbing for plots), illegal future attempts, and completion round among successes. Assert per-round depth increment≤1, Short depth≤6, successful Sufficient depth=8.

## Frozen statistical analysis

Marginal 95% Wilson intervals handle exact 0/1 cells. Paired endpoint ΔP=P(19)−P(0), discordant pairs, exact two-sided paired discordance/McNemar p, 1000 seed-pair bootstrap percentile interval, and a conservative difference interval from Bonferroni 97.5% marginal Wilson intervals. A [0,0] empirical bootstrap at constant paired outcomes is not a zero-uncertainty claim; show conservative interval alongside it.

Compare constant-success and degree-logistic Bernoulli working models when identifiable, reporting slope per unit degree, log likelihoods, AIC, seed-cluster sandwich SE, and 1000 whole-seed cluster bootstrap replicates. All-zero/all-one and separated outcomes are nonidentifiable, without fake slopes or p-values. Bootstrap interval is marked unreliable if fewer than 95% resamples are estimable, its distribution is degenerate, or fewer than six clusters exist. Cluster Wald p is only supplied with at least six clusters, nonzero SE and nondegenerate bootstrap. With 12 GPT clusters these tests are exploratory; likelihood/AIC are descriptive working-model diagnostics. Analysis RNG seed=77201.

One primary BH FDR family contains all eight endpoint tests plus only estimable/calibrated degree-slope tests. Report effects, raw p and q separately; nonidentifiable logistic fits contribute no p. Separate exploratory family examines all 15 ordered degree pairs for all eight condition×agent groups (up to 120 exact paired tests), RNG seed=77202. No post-hoc selection of pairs outside this family.

Descriptive crossover thresholds are fixed: LOW-SATURATED iff all six observed probabilities≤0.2; HIGH-SATURATED iff all≥0.8; INTERIOR-CROSSOVER iff some lower sampled degree has P≤0.2 and a larger sampled degree P≥0.8; otherwise NO-CLEAR-PATTERN. An interior crossover is a bracket inside the finite scan, not a fitted d_c. A credible paired crossover additionally requires ≥75% paired seeds changing failure→success over such a bracket and exploratory-family BH q≤0.05. Neither lack of significance nor high saturation proves exact degree invariance.

Final classification: INVALID DUE TO IMPLEMENTATION/PROTOCOL FAILURE if structural or deterministic gates fail; INCONCLUSIVE if incomplete, noisy, no clear expected regime or inadequate regime separation; EVIDENCE OF DEGREE-DEPENDENT TRANSITION if any credible paired interior crossover occurs; otherwise CONSISTENT WITH NO-TRANSITION THEORY only if all eight groups match expected saturation labels, observed within-group range≤0.2, all structural gates pass and the experiment is complete. This descriptive consistency is conditional on the finite scan; it does not establish equivalence or exclude small effects. No follow-up sampling to improve classification.

## Historical comparison and planned figures

Offline historical evidence is frozen in `data/needle_chain/historical_hierarchical`, from `absorb-state-new` commit 236a8e043c433239b03ef809f607b365fc933c81: 24 deterministic N=40, T=8 episodes at degrees 0,1,2,3,4,5,6,8, three seeds 95000–95002. It initializes only incorrect prefixes and records wrong-lineage extinction, not verified task success. Therefore historical P_success cannot honestly be reconstructed. `status: ok` is execution status, not success; existing success-model utilities require missing success outcomes. Figure 5A will explicitly show observed extinction with Wilson intervals as a mechanism comparison; its finite lineage-growth crossing is a different estimand. No new historical model calls. No substitution of the older factorizable threshold tables for Hierarchical success.

Five figures, each PNG and vector PDF: (1) Needle success with deterministic/GPT Wilson intervals; (2) Chain success; (3) Needle realized coverage and duplicate fraction vs degree for both resources; (4) Chain mean depth vs round by degree/agent; (5) historical Hierarchical extinction N=40 versus new GPT Needle and Chain task success N=20, separate degree axes and clearly distinct outcome labels. Tables export every plotted quantity. Figures will be visually inspected after execution.

## Completion and audit

Produce complete report, by-degree and mechanism CSVs, lossless processed episode archive, diagnostics, validation.json, data_audit.json, hosted_provenance.json and full_v2_freeze.json. Audit 5088 unique episode IDs, every cell count, exact snapshot for every successful response, unique logical request keys and response IDs, retries/errors, no missing cells, all source SHAs, protected artifact hashes and frozen design hashes. Record deterministic/GPT stage wall time, HTTP attempts, saved successes, retries/failures, prompt/cached/output tokens, cost estimate and episodes/minute. Inspect figures, run relevant tests and final preservation checks, commit/push final results on the same branch, verify both main branches unchanged, and stop. Post-execution edits are limited to derived reporting/plotting and documented analysis safeguards; scientific configuration remains immutable.
