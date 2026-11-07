# Needle / Chain pilot v2 preregistration

Frozen before any v2 pilot episode or live request. The commit containing
this design, configuration, prompts and tests is the v2 execution source.
Episode manifests record its SHA. This is one new design and one execution;
there will be no outcome-driven budget increases, seed replacements, prompt
changes, exclusions or selective reruns. The full experiment is not authorized.

## Motivation and preservation

V1 is frozen at `bfb1e3220a1be90e05c432bd33a363077bdb2543` and remains valid
historical evidence. Its processed files and original preregistration are
protected by `v1_frozen_hashes.json`. Existing local raw files are also hashed
before v2 and will be checked afterward. V2 uses distinct configuration,
run and result directories; it does not rewrite or pool v1 observations.

V1 Needle-High had S=48, N=6, T=2 and batch=4: exactly 48 nominal slots,
with no redundancy. The deterministic reference covered all IDs and always
succeeded. GPT succeeded in 1/2 seeds at each degree; realized coverage ranged
from 31/48 to 45/48. The three failures omitted target-containing public
assignments. All 11 changed high-resource batches were in round 1, before
neighbor evidence. This motivates resource slack for a stochastic policy;
it does not invalidate v1 or establish a communication-driven transition.

## Exact frozen changes

Only Needle-High changes its task resources: **S=48, N=6, T=2, batch=8**,
giving **96 slots = 2S**. Its public schedule becomes two complete coverage
passes. Pass 1 uses the unchanged seed-shuffled v1 candidate ordering; pass 2
is independently shuffled in a distinct RNG domain (`NEEDLE_REPEAT_ORDER`,
pass index 1). Each pass is a permutation of 0–47. Slots are ordered by round,
then agent ID, then batch position. Thus all 48 IDs are assigned before any
scheduled repetition, with exactly one complete pass per round.

The schedule takes only public resource dimensions and the task seed. It
never reads the hidden target, private actions, observed outcomes or degree.
The target uses the unchanged separate RNG domain. There is no adaptive
central allocation. Every degree receives identical assignments and prompt
instructions; only naturally received neighbor receipts differ.

Needle-High alone receives stronger default-assignment guidance: follow the
collision-free assignment within each pass; avoid arbitrary low-ID/pattern
substitution unless observed information gives a concrete reason; departures
remain legal. Repetition across the two passes is explicitly intentional.
The complete prompt is `NEEDLE_HIGH_V2_PROMPT` in `policies.py`. The schema
still allows any integer selections, including duplicates; the environment
alone determines validity. No scientific hypothesis is in the agent prompt.

This jointly changes resource slack, the second schedule pass and task-level
guidance. A v1–v2 difference cannot isolate their individual causal effects.
V2 is a new preregistered condition, not a statistical correction to v1.

## Unchanged conditions and common setup

| Condition | S or C | T | Batch | Nominal slots | GPT paired seeds |
|---|---:|---:|---:|---:|---|
| Needle-Low | S=1200 | 2 | 4 | 48 | 91000, 91001 |
| Needle-High v2 | S=48 | 2 | 8 | 96 | 91000–91005 |
| Chain-Short-Horizon | C=8 | 6 | 1 | 36 | 91000, 91001 |
| Chain-Sufficient-Horizon | C=8 | 10 | 1 | at most 60 | 91000, 91001 |

All conditions use N=6 and d in {0,2,5}; task seeds are paired across degree.
All deterministic conditions use seeds 91000–91039, 40 per cell (480 total).
There are 18 mandatory Needle-High GPT episodes plus 18 optional unchanged
sanity-check episodes (36 total). No v1 outcome is reused as a v2 replicate.
The unchanged conditions retain their exact resource definitions, prompts,
target/key generation, communication and action semantics. Their new GPT
samples are sanity checks, not replacements for the v1 observations.

Needle remains atomic, with one hidden target and binary verification. No
rejection affects another candidate. The complete fixed budget is used even
after target discovery. Chain's environment and one-level-per-round barrier
are unchanged; it stops at completion. Graphs use the same seeded exact-degree
Router, graph_seed=task_seed+400000. No hidden degree-dependent coordinator,
global private history or extra message-generation API call is introduced.

## Model, execution and stopping

Model: **gpt-5.4-nano-2026-03-17**, pinned by the existing provider lock.
Temperature=1, reasoning_effort=none, max output tokens=256, concurrency=6,
timeout=60s, at most two transport retries, no behavioral repair/fallback,
no provider sampling seed. The existing hosted Actions secret is consumed
in place in `wenwenzheng-gif/absorb-state-new`; it is not exposed or migrated.

Sequence: commit and push this preregistration and implementation; tests;
480 deterministic episodes; analysis and structural checks; 36 GPT episodes;
complete analysis, figures and v2 report; verify v1 hashes; tests; commit/push;
STOP. Maximum decision calls=864; correct-chain stopping gives 792 (216 for
Needle-High, 72 for Needle-Low, 216 for Chain-Short and 288 for Chain-Sufficient).
V1 measured 648 calls in 99.65s; comparable throughput suggests roughly two
minutes for v2 model execution, with setup and larger batches additional.
This is a planning estimate, not a wall-clock kill or a measured v2 result.
All planned seeds remain fixed. Durable response caches and round checkpoints
resume technical interruptions; completed successful calls are not resampled.

## Predictions and diagnostics

Needle-Low should remain low at every degree (reference probability .04).
Needle-High v2 should be high at every degree (reference probability 1).
Both deterministic coverage passes must be mechanically verified. GPT's
diagnostic targets at each degree are success >=.8 (at least 5/6) and mean
realized unique coverage >=.9, ideally both near 1. Chain-Short must have
success 0 and Chain-Sufficient reference success 1; GPT ideally remains high.
These are readiness diagnostics, never data inclusion rules.

Every Needle-High episode reports valid total tests, unique IDs, coverage/48,
duplicate counts/fractions, scheduled IDs followed/replaced, target presence
in the nominal schedule, actual testing and first discovery round. Schedule
following is multiset overlap within an agent-round batch; reordering the same
IDs is not a departure. Replaced fraction is (nominal slots - matched slots)
/ nominal slots; invalid or absent selections count as unmatched. Deliberate
second-pass repetition means reference duplicate fraction is **.5**, so a
larger duplicate fraction than v1 is not by itself worse coordination.

Coverage/48 is reported as P_coverage, alongside the observed success
indicator and their degree means. It is a coverage diagnostic, not a calibrated
success probability for an adaptive target-dependent policy: later choices
may respond to earlier TARGET feedback. The reference has complete coverage
and exact success 1. Empirical successes are never replaced by coverage.

Primary analysis: success vs degree, Wilson 95% intervals, paired endpoint
Delta P=P(5)-P(0), exact paired discordance tests and conservative endpoint
intervals. The existing constant-vs-logistic diagnostic is used only when
identifiable; constant outcomes have no slope. Cluster-bootstrap reliability
guards and BH/FDR remain as documented in v1 analysis notes. Six paired seeds
do not establish equivalence or absence of a transition. No finite-dc fit.

A dedicated table and three-panel figure compare v1 and v2 Needle-High:
separate success curves, realized coverage, duplicate fraction and schedule
departure fraction. V1 and v2 are never pooled; overlapping task seeds are
acknowledged. All failures, invalid actions and technical errors remain in
the analysis and are diagnosed using the established A–F categories.

## Readiness decision fixed in advance

READY FOR FULL RUN requires completed data, structural checks, deterministic
theory checks, GPT Needle-High mean coverage >=.9 and success >=.8 at every
degree, and no large observed degree contrast (range >=.5). These are
operational pilot diagnostics, not statistical proof. Deterministic Low is
assessed against its .04 baseline using the Wilson interval.

NOT READY applies to structural/task/resource failures, GPT Needle-High mean
coverage <.9 at any degree, or a demonstrated implementation issue. If
structure is correct but incomplete/noisy success outcomes miss the high
success target without poor coverage, classify INCONCLUSIVE. A large degree
contrast requires diagnosis and prevents READY. No automatic retuning follows
any status. Even READY requires separate authorization for a full run.
