# LLM experiments

Run commands from the repository root. The deterministic pipeline is documented
in [reproducibility.md](reproducibility.md). Frozen multi-model configurations and
pooled results are indexed in [SUPPLEMENTARY_README.md](../SUPPLEMENTARY_README.md).

## 14. LLM agent experiments

`src/agent_system/` re-implements the same protocol (`protocol_version = agent_v1`) behind a policy interface, so uniform, adaptive, LLM and replay decision makers share one runner, one environment and one experiment budget. The frozen pipeline above is untouched; agent runs write only to `runs/agents/<run_id>/`.

With the synthetic environment and the uniform policy, the runner reproduces `src.scaling_model.run_episode` step for step on every scientific field. The physics environment turns each component `(family, variant)` into an additive term of `dv/dt`, and an experiment returns two separate channels: the exact label of one component, and a short trajectory of the true system under a chosen public initial condition.

### Final experiment configurations

| Config | Policy | Cell | Degrees | Task seeds |
|---|---|---|---|---|
| `llm_toy.json` | `llm_full` | synthetic M=8, K=2, T=4, b=2, m=1, N=40 | 9 points in 0..39 | 87000-87003 |
| `llm_physics.json` | `llm_full` | physics support library \|V\|=16, T=4, b=2, m=1, N=40 | 11 points in 0..39 | 82240-82243 |
| `llm_tcas.json` | `llm_full` | tcas component-level \|V\|=46, T=6, b=2, m=1, N=40 | 12 points in 0..39 | 87200-87203 |
| `rule_toy.json`, `rule_physics.json`, `rule_tcas.json` | `uniform` | same cell as the LLM config | every integer 0..39 | the LLM seeds plus two more |
| `llm_smoke.json` | `llm_full` | toy cell | 0 and 8 | 87000 |

All configs live in `configs/agents/`. `synthetic_parity.json` is a small rule run for checking the runner against the frozen reference.

```bash
python -m pip install -r requirements.txt -r requirements-agents.txt
cp .env.example .env    # then set OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY

# check a config and estimate scale; no model calls
python -m src.agent_system.cli validate --config configs/agents/llm_toy.json --env-file .env

# end-to-end check of the LLM setting on a few episodes
python -m src.agent_system.cli sweep --config configs/agents/llm_smoke.json --env-file .env

# the final runs; `shard` runs one process per task seed and merges the results,
# which changes wall time only, never a recorded number
python -m src.agent_system.shard --config configs/agents/llm_toy.json     --env-file .env --workers 4
python -m src.agent_system.shard --config configs/agents/llm_physics.json --env-file .env --workers 4
python -m src.agent_system.shard --config configs/agents/llm_tcas.json    --env-file .env --workers 4

# paired rule controls need no key and never reach the network
python -m src.agent_system.shard --config configs/agents/rule_toy.json     --workers 6
python -m src.agent_system.shard --config configs/agents/rule_physics.json --workers 6
python -m src.agent_system.shard --config configs/agents/rule_tcas.json    --workers 6

# offline re-verification and analysis
# Set this to the actual cell run directory printed by the runner.
RUN_DIR=runs/agents/YOUR_RUN_ID
python -m src.agent_system.cli replay --run-dir "$RUN_DIR"
python -m src.agent_system.analysis.summarize       --run-dir "$RUN_DIR"
python -m src.agent_system.analysis.critical_degree --run-dir "$RUN_DIR"
python -m src.agent_system.analysis.plots           --run-dir "$RUN_DIR"

# overlay a run with its rule control
# Set these to the actual completed run directories; default IDs include timestamps.
LLM_RUN_DIR=runs/agents/YOUR_LLM_RUN_ID
RULE_RUN_DIR=runs/agents/YOUR_RULE_RUN_ID
python -m src.agent_system.analysis.plots \
  --compare "$LLM_RUN_DIR" "$RULE_RUN_DIR" \
  --labels "LLM" "rule" --out-dir runs/agents/compare_toy
```

Four success criteria are reported separately and never collapsed into one flag: `symbolic_recovered` (the agent generated the exact hidden component set), `evidence_verified` (its own labels cover every component of a complete candidate), `predictive_success` (a complete candidate beats the pre-registered hold-out error), and `verified_predictive_success` (both). An illegal model action that survives repair fails the episode loudly rather than falling back to a random choice.

`critical_degree` reports the pooled lineage ratio under two numerator conventions side by side: `pre_own` (the default, and the convention of the frozen `src/utils.py`, so one ratio measures `b (1 - p_nbr)^d`) and `post_own` (the numerator read after the agent's own experiment, so one ratio measures `b (1 - p_own)(1 - p_nbr)^d`).

### Objective × communication experiments (C0 / C1 / I1)

These runs compare three LLM policies. They differ only in two `policy.llm` fields, and all three use `llm_full` with the v4 prompts.

| Policy | `team_objective` | `communication_mode` | What the model is told / does |
|---|---|---|---|
| C0 | `simple` | `fixed` | Improve the team's result. Every own result is sent to direct neighbours automatically and truthfully. |
| C1 | `simple` | `free` | Same objective as C0. After each round the model chooses which of its own results to report and what verdict to claim, and may add up to 500 characters of text. It may omit results, stay silent or misreport. |
| I1 | `individual` | `free` | Improve its own result. Communication is the same as C1. |

Neither objective prescribes a search or division-of-labour strategy.

Under `free`:
- Messages arrive at the start of the next round, one hop, never forwarded.
- A received report is treated as a claim for planning and pruning. An agent's own observation overrides any claim, and among claims the first one in (round, sender, event) order wins.
- Scoring checks actual outcomes, so a false claim alone never produces `evidence_verified` or `verified_predictive_success`.
- Per-round stats in `episodes.jsonl` include `outgoing_reports`, `outgoing_false_reports`, `recv_false_reports`, `recv_conflicting_reports`, `outgoing_messages`, `comm_calls` and `silent_agents`. `silent_agents` counts only eligible agents (degree > 0; the communication phase never runs in the final round) that ended the round having sent nothing, whether by choice or because the call failed after repair -- a degree-0 agent is never counted, since it never had anyone to send to.

The cell parameters are fixed in `scripts/make_configs.py` (`CELL_SPECS`). Reported d_c uses convention `pre_own`, empirical summed over **all rounds t>=1** (`critical_degree`'s default), against theory `delayed_dc`: the empirical value is `critical_degree`'s interpolation estimator with convention `pre_own` (the A-to-A count read right after an agent receives its neighbours' branches, before that agent's own experiments and pruning, so one round's ratio isolates `b (1-p_nbr)^d` -- the paper's derivation of d_c excludes self-pruning, so `pre_own` is the convention comparable to theory). Round 1 has no neighbour messages yet, and under `pre_own` the agent's own pruning is excluded, so round 1's ratio is just `b` with no d-dependence at all (measured log ratio is ~0.64-0.69, i.e. ~ln 2, at every degree) -- that fixed `ln b` term is exactly the `(T-1) ln b` numerator of `delayed_dc` (`src/agent_system/analysis/references.py`; its survival-term denominator is summed from t=2, since that factor genuinely depends on communication not yet delivered in round 1, while its `(T-1) ln b` numerator is kept in full), so summing the empirical `pre_own` ratio from t=1 is the *exact* counterpart to `delayed_dc`, not an approximation of it. Each cell also reports the measured degrees bracketing the interpolation crossing, `bracket=[d_minus, d_plus]`. `pre_own` summed from t=2 (`emp_pre_own_t2`, the previous version's main empirical value), `post_own` summed from t=1 (`emp_post_own_all_rounds`, a self-pruning diagnostic against the current main column) and `exact_dc` (`theory_exact`, sum from t=1) are also still computed and kept in each `dc_table.csv`, just not in the main tables.

| Cell | N | M | K | \|V\| | T | m | n0 | Degrees | Seed list | Theory d_c (delayed) |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic | 40 | 8 | 2 | 16 | 4 | 1 | 8 | 0, 8, 16, 24, 39 | 87000–87003 | 6.10 |
| tcas | 40 | 12 | 10 | 46 (component-level) | 6 | 1 | 8 | 0, 6, 10, 14, 18, 24, 39 | 87200–87203 | 10.90 |
| physics | 40 | 16 | 1 | 16 (support library) | 4 | 1 | 8 | 0, 4, 8, 12, 16, 24, 39 | 82240–82243 | 6.10 |

A folder of 9 configs is named by model and (1-based) task-seed index, `<model-short>_seed<i>`. For example, `configs/agents/qwen3.5-4b_seed1/{synthetic,tcas,physics}_{C0,C1,I1}.json` is Qwen3.5-4B with the 1st task seed of each cell's seed list; `qwen3.5-4b_seed2` and `qwen3.5-4b_seed3` are the 2nd and 3rd. Each such folder's 9 configs carry exactly one task seed. Results go to `runs/agents/<tag>/`. `scripts/merge_runs.py` pools several single-seed run batches (e.g. `qwen3.5-4b_seed1` + `_seed2` + `_seed3`) into one combined analysis tag such as `qwen3.5-4b_seed1-3`, which lives only under `runs/agents/` (merged records, `dc_table.*`) and never gets its own `configs/agents/` folder.

```bash
# 1. local vLLM only: point .env at the served model and check the server
#    (health, model id, one tiny request; this makes a model call)
scripts/check_vllm.sh

# 2. generate the 9 configs for one or more task-seed indices (1-based)
python3 scripts/make_configs.py --model Qwen/Qwen3.5-4B --seed-index 1        # -> tag qwen3.5-4b_seed1
python3 scripts/make_configs.py --model Qwen/Qwen3.5-4B --seed-index 1 2 3    # -> _seed1, _seed2, _seed3, one folder each

# 3. see what would run, then smoke one degree of one cell
scripts/run_experiments.sh --tag qwen3.5-4b_seed1 --dry-run
scripts/run_experiments.sh --tag qwen3.5-4b_seed1 --smoke --env tcas --policy C1

# 4. run all 9 cells: up to 3 cells at once (-j), every degree of a cell in its own
#    process (--split degree), at most 40 in-flight LLM requests per process
scripts/run_experiments.sh --tag qwen3.5-4b_seed1 -j 3 --split degree --workers 7 --concurrency 40

# 5. empirical vs theory d_c table (+ communication stats for C1/I1)
scripts/analyze.sh --tag qwen3.5-4b_seed1    # -> runs/agents/qwen3.5-4b_seed1/dc_table.{csv,md}
```

- `run_experiments.sh` checks every selected config before launching anything. The config must be `llm_full` with `prompt_version` v4, its `policy.llm.model` must match `OPENAI_MODEL` in `.env`, and `validate` must pass.
- It skips cells that have already finished unless `--force` is given. `--env` and `--policy` select a subset, for example `--env tcas --policy C1,I1`.
- Logs go to `runs/agents/<tag>/logs/`.
- Sharding by degree or seed, and the `--concurrency` setting, change wall time only. Episode seeds depend on the task seed alone, and the merged run directory has the same layout as an unsharded run.
- The model that actually answers is the one in `OPENAI_MODEL` in `.env`. The `model` field in a config is a record and a guard.

## 15. Agent context and action space

The scheduler follows a fixed workflow: receive evidence, prune, branch, allocate experiment slots and execute experiments, prune and log, then deliver messages next round. The model selects actions only at designated stages. Each model call is stateless: `AgentRuntime` holds local state, a whitelist constructs `AgentView`, and one template per phase renders it. The templates are `branch.txt`, `experiment.txt` and `prune.txt` in `src/agent_system/policies/prompts/`, and every LLM episode records their SHA-256 as `prompt_sha256`.

### Context composition

| Context block | Contents | Switch |
|---|---|---|
| **Task and rules** | Search objective, component composition rules, current phase responsibilities, JSON output format, and action constraints | Always |
| **Public component library** | Every available component's `family_id`, `variant_id`, name, expression, and public parameters; physics tasks include equation terms | Always in branch and experiment prompts; the prune prompt lists the components of each hypothesis under review instead |
| **Progress** | Current round and total rounds; branching also includes target task size, parent depth, and child depth | Always; remaining rounds and nominal budget come with `context_identity` |
| **Local evidence** | Labels and values still allowed per family, including both own and received evidence | Always; TCAS configuration-level verdicts apply to whole configurations |
| **Current decision objects** | Branch: this batch's parents, their `legal_additions`, and `required_additions`; experiment: components available for this slot and their generating parents; prune: this batch's hypotheses and newly arrived evidence | Always; large hypothesis collections are split across model calls |
| **Public experiment library** | Available `experiment_id` values, condition descriptions, and public settings such as initial conditions | When the model chooses conditions and the environment provides a library |
| **Private observations** | The agent's own initial and experimental observations | Omitted when empty; `physics_observations` selects `"summary"` (default) or `"raw"` trajectories |
| **Identity and communication** | Agent ID, team size, neighbour IDs, degree, nominal budget, and the one-round, one-hop communication rule | `context_identity`, default on |
| **Evidence provenance and history** | Own versus received labels with source and round, plus the agent's tests and outcomes in earlier rounds | `evidence_attribution`, default on |
| **Team objective** | Asks for less redundant exploration; `device` adds a rule that differentiates choices by agent identity | `team_objective`, default `"device"`; `"soft"` or `"off"` for ablations |

Two further switches change no prompt text: `shuffle_candidates` permutes the candidate order per agent and round, and `per_agent_seed` sends a per-request sampling seed. Both default to on. Neither exposes hidden truth. A config that names no switch gets the final setting above.

Hidden truth, other agents' hypotheses and reasoning, evidence not yet received, and held-out evaluation data are excluded from context. Agents send only their own experimental results to direct neighbours for delivery next round; received evidence is never forwarded, and private trajectories are never shared. See [`views.py`](../src/agent_system/views.py) for the context whitelist and [`openai_policy.py`](../src/agent_system/policies/openai_policy.py) for prompt rendering.

### Action space and parameters

The current phase and offered candidates define the action space. A component label such as `c3v2` denotes `(family_id=3, variant_id=2)`; a hypothesis ID such as `h_c0v1_c2v0` denotes the unordered component set `{c0v1, c2v0}`. The table below lists the JSON parameters the model actually returns.

| Action | Parameters | Constraints and effect |
|---|---|---|
| **Branch: extend hypotheses** | `updates: object<parent_id, component_label[]>`; `reason: string` | Must cover every parent in the batch. Each array contains distinct entries from that parent's `legal_additions`, with exactly `required_additions = min(b, number of legal extensions)` entries. Each selected component creates a separate child with one more component than its parent; identical children merge |
| **Experiment: perform one test** | `choice: string` in the form `"<component>\|<parent_id>"`; conditional `experiment_id: string`; `reason: string` | `choice` must be an offered component–generating-parent combination. When the model chooses conditions and a library exists, `experiment_id` must name an offered condition; otherwise the planner/environment determines conditions. One action consumes one experiment slot, and the environment returns the outcome |
| **Prune: remove refuted hypotheses** | `drop: hypothesis_id[]`; `reason: string` | Called only by `llm_prune`. IDs must belong to the current batch; return `[]` if none should be removed. Runs after receiving evidence and after the round's local experiments, only when new evidence and a nonempty frontier exist |

For example, if parent `h_c0v1` requires two legal extensions, the model can return:

```json
{"updates":{"h_c0v1":["c3v2","c4v0"]},"reason":"Explore two plausible extensions."}
```

This creates two children, `{c0v1,c3v2}` and `{c0v1,c4v0}`. Experiment and prune outputs take the following forms; all IDs must actually appear in the offered candidates:

```json
{"choice":"c3v2|h_c0v1","experiment_id":"<offered experiment_id>","reason":"Test an unresolved component."}
```

```json
{"drop":["h_c0v1_c3v2"],"reason":"Contradicts a known negative label."}
```

All three actions include a short `reason`. With `reasoning="reason_first"`, the schema also requires a length-limited `analysis: string` field before the action. `reasoning="thinking"` uses the backend's thinking settings without adding action parameters. Illegal outputs are repaired up to `repair_attempts`; an action that remains illegal causes a reported failure rather than a silent random fallback.

`policy.kind` determines which decisions the model makes:

| Policy | Branching | Test component and parent | Experiment conditions | Pruning |
|---|---|---|---|---|
| `llm_branch` | LLM | Random component; parent selected by code | Planner | Deterministic code |
| `llm_branch_test` | LLM | LLM | Planner | Deterministic code |
| `llm_full` | LLM | LLM | LLM, when a library exists | Deterministic code |
| `llm_prune` | LLM | LLM | LLM, when a library exists | LLM |

The default `llm_prune_apply="validated"` applies a requested deletion only if the hypothesis contradicts local evidence; `"raw"` applies the model's choices directly. Other policies prune immediately after each experiment, while `llm_prune` makes its local pruning decision after the round's experiments. Budget allocation, message delivery, and episode termination belong to the scheduler, not the model's action space. The default global experiment budget is `N × m` per round, and an individual agent may receive more or fewer than `m` slots. Environment configuration can also override the chosen experiment condition, as with TCAS `max_containment`.

## 16. Run data

Run outputs (`manifest.json`, `episodes.jsonl`, `events.jsonl`, the LLM `requests.jsonl` and `responses.jsonl` used for replay, `analysis/` and `figures/`) go under `runs/`.

LLM runs cannot be reproduced bit for bit, so keep their `requests.jsonl` and `responses.jsonl`; `cli replay` re-runs an episode offline from them.

## Hosted API and local serving

The `check_vllm.sh` and `run_experiments.sh` convenience scripts assume a local
vLLM `/health` endpoint. For a hosted OpenAI endpoint, set the hosted values
shown in `.env.example`, then use the existing Python entry points directly:

```bash
# Validation checks configuration and request counts without contacting the model.
python -m src.agent_system.cli validate \
  --config configs/agents/gpt-5.4-nano_seed1/synthetic_C0.json --env-file .env

# This command makes paid model calls. Use the same pattern for the other
# existing environments, policies and seed folders; do not regenerate frozen configs.
python -m src.agent_system.shard \
  --config configs/agents/gpt-5.4-nano_seed1/synthetic_C0.json \
  --env-file .env --workers 5 --split degree

scripts/analyze.sh --tag gpt-5.4-nano_seed1
```

For local Qwen serving, install vLLM separately in an appropriate GPU environment
and run `serving/serve_vllm.sh Qwen/Qwen3.5-4B` in a separate terminal. The repository's Python requirements
do not install the serving stack. Keep `.env` local; it is ignored by Git.

The `make_configs.py` commands above describe how the archived configurations were
constructed and how to create a new experiment. They are not needed to reproduce
the included frozen configurations; generating the same tags may overwrite them.

Only pooled tables are included in this repository. Raw LLM request/reply logs
are not attached, and no public raw-log download URL is supplied. Offline replay
requires those logs from an actual run. Fresh LLM calls are not bit-reproducible.
