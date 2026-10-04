# Supplementary code and results

This repository contains the code, configurations and processed results for
"Absorbing State Phase Transitions in Multi-Agent Search" by Wenwen Zheng,
Yuzhe Yang, Helen Qu, Xin Eric Wang and Haewon Jeong
([arXiv:2609.38327](https://arxiv.org/abs/2609.38327)).

- [README.md](README.md) is the public overview and quick start.
- [docs/reproducibility.md](docs/reproducibility.md) preserves the model (§1–8), the d_c estimator and theory (§9–10), and the rule-based results (§11–13).
- [docs/llm_experiments.md](docs/llm_experiments.md) preserves the LLM-agent protocol and action-space documentation (§14–16).
- This file indexes the archived experiments, rerun commands and result tables.

## Contents

| Path | What it is |
|---|---|
| `src/` | Rule-based population simulator and its analysis pipeline (`scaling_model.py`, `scan_dc.py`, `bootstrap_dc.py`, `fit_scaling.py`, ...). |
| `src/agent_system/` | LLM-agent experiment stack: environments (synthetic, TCAS, physics), the scheduler (`runner.py`), the LLM policy and prompt templates (`policies/`), sharded execution (`shard.py`), and the d_c analysis (`analysis/`). |
| `configs/` | Frozen parameter registry and clean-regime criteria for the rule-based study (`final_parameter_registry.*`, `clean_criteria.json`, `seeds.json`). |
| `configs/agents/<tag>_seed<i>/` | The exact configs of every LLM-agent run: 9 files per folder (3 environments × policies C0/C1/I1), one task seed per folder, seeds 1–6. |
| `scripts/` | Drivers for the LLM-agent runs: `make_configs.py` (generates `configs/agents`), `run_experiments.sh`, `merge_runs.py` (pools seeds), `analyze.sh` / `dc_table.py` (d_c tables), `check_vllm.sh` (endpoint check). |
| `serving/serve_vllm.sh` | The vLLM command used to serve the Qwen3.5 models. |
| `data/`, `provenance/` | Processed data and bootstrap draws for the rule-based results; reference values and file hashes. |
| `results/llm_agents/<tag>/dc_table.{md,csv}` | Pooled (six-seed) d_c tables of every LLM-agent experiment, as produced by `scripts/analyze.sh`. |
| `tests/` | Unit and regression tests (`python -m pytest -q tests`). |

## Installation

```bash
conda env create -f environment.yml      # or: python -m pip install -r requirements.txt
conda activate dc-scaling-reproducible   # when using Conda
python -m pip install -r requirements-agents.txt  # also needed by the complete test suite
python reproduce.py --test
# Optional alternative test runner:
python -m pip install pytest
python -m pytest -q tests
```

Python 3.12 is recommended. All commands assume the repository root.

## A. Rule-based simulations

```bash
python reproduce.py --test     # unit and regression tests
python reproduce.py --quick    # refit and redraw the figures from the compact processed data in data/
python reproduce.py --full     # rerun every frozen simulation, then the bootstrap, classification and fit
```

`--quick` checks regenerated tables against `data/` and the fit uses the numerical references in `provenance/`. See [docs/reproducibility.md](docs/reproducibility.md), §13.

## B. LLM-agent experiments

### Setup

- **Grid.** Each experiment ("tag") is 3 environments × 3 policies × 6 task seeds × 5–7 communication degrees: 342 episodes.
- **Policies.**
  - C0: team objective, fixed communication.
  - C1: team objective, free communication.
  - I1: individual objective, free communication.
- **Environment parameters** (agents N = 40, branching b = 2) are fixed in `scripts/make_configs.py` (`CELL_SPECS`):

| Environment | T | \|V\| | Degrees | Task seeds (#1–#6) | Theory d_c (`delayed_dc`) |
|---|---|---|---|---|---|
| synthetic | 4 | 16 | 0, 8, 16, 24, 39 | 87000–87005 | 6.10 |
| tcas | 6 | 46 | 0, 6, 10, 14, 18, 24, 39 | 87200–87205 | 10.90 |
| physics | 4 | 16 | 0, 4, 8, 12, 16, 24, 39 | 82240–82245 | 6.10 |

- **Shared across all tags.** Environments, degrees, task seeds, prompts (v4) and repair budget (3) are identical in every tag. Only the model and its sampling / reasoning settings change:

| Tag | Model | Setting | Configs generated with |
|---|---|---|---|
| `qwen3.5-4b` | Qwen/Qwen3.5-4B, local vLLM | temperature 0.2, thinking off | `make_configs.py --model Qwen/Qwen3.5-4B --seed-index 1 2 3 4 5 6` |
| `qwen3.5-4b-t1` | Qwen/Qwen3.5-4B | temperature 1 | `... --temperature 1 --tag-suffix=-t1` |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B, local vLLM | temperature 0.2, thinking off | `make_configs.py --model Qwen/Qwen3.5-9B --seed-index 1 2 3 4 5 6` |
| `qwen3.5-9b-t1` | Qwen/Qwen3.5-9B | temperature 1 | `... --temperature 1 --tag-suffix=-t1` |
| `gpt-5.4-nano` | gpt-5.4-nano, OpenAI API | temperature 0.2, reasoning_effort none | `make_configs.py --model gpt-5.4-nano --seed-index 1 2 3 4 5 6 --api openai --temperature 0.2` |
| `gpt-5.4-nano-t1` | gpt-5.4-nano | temperature 1 | `... --api openai --temperature 1 --tag-suffix=-t1` |
| `gpt-6-luna` | gpt-6-luna, OpenAI API | reasoning_effort none (temperature fixed at 1 by the API) | `make_configs.py --model gpt-6-luna --seed-index 1 2 3 4 5 6 --api openai` |
| `gpt-6-luna-re-<effort>` | gpt-6-luna | reasoning_effort low / medium (API default) / high / xhigh | `... --api openai --reasoning-effort <effort> --max-output-tokens 128000 --timeout 1800 --tag-suffix=-re-<effort>` |

### Running one tag

The following convenience-script commands assume a **local vLLM** endpoint.
For hosted OpenAI models, use the direct Python CLI/shard commands in
[docs/llm_experiments.md](docs/llm_experiments.md#hosted-api-and-local-serving).
The endpoint check and experiment commands make model calls. Existing frozen
configs are already included; config generation is not required for these runs.

```bash
cp .env.example .env                    # set OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY
serving/serve_vllm.sh Qwen/Qwen3.5-4B   # run in a separate terminal; Qwen tags only
scripts/check_vllm.sh                   # endpoint health, model id, one test request
for i in 1 2 3 4 5 6; do
  scripts/run_experiments.sh --tag qwen3.5-4b_seed$i -j 4 --split degree --workers 7 --concurrency 40
  scripts/analyze.sh --tag qwen3.5-4b_seed$i
done
python3 scripts/merge_runs.py --out-tag qwen3.5-4b_seed1-6 --from-tags qwen3.5-4b_seed{1,2,3,4,5,6}
scripts/analyze.sh --tag qwen3.5-4b_seed1-6   # -> runs/agents/qwen3.5-4b_seed1-6/dc_table.{md,csv}
```

- **What the runner checks.** `run_experiments.sh` refuses to start unless each config's model matches the served model, and it skips cells that are already complete.
- **Speed settings.** Sharding (`--split degree --workers`) and `--concurrency` change wall time only, not episode seeds.
- **Failed shards.** A shard whose episode ends `policy_failed` can be rerun with the unchanged protocol. `merge_runs.py --allow-failed` pools the remaining `ok` episodes. Every pooled table in `results/` is from `ok` episodes only; `qwen3.5-9b` contains one `policy_failed` episode (tcas I1, d = 0).
- **Replay.** LLM calls are not bit-reproducible. Each run records every request and reply (`requests.jsonl`, `responses.jsonl`), and `python -m src.agent_system.cli replay --run-dir <cell dir>` replays a cell offline.

### d_c estimator

- **Log ratio.** Per round, the pooled lineage ratio Σ `old_pre`_t / Σ `A_post`_(t−1) uses convention `pre_own`. The count is taken after receive and branch, before the agent's own experiments and pruning, so it excludes self-pruning as the theory does.
- **Sum.** log R(d) is summed over all rounds t = 1…T−1. Its zero is the empirical d_c, found by linear interpolation between the two bracketing measured degrees.
- **Interval.** Brackets are a bootstrap over tasks.
- **Theory.** `delayed_dc` = −(T−1) ln b / Σ_(t=2)^(T−1) ln(1 − m t/|V|).

See [docs/llm_experiments.md](docs/llm_experiments.md), §14, and `src/agent_system/analysis/`.

### Results (six seeds pooled; d_c [bootstrap interval])

**`qwen3.5-4b`** — Qwen/Qwen3.5-4B (local vLLM), temperature 0.2

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.49 [6.01, 7.00] | 13.40 [11.73, 14.98] | 13.23 [12.31, 13.62] |
| C1 | 12.87 [10.17, 16.10] | 24.20 [17.78, 26.08] | 11.43 [10.11, 13.14] |
| I1 | 13.33 [10.63, 16.49] | 18.91 [13.80, 21.46] | 12.95 [11.40, 17.34] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`qwen3.5-4b-t1`** — Qwen/Qwen3.5-4B (local vLLM), temperature 1

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.55 [6.21, 6.88] | 12.79 [11.40, 13.74] | 8.77 [7.71, 9.96] |
| C1 | 12.82 [12.20, 13.41] | 17.69 [16.50, 25.15] | 13.94 [12.17, 14.57] |
| I1 | 13.47 [11.64, 16.04] | 20.13 [16.78, 24.79] | 14.31 [11.46, 16.95] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`qwen3.5-9b`** — Qwen/Qwen3.5-9B (local vLLM), temperature 0.2

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.93 [6.14, 7.59] | 16.27 [12.52, 18.53] | 9.33 [7.55, 10.27] |
| C1 | 17.14 [13.06, 19.98] | no crossing up to d=39 | 12.40 [11.32, 14.14] |
| I1 | 14.82 [12.17, 24.24] | no crossing up to d=39 | 17.45 [9.77, 19.90] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`qwen3.5-9b-t1`** — Qwen/Qwen3.5-9B (local vLLM), temperature 1

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.64 [6.22, 6.99] | 12.90 [10.92, 15.15] | 8.34 [7.68, 8.85] |
| C1 | 16.01 [13.80, 17.35] | 29.15 [23.62, 37.27] | 17.07 [15.04, 18.91] |
| I1 | 16.06 [14.12, 18.23] | 27.31 [24.76, 29.56] | 16.77 [14.17, 18.50] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-5.4-nano`** — gpt-5.4-nano (OpenAI API), temperature 0.2, reasoning_effort none

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 7.27 [6.65, 8.04] | 10.41 [8.95, 12.11] | 22.83 [19.79, 26.66] |
| C1 | 7.34 [6.81, 7.92] | 11.15 [8.92, 13.62] | 23.47 [15.21, 26.68] |
| I1 | 7.69 [7.09, 8.36] | 11.74 [9.38, 13.48] | 24.56 [15.17, 28.12] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-5.4-nano-t1`** — gpt-5.4-nano (OpenAI API), temperature 1, reasoning_effort none

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.39 [5.75, 7.14] | 10.52 [8.67, 11.56] | 10.95 [9.80, 11.99] |
| C1 | 7.87 [7.05, 9.00] | 12.88 [9.79, 14.54] | 13.04 [11.28, 16.67] |
| I1 | 7.29 [6.76, 7.82] | 14.41 [9.73, 16.14] | 12.79 [11.81, 13.44] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-6-luna`** — gpt-6-luna (OpenAI API), temperature 1 (fixed by the API), reasoning_effort none

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.88 [6.37, 7.33] | 10.05 [9.56, 11.25] | 8.17 [7.33, 9.00] |
| C1 | 7.06 [6.76, 7.34] | 13.71 [9.97, 18.01] | 8.48 [7.91, 9.15] |
| I1 | 6.46 [6.01, 6.98] | 13.13 [9.29, 14.93] | 7.88 [7.39, 8.75] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-6-luna-re-low`** — gpt-6-luna (OpenAI API), reasoning_effort low

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.78 [6.31, 7.16] | 15.64 [13.28, 17.08] | 8.45 [7.45, 9.05] |
| C1 | 7.11 [6.45, 7.86] | 37.59 [19.94, 38.76] | 8.84 [7.08, 10.14] |
| I1 | 6.85 [6.41, 7.33] | 29.16 [20.90, 33.42] | 8.76 [7.45, 9.57] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-6-luna-re-medium`** — gpt-6-luna (OpenAI API), reasoning_effort medium (API default)

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.80 [6.31, 7.18] | 17.76 [14.47, 22.39] | 9.74 [8.25, 11.13] |
| C1 | 7.33 [6.76, 7.95] | 34.25 [25.12, 38.46] | 10.13 [7.93, 14.76] |
| I1 | 7.34 [6.94, 7.77] | no crossing up to d=39 | 10.92 [9.18, 13.14] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-6-luna-re-high`** — gpt-6-luna (OpenAI API), reasoning_effort high

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 6.98 [6.80, 7.17] | 15.30 [12.54, 19.92] | 12.18 [10.74, 13.77] |
| C1 | 7.02 [6.72, 7.36] | no crossing up to d=39 | 17.20 [10.88, 19.40] |
| I1 | 7.07 [6.79, 7.42] | no crossing up to d=39 | 16.58 [7.81, 21.46] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

**`gpt-6-luna-re-xhigh`** — gpt-6-luna (OpenAI API), reasoning_effort xhigh

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | 7.08 [6.65, 7.57] | 18.82 [13.18, 21.96] | 19.64 [13.54, 28.14] |
| C1 | 7.23 [6.94, 7.54] | 26.97 [19.63, 37.54] | 22.01 [11.69, 37.59] |
| I1 | 7.57 [7.22, 7.89] | 28.98 [18.57, 36.96] | 27.70 [14.83, 37.46] |
| theory (`delayed_dc`) | 6.10 | 10.90 | 6.10 |

- **"no crossing up to d=39".** log R stays positive at every measured degree; d_c exceeds N − 1 = 39.
- **Coarse crossings.** Where the crossing falls between the widely spaced degrees 24 and 39, the interpolation is coarse and the interval wide.
- **Per-cell brackets and communication statistics.** Each `results/llm_agents/<tag>/dc_table.md` also lists the bracketing degrees and, for C1/I1, the number of reports, the false-report rate and the silent rate.

### Notes on the runs

- **Output cap for reasoning models.** OpenAI counts reasoning tokens against `max_completion_tokens`, so the reasoning-effort runs use 128000 (the model maximum). The earliest seed-1 cells used 32768; their `manifest.json` records the cap actually used.
- **Runaway reasoning.** In these runs a few branch replies (1–4 per effort level) reasoned until the cap and returned an empty answer. Repairing the same call succeeded each time, and no episode failed.
- **Degenerate whitespace (Qwen3.5-9B, temperature 1).** Some replies repeated whitespace inside the constrained JSON until the 3072-token cap; they were repaired.
- **Branch-schema fix.** A parent hypothesis with no legal extension used to produce an empty `enum` in the branch schema, which vLLM's grammar backend rejects. The fix is in `branch_schema`; request bodies are unchanged otherwise. One affected shard (`qwen3.5-4b-t1`, seed 1, tcas I1, d = 24) was rerun under the unchanged protocol.

## Raw run data

- **Size.** The complete raw records are about 3 GB per tag: every prompt, every raw model reply, every experiment and per-round statistics.
- **Included data.** Pooled tables in `results/` are included and were generated from the raw records by the scripts above.
- **Raw-record availability.** The complete raw LLM records are not included in this repository, and the supplied package does not contain a public download URL. Offline replay requires saved request/reply logs from an actual run. Add an author-approved data location here if those records are released.
