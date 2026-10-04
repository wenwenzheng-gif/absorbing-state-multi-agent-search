# d_c table -- gpt-6-luna-re-medium_seed1-6

d_c: empirical = `critical_degree` interpolation estimator, convention `pre_own` (pre_own: before the agent's own experiments/pruning -- the frozen `src/utils.py` lineage-ratio convention: the A-to-A count is read right after an agent receives its neighbours' branches, *before* that agent's own experiments and pruning -- so one round's ratio isolates `b (1-p_nbr)^d`, self-pruning excluded per the paper's own derivation), cumulative pooled log R summed over **all rounds t>=1** (`critical_degree`'s default, `--first-round 1`); round 1 contributes a fixed `ln b` with **no d-dependence** (no neighbour messages have arrived yet under one-round-delayed communication, and under `pre_own` the agent's own pruning is excluded, so round 1's ratio is just `b`) -- that `ln b` is exactly the `(T-1) ln b` term in `delayed_dc`'s numerator, so summing `pre_own` from t=1 is the correct empirical counterpart to `delayed_dc`, not merely a superset of it; theory = `delayed_dc` (`src/agent_system/analysis/references.py`, denominator survival terms summed from t=2, since that factor genuinely depends on communication that hasn't arrived yet in round 1, while the `(T-1) ln b` numerator is kept in full -- both likewise excluding self-pruning), b=2, q_model=1, m and T from the config; |V| is fixed per cell: synthetic=16 (M*K), tcas=46 (component-level evidence, NOT M*K=120), physics=16 (support library). Each emp cell also reports `bracket=[d_minus, d_plus]`, the two measured degrees the interpolation crossed between. Secondary columns in `dc_table.csv` only (not in this table): `emp_pre_own_t2` (pre_own summed from t=2 instead of t=1 -- the previous version of this table's main column, kept for comparison), `emp_post_own_all_rounds` (same t>=1 sum but convention `post_own`, which additionally folds in the agent's own self-pruning factor `(1-p_own)` -- a diagnostic of how much self-pruning contributes against the new main column, not a value comparable to theory), and `theory_exact` (`exact_dc`, sum from t=1). A cell with non-`"ok"` episodes (e.g. `policy_failed`, merged in via `merge_runs.py --allow-failed`) shows `failed: <n> (d=<degrees>)` below its emp/theory line; those episodes are excluded from the empirical d_c (`critical_degree.pooled_log_ratio` only pools `status == "ok"` episodes) -- a degree with fewer surviving (ok) episodes than the cell's other degrees is still pooled and estimated, just over fewer tasks, so its point is noisier but not dropped.

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | emp=6.80 [6.31, 7.18] bracket=[0, 8]<br>theory=6.10 | emp=17.76 [14.47, 22.39] bracket=[14, 18]<br>theory=10.90 | emp=9.74 [8.25, 11.13] bracket=[8, 12]<br>theory=6.10 |
| C1 | emp=7.33 [6.76, 7.95] bracket=[0, 8]<br>theory=6.10 | emp=34.25 [25.12, 38.46] bracket=[24, 39]<br>theory=10.90 | emp=10.13 [7.93, 14.76] bracket=[8, 12]<br>theory=6.10 |
| I1 | emp=7.34 [6.94, 7.77] bracket=[0, 8]<br>theory=6.10 | emp=(no crossing: no_crossing_all_positive) [20.74, 36.80]<br>theory=10.90 | emp=10.92 [9.18, 13.14] bracket=[8, 12]<br>theory=6.10 |

### Failed episodes

(none)

## Communication (C1 / I1 only)

Computed from `episodes.jsonl` transitions, which this repo's runner (`src/agent_system/runner.py`) does record for free-communication cells (C1, I1, communication_mode free). `reports` = total `outgoing_reports`; `false` = `outgoing_false_reports / outgoing_reports`; `silent` = `1 - outgoing_messages / comm_calls`, computed from `comm_calls` (successful communication-phase calls) rather than from the runner's own `silent_agents` counter, so it stays correct regardless of how `silent_agents` is defined. Shown as TODO only when a cell has no run, or its episodes predate these counters.

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C1 | reports=1463, false=0.000, silent=0.050 | reports=4842, false=0.000, silent=0.014 | reports=2755, false=0.000, silent=0.000 |
| I1 | reports=1463, false=0.000, silent=0.090 | reports=4521, false=0.001, silent=0.059 | reports=2674, false=0.000, silent=0.022 |
