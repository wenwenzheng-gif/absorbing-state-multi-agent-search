# d_c table -- gpt-6-luna-re-xhigh_seed1-6

d_c: empirical = `critical_degree` interpolation estimator, convention `pre_own` (pre_own: before the agent's own experiments/pruning -- the frozen `src/utils.py` lineage-ratio convention: the A-to-A count is read right after an agent receives its neighbours' branches, *before* that agent's own experiments and pruning -- so one round's ratio isolates `b (1-p_nbr)^d`, self-pruning excluded per the paper's own derivation), cumulative pooled log R summed over **all rounds t>=1** (`critical_degree`'s default, `--first-round 1`); round 1 contributes a fixed `ln b` with **no d-dependence** (no neighbour messages have arrived yet under one-round-delayed communication, and under `pre_own` the agent's own pruning is excluded, so round 1's ratio is just `b`) -- that `ln b` is exactly the `(T-1) ln b` term in `delayed_dc`'s numerator, so summing `pre_own` from t=1 is the correct empirical counterpart to `delayed_dc`, not merely a superset of it; theory = `delayed_dc` (`src/agent_system/analysis/references.py`, denominator survival terms summed from t=2, since that factor genuinely depends on communication that hasn't arrived yet in round 1, while the `(T-1) ln b` numerator is kept in full -- both likewise excluding self-pruning), b=2, q_model=1, m and T from the config; |V| is fixed per cell: synthetic=16 (M*K), tcas=46 (component-level evidence, NOT M*K=120), physics=16 (support library). Each emp cell also reports `bracket=[d_minus, d_plus]`, the two measured degrees the interpolation crossed between. Secondary columns in `dc_table.csv` only (not in this table): `emp_pre_own_t2` (pre_own summed from t=2 instead of t=1 -- the previous version of this table's main column, kept for comparison), `emp_post_own_all_rounds` (same t>=1 sum but convention `post_own`, which additionally folds in the agent's own self-pruning factor `(1-p_own)` -- a diagnostic of how much self-pruning contributes against the new main column, not a value comparable to theory), and `theory_exact` (`exact_dc`, sum from t=1). A cell with non-`"ok"` episodes (e.g. `policy_failed`, merged in via `merge_runs.py --allow-failed`) shows `failed: <n> (d=<degrees>)` below its emp/theory line; those episodes are excluded from the empirical d_c (`critical_degree.pooled_log_ratio` only pools `status == "ok"` episodes) -- a degree with fewer surviving (ok) episodes than the cell's other degrees is still pooled and estimated, just over fewer tasks, so its point is noisier but not dropped.

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C0 | emp=7.08 [6.65, 7.57] bracket=[0, 8]<br>theory=6.10 | emp=18.82 [13.18, 21.96] bracket=[18, 24]<br>theory=10.90 | emp=19.64 [13.54, 28.14] bracket=[16, 24]<br>theory=6.10 |
| C1 | emp=7.23 [6.94, 7.54] bracket=[0, 8]<br>theory=6.10 | emp=26.97 [19.63, 37.54] bracket=[24, 39]<br>theory=10.90 | emp=22.01 [11.69, 37.59] bracket=[16, 24]<br>theory=6.10 |
| I1 | emp=7.57 [7.22, 7.89] bracket=[0, 8]<br>theory=6.10 | emp=28.98 [18.57, 36.96] bracket=[24, 39]<br>theory=10.90 | emp=27.70 [14.83, 37.46] bracket=[24, 39]<br>theory=6.10 |

### Failed episodes

(none)

## Communication (C1 / I1 only)

Computed from `episodes.jsonl` transitions, which this repo's runner (`src/agent_system/runner.py`) does record for free-communication cells (C1, I1, communication_mode free). `reports` = total `outgoing_reports`; `false` = `outgoing_false_reports / outgoing_reports`; `silent` = `1 - outgoing_messages / comm_calls`, computed from `comm_calls` (successful communication-phase calls) rather than from the runner's own `silent_agents` counter, so it stays correct regardless of how `silent_agents` is defined. Shown as TODO only when a cell has no run, or its episodes predate these counters.

| Policy | synthetic | tcas | physics |
|---|---|---|---|
| C1 | reports=1503, false=0.000, silent=0.083 | reports=5100, false=0.000, silent=0.032 | reports=2701, false=0.000, silent=0.007 |
| I1 | reports=1436, false=0.000, silent=0.141 | reports=5042, false=0.000, silent=0.090 | reports=2704, false=0.000, silent=0.027 |
