# GPT-6-Sol / medium / twelve seeds

Reference commit: `6e1fd0e4112492e208be394ffa945e142a2d433a` on `experiment/gpt6-luna-3tasks-6seeds-parallel`.

The 684 single-episode jobs copy the original `configs/agents/gpt-6-luna_seed1/{task}_{policy}.json` C0/C1/I1 configurations. Seeds extend consecutively to twelve: Synthetic 87000–87011, TCAS 87200–87211, Physics 82240–82251. Scientific changes are limited to the requested model, reasoning effort, and seed count. Temperature is absent from API requests. The original 3072 output-token budget, repair budget, agent count, v4 prompts, protocol, environments, and degree grids remain unchanged. Operational changes are sharding, durable transport journaling, request pacing, concurrency gating, and a longer HTTP read timeout.

Artifacts are in `runs/gpt-6-sol-medium-12seeds`. Each job saves full request bodies and raw HTTP response bodies in `raw.sqlite`, in addition to native request, response, event, episode, manifest, and checkpoint files. SQLite uses WAL and FULL synchronization. Keys include the native run/episode/agent/round/phase/batch/repair identity. Successful HTTP responses are reused on trajectory reconstruction. Explicit rate-limit refusals and connection failures before a connection is established may retry. Uncertain outcomes are held and never automatically resent. Native repair decisions remain unchanged and have distinct request identities.

`progress.json` and append-only `progress.jsonl` contain task/job counts and actual unique request usage. `proxy_metrics.jsonl` records proxy health, adaptive concurrency, and throughput projections. The projection based on an unpruned upper-bound workload is not a reliable completion ETA. Proxy metrics include any other proxy users; job SQLite usage is specific to this batch.

The initial burst produced four held trajectories (30 connection-reset outcomes plus canceled outstanding requests). These are excluded from successful results, preserved for inspection, and are not automatically rerun. Client-side pacing was then added, and concurrency is now adapted to observed rate limits. Some raw responses from those initial disconnected requests may be unavailable; the journal records their uncertainty explicitly.

Processes were launched with disconnected stdin and `start_new_session=True`, so VS Code disconnection does not stop them. The supervisor adopts live workers after restart and uses an exclusive lock. Each worker also has an exclusive lock. To inspect:

```bash
cd /data4/toby/absorbing-state-multi-agent-search
.venv-sol/bin/python scripts/sol/run.py status
cat runs/gpt-6-sol-medium-12seeds/validation.json
```

To resume a stopped supervisor (do not remove locks or raw journals):

```bash
nohup .venv-sol/bin/python -u scripts/sol/run.py supervise >> runs/gpt-6-sol-medium-12seeds/supervisor.log 2>&1 < /dev/null &
```

If needed, restart `scripts/sol/controller.py` and `scripts/sol/validate.py` similarly. Both have singleton locks. The validator checks each completed experiment against raw response audits, performs an offline replay for each task/policy's first completion, and runs the repository's summary and critical-degree analysis after the queue reaches a terminal state. Held and failed experiments are not counted as complete. No uploads, pushes, or external messages are performed.

## Current execution policy

The three-hour target is no longer a deadline. Reliability and completion of all recoverable trajectories take priority. The safe request-start ceiling is 3.1 requests/second and global admission is capped at 64; upward probes are disabled. Controls can reduce pressure if rate limits return. Experimental workers must not be interrupted or restarted for tuning.

`diagnostics/detachment_verification.json` records the independent sessions, absence of controlling terminals, `/dev/null` input, and persistent log destinations of all workers and supporting processes. No Codex or VS Code connection is needed for continued execution.

`observed_progress_eta` is the authoritative ETA field in `dashboard.json`. It supersedes the legacy completion-count extrapolation and unpruned projection. The offline performance watcher refreshes it every five minutes using completed rounds, matching task, policy, exact communication degree, and round. It requires two independent seeds for each remaining stratum before reporting a whole-batch estimate. Until then the ETA is explicitly unavailable; the early 17–21 hour projections were based on insufficient representative coverage and should not be treated as forecasts. Held trajectories have no completion ETA without safe response recovery.
