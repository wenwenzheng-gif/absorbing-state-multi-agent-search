# Needle / Chain experiments

The package `src.agent_system.no_transition` reuses the repository's graph,
RNG, OpenAI backend, decision context, atomic storage, and result/figure
conventions. Its scheduler replaces the literal-branching runner because
atomic IDs and strictly locked pointers are incompatible with that runner's
mandatory branching, pruning and final hypothesis recovery.

Run commands from the repository root with the same Python environment used
for the existing agent experiments (`requirements.txt`,
`requirements-agents.txt`, and pytest).

```bash
# Relevant new tests plus shared graph/protocol/provider regressions.
python -m src.agent_system.no_transition.cli --test

# Validate either configuration without executing an episode or API request.
python -m src.agent_system.no_transition.cli --mode pilot --validate-only
python -m src.agent_system.no_transition.cli --mode full --validate-only

# Pilot, including tests, reference runs, GPT, all analysis and figures.
# Existing exported OPENAI_* variables override the environment file.
python -m src.agent_system.no_transition.cli --mode pilot --env-file /path/to/existing.env

# Resume: repeat the identical pilot command. Completed episodes and saved
# individual calls are skipped. Do not change the frozen config/run directory.

# Reanalyze frozen committed results offline (no credentials, no API requests).
python -m src.agent_system.no_transition.cli --mode pilot --analyze-existing \
  --run-dir results/needle_chain/pilot --output-dir reproduced/needle_chain/pilot

# Future full run — IMPLEMENTED ONLY. DO NOT EXECUTE during the pilot task.
python -m src.agent_system.no_transition.cli --mode full --execute-full \
  --env-file /path/to/existing.env
```

Do not put credentials in command-line arguments or tracked files. The model
is pinned by `configs/needle_chain/provider.json` to
`gpt-5.4-nano-2026-03-17`; a conflicting `OPENAI_MODEL` is rejected. The original
resource JSON retains the family alias to preserve completed deterministic
episode identities. Effective settings are saved in every GPT episode and
`provider_settings.json`. No new
secret is required when a working existing environment is supplied.

`runs/needle_chain/{pilot,full}` is ignored local working storage: per-episode
manifests, atomic round checkpoints, request/response ledgers and durable
response/decision caches. Only one process may own a run directory. Failures
are retained as outcomes. A successfully persisted model response is never
sampled again. If the process dies while the provider is handling a request
and no response has been persisted, remote execution is unknowable; such an
attempt may be repeated and remains in the attempt ledger. This is a remote
idempotency limitation, not a claim of exactly-once provider billing.

`results/needle_chain/pilot` holds portable, compact, committed deliverables:
the full processed episode trace archive; aggregate CSVs;
paired statistical diagnostics; measured runtime and cost projections; three
PNG/PDF figure pairs; and `pilot_report.md`. Secret values never enter these
artifacts. Following the existing repository convention, raw prompt/response
ledgers, their compact archive, per-call caches and Python bytecode remain in
ignored local working storage and are not committed.

The analysis handles exact 0/1 outcomes without a finite-dc estimate. All
theory labels are distinguished from empirical diagnostics. A degenerate
paired bootstrap with tiny samples is explicitly flagged. Read the
[preregistration](preregistration.md) for budgets, seeds, hypotheses, failure
handling and limits of inference.

There is no automatic full-run launch, push, merge or workflow dispatch in
the CLI. After reviewing the completed pilot artifacts, the operator commits
and pushes the experimental branch. `main` is not merged or rewritten.

The user's follow-up authorized reuse of the prior hosted provider setup.
`.github/workflows/needle-chain-hosted-pilot.yml` runs only in the prior
`wenwenzheng-gif/absorb-state-new` Actions environment, on a dedicated branch
with the same experiment name. It checks out the identical frozen commit from
the canonical `absorbing-state-multi-agent-search` repository and consumes the
existing `OPENAI_API_KEY` secret in place. The workflow has read-only repository
permissions, restores completed episodes from their committed archive, runs
only the pilot, and uploads checkpoints/results as a recoverable artifact.
It does not push code or results. Canonical results are imported, reviewed,
committed and pushed to the requested repository's experimental branch.
