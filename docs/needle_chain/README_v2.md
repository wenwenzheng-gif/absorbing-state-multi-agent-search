# Preregistered Needle / Chain pilot v2

V1 is immutable at `bfb1e3220a1be90e05c432bd33a363077bdb2543`.
Read [preregistration_v2.md](preregistration_v2.md) before execution. V2 adds
two complete Needle-High coverage passes (96 slots for 48 candidates) and
stronger default-assignment guidance without restricting legal selections.
Needle-Low and Chain resources, prompts and environment logic are unchanged.

Run from the repository root in the declared Python environment:

```bash
python -m src.agent_system.no_transition.cli --test
python -m src.agent_system.no_transition.cli --mode pilot \
  --config configs/needle_chain/pilot_v2.json --validate-only

# The already authorized hosted workflow injects the existing provider
# environment in place. Locally this command uses exported OPENAI_* values.
python -m src.agent_system.no_transition.cli --mode pilot \
  --config configs/needle_chain/pilot_v2.json

# Repeat the identical command only to resume an interrupted run.
# Persisted episodes/calls are reused, never favorably resampled.

# Offline reproduction from the compact committed v2 archive, no API calls.
python -m src.agent_system.no_transition.cli --mode pilot \
  --config configs/needle_chain/pilot_v2.json --analyze-existing \
  --run-dir results/needle_chain/pilot_v2 \
  --output-dir reproduced/needle_chain/pilot_v2
```

Default paths are `runs/needle_chain/pilot_v2` and
`results/needle_chain/pilot_v2`; directing v2 into a v1 directory is rejected.
Protected v1 hashes are checked before execution and each analysis.
The new workflow is pilot-only. The historical v1 workflow is now manual-only
to prevent it from launching on creation of the temporary hosted v2 branch.
No secret is copied, printed, committed or modified.

Outputs include `pilot_report_v2.md`, 516 processed episodes (480 reference,
36 GPT), per-episode High coverage/departure diagnostics, separate v1/v2
comparison tables, four PNG/PDF figure pairs, Wilson/paired/logistic
diagnostics, readiness and measured runtime. Missing or failed observations
are visible, not replaced. Raw request/response ledgers remain under ignored
run storage, consistent with v1 conventions.

The readiness status is an operational pilot decision, not proof of absence
of a transition. Even READY FOR FULL RUN requires separate authorization.
No v2 full design is registered here, and `full.json` remains the historical
v1 full configuration.

The completed [v2 report](../../results/needle_chain/pilot_v2/pilot_report_v2.md)
classifies the pilot **READY FOR FULL RUN** under the preregistered operational
criteria. GPT High succeeded in 6/6 seeds at each degree with full coverage;
the six-seed sample does not prove absence of a transition. The recommended
next step is to review this evidence and separately authorize a v2 full design
that retains resource slack. Do not launch the unchanged v1 `full.json` as v2.

Hosted run [42415318874](https://github.com/wenwenzheng-gif/absorb-state-new/actions/runs/42415318874)
finished once, with 792 successful calls, zero retries and no additional model
experiments. Its temporary runner branch was removed after verified import.
[Post-execution analysis notes](analysis_notes_v2.md) distinguish reporting
additions from the frozen design. `data_audit.json`, `validation.json` and
`hosted_provenance.json` accompany the results. No full experiment was launched.
