# Full v2 Needle / Chain experiment

The full experiment uses N=20, degrees [0,4,8,12,16,19], 200 deterministic seeds and 12 GPT seeds per condition and degree: 4,800 reference and 288 GPT episodes. The scientific design was committed and pushed before live execution at **c9246bf47a2f107e79413be6e9f4bda3e3ae9f78**.

- [Frozen preregistration](preregistration_full_v2.md)
- [Full report](../../results/needle_chain/full_v2/full_report.md)
- [Configuration](../../configs/needle_chain/full_v2.json)
- [Scientific file hashes](full_v2_freeze.json)
- [Protected pilot hashes](protected_pilots_full_v2.json)

The existing v1 and v2 pilot artifacts are preserved. The old `full.json` is not this full-v2 design; always select `full_v2.json` explicitly.

## Reproduce derived results without model calls

Install the repository's declared dependencies, then run from the repository root:

```bash
MPLCONFIGDIR="$PWD/runs/needle_chain/matplotlib-cache" \
XDG_CACHE_HOME="$PWD/runs/needle_chain/xdg-cache" \
python -m src.agent_system.no_transition.cli \
  --mode full --config configs/needle_chain/full_v2.json \
  --analyze-existing --run-dir results/needle_chain/full_v2 \
  --output-dir reproduced/needle_chain/full_v2
```

The committed lossless episode archive contains actions, observations, graph, task identity and episode-level usage. It reproduces primary tables, paired diagnostics, mechanism metrics and all five PNG/PDF figure pairs. Provider request/response identity auditing additionally requires the saved raw hosted checkpoint. The original `validation.json`, `data_audit.json`, `provider_response_inventory.jsonl.gz` and `hosted_provenance.json` preserve that completed audit; an archive-only reproduction honestly marks raw-call verification unavailable instead of claiming to repeat it. Do not direct reproduction output into the committed result folder.

```bash
MPLCONFIGDIR="$PWD/runs/needle_chain/matplotlib-cache" \
XDG_CACHE_HOME="$PWD/runs/needle_chain/xdg-cache" \
python -m pytest -q
```

## Execution provenance and recovery

The existing authorized hosted environment in `wenwenzheng-gif/absorb-state-new` checked out the exact canonical frozen commit. Only the existing secret reference name `OPENAI_API_KEY` is recorded. Its value was not copied, exposed or modified. The workflow guards its runner repository, saves checkpoints after every condition, and can restore the latest same-run artifact on an infrastructure retry. Runtime counts distinguish logical requests, HTTP attempts, retries and saved successful responses.

Execution requires the frozen source and raw checkpoints, not merely the processed archive. A restart skips completed episodes and reuses saved decisions. No post-hoc seeds, behavioral reruns, prompt changes, additional degrees, model substitutions or automatic extensions are permitted. Once all 5,088 episodes are complete, no further execution is needed.

## Interpretation

Theoretical d_c labels are not estimated from flat success curves. A finite scan cannot prove that no transition exists for all degrees or policies. Needle-High is a saturated resource control; round-one discovery limits claims about communication's causal effect. The historical N=40 dataset records wrong-lineage extinction rather than verified task success, so Figure 5 labels it explicitly and uses separate degree axes from the new N=20 outcomes.

The temporary runner branch remains at the preregistration SHA because automatic approval review rejected remote branch deletion as not explicitly authorized. Both main branches remain unchanged. This operational cleanup limitation is documented in the final report and provenance; it does not affect the scientific results.
