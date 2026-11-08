# Qwen 3.5 six-seed suites: prepared, blocked before execution

The 36 configs under `configs/qwen35-six-seed/` use the same `llm_toy`,
`llm_tcas`, and `llm_physics` sources as `scripts/luna/run.py`, setting independent
seeds 0–5. The scientific fields are unchanged. Model IDs, prospective local
endpoints, run names, and output paths are the only other substitutions.
Native Qwen defaults retain temperature 0.2, thinking disabled, prompt v4,
fixed communication, and redistribute budget. Luna uses its OpenAI-specific
reasoning/temperature settings; those are not valid substitutes for Qwen.
Historical Qwen C0/C1/I1 ablations have different degree grids and are not used.

`python3 scripts/qwen35/prepare.py` reproduces the configs without launching work.
`results/qwen35/prepared_runs.json` accounts for all 36 unstarted seeds.
`results/qwen35/preflight.json` records the checkpoint identities and blocker.

All four RTX A6000 GPUs are allocated to the pre-existing
`pgym-qwen27b.service` (Qwen3.8-27B), leaving about 2.1 GB free per device.
Neither requested model fits without changing precision or displacing that
service. No requested model endpoint is running. Do not stop the service
without authorization to interrupt that separate workload.

No smoke, capacity test, experiment, aggregation, or upload has run. No durable
runner has been launched; closing a laptop therefore does not leave this suite
running. Prepared endpoints are reserved plans, not verified servers. GPU
allocation and throughput must be measured once capacity becomes available.
Do not label blocked seeds FAILED_AFTER_RETRIES: no scientific attempts occurred.

Once capacity is available, finish serving supervision, endpoint capacity tests,
all six smoke pipelines, checkpoint-aware scheduling and per-task aggregation
using the native analysis modules before claiming any execution success.
Inspect Hugging Face history before choosing immutable new upload destinations
under each model; no remote paths have been written or reserved yet.
