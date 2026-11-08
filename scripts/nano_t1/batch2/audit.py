#!/usr/bin/env python3
"""Fenced Batch 2 preflight: original t1 seeds 1-6 plus validated Batch 1 seeds 7-9.

Only seed indices 10, 11, 12 are eligible. No API requests are made here.
"""
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/gpt-5.4-nano-t1-none-new3-batch2"
MODEL="gpt-5.4-nano-2026-03-17"
TASKS={"synthetic":87000,"tcas":87200,"physics":82240}
ARMS=("C0","C1","I1")
NEW=(10,11,12)
DATASET="TobyYang7/Branching-Search-runs"

def main():
    report={"source_dataset_family":"gpt-5.4-nano-t1",
            "source_snapshot":MODEL,
            "source_sampling":{"temperature":1.0,"reasoning_effort":"none","enable_thinking":False},
            "previous_valid_seed_indices":list(range(1,10)),
            "batch2_seed_indices":list(NEW),
            "batch2_task_seed_values":{task:[start+i-1 for i in NEW] for task,start in TASKS.items()},
            "expected_scientific_jobs":9,
            "expected_condition_runs":27,
            "expected_episodes":171,
            "out_dir":str(OUT),
            "verified_utc":datetime.now(timezone.utc).isoformat(),
            "status":"BLOCKED","blockers":[]}
    failures=report["blockers"]
    first=ROOT/"results/nano-t1-new3/final_report.json"
    if not first.exists(): failures.append("Valid Batch 1 final report missing")
    else:
        prior=json.loads(first.read_text())
        if not (prior.get("completed_episodes")==171 and prior.get("completed_condition_runs")==27
                and prior.get("model")==MODEL and prior.get("new_seed_indices")==[7,8,9]
                and prior.get("temperature")==1.0 and prior.get("reasoning_effort")=="none"
                and prior.get("thinking_disabled") is True
                and prior.get("secret_scan_passed") is True
                and prior.get("status") in ("VALIDATED_LOCAL_ONLY","COMPLETE")):
            failures.append("Batch 1 is not an independently validated source for Batch 2")
        if len(prior.get("results",{}))!=27 or any(
                r.get("status")!="COMPLETE" for r in prior.get("results",{}).values()):
            failures.append("Batch 1 condition-level completion is not 27/27")
        report["batch1_provenance"]={
            "first_batch_source_run":prior.get("source_workflow_run"),
            "first_batch_sha_report":first.as_posix(),
            "completed_episodes":prior.get("completed_episodes"),
            "verified_status":prior.get("status")}
    baseline_count=0
    for original_index in range(1,7):
        for task,start in TASKS.items():
            for arm in ARMS:
                p=ROOT/f"configs/agents/gpt-5.4-nano-t1_seed{original_index}/{task}_{arm}.json"
                if not p.exists():
                    failures.append(f"Historical validated seed config missing: {original_index}/{task}/{arm}")
                    continue
                c=json.loads(p.read_text())
                llm=c["policy"]["llm"]
                if (c.get("task_seeds")!=[start+original_index-1] or
                    llm.get("temperature")!=1.0 or llm.get("reasoning_effort")!="none" or
                    llm.get("enable_thinking") is not False or llm.get("max_output_tokens")!=3072 or
                    llm.get("model")!="gpt-5.4-nano" or llm.get("prompt_version")!="v4"):
                    failures.append(f"Historical seed protocol differs: {original_index}/{task}/{arm}")
                baseline_count+=1
    report["validated_historical_config_count"]=baseline_count
    for seed in NEW:
        p=ROOT/f"configs/agents/gpt-5.4-nano-t1_seed{seed}"
        if p.exists(): failures.append("Potential valid seed overlap in GitHub configs: "+str(seed))
    # A hosted run starts from an absolutely fresh output tree. The explicit
    # --resume workflow is separate and restores only Batch 2's own ledgers.
    if OUT.exists() and any(OUT.iterdir()):
        failures.append("Expected clean Batch 2 output tree already contains files")
    if os.environ.get("OPENAI_MODEL")!=MODEL:
        failures.append("OPENAI_MODEL environment differs from exact Batch 1 snapshot")
    if os.environ.get("OPENAI_BASE_URL")!="https://api.openai.com/v1":
        failures.append("API endpoint differs from validated Batch 1")
    # Public HF directory check guards against later valid t1 data using our
    # candidates, without requiring HF write authorization or blocking on it.
    try:
        from huggingface_hub import HfApi
        api=HfApi()
        seen={x.path for x in api.list_repo_tree(DATASET,repo_type="dataset",
                                        path_in_repo="gpt-5.4-nano",recursive=False)}
        report["hf_original_valid_archive_present"] = (
             "gpt-5.4-nano/gpt-5.4-nano-t1_runs.zip" in seen)
        if not report["hf_original_valid_archive_present"]:
            failures.append("Original valid Nano t1 six-seed archive missing")
        conflicts=[p for p in seen if ("gpt-5.4-nano-t1" in p and
                    re.search(r"(?:seed[s_]?)?(?:10[-_]?12|10|11|12)(?:[^0-9]|$)",p))]
        if conflicts: failures.append("Potential Batch 2 seed conflict in HF dataset: "+str(conflicts[:10]))
        report["hf_no_seed_10_11_12_filename_conflicts"]=not conflicts
    except Exception as e:
        failures.append("Could not check public HF data for seed collisions: "+type(e).__name__)
    report["status"]="PASS" if not failures else "BLOCKED"
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"status":report["status"],"new_seed_indices":report["batch2_seed_indices"],
                      "batch2_task_seeds":report["batch2_task_seed_values"],
                      "blockers":failures},indent=2))
    return 0 if not failures else 1

if __name__=="__main__":sys.exit(main())
