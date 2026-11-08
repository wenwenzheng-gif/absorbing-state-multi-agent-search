#!/usr/bin/env python3
"""Checkpoint-aware GitHub Actions re-dispatch until all nine t1 workers finish."""
import json
import os
from pathlib import Path
import subprocess
import sys
REPO="wenwenzheng-gif/absorbing-state-multi-agent-search"
BRANCH="experiment/gpt54-nano-t1-none-batch2-seeds10-12"
OUT=Path("outputs/gpt-5.4-nano-t1-none-new3-batch2/collector.json")
TASKS=("synthetic","tcas","physics")
SEEDS=(10,11,12)

def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError("GitHub workflow command failed: "+cmd[1])
    return p.stdout

def main():
    run_id=os.environ["GITHUB_RUN_ID"]
    iteration=int(os.getenv("NANO_T1_ITERATION","1"))
    artifacts=json.loads(run(["gh","api",f"repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100"]))["artifacts"]
    by_name={a["name"]:a for a in artifacts}
    progress={}
    blockers=[]
    scratch=Path("outputs/gpt-5.4-nano-t1-none-new3-batch2/_collector")
    for task in TASKS:
        for seed in SEEDS:
            key=f"{task}/seed_{seed}"
            name=f"nano-t1-batch2-{task}-seed{seed}-status"
            data_name=f"nano-t1-batch2-{task}-seed{seed}-checkpoint"
            if name not in by_name or data_name not in by_name:
                blockers.append(f"Missing status/data checkpoint pair for {key}")
                continue
            dest=scratch/task/f"seed_{seed}"
            dest.mkdir(parents=True,exist_ok=True)
            run(["gh","run","download",run_id,"--repo",REPO,"--name",name,"--dir",str(dest)])
            p=dest/"seed_status.json"
            if not p.exists():
                blockers.append(f"Missing seed_status.json for {key}")
                continue
            st=json.loads(p.read_text())
            assert st["model"]=="gpt-5.4-nano-2026-03-17"
            assert st["temperature"]==1.0 and st["reasoning_effort"]=="none"
            assert st["seed_index"]==seed and st["task"]==task
            if st["status"]=="FAILED_AFTER_RETRIES":
                blockers.append(f"Unrecoverable seed status for {key}")
            progress[key]={"status":st["status"],
                           "completed_episodes":st["completed_episodes"],
                           "target_episodes":st["target_episodes"]}
    total=sum(x["completed_episodes"] for x in progress.values())
    target=sum(x["target_episodes"] for x in progress.values())
    state="BLOCKED" if blockers else "COMPLETE" if len(progress)==9 and all(
         x["status"]=="COMPLETE" for x in progress.values()) else "CONTINUE"
    result={"workflow_run_id":run_id,"iteration":iteration,"state":state,
            "completed_episodes":total,"target_episodes":target,
            "workers":progress,"blockers":blockers}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2))
    print("RUN",run_id,"PASS",iteration,"STATUS",state,"EPISODES",total,"/",target,flush=True)
    for b in blockers: print("BLOCKER:",b,flush=True)
    # Existing workflow is scoped to this experimental branch, so direct
    # workflow_dispatch is intentionally disabled. A separate authorized
    # branch-marker watcher resumes or finalizes only after inspecting this
    # collector's complete verified status and checkpoint artifacts.
    print("CHECKPOINT_READY: external branch-marker controller may continue or finalize.",flush=True)
    return 0 if state!="BLOCKED" else 1

if __name__=="__main__":sys.exit(main())
