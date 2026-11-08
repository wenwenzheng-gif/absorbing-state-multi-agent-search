#!/usr/bin/env python3
"""Finalize stored seed artifacts without rerunning expensive model requests."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/gpt-5.4-nano"
TASKS={"synthetic":9,"tcas":12,"ode":11}
REPO="wenwenzheng-gif/absorbing-state-multi-agent-search"

def atomic(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    tmp.replace(path)

def lines(path):
    if not path.exists(): return []
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]

def download(run_id):
    dest=OUT/"_artifacts"
    dest.mkdir(parents=True,exist_ok=True)
    # gh is authenticated with GH_TOKEN. Do not print its value.
    result=subprocess.run(["gh","run","download",str(run_id),"--repo",REPO,
                           "--dir",str(dest)],cwd=ROOT,check=False)
    if result.returncode: raise RuntimeError("Could not download completed GitHub Actions artifacts.")
    return dest

def restore(dest):
    missing=[]
    for task,target in TASKS.items():
        for seed in range(6):
            source=dest/f"nano-{task}-seed-{seed}"
            d=OUT/task/f"seed_{seed}"
            if not source.is_dir():
                missing.append(f"{task}/seed_{seed}")
                continue
            d.mkdir(parents=True,exist_ok=True)
            for file in source.rglob("*"):
                if not file.is_file(): continue
                rel=file.relative_to(source)
                target_path=d/rel
                target_path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(file,target_path)
    if missing:
        raise RuntimeError("Missing seed artifacts: "+", ".join(missing))

def scan():
    violations=[]
    # No secrets are emitted in the report. Reject known token patterns and
    # security-sensitive filenames, rather than storing discovered values.
    pattern=re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,}|gh[opsu]_[A-Za-z0-9_]{25,})\b")
    for file in OUT.rglob("*"):
        if not file.is_file(): continue
        if file.suffix not in (".json",".jsonl",".csv",".md",".txt",".png"): continue
        if file.name in (".env","auth.json") or pattern.search(file.read_bytes()):
            violations.append(str(file.relative_to(OUT)))
    return violations

def aggregate(task):
    d=OUT/task
    states=[]
    counts=collections.Counter()
    for seed in range(6):
        p=d/f"seed_{seed}"
        state=json.loads((p/"status.json").read_text())
        rows=lines(p/"episodes.jsonl")
        good={r["episode_id"] for r in rows if r.get("status")=="ok"}
        if state.get("status") not in ("COMPLETE","FAILED_AFTER_RETRIES"):
            raise RuntimeError(f"{task} seed {seed}: unresolved status")
        if state["status"]=="COMPLETE" and len(good)!=TASKS[task]:
            raise RuntimeError(f"{task} seed {seed}: falsely complete")
        states.append({"seed":seed,"state":state["status"],"completed_episodes":len(good),
                       "target_episodes":TASKS[task]})
        for kind in ("requests","responses"):
            counts[kind]+=len(lines(p/f"{kind}.jsonl"))
    summary_dir=d/"aggregate"
    summary_dir.mkdir(exist_ok=True)
    manifest=json.loads((d/"seed_0"/"manifest.json").read_text())
    manifest["config"]["task_seeds"]=list(range(6))
    atomic(summary_dir/"manifest.json",manifest)
    for kind in ("episodes","events"):
        with (summary_dir/f"{kind}.jsonl").open("wb") as output:
            for seed in range(6):
                p=d/f"seed_{seed}"/f"{kind}.jsonl"
                if p.exists():
                    output.write(p.read_bytes())
    analysis={}
    for name in ("summarize","critical_degree"):
        cmd=[sys.executable,"-m",f"src.agent_system.analysis.{name}",
             "--run-dir",str(summary_dir)]
        proc=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (summary_dir/f"{name}.log").write_text(proc.stdout[-100000:])
        analysis[name]=proc.returncode
    result={"task":task,"model":"gpt-5.4-nano","reasoning_effort":"medium",
            "seeds":states,"completed_episodes":sum(x["completed_episodes"] for x in states),
            "target_episodes":6*TASKS[task],"ledger_counts":dict(counts),
            "analysis_exit_codes":analysis}
    atomic(d/"summary.json",result)
    return result

def upload(summaries):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"status":"BLOCKED","reason":"huggingface_hub package unavailable"}
    token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {"status":"BLOCKED","reason":"HF_TOKEN not injected; no token requested or exposed"}
    api=HfApi(token=token)
    statuses={}
    for task in TASKS:
        try:
            api.upload_folder(repo_id="TobyYang7/Branching-Search-runs",repo_type="dataset",
                folder_path=str(OUT/task),path_in_repo=f"gpt-5.4-nano/{task}",
                allow_patterns=["*.json","*.jsonl","*.csv","*.md","*.png"],
                ignore_patterns=["*.log","*.pid","*token*","*auth*","*secret*"],
                commit_message=f"GPT-5.4 Nano {task} six seeds")
            statuses[task]="UPLOADED"
        except Exception as err:
            statuses[task]="BLOCKED_"+type(err).__name__
    return {"status":"FINISHED","per_task":statuses}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-id",type=int,required=True)
    args=ap.parse_args()
    dest=download(args.run_id)
    restore(dest)
    violations=scan()
    atomic(OUT/"secret_audit.json",{"passed":not violations,"violations":violations})
    if violations:
        raise SystemExit("SECRET AUDIT FAILED; stop before publishing")
    results={task:aggregate(task) for task in TASKS}
    total=sum(v["completed_episodes"] for v in results.values())
    report={"model":"gpt-5.4-nano","resolved_model_id":"gpt-5.4-nano-2026-03-17",
        "reasoning_effort":"medium","seed_ids":list(range(6)),"planned_runs":18,
        "planned_episodes":192,"completed_episodes":total,"tasks":results,
        "source_workflow_run_id":args.run_id,
        "compatibility_note":"Hosted API environment injection; output token ceiling 8192 vs Luna 3072",
        "branch":"experiment/gpt54-nano-3tasks-6seeds-parallel",
        "secrets_scan_passed":True}
    # The dataset receives the combined report under its own model directory.
    atomic(OUT/"summary"/"combined_summary.json",report)
    uploads=upload(results)
    report["uploads"]=uploads
    atomic(OUT/"summary"/"final_report.json",report)
    atomic(ROOT/"results/gpt-5.4-nano/final_report.json",report)
    print(json.dumps({"completed_episodes":total,"planned_episodes":192,
                      "seed_states":{t:[r["state"] for r in v["seeds"]] for t,v in results.items()},
                      "upload_status":uploads},indent=2))
    if total!=192 or any(s["state"]!="COMPLETE" for r in results.values() for s in r["seeds"]):
        raise SystemExit("INCOMPLETE: science outputs retained; some seeds need retries")

if __name__=="__main__":
    main()
