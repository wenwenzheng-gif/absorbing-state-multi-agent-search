#!/usr/bin/env python3
"""Strictly validate/aggregate only three new Nano t1 seeds and add a new HF ZIP.

The original six-seed t1_runs.zip is read-only and never replaced. No failed,
incomplete or cancelled-medium episode enters the new archive.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/gpt-5.4-nano-t1-none-new3"
MODEL="gpt-5.4-nano-2026-03-17"
FAMILY="gpt-5.4-nano-t1"
TASKS={"synthetic":([0,8,16,24,39],87000),
       "tcas":([0,6,10,14,18,24,39],87200),
       "physics":([0,4,8,12,16,24,39],82240)}
ARMS=("C0","C1","I1")
SEEDS=(7,8,9)
DATASET="TobyYang7/Branching-Search-runs"
ORIGINAL="gpt-5.4-nano/gpt-5.4-nano-t1_runs.zip"
NEW_ZIP_PATH="gpt-5.4-nano/gpt-5.4-nano-t1_new_seeds7-9_runs.zip"
ALLOWED={"episodes.jsonl","events.jsonl","requests.jsonl","responses.jsonl",
         "checkpoint.json","manifest.json","run_summary.json","outgoing_probe.json",
         "config.json","arm_status.json"}

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")
    tmp.replace(path)

def gh(*args):
    c=subprocess.run(["gh",*args],cwd=ROOT,capture_output=True,text=True)
    if c.returncode: raise RuntimeError("GitHub artifact download failed")
    return c.stdout

def restore(run_id):
    for task in TASKS:
        for seed in SEEDS:
            dest=OUT/task/f"seed_{seed}"
            dest.mkdir(parents=True,exist_ok=True)
            gh("run","download",str(run_id),"--repo",
               "wenwenzheng-gif/absorbing-state-multi-agent-search",
               "--name",f"nano-t1-{task}-seed{seed}-checkpoint","--dir",str(dest))

def count_and_verify(path,task,seed,arm):
    degrees,start=TASKS[task]
    expected={f"d{d}_t{start+seed-1}_r0" for d in degrees}
    if not (path/"episodes.jsonl").exists():raise ValueError("missing episode ledger")
    rows=[json.loads(line) for line in (path/"episodes.jsonl").open() if line.strip()]
    ids=[r.get("episode_id") for r in rows]
    if len(rows)!=len(expected) or set(ids)!=expected or any(r.get("status")!="ok" for r in rows):
        raise ValueError("missing, failed, duplicated, or unexpected degree episode")
    if len(ids)!=len(set(ids)):raise ValueError("duplicate episode ID")
    proof=json.loads((path/"outgoing_probe.json").read_text())
    for key,value in [("model",MODEL),("temperature",1.0),
                      ("reasoning_effort","none"),("enable_thinking",False),
                      ("max_completion_tokens",3072)]:
        if proof.get(key)!=value:raise ValueError("outgoing request invariants mismatch: "+key)
    config=json.loads((path.parent/"configs"/(arm+".json")).read_text())
    llm=config["policy"]["llm"]
    for k,v in [("model",MODEL),("temperature",1.0),
                ("reasoning_effort","none"),("enable_thinking",False),
                ("max_output_tokens",3072)]:
        if llm.get(k)!=v:raise ValueError("model/sampling config mismatch: "+k)
    if config["degrees"]!=degrees or config["task_seeds"]!=[start+seed-1]:
        raise ValueError("scientific degree or task seed changed")
    historical=json.loads((ROOT/f"configs/agents/{FAMILY}_seed6/{task}_{arm}.json").read_text())
    differences=[]
    for k in ("protocol_version","agents","branching","experiments_per_agent",
              "initial_width","degrees","repeats","candidate_pool","environment",
              "epsilon_predictive"):
        if config.get(k)!=historical.get(k):differences.append(k)
    if config["policy"]["kind"]!=historical["policy"]["kind"]:
        differences.append("policy.kind")
    for k,v in historical["policy"].items():
        if k!="llm" and config["policy"].get(k)!=v:
            differences.append("policy."+k)
    allowed_transport_fields={"model","base_url","concurrency","max_wall_seconds"}
    for k,v in historical["policy"]["llm"].items():
        if k not in allowed_transport_fields and llm.get(k)!=v:
            differences.append("policy.llm."+k)
    if differences:raise ValueError("scientific protocol drift: "+",".join(differences))
    manifest=json.loads((path/"manifest.json").read_text())
    mconfig=manifest.get("config",{})
    if mconfig.get("policy",{}).get("llm",{}).get("model")!=MODEL:
        raise ValueError("manifest reports the wrong model")
    response_count=0
    requests=0
    for name in ("responses","requests"):
        file=path/(name+".jsonl")
        if not file.is_file(): raise ValueError("missing "+name+" ledger")
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():continue
                obj=json.loads(line)
                if obj.get("model")!=MODEL:
                    raise ValueError("mismatched model in "+name+" ledger")
                if obj.get("episode_id") and obj["episode_id"] not in expected:
                    raise ValueError("unknown episode ID in "+name+" ledger")
                if name=="responses":response_count+=1
                else:requests+=1
    if response_count==0: raise ValueError("No model calls in completed run")
    return {"episodes":len(rows),"model_responses":response_count,
            "requests":requests,"status":"COMPLETE"}

def scan_secrets(paths):
    # Compare against actual tokens without displaying them, as well as strong
    # credential-like patterns. Audit only what can enter uploaded archives.
    secrets=[x.encode() for k,x in os.environ.items()
             if ("KEY" in k or "TOKEN" in k) and x and len(x)>=12]
    pattern=re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,}|gh[opsu]_[A-Za-z0-9_]{25,})\b")
    violations=[]
    for file in paths:
        contents=file.read_bytes()
        if pattern.search(contents) or any(secret in contents for secret in secrets):
            violations.append(str(file.relative_to(ROOT)))
    return violations

def analysis(run_dirs):
    # Pure analysis projection: original source manifests and JSONL remain unmodified.
    run_root=ROOT/"runs"/"agents"
    config_root=ROOT/"configs"/"agents"
    for seed in SEEDS:
        tag=f"{FAMILY}_seed{seed}"
        (run_root/tag).mkdir(parents=True,exist_ok=True)
        (config_root/tag).mkdir(parents=True,exist_ok=True)
        for task in TASKS:
            for arm in ARMS:
                name=f"{tag}_{task}_{arm}"
                source=run_dirs[(task,seed,arm)]
                dest=run_root/tag/name
                # Existing merge_runs.py reads both configs/agents/<tag>/
                # and runs/agents/<tag>/. Build an analysis-only projection.
                cfg=source.parent/"configs"/(arm+".json")
                shutil.copy2(cfg,config_root/tag/(task+"_"+arm+".json"))
                if dest.is_symlink():
                    dest.unlink()
                elif dest.exists():
                    raise ValueError("Refusing to overwrite existing analysis directory: "+str(dest))
                dest.symlink_to(source,target_is_directory=True)
    output_tag=FAMILY+"_seed7-9"
    cmd=[sys.executable,"scripts/merge_runs.py","--out-tag",output_tag,
         "--from-tags",*[f"{FAMILY}_seed{n}" for n in SEEDS]]
    merged=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    if merged.returncode:
        raise RuntimeError("Existing merge_runs analysis failed: "+merged.stderr[-300:])
    table=subprocess.run([sys.executable,"scripts/dc_table.py","--tag",output_tag],
          cwd=ROOT,capture_output=True,text=True)
    if table.returncode:
        raise RuntimeError("Existing dc_table analysis failed: "+table.stderr[-300:])
    return output_tag

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-run-id",type=int,required=True)
    args=parser.parse_args()
    report={"model":MODEL,"dataset_family":FAMILY,"temperature":1.0,
            "reasoning_effort":"none","thinking_disabled":True,
            "new_seed_indices":list(SEEDS),
            "planned_task_seed_workers":9,"planned_condition_runs":27,
            "target_episodes":171,"source_workflow_run":args.source_run_id,
            "target_hf_repo":DATASET,"target_hf_path":NEW_ZIP_PATH,
            "original_hf_path_unchanged":ORIGINAL,"status":"VALIDATING"}
    summary=OUT/"summary"
    summary.mkdir(parents=True,exist_ok=True)
    try:
        restore(args.source_run_id)
        results={}
        dirs={}
        upload_files=[]
        for task in TASKS:
            for seed in SEEDS:
                seed_root=OUT/task/f"seed_{seed}"
                st=json.loads((seed_root/"seed_status.json").read_text())
                if st.get("status")!="COMPLETE":raise ValueError("Seed incomplete: "+task+" "+str(seed))
                for arm in ARMS:
                    tag=f"{FAMILY}_seed{seed}"
                    runid=f"{tag}_{task}_{arm}"
                    d=seed_root/runid
                    checks=count_and_verify(d,task,seed,arm)
                    results[runid]=checks
                    dirs[(task,seed,arm)]=d
                    upload_files.extend(x for x in d.iterdir() if x.is_file() and x.name in ALLOWED)
                    upload_files.append(seed_root/"configs"/(arm+".json"))
        if sum(v["episodes"] for v in results.values())!=171:raise ValueError("Wrong episode total")
        violations=scan_secrets(upload_files)
        report["secret_scan_passed"]=not violations
        if violations:raise ValueError("Credential-bearing output quarantined; upload forbidden")
        report["results"]=results
        report["completed_episodes"]=171
        report["completed_condition_runs"]=27
        report["total_model_responses"]=sum(v["model_responses"] for v in results.values())
        report["total_requests"]=sum(v["requests"] for v in results.values())
        tag=analysis(dirs)
        report["pooled_analysis_tag"]=tag
        report["status"]="VALIDATED"
        atomic(summary/"validation.json",report)
        # Preserve the full original runs/agents path convention in a NEW zip.
        zip_path=summary/"gpt-5.4-nano-t1_new_seeds7-9_runs.zip"
        with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
            for task in TASKS:
                for seed in SEEDS:
                    seed_root=OUT/task/f"seed_{seed}"
                    tag=f"{FAMILY}_seed{seed}"
                    for arm in ARMS:
                        runid=f"{tag}_{task}_{arm}"
                        d=seed_root/runid
                        for f in sorted(d.iterdir()):
                            if f.is_file() and f.name in ALLOWED:
                                zout.write(f,arcname=f"runs/agents/{tag}/{runid}/{f.name}")
                        cfg=seed_root/"configs"/(arm+".json")
                        zout.write(cfg,arcname=f"configs/agents/{tag}/{task}_{arm}.json")
            # Canonical pooled new-3 analysis, no old or cancelled data.
            for f in sorted((ROOT/"runs/agents"/(FAMILY+"_seed7-9")).rglob("*")):
                if f.is_file() and f.suffix in (".json",".jsonl",".csv",".md"):
                    zout.write(f,arcname=str(f.relative_to(ROOT)))
            # Include the repository's existing analysis-ready critical-degree
            # tables and strict pre-upload validation, not only raw ledgers.
            analysis_tables=ROOT/"runs"/"agents"/(FAMILY+"_seed7-9")
            if not analysis_tables.is_dir():
                raise ValueError("Expected pooled dc_table analysis directory missing")
            tables=[p for p in analysis_tables.rglob("*") if p.is_file() and p.suffix in (".csv",".md",".json")]
            if not tables:
                raise ValueError("Pooled critical-degree analysis returned no tables")
            for file in sorted(tables):
                zout.write(file,arcname=str(file.relative_to(ROOT)))
            zout.write(summary/"validation.json",arcname="NEW3_VALIDATION.json")
            payload=json.dumps({"dataset_family":FAMILY,"actual_model_snapshot":MODEL,
                   "temperature":1.0,"reasoning_effort":"none","enable_thinking":False,
                   "new_seed_indices":list(SEEDS),"completed_episodes":171,
                   "source_workflow_run":args.source_run_id},indent=2)
            zout.writestr("NEW3_PROVENANCE.json",payload)
        report["zip_bytes"]=zip_path.stat().st_size
        # Never overwrite the existing valid t1 ZIP. Authenticate only if the
        # environment already provides an authorized Hugging Face token.
        from huggingface_hub import HfApi
        auth=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        api=HfApi(token=auth or None)
        found={x.path for x in api.list_repo_tree(DATASET,repo_type="dataset",
                             path_in_repo="gpt-5.4-nano",recursive=False)}
        if ORIGINAL not in found:raise ValueError("Original valid Nano ZIP disappeared from HF")
        if NEW_ZIP_PATH in found:raise ValueError("New3 HF ZIP already exists; refuse overwrite")
        if auth:
            api.upload_file(path_or_fileobj=str(zip_path),path_in_repo=NEW_ZIP_PATH,
                            repo_id=DATASET,repo_type="dataset",
                            commit_message="Add exact GPT-5.4 Nano t1 independent seeds7-9")
            report["hf_upload_status"]="UPLOADED"
            report["status"]="COMPLETE"
        else:
            report["hf_upload_status"]="BLOCKED_NO_HF_AUTH"
            report["status"]="VALIDATED_LOCAL_ONLY"
    except Exception as e:
        report["status"]="BLOCKED"
        report["blocker_type"]=type(e).__name__
        report["blocker"]=str(e)[:350]
        print("FINALIZATION BLOCKED:",type(e).__name__,str(e)[:160],flush=True)
    finally:
        atomic(summary/"final_report.json",report)
        atomic(ROOT/"results/nano-t1-new3/final_report.json",report)
    print(json.dumps({"status":report["status"],"completed_episodes":report.get("completed_episodes"),
                     "target_episodes":171,"hf_upload_status":report.get("hf_upload_status")},indent=2))
    return 0 if report["status"] in ("COMPLETE","VALIDATED_LOCAL_ONLY") else 1

if __name__=="__main__":sys.exit(main())
