#!/usr/bin/env python3
"""Publish an immutable checked Batch 2 release asset for deferred HF upload.

GitHub release assets are durable relative to expiring 90-day workflow artifacts.
This never invokes the Hugging Face write API.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/gpt-5.4-nano-t1-none-new3-batch2/summary"
REPORT=ROOT/"results/nano-t1-batch2/final_report.json"
REPO="wenwenzheng-gif/absorbing-state-multi-agent-search"
TAG="nano-t1-batch2-validated-2026-11-08"
BRANCH="experiment/gpt54-nano-t1-none-batch2-seeds10-12"
MODEL="gpt-5.4-nano-2026-03-17"
ARCHIVE="gpt-5.4-nano-t1_batch2_seeds10-12_runs.zip"

def command(*args):
    proc=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError("Failed to publish validated asset: "+proc.stderr[-350:])
    return proc.stdout

def main():
    if not os.getenv("GH_TOKEN"):raise RuntimeError("GH_TOKEN unavailable for durable release backup")
    report=json.loads(REPORT.read_text())
    expected={
        "batch":2,"status":"VALIDATED_LOCAL_ONLY","model":MODEL,
        "new_seed_indices":[10,11,12],"temperature":1.0,
        "reasoning_effort":"none","thinking_disabled":True,
        "completed_episodes":171,"completed_condition_runs":27,
        "secret_scan_passed":True
    }
    for k,v in expected.items():
        if report.get(k)!=v:raise ValueError("Batch 2 validated report mismatch: "+k)
    zip_path=OUT/ARCHIVE
    pending_path=OUT/"UPLOAD_PENDING.json"
    if not zip_path.is_file() or not pending_path.is_file():
        raise ValueError("Batch 2 ZIP/pending HF manifest missing")
    digest=hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if report.get("archive_sha256")!=digest:raise ValueError("Batch 2 SHA256 differs from validation report")
    pending=json.loads(pending_path.read_text())
    if pending.get("no_hf_upload_attempted") is not True:
        raise ValueError("Unexpected upload state")
    if pending["additive_uploads"][1]["sha256"]!=digest:
        raise ValueError("Pending Batch 2 upload SHA256 mismatch")
    present=subprocess.run(["gh","release","view",TAG,"--repo",REPO,"--json","tagName,assets"],
                           cwd=ROOT,capture_output=True,text=True)
    if present.returncode==0:
        assets={x["name"]:x for x in json.loads(present.stdout).get("assets",[])}
        if ARCHIVE not in assets or "UPLOAD_PENDING.json" not in assets:
            raise ValueError("Release tag already exists with missing artifacts")
        if int(assets[ARCHIVE].get("size") or 0)!=zip_path.stat().st_size:
            raise ValueError("Existing published release ZIP size mismatch")
        print("Existing immutable validated Batch 2 release confirmed.")
    else:
        notes=OUT/"BATCH2_RELEASE_NOTES.md"
        notes.write_text("Fully verified Nano t1 Batch 2 seeds 10–12 (171/171 episodes, "+
           "27/27 condition runs). Exact model snapshot gpt-5.4-nano-2026-03-17; "+
           "temperature=1, reasoning_effort=none, thinking disabled. "+
           "HF unavailable: this archive remains available for later additive upload. "+
           "Original six seeds and validated Batch 1 are unchanged.\n")
        command("gh","release","create",TAG,
                str(zip_path)+"#"+ARCHIVE,
                str(pending_path)+"#UPLOAD_PENDING.json",
                str(REPORT)+"#batch2_final_report.json",
                "--repo",REPO,"--target",BRANCH,
                "--title","Validated Nano t1 Batch 2 (seeds 10–12)",
                "--notes-file",str(notes),"--latest=false")
    print(json.dumps({"release_tag":TAG,"archive":ARCHIVE,
                      "bytes":zip_path.stat().st_size,"sha256":digest,
                      "completed_episodes":171,"hf_upload_pending":True},indent=2))

if __name__=="__main__":main()
