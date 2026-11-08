#!/usr/bin/env python3
"""Durable, immutable GitHub release backup for the already validated Batch 1.

The GH Actions artifact expires; a tagged release asset survives beyond that
retention. This does not change / upload anything to Hugging Face.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/"outputs/gpt-5.4-nano-t1-none-new3-batch1-preserved"
REPO="wenwenzheng-gif/absorbing-state-multi-agent-search"
BATCH1_RUN=42519445387
BATCH1_ARTIFACT="nano-t1-final-branch-42519445387"
ZIPNAME="gpt-5.4-nano-t1_new_seeds7-9_runs.zip"
TAG="nano-t1-batch1-validated-2026-11-08"
MODEL="gpt-5.4-nano-2026-03-17"

def command(*args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    if p.returncode:raise RuntimeError("Backup command failed: "+args[0]+" "+str(args[1:3])+" "+p.stderr[-300:])
    return p.stdout

def main():
    if not os.environ.get("GH_TOKEN"):raise SystemExit("Repository Actions token missing")
    DEST.mkdir(parents=True,exist_ok=True)
    first=json.loads((ROOT/"results/nano-t1-new3/final_report.json").read_text())
    assert first["status"]=="VALIDATED_LOCAL_ONLY"
    assert first["completed_episodes"]==171
    assert first["completed_condition_runs"]==27
    assert first["new_seed_indices"]==[7,8,9]
    assert first["model"]==MODEL
    assert first["reasoning_effort"]=="none"
    assert first["temperature"]==1.0
    assert first["thinking_disabled"] is True
    assert first["secret_scan_passed"] is True
    src=DEST/"downloaded"
    src.mkdir(exist_ok=True)
    command("gh","run","download",str(BATCH1_RUN),"--repo",REPO,"--name",BATCH1_ARTIFACT,"--dir",str(src))
    matches=list(src.rglob(ZIPNAME))
    if len(matches)!=1:raise ValueError("Expected one validated Batch 1 archive, found "+str(len(matches)))
    zip_path=DEST/ZIPNAME
    shutil.copyfile(matches[0],zip_path)
    count=0
    expected={("synthetic",i,a) for i in (7,8,9) for a in ("C0","C1","I1")} | {
        (task,i,a) for task in ("tcas","physics") for i in (7,8,9) for a in ("C0","C1","I1")}
    observed=set()
    with zipfile.ZipFile(zip_path) as archive:
        namelist=archive.namelist()
        assert "NEW3_PROVENANCE.json" in namelist
        assert "NEW3_VALIDATION.json" in namelist
        proof=json.loads(archive.read("NEW3_PROVENANCE.json"))
        assert proof["new_seed_indices"]==[7,8,9] and proof["actual_model_snapshot"]==MODEL
        assert proof["completed_episodes"]==171
        for member in namelist:
            if not member.endswith("/episodes.jsonl") or "runs/agents/" not in member:continue
            # The ZIP also contains an analysis-only pooled seed7-9 view. Skip it.
            if "/gpt-5.4-nano-t1_seed7-9/" in member:continue
            m=member.split("/")
            runid=m[-2]
            found=next(((t,i,a) for t,i,a in expected if runid==f"gpt-5.4-nano-t1_seed{i}_{t}_{a}"),None)
            if not found:raise ValueError("Unexpected Batch 1 episode file: "+member)
            if found in observed:raise ValueError("Duplicate Batch 1 run: "+member)
            observed.add(found)
            lines=[json.loads(x) for x in archive.read(member).splitlines() if x.strip()]
            assert lines and all(z.get("status")=="ok" for z in lines)
            count+=len(lines)
    assert count==171 and observed==expected,(count,len(observed))
    digest=hashlib.sha256(zip_path.read_bytes()).hexdigest()
    manifest={"dataset_family":"gpt-5.4-nano-t1","batch":"1",
              "model_snapshot":MODEL,"seed_indices":[7,8,9],
              "completed_episodes":171,"validated_condition_runs":27,
              "sha256":digest,"archive_name":ZIPNAME,
              "hf_upload_status":"PENDING_HF_WRITE_CREDENTIAL",
              "intended_hf_repo":"TobyYang7/Branching-Search-runs",
              "intended_hf_path":"gpt-5.4-nano/"+ZIPNAME,
              "source_run":BATCH1_RUN,"source_validation_run":42519445387,
              "original_valid_hf_archive_unchanged":True}
    manifest_path=DEST/"batch1_upload_pending.json"
    manifest_path.write_text(json.dumps(manifest,indent=2))
    shutil.copy2(ROOT/"results/nano-t1-new3/final_report.json",DEST/"batch1_final_report.json")
    existing=subprocess.run(["gh","release","view",TAG,"--repo",REPO,"--json","tagName,assets"],
                            cwd=ROOT,capture_output=True,text=True)
    if existing.returncode==0:
        old=json.loads(existing.stdout)
        files={a["name"] for a in old.get("assets",[])}
        assert ZIPNAME in files and "batch1_upload_pending.json" in files
        print("RELEASE_ALREADY_EXISTS: verified assets; no overwrite")
    else:
        notes=DEST/"RELEASE_NOTES.md"
        notes.write_text("Validated Nano t1 Batch 1 seeds 7-9; 171/171 episodes, exact dated snapshot. "+
            "Preserved as a release asset pending separate Hugging Face upload. "+
            "No original Nano, Luna or Qwen data were modified.\n")
        command("gh","release","create",TAG,
                str(zip_path)+"#"+ZIPNAME,
                str(manifest_path)+"#batch1_upload_pending.json",
                str(DEST/"batch1_final_report.json")+"#batch1_final_report.json",
                "--repo",REPO,"--target","experiment/gpt54-nano-t1-none-batch2-seeds10-12",
                "--title","Validated Nano t1 Batch 1 (seeds 7–9)",
                "--notes-file",str(notes),"--latest=false")
        print("DURABLE_BATCH1_RELEASE_CREATED")
    print(json.dumps({"tag":TAG,"zip_bytes":zip_path.stat().st_size,
           "sha256":digest,"episodes":count,"conditions":len(observed)},indent=2))
if __name__=="__main__":main()
