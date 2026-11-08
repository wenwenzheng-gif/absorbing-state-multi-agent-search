#!/usr/bin/env python3
"""Read-only original Nano t1 provenance audit. Never print credentials or raw trajectories."""
import json, os, re, sys
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/gpt-5.4-nano-t1-none-new3"/"audit.json"
DATASET="TobyYang7/Branching-Search-runs"
PREFIX="gpt-5.4-nano-t1"
EXPECTED={"synthetic":87000,"tcas":87200,"physics":82240}
POLICIES=("C0","C1","I1")
REQUESTED_NEW=(7,8,9)
REQUIRED={"temperature":1.0,"reasoning_effort":"none","enable_thinking":False}
MODEL_SNAPSHOT_PATTERN=re.compile(r"^gpt-5\.4-nano-\d{4}-\d{2}-\d{2}$")

def save(report):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    t=OUT.with_suffix(".tmp")
    t.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    t.replace(OUT)

def run():
    report={"time_utc":datetime.now(timezone.utc).isoformat(),
            "dataset_family":PREFIX,"requested_new_seed_indices":list(REQUESTED_NEW),
            "valid":False,"checks":{},"blockers":[]}
    configs={}
    for si in range(1,7):
        for task in EXPECTED:
            for arm in POLICIES:
                path=ROOT/f"configs/agents/{PREFIX}_seed{si}/{task}_{arm}.json"
                if not path.exists():
                    report["blockers"].append("Original config missing: "+str(path.relative_to(ROOT)))
                    continue
                c=json.loads(path.read_text())
                llm=c["policy"]["llm"]
                if any(llm.get(k)!=v for k,v in REQUIRED.items()):
                    report["blockers"].append(f"Original protocol mismatch: {si}/{task}/{arm}")
                if c["task_seeds"]!=[EXPECTED[task]+si-1]:
                    report["blockers"].append(f"Unexpected original task seed: {si}/{task}/{arm}")
                configs[(si,task,arm)]=c
    report["checks"]["original_configs_checked"]=len(configs)
    report["checks"]["original_seed_indices"]=list(range(1,7))
    report["checks"]["original_aliases"]=sorted({c["policy"]["llm"].get("model") for c in configs.values()})
    report["checks"]["required_sampling"]=REQUIRED
    if len(configs)!=54:
        save(report);return 2
    for task in EXPECTED:
        for arm in POLICIES:
            baseline=configs[(1,task,arm)]
            for si in range(2,7):
                cmp=configs[(si,task,arm)]
                # The original six configs differ only in names/output root and seed.
                for field in ("protocol_version","agents","branching","experiments_per_agent",
                              "initial_width","degrees","candidate_pool","environment","repeats",
                              "policy","epsilon_predictive"):
                    if cmp.get(field)!=baseline.get(field):
                        report["blockers"].append(f"Original config drift {si}/{task}/{arm}/{field}")
    # Public Hugging Face access is sufficient, HF_TOKEN only if already configured.
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
        api=HfApi(token=token)
        info=api.repo_info(repo_id=DATASET,repo_type="dataset")
        report["checks"]["hf_repo_accessible"]=True
        report["checks"]["hf_revision"]=info.sha
        root=list(api.list_repo_tree(DATASET,repo_type="dataset",path_in_repo="",recursive=False))
        report["checks"]["hf_top_level"]=[x.path for x in root if re.search("nano",x.path,re.I)][:30]
        files=[obj.path for obj in api.list_repo_tree(DATASET,repo_type="dataset",
                    path_in_repo=PREFIX,recursive=True) if obj.path and not obj.path.endswith("/")]
        report["checks"]["hf_files_in_original_family"]=len(files)
        report["checks"]["hf_path_samples"]=files[:35]
        if not files: report["blockers"].append("Valid Nano t1 dataset path has no files.")
        # Find historical seed IDs from paths, ensure candidate identifiers truly unused.
        ids=sorted({int(n) for p in files for n in re.findall(r"(?:^|[/_-])seed[_-]?(\d+)(?:[/_.-]|$)",p)})
        report["checks"]["hf_seed_indices_observed_in_paths"]=ids
        if any(x in ids for x in REQUESTED_NEW):
            report["blockers"].append("Requested seed7-9 index overlaps HF existing paths.")
        names=[p for p in files if p.endswith("/manifest.json") or p.endswith("/config.json")]
        responses=[p for p in files if p.endswith("/responses.jsonl")]
        report["checks"]["hf_manifest_count"]=len(names)
        report["checks"]["hf_response_ledger_count"]=len(responses)
        # Sample across different tasks, seeking actual model version in recorded response.
        snapshots=set()
        sampled=[]
        for task in EXPECTED:
            selected=next((p for p in responses if task in p.lower()),None)
            if not selected: continue
            try:
                loc=hf_hub_download(repo_id=DATASET,repo_type="dataset",filename=selected,
                                    token=token)
                # Keep only model identifiers; never print or save prompts/content.
                with open(loc,encoding="utf-8") as inp:
                    for row in inp:
                        if row.strip():
                            obj=json.loads(row)
                            ident=obj.get("model")
                            if ident: snapshots.add(str(ident))
                            sampled.append({"task":task,"file":selected,"model":ident})
                            break
            except Exception as e:
                report["blockers"].append(f"Cannot read original response model for {task}: {type(e).__name__}")
        report["checks"]["original_response_model_ids"]=sorted(snapshots)
        report["checks"]["original_response_samples"]=sampled
        if not sampled:
            report["blockers"].append("Could not identify original resolved model snapshot from response records.")
        if len(snapshots)!=1 or not MODEL_SNAPSHOT_PATTERN.fullmatch(next(iter(snapshots),"")):
            report["blockers"].append("Original model version unverified or inconsistent; refusing a substitute.")
        else:
            report["resolved_original_snapshot"]=next(iter(snapshots))
    except Exception as e:
        report["checks"]["hf_repo_accessible"]=False
        report["blockers"].append("Hugging Face original data access: "+type(e).__name__)
    report["selected_new_task_seeds"]={
        task:[start+i-1 for i in REQUESTED_NEW] for task,start in EXPECTED.items()}
    report["valid"]=not report["blockers"]
    save(report)
    print(json.dumps({"audit_passed":report["valid"],
           "original_snapshot":report.get("resolved_original_snapshot"),
           "observed_existing_seed_indices":report["checks"].get("hf_seed_indices_observed_in_paths"),
           "new_seed_indices":list(REQUESTED_NEW),
           "new_task_seeds":report["selected_new_task_seeds"],
           "blockers":report["blockers"]},indent=2))
    return 0 if report["valid"] else 2

if __name__=="__main__":
    sys.exit(run())
