#!/usr/bin/env python3
"""One isolated (task, new seed index) supervisor; three original t1 arms in parallel.

Only the three new indices 10-12 are permitted. The scientific protocol is copied
from valid t1 seed6 configs. Config differences are limited to identity, seed,
exact historical model ID, connection, wall-cap removal, and API concurrency.
The 5-hour slice is a HOSTED-RUNNER checkpoint/upload boundary, not scientific
early stopping. Incomplete work is resumed by a later invocation with its own
corrected-run data; no old output tree is ever read.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/gpt-5.4-nano-t1-none-new3-batch2"
MODEL="gpt-5.4-nano-2026-03-17"
FAMILY="gpt-5.4-nano-t1"
TASK_SEEDS={"synthetic":87000,"tcas":87200,"physics":82240}
TASK_DEGREES={"synthetic":[0,8,16,24,39],
              "tcas":[0,6,10,14,18,24,39],
              "physics":[0,4,8,12,16,24,39]}
ARMS=("C0","C1","I1")

def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    temp.replace(path)

def loadrows(path):
    if not path.is_file(): return []
    result=[]
    with path.open(encoding="utf-8") as stream:
        for row in stream:
            if not row.strip(): continue
            try: result.append(json.loads(row))
            except json.JSONDecodeError: continue
    return result

def countlines(path):
    if not path.exists(): return 0
    with path.open(encoding="utf-8") as f: return sum(bool(x.strip()) for x in f)

def exact_config(task,seed,arm,task_root,concurrency):
    src=ROOT/f"configs/agents/{FAMILY}_seed6/{task}_{arm}.json"
    c=json.loads(src.read_text(encoding="utf-8"))
    assert c["task_seeds"]==[TASK_SEEDS[task]+5]
    llm=c["policy"]["llm"]
    assert llm["temperature"]==1.0
    assert llm["reasoning_effort"]=="none"
    assert llm["enable_thinking"] is False
    assert llm["max_output_tokens"]==3072
    assert llm["prompt_version"]=="v4"
    assert c["degrees"]==TASK_DEGREES[task]
    c["name"]=f"{FAMILY}_seed{seed}_{task}_{arm}"
    c["task_seeds"]=[TASK_SEEDS[task]+seed-1]
    c["output_root"]=str(task_root)
    llm["model"]=MODEL
    llm["base_url"]="https://api.openai.com/v1"
    llm["concurrency"]=concurrency
    # Original 54,000-second cap is purely infrastructure; do not truncate
    # a scientifically valid long run. Scientific stopping rules are unchanged.
    llm["max_wall_seconds"]=None
    return c

def finished_episodes(rundir,task,seed):
    expected={f"d{d}_t{TASK_SEEDS[task]+seed-1}_r0" for d in TASK_DEGREES[task]}
    rows=loadrows(rundir/"episodes.jsonl")
    good={r.get("episode_id") for r in rows if r.get("status")=="ok"}
    return len(good & expected), len(expected), good==expected and len(rows)==len(expected)

def sanitize_before_resume(rundir,attempt):
    """Quarantine interrupted/failed episode fragments; keep good episodes only."""
    episode_path=rundir/"episodes.jsonl"
    rows=loadrows(episode_path)
    good=[r for r in rows if r.get("status")=="ok"]
    good_ids={r.get("episode_id") for r in good}
    quarantined=0
    if not rows:
        # A killed request can leave partial ledgers with zero completed episodes.
        if attempt<=1:return
    quarantine=rundir/"quarantine"/f"before_attempt_{attempt}"
    for kind in ("episodes","events","requests","responses"):
        original=rundir/f"{kind}.jsonl"
        if not original.exists(): continue
        old=loadrows(original)
        accepted=(good if kind=="episodes" else
                  [row for row in old if row.get("episode_id") in good_ids])
        if len(accepted)==len(old):continue
        quarantine.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,quarantine/original.name)
        temp=original.with_suffix(".tmp")
        with temp.open("w",encoding="utf-8") as out:
            for r in accepted: out.write(json.dumps(r,ensure_ascii=False)+"\n")
        temp.replace(original)
        quarantined+=len(old)-len(accepted)
    if quarantined:
        write(quarantine/"audit.json",
              {"reason":"quarantined incomplete/failed episode fragments before safe resume",
               "records_quarantined":quarantined,"completed_episode_ids":sorted(good_ids)})

def telemetry(rundir):
    records=loadrows(rundir/"transport.jsonl")
    metrics={"api_responses":0,"http_429":0,"http_5xx":0,"timeouts":0,
             "transport_errors":0,"successful_responses_5min":0}
    now=time.time()
    for row in records:
        code=row.get("status")
        if code==200 or (isinstance(code,int) and 200<=code<300):
            metrics["api_responses"]+=1
            if now-row.get("at",0)<=300: metrics["successful_responses_5min"]+=1
        if code==429: metrics["http_429"]+=1
        if isinstance(code,int) and 500<=code<=599: metrics["http_5xx"]+=1
        if code=="TRANSPORT_ERROR":
            metrics["transport_errors"]+=1
            if "Timeout" in row.get("error_type",""):metrics["timeouts"]+=1
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",required=True,choices=tuple(TASK_SEEDS))
    p.add_argument("--seed-index",required=True,type=int,choices=(10,11,12))
    args=p.parse_args()
    task,seed=args.task,args.seed_index
    if os.environ.get("OPENAI_MODEL")!=MODEL:raise SystemExit("BLOCKED: model environment mismatch")
    if os.environ.get("OPENAI_BASE_URL")!="https://api.openai.com/v1":
        raise SystemExit("BLOCKED: endpoint mismatch")
    if not os.environ.get("OPENAI_API_KEY"):raise SystemExit("BLOCKED: API key missing")
    if os.environ.get("NANO_T1_ENFORCE")!="1":
        raise SystemExit("BLOCKED: strict request guard not enabled")
    concurrency=int(os.environ.get("NANO_T1_PER_ARM_CONCURRENCY","2"))
    if not 1<=concurrency<=12:raise SystemExit("Invalid per-arm API concurrency")
    arm_parallel=int(os.environ.get("NANO_T1_ARM_PARALLEL","3"))
    if not 1<=arm_parallel<=3: raise SystemExit("Invalid arm parallelism")
    work_slice=int(os.environ.get("NANO_T1_SLICE_SECONDS","17400"))
    if not 60<=work_slice<=18000:raise SystemExit("Invalid checkpoint slice")
    start=time.time()
    deadline=start+work_slice
    root=OUT/task/f"seed_{seed}"
    root.mkdir(parents=True,exist_ok=True)
    states={}
    tasks={}
    for arm in ARMS:
        config=exact_config(task,seed,arm,root,concurrency)
        cfg=root/"configs"/f"{arm}.json"
        if cfg.exists() and json.loads(cfg.read_text())!=config:
            raise SystemExit("BLOCKED: existing corrected-run config changed: "+str(cfg))
        write(cfg,config)
        rd=root/config["name"]
        rd.mkdir(parents=True,exist_ok=True)
        status_file=rd/"arm_status.json"
        status=json.loads(status_file.read_text()) if status_file.exists() else {}
        states[arm]=status if status else {"status":"PENDING","attempts":0,
                                        "model":MODEL,"temperature":1.0,
                                        "reasoning_effort":"none","thinking":False}
        done,total,complete=finished_episodes(rd,task,seed)
        states[arm]["completed_episodes"]=done
        states[arm]["target_episodes"]=total
        if complete:states[arm]["status"]="COMPLETE"
        tasks[arm]={"config":cfg,"rundir":rd,"proc":None,"started":None,
                    "log":None,"ready":start}
    def update_seed_status():
        data={}
        for arm in ARMS:
            rd=tasks[arm]["rundir"]
            done,total,complete=finished_episodes(rd,task,seed)
            states[arm]["completed_episodes"]=done
            states[arm]["target_episodes"]=total
            states[arm]["telemetry"]=telemetry(rd)
            states[arm]["responses"]=countlines(rd/"responses.jsonl")
            states[arm]["requests"]=countlines(rd/"requests.jsonl")
            if complete:states[arm]["status"]="COMPLETE"
            write(rd/"arm_status.json",states[arm])
            data[arm]=states[arm]
        all_done=all(data[a]["status"]=="COMPLETE" for a in ARMS)
        summary={"task":task,"seed_index":seed,"seed_value":TASK_SEEDS[task]+seed-1,
                 "status":"COMPLETE" if all_done else
                 "FAILED_AFTER_RETRIES" if any(data[a]["status"]=="FAILED_AFTER_RETRIES" for a in ARMS)
                 else "RUNNING","model":MODEL,"dataset_family":FAMILY,
                 "temperature":1.0,"reasoning_effort":"none","thinking":False,
                 "started_at":start,"updated_at":time.time(),
                 "completed_episodes":sum(data[a]["completed_episodes"] for a in ARMS),
                 "target_episodes":sum(data[a]["target_episodes"] for a in ARMS),
                 "arms":data,"per_arm_concurrency":concurrency,
                 "arm_parallel":arm_parallel}
        write(root/"seed_status.json",summary)
        return all_done
    if update_seed_status():
        print("Already COMPLETE",task,seed,flush=True);return 0
    while time.time()<deadline:
        for arm in ARMS:
            entry=tasks[arm]
            proc=entry["proc"]
            if proc is not None:
                rc=proc.poll()
                if rc is None: continue
                entry["proc"]=None
                states[arm]["last_return_code"]=rc
                done,total,complete=finished_episodes(entry["rundir"],task,seed)
                if complete:
                    states[arm]["status"]="COMPLETE"
                    print("COMPLETE",task,seed,arm,flush=True)
                else:
                    # A return code may be nonzero even after some valid episodes.
                    if states[arm]["attempts"]>=12:
                        states[arm]["status"]="FAILED_AFTER_RETRIES"
                    else:
                        states[arm]["status"]="RETRY_PENDING"
                        entry["ready"]=time.time()+min(90,2**min(states[arm]["attempts"],6))
            if entry["proc"] is not None:continue
            if states[arm]["status"] in ("COMPLETE","FAILED_AFTER_RETRIES"):continue
            if time.time()<entry["ready"] or time.time()>deadline-30:continue
            if sum(x["proc"] is not None for x in tasks.values())>=arm_parallel:continue
            states[arm]["attempts"]+=1
            attempt=states[arm]["attempts"]
            sanitize_before_resume(entry["rundir"],attempt)
            cmd=[sys.executable,"-m","src.agent_system.cli","run",
                 "--config",str(entry["config"]),"--run-id",
                 f"{FAMILY}_seed{seed}_{task}_{arm}",
                 "--output-root",str(root),"--resume"]
            env=dict(os.environ)
            env.update(NANO_T1_PROBE_PATH=str(entry["rundir"]/"outgoing_probe.json"),
                       NANO_T1_TRANSPORT_LOG=str(entry["rundir"]/"transport.jsonl"))
            log=entry["rundir"]/f"attempt_{attempt}.log"
            with log.open("ab") as sink:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=sink,stderr=subprocess.STDOUT,
                    start_new_session=True)
            entry["proc"]=p
            entry["started"]=time.time()
            states[arm]["status"]="RUNNING"
            states[arm]["worker_pid"]=p.pid
            print("START",task,seed,arm,"attempt",attempt,"pid",p.pid,flush=True)
        complete=update_seed_status()
        if complete: break
        if all(states[a]["status"] in ("COMPLETE","FAILED_AFTER_RETRIES") for a in ARMS):
            break
        time.sleep(10)
    # Restore the runner to the clean checkpoint boundary before GitHub uploads.
    for arm in ARMS:
        p=tasks[arm]["proc"]
        if p is not None and p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            if states[arm]["status"]!="COMPLETE":
                states[arm]["status"]="PAUSED_INFRA_SLICE"
        states[arm].pop("worker_pid",None)
    complete=update_seed_status()
    status=json.loads((root/"seed_status.json").read_text())
    if not complete and status["status"]!="FAILED_AFTER_RETRIES":
        status["status"]="PAUSED_INFRA_SLICE"
        write(root/"seed_status.json",status)
    print(json.dumps({"task":task,"seed_index":seed,"status":status["status"],
                      "episodes":status["completed_episodes"],
                      "target":status["target_episodes"],
                      "wall_seconds":round(time.time()-start)}),flush=True)
    # Return successfully so every partial checkpoint can be uploaded.
    return 0

if __name__=="__main__":
    sys.exit(main())
