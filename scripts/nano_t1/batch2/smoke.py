#!/usr/bin/env python3
"""Small end-to-end task smoke. Never counts as one of the three new seeds."""
import json,os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
MODEL="gpt-5.4-nano-2026-03-17"
task=sys.argv[1]
if task not in ("synthetic","tcas","physics"):raise SystemExit("Invalid task")
if os.environ.get("OPENAI_MODEL")!=MODEL or os.environ.get("NANO_T1_ENFORCE")!="1":
    raise SystemExit("Strict model or request guard absent")
original=ROOT/f"configs/agents/gpt-5.4-nano-t1_seed6/{task}_C0.json"
config=json.loads(original.read_text())
config.update(name=f"nano_t1_batch2_smoke_{task}",degrees=[0],task_seeds=[990000],
              agents=2,initial_width=1)
llm=config["policy"]["llm"]
llm.update(model=MODEL,base_url="https://api.openai.com/v1",concurrency=2,
           max_wall_seconds=None)
assert llm["temperature"]==1.0 and llm["reasoning_effort"]=="none"
assert llm["enable_thinking"] is False and llm["max_output_tokens"]==3072
root=ROOT/"outputs/gpt-5.4-nano-t1-none-new3-batch2/_smoke"/task
root.mkdir(parents=True,exist_ok=True)
config["output_root"]=str(root)
cfg=root/"smoke_config.json"
cfg.write_text(json.dumps(config,indent=2))
rundir=root/config["name"]
env=dict(os.environ,NANO_T1_PROBE_PATH=str(rundir/"outgoing_probe.json"),
         NANO_T1_TRANSPORT_LOG=str(rundir/"transport.jsonl"))
with (root/"smoke.log").open("wb") as output:
    result=subprocess.run([sys.executable,"-m","src.agent_system.cli","run",
         "--config",str(cfg),"--output-root",str(root),
         "--run-id",config["name"]],cwd=ROOT,env=env,
         stdout=output,stderr=subprocess.STDOUT)
def readrows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
episodes=readrows(rundir/"episodes.jsonl")
responses=readrows(rundir/"responses.jsonl")
probe=json.loads((rundir/"outgoing_probe.json").read_text()) if (rundir/"outgoing_probe.json").exists() else {}
good=(result.returncode==0 and len(episodes)==1 and
      episodes[0].get("status")=="ok" and len(responses)>0 and
      all(r.get("model")==MODEL for r in responses) and
      probe.get("model")==MODEL and probe.get("temperature")==1.0 and
      probe.get("reasoning_effort")=="none" and
      probe.get("enable_thinking") is False)
check={"task":task,"status":"PASS" if good else "FAILED",
       "episodes":len(episodes),"valid_episodes":sum(x.get("status")=="ok" for x in episodes),
       "response_count":len(responses),"model_ids":sorted({str(x.get("model")) for x in responses}),
       "actual_outgoing_request":probe,"exit_code":result.returncode}
(root/"smoke_check.json").write_text(json.dumps(check,indent=2))
print(task,check["status"],"responses",len(responses),flush=True)
if not good:raise SystemExit(1)
