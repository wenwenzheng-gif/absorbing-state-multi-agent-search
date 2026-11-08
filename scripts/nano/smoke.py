#!/usr/bin/env python3
"""A task-specific end-to-end smoke that never counts as a seed."""
import json, os, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
TASKS={"synthetic":"llm_toy","tcas":"llm_tcas","ode":"llm_physics"}
task=sys.argv[1]
if task not in TASKS: raise SystemExit("unknown task")
if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("missing OPENAI_API_KEY")
preflight=ROOT/"outputs/gpt-5.4-nano/preflight.json"
# The hosted run separately validates its own exact model first.
base=json.loads((ROOT/f"configs/agents/{TASKS[task]}.json").read_text())
base.update(name=f"nano_smoke_{task}",agents=2,initial_width=1,degrees=[0],task_seeds=[900000],
            output_root=str(ROOT/"outputs/gpt-5.4-nano/_smoke"))
base["policy"]["llm"].update(model="gpt-5.4-nano",base_url="https://api.openai.com/v1",
    api_style="openai",reasoning_effort="medium",temperature=None,top_p=None,
    concurrency=2,transport_retries=8,max_output_tokens=8192)
out=ROOT/"outputs/gpt-5.4-nano/_smoke"/task
out.mkdir(parents=True,exist_ok=True)
config=out/"config.json"
config.write_text(json.dumps(base,indent=2))
with (out/"smoke.log").open("wb") as log:
    rc=subprocess.run([sys.executable,"-m","src.agent_system.cli","run",
      "--config",str(config),"--output-root",str(out.parent),
      "--run-id",task,"--resume"],cwd=ROOT,
      env={**os.environ,"OPENAI_MODEL":"gpt-5.4-nano"},stdout=log,stderr=subprocess.STDOUT).returncode
def read(path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
episodes=read(out/"episodes.jsonl");responses=read(out/"responses.jsonl")
passed=(rc==0 and len(episodes)==1 and episodes[0].get("status")=="ok" and
        len(responses)>0 and all(x.get("model") in ("gpt-5.4-nano","gpt-5.4-nano-2026-03-17") for x in responses))
(out/"smoke_check.json").write_text(json.dumps({"task":task,"status":"PASS" if passed else "FAILED",
 "exit_code":rc,"episodes":len(episodes),"successful_episodes":sum(x.get("status")=="ok" for x in episodes),
 "responses":len(responses),"resolved_models":sorted(set(str(x.get("model")) for x in responses))},indent=2))
print(task,"PASS" if passed else "FAILED", "episodes",len(episodes),"responses",len(responses),flush=True)
if not passed: raise SystemExit(1)
