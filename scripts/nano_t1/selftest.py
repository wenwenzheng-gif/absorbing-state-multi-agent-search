#!/usr/bin/env python3
"""Offline reproducibility checks; no hosted API requests or credentials."""
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("nano_new3_runner",ROOT/"scripts/nano_t1/run_seed.py")
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
VALID=importlib.util.spec_from_file_location("nano_new3_finalizer",ROOT/"scripts/nano_t1/finalize.py")
fin=importlib.util.module_from_spec(VALID)
VALID.loader.exec_module(fin)

def main():
    count=0
    for task,start in mod.TASK_SEEDS.items():
        for seed in (7,8,9):
            for arm in mod.ARMS:
                with tempfile.TemporaryDirectory() as tmp:
                    c=mod.exact_config(task,seed,arm,Path(tmp),1)
                    historical=json.loads((ROOT/f"configs/agents/gpt-5.4-nano-t1_seed6/{task}_{arm}.json").read_text())
                    assert c["task_seeds"]==[start+seed-1]
                    assert c["degrees"]==historical["degrees"]
                    assert c["agents"]==historical["agents"]==40
                    assert c["initial_width"]==historical["initial_width"]==8
                    llm=c["policy"]["llm"]
                    assert llm["model"]=="gpt-5.4-nano-2026-03-17"
                    assert llm["temperature"]==1.0
                    assert llm["reasoning_effort"]=="none"
                    assert llm["enable_thinking"] is False
                    assert llm["max_output_tokens"]==3072
                    assert llm["communication_mode"]==historical["policy"]["llm"]["communication_mode"]
                    expected={k:v for k,v in historical["policy"]["llm"].items() if k not in
                              ("model","base_url","concurrency","max_wall_seconds")}
                    observed={k:v for k,v in llm.items() if k not in
                              ("model","base_url","concurrency","max_wall_seconds")}
                    assert expected==observed,task+arm
                    assert mod.TASK_DEGREES[task]==fin.TASKS[task][0]
                    assert fin.TASKS[task][1]==start
                    count+=1
    assert count==27
    assert sum(len(mod.TASK_DEGREES[t])*3*3 for t in mod.TASK_DEGREES)==171
    print("PASS: 27 new3 condition configurations; strict exact-model/science invariants; 171 episodes.")
if __name__=="__main__":main()
