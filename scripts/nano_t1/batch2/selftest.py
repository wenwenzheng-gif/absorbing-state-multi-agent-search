#!/usr/bin/env python3
"""Offline static equivalence for all 27 Batch 2 condition configs."""
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]
path=ROOT/"scripts/nano_t1/batch2/run_seed.py"
spec=importlib.util.spec_from_file_location("nano_t1_batch2_worker",path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def main():
    assert mod.MODEL=="gpt-5.4-nano-2026-03-17"
    assert mod.FAMILY=="gpt-5.4-nano-t1"
    observed=0
    for task,start in mod.TASK_SEEDS.items():
        for seed in (10,11,12):
            for arm in mod.ARMS:
                with tempfile.TemporaryDirectory() as folder:
                    cfg=mod.exact_config(task,seed,arm,Path(folder),2)
                original=json.loads((ROOT/f"configs/agents/gpt-5.4-nano-t1_seed6/{task}_{arm}.json").read_text())
                assert cfg["task_seeds"]==[start+seed-1]
                assert cfg["degrees"]==original["degrees"]
                assert cfg["environment"]==original["environment"]
                assert cfg["candidate_pool"]==original["candidate_pool"]
                assert cfg["agents"]==original["agents"]==40
                assert cfg["initial_width"]==original["initial_width"]==8
                assert cfg["branching"]==original["branching"]==2
                assert cfg["policy"]["kind"]==original["policy"]["kind"]
                for k,v in original["policy"].items():
                    if k!="llm":assert cfg["policy"][k]==v
                sampled=cfg["policy"]["llm"]
                expected=original["policy"]["llm"]
                ignored={"model","base_url","concurrency","max_wall_seconds"}
                assert {k:v for k,v in sampled.items() if k not in ignored} == {
                       k:v for k,v in expected.items() if k not in ignored}
                assert sampled["model"]==mod.MODEL
                assert sampled["temperature"]==1.0
                assert sampled["reasoning_effort"]=="none"
                assert sampled["enable_thinking"] is False
                observed+=1
    assert observed==27
    assert sum(len(mod.TASK_DEGREES[t]) for t in mod.TASK_DEGREES)*3*3==171
    print("Batch 2 PASS: exact original Nano scientific config, 27 arms, 171 episodes; only seed and infrastructure differ.")

if __name__=="__main__":main()
