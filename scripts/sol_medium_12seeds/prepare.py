#!/usr/bin/env python3
"""Prepare one original agent_v1 v4-prompt cell, changing only model/effort/seeds.

The source is the checked-in Luna-medium config in the public paper repository.
No network operations or paid requests occur here.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "configs/agents/gpt-6-luna-re-medium_seed1"
START_SEEDS = {"synthetic": 87000, "tcas": 87200, "physics": 82240}
POLICIES = ("C0", "C1", "I1")

def prepare(task: str, policy: str, index: int) -> tuple[str, Path]:
    if task not in START_SEEDS or policy not in POLICIES or index not in range(1, 13):
        raise ValueError("Expected task synthetic/tcas/physics; policy C0/C1/I1; seed index 1..12")
    original = json.loads((SOURCE / f"{task}_{policy}.json").read_text())
    data = copy.deepcopy(original)
    tag = f"sol-medium_seed{index:02d}"
    data["name"] = f"{tag}_{task}_{policy}"
    data["task_seeds"] = [START_SEEDS[task] + index - 1]
    data["output_root"] = f"runs/agents/{tag}"
    data["notes"] = original.get("notes", "") + " GPT-6 Sol medium replication, v4 prompts unchanged; 12 paired seed indices."
    llm = data["policy"]["llm"]
    llm["model"] = "gpt-6-sol"
    llm["api_style"] = "openai"
    llm["reasoning_effort"] = "medium"
    llm["temperature"] = None  # OpenAI-compatible request builder omits this field
    llm["enable_thinking"] = False
    llm["prompt_version"] = "v4"
    # The endpoint is injected at execution using a GitHub secret; never write keys.
    llm["base_url"] = ""
    if llm["max_wall_seconds"] < 10000:
        raise ValueError("Reference unexpectedly has a reduced wall-clock ceiling")
    assert data["protocol_version"] == "agent_v1"
    assert data["agents"] == 40 and data["branching"] == 2
    assert data["experiments_per_agent"] == 1 and data["initial_width"] == 8
    assert data["policy"]["kind"] == "llm_full"
    expected_degree = {"synthetic": [0,8,16,24,39], "tcas": [0,6,10,14,18,24,39],
                       "physics": [0,4,8,12,16,24,39]}[task]
    assert data["degrees"] == expected_degree
    target = ROOT / "configs/agents" / tag / f"{task}_{policy}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")
    return tag, target

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=tuple(START_SEEDS))
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--seed-index", required=True, type=int, choices=range(1,13))
    a=p.parse_args()
    tag,target=prepare(a.task,a.policy,a.seed_index)
    print(json.dumps({"tag":tag,"config":str(target.relative_to(ROOT)),
                      "seed":START_SEEDS[a.task]+a.seed_index-1,
                      "degrees":json.loads(target.read_text())["degrees"]}))

if __name__=="__main__":
    main()
