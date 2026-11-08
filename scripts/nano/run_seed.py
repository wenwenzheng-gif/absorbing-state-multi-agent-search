#!/usr/bin/env python3
"""Run one isolated seed, with exact-model preflight enforced by workflow."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = {"synthetic": "llm_toy", "tcas": "llm_tcas", "ode": "llm_physics"}
OUT = ROOT / "outputs/gpt-5.4-nano"
MODEL = "gpt-5.4-nano"

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def readrows(path):
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): out.append(json.loads(line))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--seed", required=True, type=int, choices=range(6))
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("BLOCKED: OPENAI_API_KEY is not injected")
    if os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") != "https://api.openai.com/v1":
        raise SystemExit("BLOCKED: unexpected API endpoint")
    root = OUT / args.task
    d = root / ("seed_" + str(args.seed))
    d.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "configs/agents" / (TASKS[args.task] + ".json")).read_text())
    base.update(name=f"nano_{args.task}_{args.seed}", task_seeds=[args.seed], output_root=str(root))
    base["policy"]["llm"].update(model=MODEL, base_url="https://api.openai.com/v1",
                                  api_style="openai", reasoning_effort="medium",
                                  temperature=None, top_p=None, transport_retries=8,
                                  concurrency=int(os.getenv("NANO_INTERNAL_CONCURRENCY", "4")))
    target = len(base["degrees"])
    atomic(d / "config.json", base)
    start = time.time()
    previous = (d / "status.json")
    if previous.exists() and json.loads(previous.read_text()).get("status") == "COMPLETE":
        print("Already complete", args.task, args.seed, flush=True)
        return
    attempts = []
    for attempt in range(1, 4):
        rows = readrows(d / "episodes.jsonl")
        good = [r for r in rows if r.get("status") == "ok"]
        completed = {r["episode_id"] for r in good}
        if len(completed) >= target:
            break
        # Resume only successful episodes; archive failed episode rows for audit.
        if rows and len(good) != len(rows):
            backup = d / f"attempt_{attempt}_original_episodes.jsonl"
            backup.write_bytes((d / "episodes.jsonl").read_bytes())
            (d / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in good))
        atomic(d / "status.json", {"status": "RUNNING", "task": args.task, "seed": args.seed,
                                   "attempt": attempt, "completed_episodes": len(completed),
                                   "target_episodes": target, "started": start,
                                   "model": MODEL, "reasoning_effort": "medium"})
        with (d / f"attempt_{attempt}.log").open("ab") as log:
            proc = subprocess.run([sys.executable, "-m", "src.agent_system.cli", "run",
                                   "--config", str(d / "config.json"),
                                   "--output-root", str(root),
                                   "--run-id", d.name, "--resume"],
                                  cwd=ROOT, env={**os.environ, "OPENAI_MODEL": MODEL},
                                  stdout=log, stderr=subprocess.STDOUT)
        rows = readrows(d / "episodes.jsonl")
        good = {r["episode_id"] for r in rows if r.get("status") == "ok"}
        attempts.append({"attempt": attempt, "exit_code": proc.returncode,
                         "successful_episodes": len(good), "recorded_episodes": len(rows)})
        if len(good) >= target: break
        time.sleep(min(30, 2 ** attempt))
    rows = readrows(d / "episodes.jsonl")
    good = {r["episode_id"] for r in rows if r.get("status") == "ok"}
    status = "COMPLETE" if len(good) == target else "FAILED_AFTER_RETRIES"
    atomic(d / "status.json", {"status": status, "task": args.task, "seed": args.seed,
                              "completed_episodes": len(good), "target_episodes": target,
                              "started": start, "finished": time.time(),
                              "wall_seconds": time.time() - start,
                              "model": MODEL, "reasoning_effort": "medium",
                              "attempts": attempts})
    print(args.task, args.seed, status, len(good), "/", target, flush=True)
    if status != "COMPLETE": raise SystemExit(1)

if __name__ == "__main__": main()
