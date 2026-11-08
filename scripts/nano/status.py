#!/usr/bin/env python3
"""Read-only Nano status; never print credentials or raw request bodies."""
import collections, json, pathlib, time
ROOT=pathlib.Path(__file__).resolve().parents[2]/"outputs/gpt-5.4-nano"
GOALS={"synthetic":54,"tcas":72,"ode":66}
total=0
for task,target in GOALS.items():
    done=0
    for seed in range(6):
        d=ROOT/task/f"seed_{seed}"
        status=json.loads((d/"status.json").read_text()) if (d/"status.json").exists() else {}
        episodes=[]
        if (d/"episodes.jsonl").exists():
            for line in (d/"episodes.jsonl").open():
                if line.strip():
                    try: episodes.append(json.loads(line))
                    except json.JSONDecodeError: continue
        successes={row.get("episode_id") for row in episodes if row.get("status")=="ok"}
        done+=len(successes)
        counts={}
        for name in ("requests","responses","transport_errors"):
            p=d/f"{name}.jsonl"
            counts[name]=sum(1 for _ in p.open()) if p.exists() else 0
        last=max((p.stat().st_mtime for p in d.glob("*") if p.is_file()),default=None)
        print(f"{task:9} seed {seed}: {status.get('status','PENDING'):20} "
              f"episodes {len(successes)}/{target//6} "
              f"req {counts['requests']} resp {counts['responses']} errors {counts['transport_errors']} "
              f"last {time.strftime('%Y-%m-%d %H:%M:%SZ',time.gmtime(last)) if last else '-'}")
    total+=done
    print(f"{task:9} TOTAL {done}/{target}")
print(f"ALL       TOTAL {total}/192")
