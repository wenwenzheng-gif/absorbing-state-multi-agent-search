#!/usr/bin/env python3
"""Short real-upstream capacity probe; scientific configs/tasks are not modified."""
import asyncio, json, os, statistics, time
from pathlib import Path
import httpx
MODEL="gpt-5.4-nano-2026-03-17"
OUT=Path("outputs/gpt-5.4-nano-t1-none-new3/capacity.json")
LEVELS=(8,16,24,32)
BODY={"model":MODEL,"temperature":1.0,"reasoning_effort":"none",
      "messages":[{"role":"user","content":"Return {\\\"ok\\\":true} as JSON."}],
      "max_completion_tokens":128,
      "response_format":{"type":"json_schema","json_schema":{
         "name":"nano_t1_capacity","strict":True,
         "schema":{"type":"object","additionalProperties":False,
                   "required":["ok"],"properties":{"ok":{"type":"boolean","enum":[True]}}}}}}
async def main():
    assert os.getenv("OPENAI_MODEL")==MODEL and os.getenv("OPENAI_BASE_URL")=="https://api.openai.com/v1"
    token=os.environ["OPENAI_API_KEY"]
    results=[]
    limits=httpx.Limits(max_connections=40,max_keepalive_connections=40)
    async with httpx.AsyncClient(base_url="https://api.openai.com/v1",
          headers={"Authorization":"Bearer "+token},timeout=httpx.Timeout(90),limits=limits) as client:
        for n in LEVELS:
            async def call(i):
                started=time.monotonic()
                try:
                    data=dict(BODY)
                    data["messages"]=[{"role":"user","content":f"Return true for request {n}-{i}."}]
                    r=await client.post("/chat/completions",json=data)
                    good=r.status_code==200 and r.json().get("model")==MODEL
                    return {"ok":good,"status":r.status_code,
                            "latency":time.monotonic()-started,
                            "rate_limit":r.headers.get("x-ratelimit-limit-requests")}
                except (httpx.HTTPError, ValueError):
                    return {"ok":False,"status":"transport","latency":time.monotonic()-started}
            tick=time.monotonic()
            records=await asyncio.gather(*(call(i) for i in range(n)))
            wall=max(time.monotonic()-tick,0.01)
            good=sum(r["ok"] for r in records)
            err=sum(not r["ok"] for r in records)
            row={"level":n,"requests":n,"successful":good,"errors":err,
                 "http429":sum(r["status"]==429 for r in records),
                 "http5xx":sum(isinstance(r["status"],int) and 500<=r["status"]<600 for r in records),
                 "transport_errors":sum(r["status"]=="transport" for r in records),
                 "wall_seconds":round(wall,2),
                 "successful_responses_per_min":round(60*good/wall,2),
                 "mean_latency_s":round(statistics.mean(r["latency"] for r in records),2),
                 "reported_rpm_limit":next((r["rate_limit"] for r in records if r.get("rate_limit")),None)}
            results.append(row)
            print("capacity",n,"ok",good,"error",err,"rpm",row["successful_responses_per_min"],flush=True)
            await asyncio.sleep(1)
    stable=None
    for row in results:
        if row["errors"]==0 and (stable is None or row["successful_responses_per_min"]>=stable["successful_responses_per_min"]*0.90):
            stable=row
    if stable is None:
        raise SystemExit("BLOCKED: no stable Nano capacity level")
    n=stable["level"]
    arms=1 if n<18 else 2 if n<27 else 3
    report={"model":MODEL,"results":results,"highest_stable_measured_concurrency":n,
            "initial_seed_workers":9,"per_arm_internal_concurrency":1,
            "parallel_arms_per_seed_worker":arms,"estimated_global_requests_in_flight":9*arms,
            "note":"9 independent seed workers; scientific workloads unchanged"}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2))
    with open(os.environ.get("GITHUB_OUTPUT","/dev/null"),"a") as f:
        f.write(f"arms={arms}\\nper_arm=1\\n")
    print("ADMISSION: workers 9, arms per worker",arms,"total initial API in flight",9*arms,flush=True)
if __name__=="__main__":asyncio.run(main())
