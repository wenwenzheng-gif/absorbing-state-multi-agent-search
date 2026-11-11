#!/usr/bin/env python3
"""Adapt only execution pressure, record proxy metrics and workload estimates."""
import json,time,urllib.request,fcntl,collections,os
from run import OUT,read,write
lock=(OUT/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=None; clean=0
while not (OUT/'completion.json').exists():
 try:
  with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=10) as r: health=json.load(r)
  metrics=health['metrics']; control=read(OUT/'control.json'); limit=control['upstream_limit']; reason='stable'
  if previous:
   attempts=metrics['upstream_attempts']-previous['upstream_attempts']; limited=metrics['upstream_429']-previous['upstream_429']
   if attempts>=10 and limited/attempts>0.02:
    limit=max(24,int(limit*0.85)); clean=0; reason='reduce for rate limits'
   elif attempts>=10 and limited==0:
    clean+=1
    if clean>=3: limit=min(health['upstream_limit'],108,limit+4); clean=0; reason='probe higher concurrency after clean windows'
  previous=metrics
  policy=read(OUT/'tuning_policy.json',{}); limit=min(limit,policy.get('max_concurrency',108))
  control.update(upstream_limit=limit,updated=time.time(),reason=reason); write(OUT/'control.json',control)
  progress=read(OUT/'progress.json',{}); elapsed=progress.get('elapsed_seconds',0); done=progress.get('usage',{}).get('DONE',0)
  estimate=read(OUT/'batch.json')['estimates']['requests_total_upper_bound']
  rate=done/elapsed if elapsed else 0
  record={'time':time.time(),'health':health,'execution_control':control,'observed_responses_per_second':rate,'unpruned_remaining_seconds_at_observed_rate':(estimate-done)/rate if rate else None,'note':'Unpruned workload projection, not a completion forecast; repairs add calls, pruning removes calls.'}
  write(OUT/'throughput.json',record)
  with (OUT/'proxy_metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
 except Exception as e:
  with (OUT/'controller_errors.log').open('a') as f: f.write(str(time.time())+' '+repr(e)+'\n')
 time.sleep(30)
