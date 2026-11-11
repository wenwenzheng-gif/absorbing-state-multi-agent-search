#!/usr/bin/env python3
"""Bounded, live A-channel capacity test; never signals experimental workers."""
import time,json,pathlib,fcntl,urllib.request,statistics
from run import OUT,read,write
lock=(OUT/'a_capacity_probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
RPS=pathlib.Path('/data4/toby/codex_api/upstream_rps.txt')
def health():
 with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=10) as f:return json.load(f)
def set_limits(rate,slots):
 policy=read(OUT/'tuning_policy.json',{});policy.update(max_rps=rate,max_concurrency=slots,allow_upward_probes=False,reason='Bounded measured capacity experiment authorized 2026-11-11');write(OUT/'tuning_policy.json',policy)
 c=read(OUT/'control.json');c['upstream_limit']=slots;write(OUT/'control.json',c)
 t=RPS.with_suffix('.capacity.tmp');t.write_text(str(rate)+'\n');t.replace(RPS)
def resource():
 mem={line.split(':')[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
 return {'load_average':pathlib.Path('/proc/loadavg').read_text().strip(),'memory_available_kb':mem.get('MemAvailable'),'cpu_counters':pathlib.Path('/proc/stat').read_text().splitlines()[0],'network_counters':pathlib.Path('/proc/net/dev').read_text()}
records=[]; best=(3.1,64); best_rate=0
try:
 for rate,slots in [(3.1,64),(3.2,68),(3.3,72),(3.4,72),(3.5,76)]:
  set_limits(rate,slots);started=time.time();before=health(); v0=read(OUT/'validation.json',{}).get('passed',0);samples=[];bad=False
  for i in range(12):
   time.sleep(15);h=health();samples.append(h)
   m=h['metrics'];a=before['metrics'];attempts=m['upstream_attempts']-a['upstream_attempts']
   if m['upstream_429']>a['upstream_429'] or (attempts>=30 and (m.get('upstream_5xx',0)-a.get('upstream_5xx',0))/attempts>.01):bad=True;break
  elapsed=time.time()-started;after=health();a=before['metrics'];b=after['metrics'];delta={k:b.get(k,0)-a.get(k,0) for k in ('upstream_attempts','completed_responses','upstream_429','upstream_5xx')};effective=delta['completed_responses']/elapsed
  record={'time':time.time(),'configured_rps':rate,'slots':slots,'seconds':elapsed,'delta':delta,'successful_rps':effective,'validated_experiments':read(OUT/'validation.json',{}).get('passed',0)-v0,'active_upstream_mean':statistics.mean(h['metrics']['active_upstream_requests'] for h in samples),'active_proxy_mean':statistics.mean(h['active_requests'] for h in samples),'resource':resource(),'rate_limit_seen':bad}
  records.append(record);write(OUT/'diagnostics/channel_a_capacity.json',{'stages':records,'state':'RUNNING','best':best});print(json.dumps(record),flush=True)
  if bad:break
  if best_rate and effective<best_rate*1.01:break
  best=(rate,slots);best_rate=effective
finally:
 set_limits(*best);write(OUT/'diagnostics/channel_a_capacity.json',{'stages':records,'state':'FINISHED','selected_rps':best[0],'selected_slots':best[1],'successful_rps_at_selected':best_rate,'workers':36,'worker_expansion':'Only warranted if API slots cannot be kept fed; measured admission is already saturated.','finished':time.time()})
