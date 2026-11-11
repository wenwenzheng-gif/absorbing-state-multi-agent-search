#!/usr/bin/env python3
"""Conservative live upstream pacing; does not restart or signal any worker."""
import pathlib,time,json,urllib.request,fcntl
from run import OUT,read,write
path=pathlib.Path('/data4/toby/codex_api/upstream_rps.txt')
lock=(OUT/'rate_control.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=None; clean=0; last_rate=None
while not (OUT/'completion.json').exists():
 try:
  with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=10) as f: h=json.load(f)
  m=h['metrics']; rate=float(path.read_text()); action='hold'; deltas={}
  if previous:
   deltas={k:m.get(k,0)-previous.get(k,0) for k in ('upstream_attempts','upstream_429','completed_responses','upstream_5xx')}
   n=deltas['upstream_attempts']; bad=deltas['upstream_429']
   if n>=20 and bad/n>0.02:
    rate=max(1.0,rate*0.9); clean=0; action='reduce by 10% for rate limits'
   elif n>=20 and bad==0 and deltas['upstream_5xx']==0:
    clean+=1
    if clean>=5 and read(OUT/'tuning_policy.json',{}).get('allow_upward_probes',True): rate=min(3.5,rate+0.1); clean=0; action='small probe after five clean minutes'
   else: clean=0
  ceiling=read(OUT/'tuning_policy.json',{}).get('max_rps',3.5)
  if rate>ceiling: rate=ceiling; action='respect measured stable ceiling'
  if action!='hold':
   tmp=path.with_suffix('.sol.tmp'); tmp.write_text(str(rate)+'\n'); tmp.replace(path)
  last_rate=rate; previous=m
  record={'time':time.time(),'rate_cap':rate,'action':action,'window_seconds':60,'deltas':deltas,'429_fraction':deltas.get('upstream_429',0)/max(1,deltas.get('upstream_attempts',0)),'successful_responses_per_second':deltas.get('completed_responses',0)/60}
  write(OUT/'rate_status.json',record)
  with (OUT/'rate_history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
 except Exception as e:
  with (OUT/'rate_errors.log').open('a') as f: f.write(str(time.time())+' '+repr(e)+'\n')
 time.sleep(60)
# Restore the initial shared setting only if no external operator changed it.
if last_rate is not None and float(path.read_text())==last_rate:
 tmp=path.with_suffix('.sol.tmp'); tmp.write_text((OUT/'proxy_rps_before.txt').read_text()); tmp.replace(path)
