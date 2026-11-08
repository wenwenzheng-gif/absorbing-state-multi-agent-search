#!/usr/bin/env python3
"""Capacity observations and live transport tuning; never touches scientific ledgers."""
import collections,datetime,fcntl,json,os,pathlib,signal,statistics,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/gpt-6-luna';META=OUT/'_infrastructure';PROXY=pathlib.Path('/data4/toby/codex_api');TASKS={'synthetic':9,'tcas':12,'ode':11}
def write(p,obj):
 t=p.with_suffix(p.suffix+'.tuning.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def log(obj):
 obj['time']=time.time()
 with (META/'capacity.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
 print(json.dumps(obj),flush=True)
def health():return json.load(urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5))
def paths(name):return sorted(OUT.glob(f'*/seed_*/{name}.jsonl'))
def positions():return {str(p):p.stat().st_size for name in ['responses','transport_errors'] for p in paths(name)}
def measure(seconds,limit):
 before=positions();offset=(PROXY/'server.log').stat().st_size;started=time.monotonic()
 while time.monotonic()-started<seconds:
  time.sleep(max(.01,min(5,seconds-(time.monotonic()-started))))
  if limit>4:
   with (PROXY/'server.log').open() as f:f.seek(offset);recent=f.read()
   error_count=0
   for error_path in paths('transport_errors'):
    with error_path.open() as f:
     f.seek(before.get(str(error_path),0));error_count+=sum(1 for line in f if line.strip())
   if recent.count('[upstream] 429')>=6 or error_count>=6:
    log({'kind':'early_rollback_trigger','limit':limit,'reason':'six rate-limit or transport failures in observation window'})
    break
 elapsed=time.monotonic()-started;lat=[];success=0;by_task=collections.Counter();errors=collections.Counter()
 for name in ['responses','transport_errors']:
  for p in paths(name):
   with p.open() as f:
    f.seek(before.get(str(p),0))
    for line in f:
     try:r=json.loads(line)
     except ValueError:continue
     if name=='responses':success+=1;by_task[p.parent.parent.name]+=1;lat.append(r.get('latency_s',0))
     else:errors[r['kind']]+=1
 with (PROXY/'server.log').open() as f:f.seek(offset);proxylog=f.read()
 pids=[int(p.read_text().strip()) for p in [META/'proxy.pid'] if p.exists()]
 resource=subprocess.check_output(['ps','-p',','.join(map(str,pids or [3100485])),'-o','pid=,pcpu=,rss=,nlwp='],text=True).strip()
 result={'kind':'window','upstream_limit':limit,'seconds':elapsed,'successful_responses':success,'successful_responses_by_task':dict(by_task),'responses_per_min':success*60/elapsed,'latency_mean_s':statistics.mean(lat) if lat else None,'latency_median_s':statistics.median(lat) if lat else None,'transport_errors':dict(errors),'proxy_429':proxylog.count('[upstream] 429'),'proxy_resources_pid_cpu_rss_kb_threads':resource,'progress':progress(),'active_workers':active_workers(),'admitted_workers':json.loads((OUT/'control.json').read_text()).get('concurrency')}
 log(result);return result
def active_workers():
 count=0
 for p in OUT.glob('*/seed_*/stdout.pid.json'):
  try:
   pid=json.loads(p.read_text())['pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'src.agent_system.cli' in cmd:count+=1
  except (OSError,ValueError,KeyError):pass
 return count
def progress():
 result={}
 for task,target in TASKS.items():
  result[task]={}
  for seed in range(6):
   d=OUT/task/f'seed_{seed}';p=d/'episodes.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
   result[task][str(seed)]={'ok':len({r['episode_id'] for r in rows if r['status']=='ok'}),'target':target,'status':json.loads((d/'status.json').read_text()).get('status')}
 return result
def setlimit(n):
 (PROXY/'upstream_limit.txt').write_text(str(n)+'\n')
 env=PROXY/'.env';s=env.read_text();import re
 s=re.sub(r'^MAX_CONCURRENCY=.*$',f'MAX_CONCURRENCY={n}',s,flags=re.M);env.write_text(s)
 c=json.loads((OUT/'control.json').read_text());c.update(upstream_limit=n,concurrency=18,allowed=[f'{t}/seed_{s}' for t in TASKS for s in range(6)],updated=time.time());write(OUT/'control.json',c)
 log({'kind':'set_limit','limit':n,'health':health()})
def install():
 subprocess.run([sys.executable,str(ROOT/"scripts/luna/install_proxy_capacity.py")],check=True)
 # Freeze only the coordinator; every scientific child continues unchanged.
 supervisor=int((OUT/'supervisor.pid').read_text());os.kill(supervisor,signal.SIGSTOP)
 c=json.loads((OUT/'control.json').read_text());c['upstream_limit']=0;write(OUT/'control.json',c)
 held=[]
 try:
  for i in range(4):
   f=(OUT/f'_proxy_slot_{i}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX);held.append(f)
  # All four shared slots drained: no experimental API request is interrupted.
  os.kill(3100485,signal.SIGTERM)
  for _ in range(100):
   try:health()
   except Exception:break
   time.sleep(.1)
  with (PROXY/'server.log').open('ab') as f:
   p=subprocess.Popen(['python3','server.py'],cwd=PROXY,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
  (META/'proxy.pid').write_text(str(p.pid))
  for _ in range(100):
   try:health();break
   except Exception:time.sleep(.1)
  setlimit(12)
 finally:
  for f in held:fcntl.flock(f,fcntl.LOCK_UN);f.close()
 # Restart coordinator alone, allowing the documented PID adoption of all live workers.
 os.kill(supervisor,signal.SIGKILL)
 with (OUT/'supervisor.log').open('ab') as f:
  p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','scripts/luna/run.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 (OUT/'supervisor.pid').write_text(str(p.pid));log({'kind':'installed','supervisor_pid':p.pid,'proxy':health()})
def stable(r):return r['proxy_429']<=2 and sum(r['transport_errors'].values())<=max(2,r['successful_responses']*.02)
def capacity_test():
 if 'upstream_limit' in health():
  raise SystemExit('Adaptive proxy already installed; use --monitor to preserve active requests.')
 META.mkdir(exist_ok=True);log({'kind':'baseline_start','progress':progress(),'active_workers':active_workers(),'admitted_workers':json.loads((OUT/'control.json').read_text()).get('concurrency')});base=measure(180,4);install();best=None;maximum=12
 for n in [12,16,24]:
  if n!=12:setlimit(n)
  maximum=n;r=measure(240,n)
  if not stable(r):
   setlimit(best['upstream_limit'] if best else 4);break
  if best and r['responses_per_min']<best['responses_per_min']*.95:
   setlimit(best['upstream_limit']);break
  best=r
 log({'kind':'capacity_result','baseline':base,'best':best,'maximum_tested':maximum})
def monitor():
 initial=progress();started=time.time();initial_ok=sum(v['ok'] for t in initial.values() for v in t.values())
 previous=None;tested=set()
 for line in (META/'capacity.jsonl').read_text().splitlines():
  item=json.loads(line)
  if item.get('kind')=='window':tested.add(item['upstream_limit'])
 deadline=1794154935.6253016+86400
 while True:
  n=health()['upstream_limit'];r=measure(300,n)
  done=sum(v['ok'] for t in r['progress'].values() for v in t.values())
  added=done-initial_ok
  episode_eta=(192-done)*(time.time()-started)/added if added>0 else None
  request_bound=sum((v['target']-v['ok'])*(840 if t=='tcas' else 280) for t,jobs in r['progress'].items() for v in jobs.values())
  request_eta=request_bound/max(r['responses_per_min'],.01)*60
  log({'kind':'eta','completed_episodes':done,'remaining_episodes':192-done,'episode_eta_seconds':episode_eta,'request_upper_bound_eta_seconds':request_eta,'deadline_seconds_remaining':deadline-time.time()})
  if not stable(r) and n>4:
   setlimit({48:32,32:24,24:16,16:12,12:4,8:4}.get(n,4));previous=None
  elif previous and previous['upstream_limit']<n and r['responses_per_min']<previous['responses_per_min']*.95:
   setlimit(previous['upstream_limit']);previous=None
  elif request_eta>deadline-time.time():
   levels=[4,8,12,16,24,32,48]
   candidates=[level for level in levels if level>n and level not in tested]
   if candidates:
    next_limit=candidates[0];previous=r;tested.add(next_limit);setlimit(next_limit)
  if all(v['status'] in ['COMPLETE','FAILED_AFTER_RETRIES'] for t in r['progress'].values() for v in t.values()):break
 log({'kind':'finished','progress':progress(),'active_workers':active_workers(),'admitted_workers':json.loads((OUT/'control.json').read_text()).get('concurrency')})
if __name__=='__main__':
 if '--monitor' not in sys.argv:capacity_test()
 monitor()
