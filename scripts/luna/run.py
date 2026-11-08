#!/usr/bin/env python3
"""Nine task/seed jobs extending the valid none/temperature-one experiment."""
import argparse, asyncio, collections, fcntl, hashlib, json, os, pathlib, subprocess, sys, time
from common import ROOT, OUT, REF, SEEDS, TASKS, POLICIES, SETTINGS, ENV_DEFAULT, batch, cell_dir, completed, config, expected, rows, verify_sources, write
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values
import httpx
PYTHON = str(ROOT/'.venv/bin/python')

def prepare():
    verify_sources()
    if (ROOT/'outputs/gpt-6-luna').exists():
        raise RuntimeError('Cancelled run directory exists; refusing any experiment execution.')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'batch.json').exists():
        previous=json.loads((OUT/'batch.json').read_text())
        for key in ['model','temperature','reasoning_effort','enable_thinking','seed_ids','reference_archive_sha256']:
            assert previous[key] == batch()[key], f'Foreign batch metadata: {key}'
    else:
        write(OUT/'batch.json',batch())
    for task in TASKS:
        for seed in SEEDS:
            for policy in POLICIES:
                d=cell_dir(task,seed,policy); d.mkdir(parents=True,exist_ok=True)
                c=config(task,seed,policy)
                if (d/'config.json').exists():
                    assert json.loads((d/'config.json').read_text())==c, f'Configuration changed: {d}'
                else:write(d/'config.json',c)
    for task in TASKS:
        for seed in SEEDS:
            status=OUT/'jobs'/f'{task}_seed{seed}'/'status.json'
            if not status.exists():
                write(status,{**SETTINGS,'task':task,'seed':seed,'status':'PENDING_PREFLIGHT','completed_episodes':0,'expected_episodes':expected(task),'prepared_at':time.time()})
            elif json.loads(status.read_text()).get('status')=='PENDING_PREFLIGHT':
                pending=json.loads(status.read_text());pending['prepared_at']=pending.pop('started',pending.get('prepared_at',time.time()));write(status,pending)
    if not (OUT/'control.json').exists():
        write(OUT/'control.json',{'upstream_limit':24,'concurrency':9,'allowed':[str(cell_dir(t,s,p).relative_to(OUT)) for t in TASKS for s in SEEDS for p in POLICIES],**SETTINGS})

async def preflight(env_file):
    record={**(json.loads((OUT/'batch.json').read_text()) if (OUT/'batch.json').exists() else batch()),'checked_at':time.time(),'env_file':str(env_file),'status':'CHECKING','scientific_workers_launched':0}
    try:
        v=dotenv_values(env_file);base=v.get('OPENAI_BASE_URL','').rstrip('/');key=v.get('OPENAI_API_KEY')
        assert base and key, 'API endpoint or credential missing'
        assert v.get('OPENAI_MODEL','gpt-6-luna')=='gpt-6-luna', 'Environment overrides the required model'
        record['endpoint']=base
        assert base.startswith(('http://127.0.0.1:8765','http://localhost:8765')), 'Only the authorized local codex_api endpoint may be used'
        record.update(baseline_endpoint='https://api.openai.com/v1',same_proxy_as_baseline=False,
                      baseline_temperature=1.0,baseline_temperature_omitted=True,
                      temperature_forwarded=False,sampling_seed_forwarded=False,
                      sampling_seed_handling='Proxy strips sampling seed; independent local task RNG seeds and original request seed derivation are retained.',
                      endpoint_migration_explicitly_recorded=True)
        async with httpx.AsyncClient(base_url=base,headers={'Authorization':'Bearer '+key},timeout=30) as client:
            r=await client.get('/models');record['models_http_status']=r.status_code
            if r.status_code!=200:
                record.update(status='BLOCKED_AUTH' if r.status_code in [401,403] else 'BLOCKED_ENDPOINT',reason=f'Model discovery returned HTTP {r.status_code}')
                write(OUT/'preflight.json',record);return False
            assert any(m['id']=='gpt-6-luna' for m in r.json().get('data',[])), 'Required model not available'
            body={'model':'gpt-6-luna','temperature':1.0,'reasoning_effort':'none','seed':1234567,'max_completion_tokens':64,'messages':[{'role':'user','content':'Return {"ok":true}.'}],'response_format':{'type':'json_schema','json_schema':{'name':'preflight','strict':True,'schema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}}}}
            r=await client.post('/chat/completions',json=body);record['probe_http_status']=r.status_code
            assert r.status_code==200, f'Exact sampling request rejected: HTTP {r.status_code}'
            result=r.json();actual=result.get('model','');assert actual.startswith('gpt-6-luna'), 'Backend returned another model'
            audit=result.get('sampling_audit',{})
            assert audit.get('actual_model')==actual and audit.get('upstream_response_id'), 'Actual upstream model provenance missing'
            assert audit.get('effective_temperature')==1.0, 'Effective upstream temperature is not 1.0'
            assert audit.get('effective_reasoning_effort')=='none' and audit.get('reasoning_tokens')==0
            assert audit.get('temperature_forwarded') is False and audit.get('seed_forwarded') is False
            details=(result.get('usage') or {}).get('completion_tokens_details') or {}
            assert details.get('reasoning_tokens')==0, 'Zero reasoning tokens not confirmed by backend'
            assert json.loads(result['choices'][0]['message']['content'])['ok'] is True
            record.update(status='PASS',actual_model=actual,actual_request=body,effective_temperature=1.0,effective_temperature_evidence='Raw completed upstream Responses object reports temperature=1.0 after proxy strips the unsupported explicit field.',reasoning_tokens=0,sampling_seed_forwarded=False,backend_usage=result.get('usage'),sampling_audit=audit)
    except Exception as e:
        record.update(status='BLOCKED_PREFLIGHT',reason=f'{type(e).__name__}: {str(e)[:300]}')
    write(OUT/'preflight.json',record)
    return record['status']=='PASS'

async def command(d,env_file):
    args=[PYTHON,'-m','src.agent_system.cli','run','--config',str(d/'config.json'),'--env-file',str(env_file),'--output-root',str(d.parent),'--run-id',d.name,'--resume']
    pidfile=d/'worker.pid.json'
    def same_cell(previous):
        try:
            return 'src.agent_system.cli' in previous and all(previous[previous.index(flag)+1]==args[args.index(flag)+1] for flag in ['--config','--output-root','--run-id'])
        except (ValueError,IndexError):return False
    candidates=[]
    if pidfile.exists():candidates.append(int(json.loads(pidfile.read_text())['pid']))
    candidates.extend(int(p.name) for p in pathlib.Path('/proc').iterdir() if p.name.isdigit() and int(p.name) not in candidates)
    for pid in candidates:
        proc=pathlib.Path(f'/proc/{pid}')
        try:match=same_cell((proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
        except (OSError,UnicodeError):match=False
        if match:
            write(pidfile,{'pid':pid,'adopted':time.time(),**SETTINGS})
            print(f'Adopting corrected worker {pid}: {d.name}',flush=True)
            while proc.exists():
                try:
                    if (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):break
                except OSError:break
                await asyncio.sleep(2)
            return 0
    env=dict(os.environ,OPENAI_MODEL='gpt-6-luna',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LUNA_JOB=str(d.relative_to(OUT)),LUNA_CONTROL=str(OUT/'control.json'))
    with (d/'stdout.log').open('ab') as log:
        p=await asyncio.create_subprocess_exec(*args,cwd=ROOT,env=env,stdout=log,stderr=log)
        write(pidfile,{'pid':p.pid,'started':time.time(),**SETTINGS})
        return await p.wait()

async def smoke(env_file):
    async def one(task):
        d=OUT/'_smoke'/task;d.mkdir(parents=True,exist_ok=True)
        if completed(d):return True
        write(d/'config.json',config(task,7,'C1',True))
        rc=await command(d,env_file)
        good=rc==0 and len(completed(d))==1
        write(d/'status.json',{**SETTINGS,'status':'COMPLETE' if good else 'SMOKE_FAILED','scientific':False})
        return good
    return all(await asyncio.gather(*(one(task) for task in TASKS)))

async def run_cell(task,seed,policy,env_file,semaphore):
    from audit import audit_cell
    async with semaphore:
        d=cell_dir(task,seed,policy);target=len(config(task,seed,policy)['degrees'])
        jd=OUT/'jobs'/f'{task}_seed{seed}';old=json.loads((jd/'status.json').read_text())
        started=old.get('started',time.time())
        try:
            if len(completed(d))!=target:
                history=rows(d/'attempts.jsonl')
                for attempt in range(len(history)+1,4):
                    write(jd/'status.json',{**SETTINGS,'task':task,'seed':seed,'status':'RUNNING','started':started,'expected_episodes':expected(task),'updated':time.time()})
                    write(d/'status.json',{**SETTINGS,'task':task,'seed':seed,'policy':policy,'status':'RUNNING','started':time.time()})
                    rc=await command(d,env_file)
                    with (d/'attempts.jsonl').open('a') as f:f.write(json.dumps({'attempt':attempt,'returncode':rc,'completed_episodes':len(completed(d)),'time':time.time(),**SETTINGS})+'\n')
                    if len(completed(d))==target:break
                    if attempt<3:await asyncio.sleep(2**attempt)
            if len(completed(d))!=target:raise RuntimeError('Incomplete after bounded process retries')
            if not (d/'run_summary.json').exists():await command(d,env_file)
            audit_cell(task,seed,policy);state='COMPLETE';reason=None
        except Exception as e:
            state='QUARANTINED';reason=f'{type(e).__name__}: {str(e)[:300]}'
        write(d/'status.json',{**SETTINGS,'task':task,'seed':seed,'policy':policy,'status':state,'reason':reason,'finished':time.time(),'completed_episodes':len(completed(d)),'expected_episodes':target})
        leaf_states=[json.loads((cell_dir(task,seed,p)/'status.json').read_text())['status'] if (cell_dir(task,seed,p)/'status.json').exists() else 'PENDING' for p in POLICIES]
        if all(value in ['COMPLETE','QUARANTINED'] for value in leaf_states):
            write(jd/'status.json',{**SETTINGS,'task':task,'seed':seed,'status':'COMPLETE' if all(value=='COMPLETE' for value in leaf_states) else 'QUARANTINED','started':started,'finished':time.time(),'expected_episodes':expected(task),'completed_episodes':sum(len(completed(cell_dir(task,seed,p))) for p in POLICIES)})
        print(f'{task} seed {seed} {policy}: {state}',flush=True)
        return state=='COMPLETE'

def schedule():
    # Start one cell in each of the nine independent task/seed jobs, then
    # fill every released slot with another arm, giving TCAS priority.
    initial=[(task,seed,'C0') for task in TASKS for seed in SEEDS]
    later=[(task,seed,policy) for policy in ['C1','I1'] for task in ['tcas','physics','synthetic'] for seed in SEEDS]
    cells=initial+later
    # Adopt every live corrected worker before admitting new processes.
    live=set()
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if 'src.agent_system.cli' in args and '--config' in args:live.add(args[args.index('--config')+1])
        except (OSError,UnicodeError,IndexError):pass
    return sorted(cells,key=lambda cell:str(cell_dir(*cell)/'config.json') not in live)

class Counts:
    def __init__(self):self.positions={};self.counts=collections.Counter();self.errors=collections.Counter();self.latencies=[];self.upstream_latencies=[]
    def update(self):
        for kind in ['requests','responses','api_attempts','transport_errors']:
            for p in (OUT/'runs').glob(f'agents/*/*/{kind}.jsonl'):
                with p.open() as f:
                    f.seek(self.positions.get(str(p),0))
                    while True:
                        pos=f.tell();line=f.readline()
                        if not line:break
                        try:r=json.loads(line)
                        except ValueError:f.seek(pos);break
                        self.counts[(str(p.parent),kind)]+=1
                        if kind=='transport_errors':self.errors[r['kind']]+=1
                        if kind=='responses':
                            self.latencies.append(r.get('latency_s',0))
                            if (r.get('sampling_audit') or {}).get('upstream_latency_s') is not None:self.upstream_latencies.append(r['sampling_audit']['upstream_latency_s'])
                            if r.get('finish_reason')=='stop' and not r.get('refusal'):self.counts[(str(p.parent),'useful_responses')]+=1
                    self.positions[str(p)]=f.tell()
    def total(self,kind):return sum(n for (p,k),n in self.counts.items() if k==kind)

def proxy_health():
    import urllib.request
    return json.load(urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5))

def set_global_limit(control):
    n=control['upstream_limit']
    path=pathlib.Path('/data4/toby/codex_api/upstream_limit.txt');tmp=path.with_suffix('.tuning.tmp')
    tmp.write_text(str(n)+'\n');tmp.replace(path)
    write(OUT/'control.json',control)

def active_upstream_calls():
    active=0
    for path in OUT.glob('_proxy_slot_*.lock'):
        with path.open('a') as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(handle,fcntl.LOCK_UN)
            except BlockingIOError:
                active+=1
    return active

async def monitor():
    import statistics
    counts=Counts();counts.update();last_proxy_metrics=proxy_health().get('metrics',{});last_success=counts.total('useful_responses');last_errors=counts.errors.copy();window=time.time();last_print=0;levels=[1,2,4,6,8,10,12,14,16,20,24,32,48,64,96,108];tested=set();best=None;peak_calls=0;last_failure=max([r.get('updated',0) for r in rows(OUT/'capacity.jsonl') if sum(r.get('window_errors',{}).values()) or sum(r.get('proxy_window_errors',{}).values())] or [time.time()])
    batch_meta=json.loads((OUT/'batch.json').read_text());batch_started=batch_meta.get('science_started',batch_meta['created_at']);baseline_done=None
    while True:
        await asyncio.sleep(10);counts.update();now=time.time();jobs=[]
        for task in TASKS:
            for seed in SEEDS:
                jd=OUT/'jobs'/f'{task}_seed{seed}';sp=jd/'status.json';state=json.loads(sp.read_text()) if sp.exists() else {'status':'PENDING','started':now}
                cells=[cell_dir(task,seed,p) for p in POLICIES];api=sum(counts.counts[(str(d),'api_attempts')] for d in cells);success=sum(counts.counts[(str(d),'useful_responses')] for d in cells)
                errs=collections.Counter(r['kind'] for d in cells for r in rows(d/'transport_errors.jsonl'))
                jobs.append({**SETTINGS,'task':task,'seed':seed,'status':state['status'],'completed_episodes':sum(len(completed(d)) for d in cells),'expected_episodes':expected(task),'api_requests':api,'successful_requests':success,'transport_retries':sum(errs.values()),'http_429':errs['HTTP 429'],'http_5xx':sum(n for k,n in errs.items() if k.startswith('HTTP 5')),'timeouts':sum(n for k,n in errs.items() if 'Timeout' in k),'wall_seconds':now-state.get('started',now),'last_progress_timestamp':max([p.stat().st_mtime for d in cells for p in [d/'responses.jsonl',d/'episodes.jsonl'] if p.exists()] or [state.get('started',now)])})
        done=sum(j['completed_episodes'] for j in jobs)
        if baseline_done is None:baseline_done=done
        control=json.loads((OUT/'control.json').read_text());limit=control['upstream_limit'];health=proxy_health();proxy_metrics=health.get('metrics',{});proxy_delta={k:v-last_proxy_metrics.get(k,0) for k,v in proxy_metrics.items()};delta=counts.errors-last_errors;new_errors=sum(delta.values())+proxy_delta.get('upstream_429',0)+proxy_delta.get('upstream_5xx',0)+proxy_delta.get('upstream_transport_errors',0);elapsed=now-window;success=counts.total('useful_responses')-last_success;rpm=success*60/max(elapsed,1)
        active_calls=active_upstream_calls();peak_calls=max(peak_calls,proxy_metrics.get('active_upstream_requests',0))
        progress={**SETTINGS,'proxy_admitted_requests_including_backoff':health.get('active_requests'),'proxy_active_upstream_calls':proxy_metrics.get('active_upstream_requests',0),'proxy_effective_limit':health.get('upstream_limit'),'proxy_metrics':proxy_metrics,'proxy_window_errors':{k:v for k,v in proxy_delta.items() if k in ['upstream_429','upstream_5xx','upstream_transport_errors','upstream_timeouts']},'active_upstream_calls':active_calls,'peak_active_upstream_calls':peak_calls,'pending_logical_requests':max(0,counts.total('requests')-counts.total('responses')-active_calls),'host_load_average':os.getloadavg(),'updated':now,'jobs':jobs,'completed_episodes':done,'expected_episodes':171,'upstream_limit':limit,'active_scientific_workers':sum(json.loads(p.read_text()).get('status')=='RUNNING' for p in (OUT/'runs/agents').glob('*/*/status.json')),'responses_per_min':rpm,'api_requests':counts.total('api_attempts'),'transport_errors':dict(counts.errors),'wall_seconds':now-batch_started}
        write(OUT/'progress.json',progress)
        if new_errors>=6 or elapsed>=300:
            record={**progress,'window_seconds':elapsed,'successful_responses':success,'window_errors':dict(delta),'latency_mean_s':statistics.mean(counts.latencies) if counts.latencies else None,'latency_median_s':statistics.median(counts.latencies) if counts.latencies else None,'upstream_latency_mean_s':statistics.mean(counts.upstream_latencies) if counts.upstream_latencies else None,'upstream_latency_median_s':statistics.median(counts.upstream_latencies) if counts.upstream_latencies else None}
            added=done-baseline_done;record['observed_episodes_per_hour']=added*3600/max(now-batch_started,1);record['episode_eta_seconds']=(171-done)*(now-batch_started)/added if added>0 else None
            with (OUT/'capacity.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            unstable=(new_errors>=6 and new_errors/max(1,success+new_errors)>.03) or delta['HTTP 429']>=3 or proxy_delta.get('upstream_429',0)>=3
            tested.add(limit)
            if unstable:
                control['upstream_limit']=best[0] if best and best[0]<limit else levels[max(0,levels.index(limit)-1)];best=None;last_failure=now
            elif elapsed>=300:
                if best and rpm<best[1]*.9 and limit>best[0]:control['upstream_limit']=best[0]
                else:
                    best=(limit,rpm)
                    if limit<108 and now-last_failure>=600 and (now-batch_started<1800 or (record['episode_eta_seconds'] or 0)>39600-(now-batch_started)):
                        control['upstream_limit']=levels[levels.index(limit)+1]
            set_global_limit(control);last_proxy_metrics=proxy_metrics;last_success=counts.total('useful_responses');last_errors=counts.errors.copy();window=now;counts.latencies=[];counts.upstream_latencies=[];peak_calls=0
        if now-last_print>=300:
            print(f'Progress {done}/171; {progress["active_scientific_workers"]}/9 workers; upstream {limit}; {rpm:.1f} responses/min; errors {dict(delta)}',flush=True);last_print=now

async def main(args):
    prepare()
    lock=(OUT/'coordinator.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A corrected coordinator is already active.')
    write(OUT/'supervisor.pid.json',{'pid':os.getpid(),'started':time.time(),**SETTINGS})
    last_status=None;last_notice=0
    while not await preflight(args.env_file):
        status=json.loads((OUT/'preflight.json').read_text())
        if status['status']!=last_status or time.time()-last_notice>=600:
            print(f'Preflight {status["status"]}: {status.get("reason")}',flush=True);last_status=status['status'];last_notice=time.time()
        if not args.watch_credentials:raise SystemExit(2)
        await asyncio.sleep(60)
    if args.preflight_only:return
    if not await smoke(args.env_file):raise RuntimeError('Smoke validation failed; scientific jobs not launched.')
    meta=json.loads((OUT/'batch.json').read_text())
    if 'science_started' not in meta:
        meta['science_started']=time.time();write(OUT/'batch.json',meta)
    m=asyncio.create_task(monitor())
    semaphore=asyncio.Semaphore(9)
    try:states=await asyncio.gather(*(run_cell(t,s,p,args.env_file,semaphore) for t,s,p in schedule()))
    finally:m.cancel()
    for task in TASKS:
        for seed in SEEDS:
            statuses=[json.loads((cell_dir(task,seed,p)/'status.json').read_text())['status'] for p in POLICIES]
            write(OUT/'jobs'/f'{task}_seed{seed}'/'status.json',{**SETTINGS,'task':task,'seed':seed,'status':'COMPLETE' if all(s=='COMPLETE' for s in statuses) else 'QUARANTINED','completed_episodes':sum(len(completed(cell_dir(task,seed,p))) for p in POLICIES),'expected_episodes':expected(task),'finished':time.time()})
    if not all(states):
        write(OUT/'final_status.json',{**SETTINGS,'status':'QUARANTINED','reason':'One or more incomplete jobs; upload prohibited.'});return
    import audit,importlib
    importlib.reload(audit)
    await asyncio.to_thread(audit.finish)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--env-file',type=pathlib.Path,default=pathlib.Path(ENV_DEFAULT));parser.add_argument('--watch-credentials',action='store_true');parser.add_argument('--preflight-only',action='store_true')
    asyncio.run(main(parser.parse_args()))
