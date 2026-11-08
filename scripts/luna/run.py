#!/usr/bin/env python3
"""Durable, isolated 18-job execution of the documented final protocols."""
import asyncio, collections, copy, json, os, pathlib, shutil, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/gpt-6-luna'
PYTHON=str(ROOT/'.venv/bin/python')
TASKS={'synthetic':'llm_toy','tcas':'llm_tcas','ode':'llm_physics'}
START=time.time()

def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)
def rows(p):
 if not p.exists(): return []
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def config(task,seed,smoke=False):
 c=json.loads((ROOT/f'configs/agents/{TASKS[task]}.json').read_text())
 c.update(name=f'luna_{task}_{seed}',task_seeds=[seed],output_root=str(OUT/task))
 c['policy']['llm'].update(model='gpt-6-luna',api_style='openai',reasoning_effort='medium',temperature=None,transport_retries=8)
 if smoke: c.update(degrees=[0], agents=2, initial_width=1)
 return c
async def command(args,log,env=None):
 # Adopt a live worker after a supervisor restart; never duplicate its run.
 pidfile=log.with_suffix('.pid.json')
 if pidfile.exists():
  oldpid=json.loads(pidfile.read_text())['pid']
  proc=pathlib.Path(f'/proc/{oldpid}')
  try: matches=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')==args
  except OSError: matches=False
  if matches:
   print('Adopting worker',oldpid,log,flush=True)
   while proc.exists():
    try:
     if (proc/'stat').read_text().split(') ',1)[1].startswith('Z'): break
    except OSError: break
    await asyncio.sleep(2)
   return 0
 with log.open('ab') as f:
  p=await asyncio.create_subprocess_exec(*args,cwd=ROOT,env=env,stdout=f,stderr=f)
  write(log.with_suffix('.pid.json'),{'pid':p.pid,'started':time.time()})
  return await p.wait()
async def run(task,seed,smoke=False):
 d=OUT/('_smoke' if smoke else task)/(task if smoke else f'seed_{seed}')
 d.mkdir(parents=True,exist_ok=True)
 if (d/'status.json').exists() and json.loads((d/'status.json').read_text())['status']=='COMPLETE': return True
 c=config(task,seed,smoke); write(d/'config.json',c)
 env=dict(os.environ,OPENAI_MODEL='gpt-6-luna',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LUNA_JOB=str(d.relative_to(OUT)),LUNA_CONTROL=str(OUT/'control.json'))
 start=time.time(); history=[]
 for attempt in range(1,4):
  write(d/'status.json',{'status':'RUNNING','attempt':attempt,'seed':seed,'task':task,'started':start,'reasoning_effort':'medium'})
  rc=await command([PYTHON,'-m','src.agent_system.cli','run','--config',str(d/'config.json'),'--env-file','/data4/toby/codex_api/.env','--output-root',str(d.parent),'--run-id',d.name,'--resume'],d/'stdout.log',env)
  records=rows(d/'episodes.jsonl'); successful=[r for r in records if r['status']=='ok']
  good=len(successful)==len(c['degrees'])
  history.append({'attempt':attempt,'returncode':rc,'ok_episodes':len(successful),'failed_episodes':[r for r in records if r['status']!='ok']})
  if (d/'run_summary.json').exists(): shutil.copyfile(d/'run_summary.json',d/f'attempt_{attempt}_summary.json')
  if good: break
  if attempt<3:
   # Preserve every failed trajectory while the native resume skips successful episodes.
   shutil.copyfile(d/'episodes.jsonl',d/f'attempt_{attempt}_episodes.jsonl') if (d/'episodes.jsonl').exists() else None
   (d/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in successful))
   await asyncio.sleep(2**attempt)
 write(d/'status.json',{'status':'COMPLETE' if good else 'FAILED_AFTER_RETRIES','seed':seed,'task':task,'started':start,'finished':time.time(),'wall_seconds':time.time()-start,'attempts':history,'reasoning_effort':'medium'})
 print(task,seed,'smoke' if smoke else 'experiment','COMPLETE' if good else 'FAILED_AFTER_RETRIES',flush=True)
 return good
async def upload(task):
 from huggingface_hub import HfApi, get_token
 status=OUT/task/'upload_status.json'
 if not get_token():
  write(status,{'status':'AUTHENTICATION_BLOCKER','reason':'No Hugging Face token available from environment or cached login.'}); return
 for attempt in range(3):
  try:
   # Explicit artifact allowlist. Never upload credentials or stdout logs.
   api=HfApi(); await asyncio.to_thread(api.upload_folder,repo_id='TobyYang7/Branching-Search-runs',repo_type='dataset',folder_path=str(OUT/task),path_in_repo=f'gpt-6-luna/{task}',allow_patterns=['*.json','*.jsonl','*.csv','*.md','*.png'],ignore_patterns=['*.pid.json','upload_status.json'],commit_message=f'gpt-6-luna {task}: six independent seeds')
   write(status,{'status':'UPLOADED','path':f'gpt-6-luna/{task}'}); return
  except Exception as e:
   write(status,{'status':'UPLOAD_FAILED','error_type':type(e).__name__,'attempt':attempt+1})
   await asyncio.sleep(2**(attempt+1))
async def aggregate(task):
 d=OUT/task; a=d/'aggregate'; a.mkdir(exist_ok=True)
 manifest=json.loads((d/'seed_0/manifest.json').read_text()); manifest['config']['task_seeds']=list(range(6)); write(a/'manifest.json',manifest)
 for ledger in ['episodes','events']:
  with (a/f'{ledger}.jsonl').open('wb') as dest:
   for seed in range(6):
    p=d/f'seed_{seed}/{ledger}.jsonl'
    if p.exists():
     with p.open('rb') as source: shutil.copyfileobj(source,dest)
 codes={}
 for mod in ['summarize','critical_degree']:
  codes[mod]=await command([PYTHON,'-m',f'src.agent_system.analysis.{mod}','--run-dir',str(a)],a/f'{mod}.log')
 states=[json.loads((d/f'seed_{s}/status.json').read_text()) for s in range(6)]
 write(d/'summary.json',{'task':task,'model':'gpt-6-luna','reasoning_effort':'medium','seeds':states,'analysis_exit_codes':codes})
 await upload(task)
async def family(task):
 passed=await run(task,0,True)
 if not passed:
  # Smoke failure does not silently omit seeds: bounded scientific attempts still
  # expose whether a failure was transient or task-specific.
  print('Smoke exhausted retries for',task,flush=True)
 await asyncio.gather(*(run(task,s) for s in range(6)))
 await aggregate(task)
async def monitor():
 levels=[18,12,9,6]; level=0; cursor=0; previous=0; streak=0
 while True:
  active=[]
  for task in TASKS:
   for s in range(6):
    p=OUT/task/f'seed_{s}/status.json'
    if p.exists() and json.loads(p.read_text()).get('status')=='RUNNING': active.append(f'{task}/seed_{s}')
  errors=sum(len(rows(p)) for p in OUT.glob('*/seed_*/transport_errors.jsonl'))
  delta=errors-previous; previous=errors
  streak=streak+1 if delta>=12 else 0
  if streak>=2 and level<3: level+=1; streak=0
  n=levels[level]; selected=active[:n]
  if len(active)<=n: selected=active
  from dotenv import dotenv_values
  proxy_limit=int(dotenv_values('/data4/toby/codex_api/.env').get('MAX_CONCURRENCY',4))
  write(OUT/'control.json',{'upstream_limit':proxy_limit,'concurrency':n,'allowed':selected,'transport_errors':errors,'updated':time.time()})
  if active: cursor=(cursor+n)%len(active)
  await asyncio.sleep(30)
async def main():
 from dotenv import dotenv_values
 import httpx
 v=dotenv_values('/data4/toby/codex_api/.env'); base=v['OPENAI_BASE_URL'].rstrip('/')
 async with httpx.AsyncClient() as client:
  health=await client.get(base.removesuffix('/v1')+'/health'); health.raise_for_status()
  r=await client.get(base+'/models',headers={'Authorization':'Bearer '+v['OPENAI_API_KEY']}); r.raise_for_status()
  assert any(x['id']=='gpt-6-luna' for x in r.json()['data']), 'Required model unavailable'
 write(OUT/'preflight.json',{'model':'gpt-6-luna','available':True,'health':health.status_code,'reasoning_effort':'medium','initial_concurrency':18,'internal_concurrency':12,'started':START,'pid':os.getpid()})
 m=asyncio.create_task(monitor())
 try: await asyncio.gather(*(family(t) for t in TASKS))
 finally: m.cancel()
 summaries={t:json.loads((OUT/t/'summary.json').read_text()) for t in TASKS}
 usage=collections.Counter()
 for p in OUT.glob('*/seed_*/attempt_*_summary.json'): usage.update(json.loads(p.read_text()).get('llm_usage',{}))
 write(OUT/'combined_summary.json',{'model':'gpt-6-luna','reasoning_effort':'medium','started':START,'finished':time.time(),'wall_seconds':time.time()-START,'tasks':summaries,'usage':dict(usage),'control':json.loads((OUT/'control.json').read_text()),'uploads':{t:json.loads((OUT/t/'upload_status.json').read_text()) for t in TASKS}})
 print('ALL 18 JOBS FINALIZED',flush=True)
if __name__=='__main__': asyncio.run(main())
