#!/usr/bin/env python3
"""Durable execution wrapper; scientific source and prompts stay unchanged."""
import asyncio, collections, contextvars, contextlib, fcntl, hashlib, json, os, pathlib, sqlite3, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/gpt-6-sol-medium-12seeds'
PYTHON=ROOT/'.venv-sol/bin/python'
SCRIPT=pathlib.Path(__file__).resolve()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(v,indent=2)); t.replace(p)
@contextlib.contextmanager
def allocation_lock():
 with (OUT/'allocation.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try: yield
  finally: fcntl.flock(f,fcntl.LOCK_UN)
def read(p,default=None):
 return json.loads(p.read_text()) if p.exists() else default
def rows(p):
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def prepare():
 from src.agent_system.cli import read_config
 from src.agent_system.driver import estimate_requests
 OUT.mkdir(parents=True,exist_ok=True)
 jobs=[]; estimates=collections.Counter()
 for i in range(12):
  for degree_index in range(7):
   for task in ('synthetic','tcas','physics'):
    for policy in ('C0','C1','I1'):
     source=ROOT/f'configs/agents/gpt-6-luna_seed1/{task}_{policy}.json'
     c=read(source)
     if degree_index>=len(c['degrees']): continue
     degree=c['degrees'][degree_index]; seed=c['task_seeds'][0]+i
     name=f'{task}_{policy}_s{seed}_d{degree}'
     c.update(name=name,task_seeds=[seed],degrees=[degree],output_root=str(OUT/'jobs'))
     c['policy']['llm'].update(model='gpt-6-sol',reasoning_effort='medium',temperature=None,api_style='openai',timeout_s=1800.0)
     d=OUT/'jobs'/name; d.mkdir(parents=True,exist_ok=True)
     if not (d/'config.json').exists(): write(d/'config.json',c)
     cfg=read_config(d/'config.json'); estimates.update({k:v for k,v in estimate_requests(cfg).items() if isinstance(v,(int,float))})
     jobs.append(name)
 manifest={'reference_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model':'gpt-6-sol','reasoning_effort':'medium','jobs':jobs,'total':len(jobs),'estimates':dict(estimates),'prompt_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/agent_system/policies/prompts').glob('*.txt')},'seed_ranges':{'synthetic':[87000,87011],'tcas':[87200,87211],'physics':[82240,82251]},'source_configs':'configs/agents/gpt-6-luna_seed1','transport_changes':['1800 second read timeout','durable per-request SQLite journal','retry only explicit 429 or pre-connect failure','global proxy concurrency gate'],'created':time.time()}
 if not (OUT/'batch.json').exists(): write(OUT/'batch.json',manifest)
 print(json.dumps({'jobs':len(jobs),'estimates':dict(estimates)}))

def worker(name):
 import httpx
 from src.agent_system.policies.openai_policy import LLMBackend
 from src.agent_system.schemas import PolicyFailure
 from src.agent_system.cli import main
 d=OUT/'jobs'/name
 assert read(OUT/'allocation.json',{}).get('owners',{}).get(name,'A')=='A', 'Task is exclusively allocated to another channel'
 lock=(d/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 db=sqlite3.connect(d/'raw.sqlite'); db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, state TEXT, request TEXT, response TEXT, started REAL, finished REAL, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0)'); db.commit()
 keyvar=contextvars.ContextVar('request_key'); original_complete=LLMBackend.complete; original_post=httpx.AsyncClient.post; original_close=LLMBackend.aclose; network_tasks=set()
 async def complete(self,**kw):
  token=keyvar.set(kw['cache_key'])
  try: return await original_complete(self,**kw)
  finally: keyvar.reset(token)
 async def post(self,url,**kw):
  key=keyvar.get(); body=kw['json']; assert body['model']=='gpt-6-sol' and body['reasoning_effort']=='medium' and 'temperature' not in body
  serialized=json.dumps(body,sort_keys=True)
  old=db.execute('SELECT state,request,response FROM calls WHERE key=?',(key,)).fetchone()
  if old:
   assert old[1]==serialized, 'Replay request body changed'
   if old[0]=='DONE': return httpx.Response(200,content=old[2],request=httpx.Request('POST',str(url)))
   if old[0] not in ('RETRY_SAFE',): raise PolicyFailure('AMBIGUOUS_REQUEST: refusing to resend '+key)
  # Pace starts across processes, preventing TCP backlog bursts at the local proxy.
  with (OUT/'starts.lock').open('a+') as pace:
   while True:
    try: fcntl.flock(pace,fcntl.LOCK_EX|fcntl.LOCK_NB); break
    except BlockingIOError: await asyncio.sleep(0.02)
   pace.seek(0); last=float(pace.read() or 0)
   await asyncio.sleep(max(0,0.25-(time.time()-last)))
   pace.seek(0); pace.truncate(); pace.write(str(time.time())); pace.flush()
   fcntl.flock(pace,fcntl.LOCK_UN)
  db.execute('INSERT OR REPLACE INTO calls(key,state,request,started) VALUES(?,?,?,?)',(key,'INFLIGHT',serialized,time.time())); db.commit()
  try: response=await original_post(self,url,**kw)
  except httpx.ConnectError:
   db.execute('UPDATE calls SET state=? WHERE key=?',('RETRY_SAFE',key)); db.commit(); raise
  except Exception as e:
   db.execute('UPDATE calls SET state=?,response=?,finished=? WHERE key=?',('AMBIGUOUS',json.dumps({'error':type(e).__name__}),time.time(),key)); db.commit()
   raise PolicyFailure('AMBIGUOUS_REQUEST: '+type(e).__name__+' '+key) from e
  state='DONE' if response.status_code==200 else ('RETRY_SAFE' if response.status_code==429 else 'HELD')
  try: usage=response.json().get('usage',{})
  except ValueError: usage={}; state='HELD'
  db.execute('UPDATE calls SET state=?,response=?,finished=?,input_tokens=?,output_tokens=? WHERE key=?',(state,response.text,time.time(),usage.get('prompt_tokens',0),usage.get('completion_tokens',0),key)); db.commit()
  if state=='HELD': raise PolicyFailure('HELD_HTTP_'+str(response.status_code)+' '+key)
  return response
 async def durable_post(self,url,**kw):
  task=asyncio.create_task(post(self,url,**kw)); network_tasks.add(task)
  task.add_done_callback(network_tasks.discard)
  return await asyncio.shield(task)
 async def aclose(self):
  while network_tasks: await asyncio.gather(*list(network_tasks),return_exceptions=True)
  await original_close(self)
 LLMBackend.complete=complete; LLMBackend.aclose=aclose; httpx.AsyncClient.post=durable_post
 os.environ.update(OPENAI_MODEL='gpt-6-sol',LUNA_CONTROL=str(OUT/'control.json'),LUNA_JOB='jobs/'+name)
 result=main(['run','--config',str(d/'config.json'),'--run-id',name,'--output-root',str(OUT/'jobs'),'--env-file','/data4/toby/codex_api/.env','--resume'])
 db.close(); return result

async def compatibility():
 import httpx
 from dotenv import dotenv_values
 v=dotenv_values('/data4/toby/codex_api/.env'); base=v['OPENAI_BASE_URL'].rstrip('/')
 if (OUT/'compatibility.json').exists(): print('Existing compatibility result retained'); return
 results=[]
 async with httpx.AsyncClient(timeout=1800) as client:
  health=(await client.get(base.removesuffix('/v1')+'/health')).json(); write(OUT/'proxy_baseline.json',health)
  async def one(i):
   body={'model':'gpt-6-sol','reasoning_effort':'medium','max_completion_tokens':3072,'messages':[{'role':'user','content':f'Return JSON only: {{"test": {i}, "answer": {17+i} }}. Copy the two integers exactly.'}],'response_format':{'type':'json_schema','json_schema':{'name':'compatibility','strict':True,'schema':{'type':'object','properties':{'test':{'type':'integer'},'answer':{'type':'integer'}},'required':['test','answer'],'additionalProperties':False}}}}
   p=OUT/'compatibility'/f'{i}.json'
   if p.exists(): return read(p)
   write(p,{'state':'INFLIGHT','request':body}); start=time.time()
   r=await client.post(base+'/chat/completions',headers={'Authorization':'Bearer '+v['OPENAI_API_KEY']},json=body)
   data=r.json(); obj={'state':'DONE','request':body,'status':r.status_code,'response':data,'seconds':time.time()-start}; write(p,obj)
   assert r.status_code==200 and data['choices'][0]['finish_reason']=='stop',obj
   assert json.loads(data['choices'][0]['message']['content'])=={'test':i,'answer':17+i}
   return obj
  start=time.time(); results.append(await one(0)); results+=await asyncio.gather(*(one(i) for i in range(1,13)))
  report={'passed':True,'requests':len(results),'seconds':time.time()-start,'parallel_requests':12,'parallel_max_latency':max(x['seconds'] for x in results[1:]),'proxy':health}
  report['usage']={k:sum(x['response']['usage'].get(k,0) for x in results) for k in ('prompt_tokens','completion_tokens','total_tokens')}
  write(OUT/'compatibility.json',report); print(json.dumps(report),flush=True)

def snapshot(active):
 batch=read(OUT/'batch.json'); counts=collections.Counter(); usage=collections.Counter(); tasks={}; errors=[]
 for name in batch['jobs']:
  d=OUT/'jobs'/name; state=read(d/'status.json',{'state':'PENDING'})['state']; counts[state]+=1
  task=name.split('_')[0]; tasks.setdefault(task,collections.Counter())[state]+=1
  p=d/'raw.sqlite'
  if p.exists():
   try:
    with sqlite3.connect(f'file:{p}?mode=ro',uri=True,timeout=1) as db:
     for st,n,inp,out in db.execute('SELECT state,count(*),sum(input_tokens),sum(output_tokens) FROM calls GROUP BY state'):
      usage[st]+=n; usage['input_tokens']+=inp or 0; usage['output_tokens']+=out or 0
   except sqlite3.OperationalError: pass
 start=read(OUT/'started.json')['time']; elapsed=time.time()-start
 done=counts['COMPLETE']; eta=elapsed*(batch['total']-done)/done if done else None
 result={'time':time.time(),'elapsed_seconds':elapsed,'total':batch['total'],'states':dict(counts),'tasks':{k:dict(v) for k,v in tasks.items()},'running_jobs':len(active),'usage':dict(usage),'eta_seconds_from_completed_experiments':eta,'pid':os.getpid()}
 write(OUT/'progress.json',result)
 with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
 print(json.dumps(result),flush=True)
 return result

async def supervise():
 import httpx
 lock=(OUT/'supervisor.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(OUT/'supervisor.pid.json',{'pid':os.getpid(),'time':time.time()})
 if not (OUT/'started.json').exists(): write(OUT/'started.json',{'time':time.time()})
 await compatibility()
 async with httpx.AsyncClient() as client: health=(await client.get('http://127.0.0.1:8765/health')).json()
 write(OUT/'control.json',read(OUT/'control.json',{'concurrency':18,'allowed':[],'upstream_limit':min(64,health['upstream_limit'])}))
 pending=[]; active={}
 for name in read(OUT/'batch.json')['jobs']:
  d=OUT/'jobs'/name; state=read(d/'status.json',{})
  if state.get('state') in ('COMPLETE','FAILED','HELD'): continue
  pid=state.get('pid'); proc=pathlib.Path(f'/proc/{pid}')
  if pid and proc.exists() and str(SCRIPT).encode() in (proc/'cmdline').read_bytes(): active[name]=(None,pid)
  else: pending.append(name)
 while pending or active:
  while pending and len(active)<read(OUT/'execution_limits.json',{}).get('workers',36):
   with allocation_lock():
    name=pending.pop(0); d=OUT/'jobs'/name
    if read(OUT/'allocation.json',{}).get('owners',{}).get(name,'A')!='A': continue
    # Archive prior trajectory ledgers; the raw request journal is retained for exact replay.
    if (d/'episodes.jsonl').exists() and any(x.get('status')=='ok' for x in rows(d/'episodes.jsonl')):
     write(d/'status.json',{'state':'COMPLETE','recovered':True}); continue
    for filename in ('episodes.jsonl','events.jsonl','requests.jsonl','responses.jsonl'):
     p=d/filename
     if p.exists(): p.rename(d/(filename+'.attempt-'+str(time.time_ns())))
    with (d/'stdout.log').open('ab') as log:
     p=subprocess.Popen([str(PYTHON),str(SCRIPT),'worker',name],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),start_new_session=True)
    active[name]=(p,p.pid); write(d/'status.json',{'state':'RUNNING','pid':p.pid,'started':time.time()})
  for lease_name,(_,lease_pid) in active.items():
   write(OUT/'jobs'/lease_name/'lease.json',{'channel':'A','pid':lease_pid,'heartbeat':time.time(),'expires':time.time()+90,'expiry_policy':'Never reassign without proving the prior worker stopped and reconciling raw requests.'})
  await asyncio.sleep(5)
  for name,(p,pid) in list(active.items()):
   alive=p.poll() is None if p else pathlib.Path(f'/proc/{pid}').exists()
   if alive: continue
   d=OUT/'jobs'/name; episodes=rows(d/'episodes.jsonl'); good=len(episodes)==1 and episodes[0].get('status')=='ok'
   state='COMPLETE' if good else 'FAILED'
   if (d/'raw.sqlite').exists():
    with sqlite3.connect(d/'raw.sqlite') as db:
     if db.execute("SELECT count(*) FROM calls WHERE state IN ('AMBIGUOUS','INFLIGHT','HELD')").fetchone()[0]: state='HELD'
   write(d/'status.json',{'state':state,'pid':pid,'finished':time.time(),'episodes':len(episodes),'error':episodes[-1].get('error') if episodes else 'No episode result'})
   del active[name]
  if not hasattr(supervise,'last') or time.time()-supervise.last>30:
   snapshot(active); supervise.last=time.time()
 result=snapshot(active); write(OUT/'completion.json',result)

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare': prepare()
 elif mode=='worker': sys.exit(worker(sys.argv[2]))
 elif mode=='compatibility': asyncio.run(compatibility())
 elif mode=='supervise': asyncio.run(supervise())
 elif mode=='status': print(json.dumps(read(OUT/'progress.json'),indent=2))
