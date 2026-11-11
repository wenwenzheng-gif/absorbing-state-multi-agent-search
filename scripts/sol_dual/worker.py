"""One exclusively allocated experiment, durable journal, <=$4 conservative cost cap."""
import asyncio,base64,contextvars,hashlib,json,os,pathlib,sqlite3,sys,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
NAME=sys.argv[1]; OUT=ROOT/'artifacts/channel-b'/NAME;OUT.mkdir(parents=True,exist_ok=True)
KEY=os.environ['OPENAI_API_KEY']; RUN=os.environ['GITHUB_RUN_ID']; REPO=os.environ['GITHUB_REPOSITORY']
def write(name,v):
 p=OUT/name;t=p.with_suffix('.tmp');text=json.dumps(v,indent=2);assert KEY not in text;t.write_text(text);t.replace(p)
def gh(path,body=None,method=None):
 req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=json.dumps(body).encode() if body else None,method=method,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=30) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,{}
def publish_status(payload):
 path='/contents/sol_channel_status.json';branch='sol-medium-claims/'+NAME
 status,existing=gh(path+'?ref='+branch)
 body={'message':'Update experiment heartbeat '+NAME,'branch':branch,'content':base64.b64encode(json.dumps(payload).encode()).decode()}
 if status==200:body['sha']=existing['sha']
 elif status!=404:return
 gh(path,body,method='PUT')
def main():
 allocation=json.loads((ROOT/'scripts/sol_dual/allocation.json').read_text()); assert NAME in allocation['jobs'] and allocation['owners'][NAME]=='B'
 assert os.environ.get('GITHUB_RUN_ATTEMPT')=='1','Rerun requires explicit recovery; no automatic duplicated calls'
 # GitHub's ref creation is atomic. An existing claim blocks every new execution.
 status,_=gh('/git/refs',{'ref':'refs/heads/sol-medium-claims/'+NAME,'sha':os.environ['GITHUB_SHA']})
 if status!=201:
  write('status.json',{'state':'HELD','reason':'Exclusive GitHub claim not acquired','http_status':status});return 1
 write('lease.json',{'channel':'B','experiment_id':NAME,'owner_run':RUN,'heartbeat':time.time(),'auto_reassign':False})
 import httpx
 from src.agent_system.cli import main as cli
 from src.agent_system.policies.openai_policy import LLMBackend
 from src.agent_system.schemas import PolicyFailure,BudgetPaused
 config=json.loads((ROOT/f'scripts/sol_dual/configs/{NAME}.json').read_text())
 expected=allocation['config_hashes'][NAME];assert hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected
 config['output_root']=str(OUT.parent);write('config.json',config)
 write('provenance.json',{'channel':'B','experiment_id':NAME,'config_sha256_before_runtime_paths':expected,'requested_model':'gpt-6-sol','reasoning_effort':'medium','prompt_hashes':allocation['prompt_hashes'],'workflow_run':RUN,'git_sha':os.environ['GITHUB_SHA'],'remote_sampling_seed_forwarded':False,'per_experiment_cost_cap_usd':4.0})
 for path,digest in allocation['prompt_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
 db=sqlite3.connect(OUT/'raw.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY,state TEXT,request TEXT,response TEXT,headers TEXT,started REAL,finished REAL,reserved REAL,actual_estimate REAL)');db.commit()
 original_complete=LLMBackend.complete;original_post=httpx.AsyncClient.post;original_close=LLMBackend.aclose;cv=contextvars.ContextVar('key');tasks={};pace=asyncio.Lock();last=0.;rps=.5;blocked_until=0.;started=time.time();requests=0;last_publish=0.
 async def complete(self,**kw):
  token=cv.set(kw['cache_key'])
  try:return await original_complete(self,**kw)
  finally:cv.reset(token)
 async def posted(self,url,**kw):
  nonlocal last,rps,blocked_until,requests,last_publish
  key=cv.get();body=dict(kw['json']);body.pop('seed',None);assert body['model']=='gpt-6-sol' and body.get('reasoning_effort')=='medium' and 'temperature' not in body
  serialized=json.dumps(body,sort_keys=True);old=db.execute('SELECT state,request,response FROM calls WHERE key=?',(key,)).fetchone()
  if old:
   assert old[1]==serialized
   if old[0]=='DONE':return httpx.Response(200,content=old[2],request=httpx.Request('POST',str(url)))
   if old[0]!='RETRY_SAFE':raise PolicyFailure('Uncertain request held; no resend')
  async with pace:
   await asyncio.sleep(max(0,last+1/rps-time.time(),blocked_until-time.time()));last=time.time()
   spent=db.execute('SELECT coalesce(sum(reserved),0) FROM calls').fetchone()[0]
   reserve=len(serialized.encode())*2.5/1e6+body['max_completion_tokens']*10/1e6
   if spent+reserve>4.0 or time.time()-started>4.5*3600:raise BudgetPaused('Operational cost/time ceiling; checkpoint retained')
   db.execute('INSERT OR REPLACE INTO calls(key,state,request,started,reserved,actual_estimate) VALUES(?,?,?,?,?,?)',(key,'INFLIGHT',serialized,time.time(),reserve,0));db.commit()
  try:response=await original_post(self,url,json=body)
  except httpx.ConnectError:
   db.execute("UPDATE calls SET state='RETRY_SAFE',reserved=0 WHERE key=?",(key,));db.commit();raise
  except Exception as e:
   db.execute("UPDATE calls SET state='AMBIGUOUS',response=?,finished=? WHERE key=?",(json.dumps({'error_type':type(e).__name__}),time.time(),key));db.commit();raise PolicyFailure('Ambiguous transport; held') from e
  headers={k:v for k,v in response.headers.items() if k.startswith('x-ratelimit-') or k in ('x-request-id','retry-after')};raw=response.text.replace(KEY,'<redacted>')
  try:p=json.loads(raw)
  except ValueError:p={}
  state='DONE' if response.status_code==200 else ('RETRY_SAFE' if response.status_code==429 else 'HELD')
  quota_block=response.status_code==429 and (p.get('error') or {}).get('code') in ('insufficient_quota','billing_hard_limit_reached','credit_balance_exhausted')
  if quota_block:state='HELD_QUOTA'
  if state=='DONE' and p.get('model')!='gpt-6-sol':state='HELD'
  u=p.get('usage',{});inp=u.get('prompt_tokens',0);output=u.get('completion_tokens',0);cached=u.get('prompt_tokens_details',{}).get('cached_tokens',0)
  upper=(inp*2.5+output*10)/1e6 if state=='DONE' else reserve
  actual=((inp-cached)*2+cached*.2+output*10)/1e6
  if response.status_code==429:
   upper=0;rps=max(.1,rps*.75)
   try:delay=float(response.headers.get('retry-after','30'))
   except ValueError:delay=30
   blocked_until=max(blocked_until,time.time()+delay)
  elif state=='DONE':
   requests+=1
   # Divide the advertised global quota across the two permitted matrix runners.
   try:
    rpm=float(headers['x-ratelimit-limit-requests']);tpm=float(headers['x-ratelimit-limit-tokens']);ceiling=min(2.,rpm*.7/120,tpm*.7/(120*max(inp+body['max_completion_tokens'],1)))
    if requests%30==0:rps=min(ceiling,rps+.1)
   except (KeyError,ValueError):pass
  db.execute('UPDATE calls SET state=?,response=?,headers=?,finished=?,reserved=?,actual_estimate=? WHERE key=?',(state,raw,json.dumps(headers),time.time(),upper,actual,key));db.commit()
  totals=db.execute('SELECT count(*),sum(reserved),sum(actual_estimate) FROM calls').fetchone()
  write('usage.json',{'recorded_requests':totals[0],'conservative_cost_ceiling_usd':totals[1],'standard_price_estimated_usd':totals[2],'configured_rps':rps,'elapsed_s':time.time()-started,'successful_rps':requests/(time.time()-started),'rate_headers':headers})
  write('lease.json',{'channel':'B','experiment_id':NAME,'owner_run':RUN,'heartbeat':time.time(),'auto_reassign':False})
  if time.time()-last_publish>=60:
   last_publish=time.time()
   await asyncio.to_thread(publish_status,{'experiment_id':NAME,'channel':'B','state':'RUNNING','owner_run':RUN,'heartbeat':time.time(),'successful_requests':requests,'cost_ceiling_usd':totals[1],'estimated_cost_usd':totals[2],'rps':requests/(time.time()-started),'configured_rps':rps,'auto_reassign':False})
  if state in ('HELD','HELD_QUOTA'):raise PolicyFailure('HTTP/model outcome held: '+str(response.status_code))
  return response
 async def durable_post(self,url,**kw):
  task=asyncio.create_task(posted(self,url,**kw));tasks.setdefault(id(self),set()).add(task);task.add_done_callback(tasks[id(self)].discard);return await asyncio.shield(task)
 async def close(self):
  while tasks.get(id(self._client)):await asyncio.gather(*list(tasks[id(self._client)]),return_exceptions=True)
  await original_close(self)
 LLMBackend.complete=complete;httpx.AsyncClient.post=durable_post;LLMBackend.aclose=close
 os.environ.update(OPENAI_MODEL='gpt-6-sol',OPENAI_BASE_URL='https://api.openai.com/v1')
 (OUT/'runtime.env').write_text('OPENAI_BASE_URL=https://api.openai.com/v1\nOPENAI_MODEL=gpt-6-sol\n')
 code=cli(['run','--env-file',str(OUT/'runtime.env'),'--config',str(OUT/'config.json'),'--run-id',NAME,'--output-root',str(OUT.parent),'--resume'])
 episodes=[json.loads(l) for l in (OUT/'episodes.jsonl').read_text().splitlines()];good=len(episodes)==1 and episodes[0]['status']=='ok'
 uncertain=db.execute("SELECT count(*) FROM calls WHERE state IN ('INFLIGHT','AMBIGUOUS','HELD')").fetchone()[0]
 replay=None
 if good and not uncertain:replay=cli(['replay','--run-dir',str(OUT),'--output-root',str(ROOT/'artifacts/replay')])
 state='COMPLETE' if good and not uncertain and replay==0 else 'HELD'
 write('validation.json',{'passed':state=='COMPLETE','offline_replay_returncode':replay,'uncertain_requests':uncertain,'actual_model_checked':True,'prompt_hashes_checked':True})
 write('status.json',{'state':state,'experiment_id':NAME,'channel':'B','finished':time.time(),'native_exit_code':code,'owner_run':RUN})
 publish_status({'experiment_id':NAME,'channel':'B','state':state,'owner_run':RUN,'heartbeat':time.time(),'successful_requests':requests,'cost_ceiling_usd':db.execute('SELECT sum(reserved) FROM calls').fetchone()[0],'estimated_cost_usd':db.execute('SELECT sum(actual_estimate) FROM calls').fetchone()[0],'auto_reassign':False})
 db.close();return 0 if state=='COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
