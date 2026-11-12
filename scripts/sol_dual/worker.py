"""One exclusively allocated experiment, durable recovery journal, no internal spending caps."""
import asyncio,base64,contextvars,hashlib,json,os,pathlib,sqlite3,sys,time,urllib.request,urllib.error,shutil,subprocess,random,fcntl
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
 recovery=allocation['recovery'][NAME]
 # Old ownership is terminal; compare the exact old branch head before an atomic fast-forward transfer.
 status,old_run=gh('/actions/runs/'+str(recovery['old_run']))
 assert status==200 and old_run['status']=='completed','Prior owner is still active'
 branch='sol-medium-claims/'+NAME
 status,ref=gh('/git/ref/heads/'+branch);assert status==200 and ref['object']['sha']==recovery['claim_sha'],'Claim changed; reconcile instead of duplicating work'
 status,old_commit=gh('/git/commits/'+recovery['claim_sha']);assert status==200
 claim={'experiment_id':NAME,'channel':'B','state':'RECOVERING','owner_run':RUN,'prior_owner_run':recovery['old_run'],'heartbeat':time.time(),'auto_reassign':False}
 status,tree=gh('/git/trees',{'base_tree':old_commit['tree']['sha'],'tree':[{'path':'sol_channel_status.json','mode':'100644','type':'blob','content':json.dumps(claim)}]});assert status==201
 status,commit=gh('/git/commits',{'message':'Recover existing exclusive claim '+NAME,'tree':tree['sha'],'parents':[recovery['claim_sha']]});assert status==201
 status,_=gh('/git/refs/heads/'+branch,{'sha':commit['sha'],'force':False},method='PATCH');assert status==200,'Concurrent recovery won the claim'
 restored=ROOT/'recovered'/NAME;restored.mkdir(parents=True,exist_ok=True)
 subprocess.run(['gh','run','download',str(recovery['old_run']),'--repo',REPO,'--name',f"sol-channel-b-{NAME}-{recovery['old_run']}",'--dir',str(restored)],check=True)
 sources=list(restored.glob('**/'+NAME+'/raw.sqlite'));assert len(sources)==1
 source=sources[0].parent
 old_status=json.loads((source/'status.json').read_text());assert old_status['state']=='HELD'
 with sqlite3.connect('file:'+str(source/'raw.sqlite')+'?mode=ro',uri=True) as prior:
  assert prior.execute('PRAGMA quick_check').fetchone()[0]=='ok'
  signatures={key:hashlib.sha256((request+'\0'+response).encode()).hexdigest() for key,state,request,response in prior.execute('SELECT key,state,request,response FROM calls') if state=='DONE'}
  assert signatures==recovery['journal_signatures'],'Artifact differs from reconciled saved successes'
  assert prior.execute("SELECT count(*) FROM calls WHERE state!='DONE'").fetchone()[0]==0
 for item in source.iterdir():
  if item.is_file():shutil.copy2(item,OUT/item.name)
 archive=OUT/'prior_attempt';archive.mkdir(exist_ok=True)
 for filename in ('episodes.jsonl','events.jsonl','requests.jsonl','responses.jsonl','manifest.json','run_summary.json','status.json','validation.json','provenance.json','lease.json','usage.json'):
  path=OUT/filename
  if path.exists():path.rename(archive/filename)
 local_lock=(OUT/'worker.lock').open('a');fcntl.flock(local_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write('recovery.json',{'old_run':recovery['old_run'],'run':RUN,'preserved_successes':len(signatures),'prior_claim_sha':recovery['claim_sha'],'no_spending_caps':True})
 await_delay=recovery.get('launch_delay_seconds',0);time.sleep(await_delay)
 write('lease.json',{'channel':'B','experiment_id':NAME,'owner_run':RUN,'heartbeat':time.time(),'auto_reassign':False})
 import httpx
 from src.agent_system.cli import main as cli
 from src.agent_system.policies.openai_policy import LLMBackend
 from src.agent_system.schemas import PolicyFailure,BudgetPaused
 config=json.loads((ROOT/f'scripts/sol_dual/configs/{NAME}.json').read_text())
 expected=allocation['config_hashes'][NAME];assert hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected
 config['output_root']=str(OUT.parent);write('config.json',config)
 write('provenance.json',{'channel':'B','experiment_id':NAME,'config_sha256_before_runtime_paths':expected,'requested_model':'gpt-6-sol','reasoning_effort':'medium','prompt_hashes':allocation['prompt_hashes'],'workflow_run':RUN,'git_sha':os.environ['GITHUB_SHA'],'remote_sampling_seed_forwarded':False,'internal_spending_caps':None,'restored_successful_calls':recovery['saved_successes']})
 for path,digest in allocation['prompt_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
 db=sqlite3.connect(OUT/'raw.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY,state TEXT,request TEXT,response TEXT,headers TEXT,started REAL,finished REAL,reserved REAL,actual_estimate REAL,input_tokens INTEGER DEFAULT 0,output_tokens INTEGER DEFAULT 0)');db.commit()
 original_complete=LLMBackend.complete;original_post=httpx.AsyncClient.post;original_close=LLMBackend.aclose;cv=contextvars.ContextVar('key');tasks={};pace=asyncio.Lock();last=0.;rps=.5;blocked_until=0.;started=time.time();requests=0;last_publish=0.;rate_ceiling=.5;active_network=0;network_peak=0;limited=0;reused=0
 async def complete(self,**kw):
  token=cv.set(kw['cache_key'])
  try:return await original_complete(self,**kw)
  finally:cv.reset(token)
 async def posted(self,url,**kw):
  nonlocal last,rps,blocked_until,requests,last_publish,rate_ceiling,active_network,network_peak,limited,reused
  key=cv.get();body=dict(kw['json']);body.pop('seed',None);assert body['model']=='gpt-6-sol' and body.get('reasoning_effort')=='medium' and 'temperature' not in body
  serialized=json.dumps(body,sort_keys=True);old=db.execute('SELECT state,request,response FROM calls WHERE key=?',(key,)).fetchone()
  if old:
   assert old[1]==serialized
   if old[0]=='DONE':
    reused+=1;return httpx.Response(200,content=old[2],request=httpx.Request('POST',str(url)))
   if old[0]!='RETRY_SAFE':raise PolicyFailure('Uncertain request held; no resend')
  async with pace:
   await asyncio.sleep(max(0,last+1/rps-time.time(),blocked_until-time.time()));last=time.time()
   spent=db.execute('SELECT coalesce(sum(reserved),0) FROM calls').fetchone()[0]
   reserve=len(serialized.encode())*2.5/1e6+body['max_completion_tokens']*10/1e6
   db.execute('INSERT OR REPLACE INTO calls(key,state,request,started,reserved,actual_estimate) VALUES(?,?,?,?,?,?)',(key,'INFLIGHT',serialized,time.time(),reserve,0));db.commit()
  active_network+=1;network_peak=max(network_peak,active_network)
  try:response=await original_post(self,url,json=body)
  except httpx.ConnectError:
   active_network-=1;db.execute("UPDATE calls SET state='RETRY_SAFE',reserved=0 WHERE key=?",(key,));db.commit();raise
  except Exception as e:
   active_network-=1;db.execute("UPDATE calls SET state='AMBIGUOUS',response=?,finished=? WHERE key=?",(json.dumps({'error_type':type(e).__name__}),time.time(),key));db.commit();raise PolicyFailure('Ambiguous transport; held') from e
  active_network-=1
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
   upper=0;limited+=1;rps=max(.1,rps*.5)
   try:delay=float(response.headers.get('retry-after',str(min(60.,2.**min(limited,6))+random.random())))
   except ValueError:delay=30
   blocked_until=max(blocked_until,time.time()+delay)
  elif state=='DONE':
   requests+=1
   # Divide advertised provider limits across every eligible runner, including delayed starts.
   try:
    rpm=float(headers['x-ratelimit-limit-requests']);tpm=float(headers['x-ratelimit-limit-tokens']);ceiling=min(rpm*.85/(60*len(allocation['jobs'])),tpm*.85/(60*len(allocation['jobs'])*max(inp+body['max_completion_tokens'],1)))
    rate_ceiling=ceiling
    rps=min(rps,ceiling)
    if requests%20==0:rps=min(ceiling,rps*1.5)
   except (KeyError,ValueError):pass
  db.execute('UPDATE calls SET state=?,response=?,headers=?,finished=?,reserved=?,actual_estimate=?,input_tokens=?,output_tokens=? WHERE key=?',(state,raw,json.dumps(headers),time.time(),upper,actual,inp,output,key));db.commit()
  totals=db.execute('SELECT count(*),sum(reserved),sum(actual_estimate) FROM calls').fetchone()
  write('usage.json',{'recorded_requests':totals[0],'conservative_cost_ceiling_usd':totals[1],'standard_price_estimated_usd':totals[2],'configured_rps':rps,'elapsed_s':time.time()-started,'successful_rps':requests/(time.time()-started),'successful_requests_new':requests,'reused_responses':reused,'network_active':active_network,'network_peak':network_peak,'http_429':limited,'rate_ceiling_from_provider':rate_ceiling,'rate_headers':headers})
  write('lease.json',{'channel':'B','experiment_id':NAME,'owner_run':RUN,'heartbeat':time.time(),'auto_reassign':False})
  if time.time()-last_publish>=60:
   last_publish=time.time()
   await asyncio.to_thread(publish_status,{'experiment_id':NAME,'channel':'B','state':'RUNNING','owner_run':RUN,'heartbeat':time.time(),'successful_requests':recovery['saved_successes']+requests,'new_successful_requests':requests,'reused_responses':reused,'http_429':limited,'network_peak':network_peak,'cost_ceiling_usd':totals[1],'estimated_cost_usd':totals[2],'rps':requests/(time.time()-started),'configured_rps':rps,'auto_reassign':False})
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
 publish_status({'experiment_id':NAME,'channel':'B','state':state,'owner_run':RUN,'heartbeat':time.time(),'successful_requests':recovery['saved_successes']+requests,'new_successful_requests':requests,'reused_responses':reused,'http_429':limited,'network_peak':network_peak,'cost_ceiling_usd':db.execute('SELECT sum(reserved) FROM calls').fetchone()[0],'estimated_cost_usd':db.execute('SELECT sum(actual_estimate) FROM calls').fetchone()[0],'auto_reassign':False})
 db.close();return 0 if state=='COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
