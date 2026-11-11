"""Read B heartbeats, validate artifacts, and publish without overwriting successful data."""
import base64,collections,ctypes,sqlite3,fcntl,json,os,pathlib,shutil,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
A=pathlib.Path('/data4/toby/absorbing-state-multi-agent-search');OUT=A/'runs/gpt-6-sol-medium-12seeds';REPO='wenwenzheng-gif/absorbing-state-multi-agent-search'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.monitor.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def gh(args):
 p=subprocess.run(['gh']+args,cwd=ROOT,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[:500])
 return json.loads(p.stdout) if p.stdout.strip() else None

def publish(source,name):
 current=OUT/'jobs'/name;remote_status=read(source/'status.json',{})
 if read(current/'validation.json',{}).get('passed'):return 'ALREADY_VERIFIED_PRESERVED'
 owner=read(OUT/'allocation.json')['owners'][name];assert owner=='B'
 assert not (current/'raw.sqlite').exists(),'Existing request data must not be overwritten'
 before=read(current/'config.json');after=read(source/'config.json');before.pop('output_root',None);after.pop('output_root',None);assert before==after,'Scientific config mismatch'
 stage=OUT/'channel_b/staging'/name
 if stage.exists():raise RuntimeError('Staged result requires reconciliation')
 shutil.copytree(source,stage)
 if (current/'channel_provenance.json').exists():shutil.copyfile(current/'channel_provenance.json',stage/'channel_provenance.json')
 if remote_status.get('state')=='COMPLETE':
  assert read(stage/'validation.json',{}).get('passed')
  ids=set();usage=collections.Counter()
  with sqlite3.connect(f'file:{stage}/raw.sqlite?mode=ro',uri=True) as db:
   for state,request,response in db.execute('SELECT state,request,response FROM calls'):
    assert state=='DONE','Unresolved response cannot be published as complete'
    req=json.loads(request);reply=json.loads(response)
    assert req['model']=='gpt-6-sol' and req['reasoning_effort']=='medium' and 'temperature' not in req
    assert reply['model']=='gpt-6-sol' and reply.get('id') and reply['id'] not in ids
    ids.add(reply['id'])
    for key in ('prompt_tokens','completion_tokens','total_tokens'):usage[key]+=reply.get('usage',{}).get(key,0)
  with (stage/'local_replay.log').open('w') as f:
   p=subprocess.run([str(A/'.venv-sol/bin/python'),'-m','src.agent_system.cli','replay','--run-dir',str(stage),'--output-root',str(OUT/'channel_b/local_replay')],cwd=A,stdout=f,stderr=f,env=dict(os.environ,OPENAI_API_KEY='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
  assert p.returncode==0,'Local offline replay failed'
  v=read(stage/'validation.json');v['local_offline_replay_passed']=True;v['raw_responses']=len(ids);v['usage']=dict(usage);v['local_raw_audit_passed']=True;write(stage/'validation.json',v)
 with (OUT/'allocation.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  assert read(OUT/'allocation.json')['owners'][name]=='B' and not read(current/'validation.json',{}).get('passed')
  assert not (current/'raw.sqlite').exists()
  # Linux RENAME_EXCHANGE atomically publishes the entire validated directory.
  libc=ctypes.CDLL(None,use_errno=True);rc=libc.renameat2(-100,os.fsencode(stage),-100,os.fsencode(current),2)
  if rc:raise OSError(ctypes.get_errno(),'Atomic result exchange failed')
 # Keep the allocation stub as an archive; never delete previous metadata.
 archived=OUT/'channel_b/allocation_stubs'/name;archived.parent.mkdir(parents=True,exist_ok=True);stage.rename(archived)
 return remote_status.get('state','HELD')

lock=(OUT/'monitor_b.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
imported=read(OUT/'channel_b/imported.json',{});heartbeats={}
while True:
 try:
  activation=read(OUT/'channel_b_activation.json',{})
  if activation.get('state')!='B_WORKFLOW_SUBMITTED':time.sleep(30);continue
  allocation=activation['allocation'];sha=activation['experiment_sha'];run_id=activation.get('experiment_run')
  if not run_id:
   runs=gh(['run','list','--repo',REPO,'--branch','experiment/gpt6-sol-medium-12seeds-20261111','--limit','40','--json','databaseId,headSha,name'])
   match=[r for r in runs if r['headSha']==sha and r['name']=='Sol Medium channel B exclusive budgeted experiments']
   if not match:time.sleep(30);continue
   run_id=match[0]['databaseId'];activation['experiment_run']=run_id;write(OUT/'channel_b_activation.json',activation)
  run=gh(['run','view',str(run_id),'--repo',REPO,'--json','status,conclusion,jobs,url'])
  for name in allocation['jobs']:
   try:
    v=gh(['api',f'repos/{REPO}/contents/sol_channel_status.json?ref=sol-medium-claims/{name}']);h=json.loads(base64.b64decode(v['content']));assert h['experiment_id']==name and str(h['owner_run'])==str(run_id);heartbeats[name]=h
    if name not in imported:write(OUT/'jobs'/name/'status.json',{'state':'RUNNING' if h['state']=='COMPLETE' else h['state'],'channel':'B','github_run':run_id,'heartbeat':h['heartbeat'],'awaiting_local_validation':h['state']=='COMPLETE'})
   except RuntimeError:pass
  artifacts=gh(['api',f'repos/{REPO}/actions/runs/{run_id}/artifacts'])
  for artifact in artifacts['artifacts']:
   names=[n for n in allocation['jobs'] if artifact['name']==f'sol-channel-b-{n}-{run_id}']
   if not names or names[0] in imported:continue
   name=names[0];dest=OUT/'channel_b/downloads'/artifact['name'];dest.mkdir(parents=True,exist_ok=True)
   if not any(dest.iterdir()):gh(['run','download',str(run_id),'--repo',REPO,'--name',artifact['name'],'--dir',str(dest)])
   sources=list(dest.glob('**/'+name+'/status.json'))
   if not sources:continue
   source=sources[0].parent;result=publish(source,name);imported[name]={'state':result,'artifact_id':artifact['id'],'time':time.time()};write(OUT/'channel_b/imported.json',imported)
  usage=sum(h.get('estimated_cost_usd',0) for h in heartbeats.values())+allocation['preflight']['estimated_cost_usd'];reserved=sum(h.get('cost_ceiling_usd',0) for h in heartbeats.values())
  a=read(OUT/'progress.json',{});b_running=sum(h['state']=='RUNNING' for h in heartbeats.values());unimported_calls=sum(h.get('successful_requests',0) for n,h in heartbeats.items() if n not in imported)
  write(OUT/'dual_channel_status.json',{'updated':time.time(),'channel_a':a,'channel_b':{'workflow':run,'heartbeats':heartbeats,'imported':imported,'running_workers':b_running,'estimated_cost_usd':usage,'conservative_cost_ceiling_usd':reserved+allocation['preflight']['estimated_cost_usd'],'authorized_cap_usd':100},'combined_saved_or_reported_requests':a.get('usage',{}).get('DONE',0)+unimported_calls,'independent_quota_status':'Not inferred from different credentials; assess concurrent measured throughput and 429s.'})
 except Exception as e:
  with (OUT/'channel_b_monitor_errors.log').open('a') as f:f.write(str(time.time())+' '+type(e).__name__+': '+str(e)+'\n')
 time.sleep(60)
