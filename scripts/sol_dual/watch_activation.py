"""Continue a blocked preflight without resending any uncertain request."""
import fcntl,json,pathlib,subprocess,time,sys,shutil
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path('/data4/toby/absorbing-state-multi-agent-search/runs/gpt-6-sol-medium-12seeds')
REPO='wenwenzheng-gif/absorbing-state-multi-agent-search';BRANCH='experiment/gpt6-sol-medium-12seeds-20261111'
STATE=OUT/'channel_b_activation.json'
def write(v):
 t=STATE.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(STATE)
def cmd(args):return subprocess.check_output(args,cwd=ROOT,text=True)
def gh(args):return json.loads(cmd(['gh']+args))
def commit_push(paths,message):
 cmd(['git','add']+paths)
 cmd(['git','-c','user.name=TobyYang7','-c','user.email=42986654+TobyYang7@users.noreply.github.com','commit','-m',message])
 sha=cmd(['git','rev-parse','HEAD']).strip();cmd(['git','push','origin',sha+':'+BRANCH]);return sha
lock=(OUT/'channel_b_activation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=json.loads(STATE.read_text()) if STATE.exists() else {'preflight_run':43094576266,'state':'WAITING_FOR_PREFLIGHT','budget_usd':100,'rerouted':False}
while True:
 try:
  r=gh(['run','view',str(s['preflight_run']),'--repo',REPO,'--json','status,conclusion,jobs,url']);s.update(updated=time.time(),github_status=r['status'],github_conclusion=r['conclusion'],preflight_url=r['url'])
  if r['status']!='completed':write(s);time.sleep(60);continue
  if r['conclusion']=='cancelled' and not s.get('rerouted'):
   # A canceled queue entry is retried only with positive evidence no API step ran.
   api_steps=[step for j in r['jobs'] for step in (j.get('steps') or []) if 'exact model' in step.get('name','').lower()]
   if any(step.get('status')!='queued' and step.get('conclusion')!='skipped' for step in api_steps):
    s.update(state='HELD_PREFLIGHT_OUTCOME_UNCERTAIN');write(s);break
   if any(j.get('steps') for j in r['jobs']):
    s.update(state='HELD_PREFLIGHT_REQUIRES_ARTIFACT_AUDIT');write(s);break
   p=ROOT/'.github/workflows/sol-dual-preflight.yml';text=p.read_text();assert 'ubuntu-latest' in text;p.write_text(text.replace('ubuntu-latest','ubuntu-22.04'))
   sha=commit_push(['.github/workflows/sol-dual-preflight.yml'],'Move confirmed unexecuted Sol preflight to alternate Linux image')
   s.update(rerouted=True,reroute_sha=sha,state='WAITING_FOR_REROUTED_PREFLIGHT');write(s)
   for _ in range(60):
    runs=gh(['run','list','--repo',REPO,'--branch',BRANCH,'--limit','20','--json','databaseId,headSha,name'])
    match=[v for v in runs if v['headSha']==sha and v['name']=='Sol Medium channel B exact-model preflight']
    if match:s['preflight_run']=match[0]['databaseId'];write(s);break
    time.sleep(10)
   else:raise RuntimeError('Rerouted workflow ID unavailable; no further push or API retry is allowed')
   continue
  artifact=OUT/'channel_b_preflight'/str(s['preflight_run']);artifact.mkdir(parents=True,exist_ok=True)
  if not (artifact/'summary.json').exists():cmd(['gh','run','download',str(s['preflight_run']),'--repo',REPO,'--name','sol-channel-b-preflight-'+str(s['preflight_run']),'--dir',str(artifact)])
  report=json.loads((artifact/'summary.json').read_text());s['preflight_result']=report
  if not report.get('passed'):
   s.update(state='BLOCKED_API_PREFLIGHT');write(s);break
  if not s.get('allocation_published'):
   sys.path.insert(0,str(ROOT/'scripts/sol_dual'));from partition import reserve
   reserve(OUT,ROOT/'scripts/sol_dual',artifact/'summary.json')
   s.update(allocation_published=True,state='PARTITION_RESERVED');write(s)
  target=ROOT/'.github/workflows/sol-dual-experiments.yml';shutil.copyfile(ROOT/'scripts/sol_dual/experiment-workflow.yml.template',target)
  sha=commit_push(['scripts/sol_dual/allocation.json','scripts/sol_dual/configs','.github/workflows/sol-dual-experiments.yml'],'Launch disjoint channel B experiments within approved 100 dollar cap')
  s.update(state='B_WORKFLOW_SUBMITTED',experiment_sha=sha,allocation=json.loads((ROOT/'scripts/sol_dual/allocation.json').read_text()));write(s)
  break
 except Exception as e:
  s.update(state='BLOCKED_NEEDS_RECONCILIATION',error=type(e).__name__+': '+str(e),updated=time.time());write(s);break
