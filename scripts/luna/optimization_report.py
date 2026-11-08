#!/usr/bin/env python3
"""Read-only final accounting for the Luna infrastructure intervention."""
import collections,hashlib,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/gpt-6-luna';META=OUT/'_infrastructure'
def build():
 records=[json.loads(line) for line in (META/'capacity.jsonl').read_text().splitlines() if line.strip()]
 baseline=next(r for r in records if r['kind']=='window' and r['upstream_limit']==4)
 limit=int(pathlib.Path('/data4/toby/codex_api/upstream_limit.txt').read_text())
 windows=[r for r in records if r['kind']=='window' and r['upstream_limit']==limit and r.get('successful_responses',0)>0]
 rate=sum(r['successful_responses'] for r in windows)*60/sum(r['seconds'] for r in windows) if windows else 0
 snap=pathlib.Path((META/'latest_snapshot.txt').read_text());inventory=json.loads((snap/'inventory.json').read_text());violations=[];jobs={};totals=collections.Counter();errors=collections.Counter()
 for job,files in inventory.items():
  d=OUT/job
  for name,meta in files.items():
   if not isinstance(meta,dict) or 'prefix_sha256' not in meta:continue
   p=d/name
   if not p.exists() or p.stat().st_size<meta['size']:violations.append(f'{job}/{name}: missing or truncated');continue
   with p.open('rb') as f:digest=hashlib.sha256(f.read(meta['size'])).hexdigest()
   if digest!=meta['prefix_sha256']:violations.append(f'{job}/{name}: prior bytes changed')
  eps=[json.loads(line) for line in (d/'episodes.jsonl').read_text().splitlines() if line.strip()] if (d/'episodes.jsonl').exists() else []
  for old in files['completed_episodes']:
   if old not in eps:violations.append(f'{job}: prior completed episode missing')
  state=json.loads((d/'status.json').read_text());jobs[job]={'status':state['status'],'completed_episodes':len({r['episode_id'] for r in eps if r['status']=='ok'})}
  for ledger in ['requests','responses','events']:
   p=d/f'{ledger}.jsonl'
   if p.exists():
    with p.open() as f:totals[ledger]+=sum(1 for line in f if line.strip())
  p=d/'transport_errors.jsonl'
  if p.exists():
   for line in p.open():
    if line.strip():errors[json.loads(line)['kind']]+=1
 completed=collections.Counter()
 for job,state in jobs.items():completed[job.split('/')[0]]+=state['completed_episodes']
 proxy429=sum(r.get('proxy_429',0) for r in records if r['kind']=='window')
 return {'time':time.time(),'final_upstream_limit':limit,'maximum_tested_concurrency':max(r['upstream_limit'] for r in records if r['kind']=='window'),'baseline_successful_responses_per_min':baseline['responses_per_min'],'post_optimization_successful_responses_per_min':rate,'speedup':rate/baseline['responses_per_min'],'experiment_wall_seconds':time.time()-1794116541.016769,'optimization_wall_seconds':time.time()-1794154935.6253016,'completed_episodes_by_task':dict(completed),'jobs':jobs,'failed_seeds':[job for job,state in jobs.items() if state['status']=='FAILED_AFTER_RETRIES'],'ledger_totals':dict(totals),'transport_errors':dict(errors),'observed_proxy_429':proxy429,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'hugging_face':{task:json.loads((OUT/task/'upload_status.json').read_text()) if (OUT/task/'upload_status.json').exists() else {'status':'NOT_UPLOADED'} for task in ['synthetic','tcas','ode']},'prior_data_preserved':not violations,'preservation_violations':violations}
if __name__=='__main__':
 if '--wait' in sys.argv:
  while not (OUT/'combined_summary.json').exists():
   time.sleep(30)
 report=build();(META/'final_optimization_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
