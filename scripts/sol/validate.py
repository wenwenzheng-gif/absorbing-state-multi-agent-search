#!/usr/bin/env python3
"""Detached incremental offline validation and final aggregation. No API calls."""
import collections,json,pathlib,subprocess,sys,time,sqlite3,hashlib,os,fcntl
from run import ROOT,OUT,PYTHON,read,write,rows

def audit_config():
 batch=read(OUT/'batch.json'); failures=[]
 for name in batch['jobs']:
  c=read(OUT/'jobs'/name/'config.json'); task,policy=name.split('_')[:2]
  ref=read(ROOT/f'configs/agents/gpt-6-luna_seed1/{task}_{policy}.json')
  for k in ('name','task_seeds','degrees','output_root'): c.pop(k); ref.pop(k)
  for k in ('model','reasoning_effort','temperature','timeout_s','api_style'): c['policy']['llm'].pop(k,None); ref['policy']['llm'].pop(k,None)
  if c!=ref: failures.append(name)
 for p,h in batch['prompt_hashes'].items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h: failures.append(p)
 return {'passed':not failures,'checked_jobs':len(batch['jobs']),'failures':failures,'prompt_hashes_unchanged':not any('/prompts/' in p for p in failures)}

def validate_job(name):
 d=OUT/'jobs'/name; c=read(d/'config.json'); records=rows(d/'episodes.jsonl'); failures=[]; response_ids=[]; usage=collections.Counter()
 if len(records)!=1 or records[0].get('status')!='ok': failures.append('episode not successful')
 with sqlite3.connect(f'file:{d}/raw.sqlite?mode=ro',uri=True) as db:
  for state,req,resp in db.execute('select state,request,response from calls'):
   if state!='DONE': failures.append('unresolved raw request: '+state); continue
   request=json.loads(req); response=json.loads(resp)
   if request['model']!='gpt-6-sol' or request.get('reasoning_effort')!='medium' or 'temperature' in request: failures.append('request settings mismatch')
   audit=response.get('sampling_audit',{})
   if audit.get('actual_model')!='gpt-6-sol' or audit.get('effective_reasoning_effort')!='medium' or audit.get('temperature_forwarded'): failures.append('upstream settings mismatch')
   response_ids.append(audit.get('upstream_response_id') or response.get('id'))
   for k in ('prompt_tokens','completion_tokens','total_tokens'): usage[k]+=response.get('usage',{}).get(k,0)
 if len(response_ids)!=len(set(response_ids)): failures.append('duplicate upstream response IDs')
 result={'passed':not failures,'failures':dict(collections.Counter(failures)),'raw_responses':len(response_ids),'usage':dict(usage),'checked':time.time()}
 write(d/'validation.json',result); return result

def aggregate():
 batch=read(OUT/'batch.json'); reports={}
 for task in ('synthetic','tcas','physics'):
  for policy in ('C0','C1','I1'):
   names=[n for n in batch['jobs'] if n.startswith(f'{task}_{policy}_')]; dest=OUT/'aggregate'/f'{task}_{policy}'; dest.mkdir(parents=True,exist_ok=True)
   manifest=read(OUT/'jobs'/names[0]/'manifest.json')
   manifest['config']['task_seeds']=sorted({read(OUT/'jobs'/n/'config.json')['task_seeds'][0] for n in names})
   manifest['config']['degrees']=sorted({read(OUT/'jobs'/n/'config.json')['degrees'][0] for n in names})
   manifest['shards']=names; manifest['run_id']=f'{task}_{policy}'; write(dest/'manifest.json',manifest)
   for ledger in ('episodes','events'):
    with (dest/f'{ledger}.jsonl').open('w') as f:
     for n in names:
      source=OUT/'jobs'/n/f'{ledger}.jsonl'
      if source.exists():
       with source.open() as inp:
        for line in inp: f.write(line)
   codes={}
   for module in ('summarize','critical_degree'):
    with (dest/f'{module}.log').open('w') as log:
     codes[module]=subprocess.run([str(PYTHON),'-m',f'src.agent_system.analysis.{module}','--run-dir',str(dest)],cwd=ROOT,stdout=log,stderr=log).returncode
   reports[f'{task}_{policy}']=codes
 write(OUT/'analysis.json',reports)

def main():
 lock=(OUT/'validation.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 config=audit_config(); write(OUT/'config_validation.json',config); assert config['passed'],config
 validated={}; replayed={}
 while True:
  for name in read(OUT/'batch.json')['jobs']:
   d=OUT/'jobs'/name
   if read(d/'status.json',{}).get('state')!='COMPLETE' or name in validated: continue
   validated[name]=validate_job(name)
   cell='_'.join(name.split('_')[:2])
   if cell not in replayed:
    with (d/'replay.log').open('w') as log:
     code=subprocess.run([str(PYTHON),'-m','src.agent_system.cli','replay','--run-dir',str(d),'--output-root',str(OUT/'replay')],cwd=ROOT,stdout=log,stderr=log,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
    replayed[cell]={'job':name,'returncode':code}
  write(OUT/'validation.json',{'config':config,'validated':len(validated),'passed':sum(v['passed'] for v in validated.values()),'failed':{k:v for k,v in validated.items() if not v['passed']},'offline_replays':replayed,'updated':time.time()})
  if (OUT/'completion.json').exists(): aggregate(); break
  time.sleep(30)
if __name__=='__main__': main()
