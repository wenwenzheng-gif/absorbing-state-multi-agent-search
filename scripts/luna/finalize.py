#!/usr/bin/env python3
"""Detached completion watcher: final accounting, secret audit, Git publication."""
import collections,json,os,pathlib,re,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/gpt-6-luna'
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
 while not (OUT/'combined_summary.json').exists():
  time.sleep(30)
 result=json.loads((OUT/'combined_summary.json').read_text())
 from dotenv import dotenv_values
 from huggingface_hub import get_token
 values=dotenv_values('/data4/toby/codex_api/.env')
 secrets=[v.encode() for k,v in values.items() if v and len(v)>=8 and ('KEY' in k or 'TOKEN' in k)]
 token=get_token()
 if token: secrets.append(token.encode())
 violations=[]
 for p in OUT.rglob('*'):
  if not p.is_file(): continue
  content=p.read_bytes()
  if any(s in content for s in secrets) or re.search(rb'\bhf_[A-Za-z0-9]{25,}\b|\bsk-[A-Za-z0-9_-]{25,}\b',content): violations.append(str(p.relative_to(ROOT)))
 write(OUT/'secret_audit.json',{'passed':not violations,'violations':violations,'scanned_root':str(OUT),'timestamp':time.time()})
 if violations: raise RuntimeError('Secret audit failed; publication stopped. See local secret_audit.json.')
 states={}
 for task in ['synthetic','tcas','ode']:
  states[task]={}
  for seed in range(6):
   state=json.loads((OUT/task/f'seed_{seed}/status.json').read_text())
   assert state['status'] in ('COMPLETE','FAILED_AFTER_RETRIES')
   states[task][str(seed)]=state['status']
  assert (OUT/task/'summary.json').exists()
 # Count every retained response exactly once, including interrupted smoke work.
 usage=collections.Counter();smoke=collections.Counter()
 for p in OUT.rglob('responses.jsonl'):
  target=smoke if any(part.startswith('_smoke') for part in p.parts) else usage
  for line in p.open():
   row=json.loads(line);u=row.get('usage') or {}
   target['model_calls']+=1
   for key in ['prompt_tokens','completion_tokens','total_tokens']: target[key]+=u.get(key,0)
 report={'model':'gpt-6-luna','reasoning_effort':'medium','seed_ids':list(range(6)),'states':states,'experiment_usage':dict(usage),'smoke_usage':dict(smoke),'wall_seconds':time.time()-1794116541.016769,'maximum_job_concurrency':18,'final_admission_limit':result['control']['concurrency'],'retries':{t:{str(s):len(result['tasks'][t]['seeds'][s]['attempts'])-1 for s in range(6)} for t in states},'local_outputs':str(OUT),'uploads':result['uploads'],'secret_audit_passed':True,'branch':'experiment/gpt6-luna-3tasks-6seeds-parallel'}
 write(OUT/'final_report.json',report)
 write(ROOT/'results/gpt-6-luna/final_report.json',report)
 write(ROOT/'results/gpt-6-luna/combined_summary.json',result)
 for task in states:
  write(ROOT/f'results/gpt-6-luna/{task}_summary.json',result['tasks'][task])
 cmd=['git','-c','user.name=Codex','-c','user.email=codex@openai.com']
 subprocess.run(['git','add','scripts/luna','src/agent_system/policies/openai_policy.py','.gitignore','results/gpt-6-luna'],cwd=ROOT,check=True)
 subprocess.run(cmd+['commit','-m','Finalize GPT-6-Luna three-task six-seed results and accounting'],cwd=ROOT,check=True)
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 for attempt in range(4):
  p=subprocess.run(['git','push','origin',report['branch']],cwd=ROOT)
  if p.returncode==0: break
  time.sleep(min(60,2**(attempt+1)))
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+report['branch']],cwd=ROOT,text=True).split()[0]
 write(OUT/'publication.json',{'commit':sha,'remote_commit':remote,'verified':sha==remote})
 print('FINALIZED',sha,'remote verified:',sha==remote,flush=True)
if __name__=='__main__': main()
