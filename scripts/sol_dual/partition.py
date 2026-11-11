"""Publish an immutable disjoint partition only after a successful exact-model preflight."""
import argparse,fcntl,hashlib,json,pathlib,time

def reserve(out,destination,preflight):
 report=json.loads(preflight.read_text())
 assert report.get('passed') is True and report.get('actual_model')=='gpt-6-sol' and report.get('http_status')==200,'Channel B has not passed real API validation'
 assert report.get('estimated_cost_usd',1)<.1,'Preflight cost exceeds reserved allowance'
 destination.mkdir(parents=True,exist_ok=True)
 with (out/'allocation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  allocation=json.loads((out/'allocation.json').read_text());batch=json.loads((out/'batch.json').read_text())
  assert len(allocation['owners'])==684 and set(allocation['owners'])==set(batch['jobs'])
  assert not allocation.get('b_enabled'),'Partition already published; never issue a second allocation blindly'
  candidates=[]
  # Cheap high-degree cells first; all scientific parameters and the global plan remain fixed.
  for name in reversed(batch['jobs']):
   d=out/'jobs'/name;c=json.loads((d/'config.json').read_text());state=json.loads((d/'status.json').read_text()).get('state') if (d/'status.json').exists() else 'PENDING'
   if allocation['owners'][name]=='A' and state=='PENDING' and not (d/'raw.sqlite').exists() and not (d/'worker.lock').exists() and c['degrees']==[39] and '_C0_' in name:candidates.append(name)
  names=candidates[:24];assert len(names)==24,'Need 24 genuinely unclaimed jobs'
  hashes={};(destination/'configs').mkdir(exist_ok=True)
  for name in names:
   c=json.loads((out/'jobs'/name/'config.json').read_text());hashes[name]=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   (destination/'configs'/f'{name}.json').write_text(json.dumps(c,indent=2))
   allocation['owners'][name]='B'
  allocation.update(b_enabled=True,b_jobs=names,partitioned_at=time.time(),b_budget_usd=100,b_per_job_cap_usd=4,b_preflight_usd=report['estimated_cost_usd'],allocation_policy='No lease expiry or failure automatically transfers an ID between channels.')
  remote={'b_enabled':True,'jobs':names,'owners':allocation['owners'],'config_hashes':hashes,'prompt_hashes':batch['prompt_hashes'],'budget_usd':100,'per_job_cap_usd':4,'preflight':report,'source_reference_commit':batch['reference_commit']}
  remote['partition_sha256']=hashlib.sha256(json.dumps(remote,sort_keys=True).encode()).hexdigest()
  (destination/'allocation.json').write_text(json.dumps(remote,indent=2));allocation['b_partition_sha256']=remote['partition_sha256']
  tmp=out/'allocation.partition.tmp';tmp.write_text(json.dumps(allocation,indent=2));tmp.replace(out/'allocation.json')
  for name in names:
   p=out/'jobs'/name/'channel_provenance.json';v=json.loads(p.read_text());v.update(channel='B',allocation_sha256=remote['partition_sha256']);tmp=p.with_suffix('.partition.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
 print(json.dumps({'b_jobs':names,'max_cost_usd':24*4+report['estimated_cost_usd'],'partition_sha256':remote['partition_sha256']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--destination',type=pathlib.Path,required=True);p.add_argument('--preflight',type=pathlib.Path,required=True);a=p.parse_args();reserve(a.out,a.destination,a.preflight)
