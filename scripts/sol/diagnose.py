#!/usr/bin/env python3
import pathlib,json,sqlite3,collections,time,statistics,math
from run import OUT,read,write,rows
now=time.time(); lat=collections.defaultdict(list); workers=[]; finished_rounds=[]; successful=0; total_tokens=0
for name in read(OUT/'batch.json')['jobs']:
 d=OUT/'jobs'/name
 if not (d/'raw.sqlite').exists(): continue
 counts=collections.Counter(); usage=collections.Counter(); response_times=[]
 with sqlite3.connect(f'file:{d}/raw.sqlite?mode=ro',uri=True) as db:
  for state,req,resp,start,finish in db.execute('select state,request,response,started,finished from calls'):
   counts[state]+=1
   if state!='DONE': continue
   payload=json.loads(resp); audit=payload.get('sampling_audit',{}); successful+=1
   n=payload.get('usage',{}).get('total_tokens',0); total_tokens+=n
   response_times.append(finish)
   if finish>=now-300:
    usage['last_300s_requests']+=1; usage['last_300s_tokens']+=n
   for scope in ('all','recent_300s'):
    if scope=='recent_300s' and finish<now-300: continue
    lat[scope+'_http_wall'].append(finish-start)
    for key in ('upstream_latency_s','upstream_network_latency_s'):
     if audit.get(key) is not None: lat[scope+'_'+key].append(audit[key])
    if audit.get('upstream_latency_s') is not None and audit.get('upstream_network_latency_s') is not None:
     lat[scope+'_proxy_pacing_wait'].append(max(0,audit['upstream_latency_s']-audit['upstream_network_latency_s']))
 records={r['cache_key']:r for r in rows(d/'responses.jsonl')}; round_counts=collections.Counter(r['round'] for r in records.values()); max_round=max(round_counts,default=0)
 phase_counts=collections.Counter((r['round'],r['phase']) for r in records.values())
 for r in records.values():
  if r.get('latency_s') is not None: lat['all_end_to_end_including_local_queue'].append(r['latency_s'])
 episodes=rows(d/'episodes.jsonl'); episode=next((e for e in episodes if e.get('status')=='ok'),None)
 cfg=read(d/'config.json'); task,policy=name.split('_')[:2]; planned=cfg['environment']['task_size']-1
 rounds_done=[t['round'] for t in episode['transitions']] if episode else list(range(1,max_round))
 for round_number in rounds_done:
  transition=next((t for t in (episode or {}).get('transitions',[]) if t['round']==round_number),{})
  finished_rounds.append({'job':name,'task':task,'policy':policy,'degree':cfg['degrees'][0],'round':round_number,'requests':round_counts[round_number],'phases':{p:phase_counts[round_number,p] for p in ('branch','experiment','communication')},'frontier_empty_fraction':transition.get('frontier_empty_fraction'),'own_pruned':transition.get('own_pruned_hypotheses'),'received_pruned':transition.get('recv_pruned_hypotheses'),'confirmed_by':'final episode' if episode else 'later round observed'})
 workers.append({'job':name,'task':task,'policy':policy,'degree':cfg['degrees'][0],'state':read(d/'status.json',{}).get('state'),'successful_requests':counts['DONE'],'last_300s_requests':usage['last_300s_requests'],'last_300s_tokens':usage['last_300s_tokens'],'requests_per_second_last_300s':usage['last_300s_requests']/300,'planned_rounds':planned,'confirmed_completed_rounds':len(rounds_done),'round_requests':dict(round_counts),'raw_states':dict(counts)})
def stats(values):
 if not values: return {'n':0}
 a=sorted(values); return {'n':len(a),'median':statistics.median(a),'p95':a[max(0,math.ceil(.95*len(a))-1)],'mean':statistics.mean(a),'min':a[0],'max':a[-1]}
groups=collections.defaultdict(list)
for r in finished_rounds: groups[(r['task'],r['policy'],r['round'])].append(r['requests'])
group_report=[{'task':k[0],'policy':k[1],'round':k[2],**stats(v)} for k,v in sorted(groups.items())]
# Do not project slow/unobserved strata from the fastest completed trajectories.
# Match task, policy, exact degree, and round; require two independent seeds.
by_job={w['job']:w for w in workers}
strata=collections.defaultdict(dict)
for r in finished_rounds:
 seed=read(OUT/'jobs'/r['job']/'config.json')['task_seeds'][0]
 strata[(r['task'],r['policy'],r['degree'],r['round'])][seed]=r['requests']
needed=collections.Counter(); covered=0; unknown=0; remaining=0.0; held=0
for name in read(OUT/'batch.json')['jobs']:
 c=read(OUT/'jobs'/name/'config.json'); task,policy=name.split('_')[:2]; w=by_job.get(name,{})
 if w.get('state')=='COMPLETE': continue
 if w.get('state')=='HELD': held+=1; continue
 for rn in range(1,c['environment']['task_size']):
  if rn<=w.get('confirmed_completed_rounds',0): continue
  key=(task,policy,c['degrees'][0],rn); needed[key]+=1
  observations=strata.get(key,{})
  if len(observations)>=2:
   covered+=1; remaining+=max(0,statistics.mean(observations.values())-w.get('round_requests',{}).get(rn,0))
  else: unknown+=1
rate=sum(w['last_300s_requests'] for w in workers)/300
active=[w for w in workers if w['state']=='RUNNING']
forecast={'status':'ESTIMATE_AVAILABLE' if unknown==0 and rate else 'INSUFFICIENT_REPRESENTATIVE_COVERAGE','remaining_requests_excluding_held':remaining if unknown==0 else None,'remaining_hours_excluding_held':remaining/rate/3600 if unknown==0 and rate else None,'held_trajectories':held,'observed_completed_rounds':len(finished_rounds),'remaining_round_instances':covered+unknown,'covered_round_instances':covered,'uncovered_round_instances':unknown,'required_strata':len(needed),'strata_with_two_independent_seeds':sum(len(strata.get(k,{}))>=2 for k in needed),'coverage_fraction':covered/(covered+unknown) if covered+unknown else 1,'method':'Exact task/policy/degree/round strata, at least two independent seeds per stratum; no cross-stratum extrapolation.','supersedes_fields':['empirical_eta_seconds','unpruned_remaining_hours'],'limitations':['A whole-batch ETA is withheld until all remaining strata have representative observations.','Held trajectories remain excluded because uncertain requests are not retried.','Projected future throughput may change with provider quotas.']}
report={'time':now,'latency_seconds':{k:stats(v) for k,v in lat.items()},'successful_requests':successful,'total_tokens':total_tokens,'recent_300s':{'requests':sum(w['last_300s_requests'] for w in workers),'requests_per_second':rate,'tokens_per_minute':sum(w['last_300s_tokens'] for w in workers)/5},'per_running_worker_rps':stats([w['requests_per_second_last_300s'] for w in active]),'completed_rounds':finished_rounds,'round_group_stats':group_report,'workers':workers,'observed_round_projection':forecast}
write(OUT/'diagnostics/performance.json',report)
summary={k:v for k,v in report.items() if k not in ('workers','completed_rounds')}; print(json.dumps(summary,indent=2))
