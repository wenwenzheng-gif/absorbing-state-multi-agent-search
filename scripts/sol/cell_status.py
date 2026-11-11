#!/usr/bin/env python3
"""Record task/policy/seed cell completion independently of worker lifetimes."""
import collections,time,fcntl
from run import OUT,read,write
lock=(OUT/'cell_status.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 cells={}
 for name in read(OUT/'batch.json')['jobs']:
  cell=name.rsplit('_d',1)[0]; state=read(OUT/'jobs'/name/'status.json',{'state':'PENDING'})['state']
  cells.setdefault(cell,collections.Counter())[state]+=1
 complete=sum(set(v)=={'COMPLETE'} for v in cells.values())
 p=read(OUT/'progress.json',{}); r=read(OUT/'rate_status.json',{}); t=read(OUT/'throughput.json',{})
 perf=read(OUT/'diagnostics/performance.json',{}); forecast=perf.get('observed_round_projection',{})
 write(OUT/'dashboard.json',{'updated':time.time(),'observed_progress_eta':forecast,'observed_progress_eta_updated':perf.get('time'),'total_cells':len(cells),'completed_cells':complete,'total_degree_experiments':684,'completed_degree_experiments':p.get('states',{}).get('COMPLETE',0),'running_jobs':p.get('running_jobs',0),'states':p.get('states',{}),'usage':p.get('usage',{}),'rate_limit_frequency':r,'unpruned_remaining_hours':t.get('unpruned_remaining_seconds_at_observed_rate',0)/3600,'empirical_eta_seconds':p.get('eta_seconds_from_completed_experiments'),'cells':{k:dict(v) for k,v in cells.items()}})
 if (OUT/'completion.json').exists(): break
 time.sleep(30)
