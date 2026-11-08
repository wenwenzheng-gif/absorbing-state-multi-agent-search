#!/usr/bin/env python3
"""Low-overhead live progress and local proxy resource observations."""
import json,os,pathlib,time,urllib.request
from common import OUT,SETTINGS
previous=None
while True:
    p=OUT/'progress.json'
    progress=json.loads(p.read_text()) if p.exists() else {}
    try:health=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5))
    except Exception as e:health={'ok':False,'error':type(e).__name__}
    resource={}
    try:
        pid=json.loads((OUT/'operations/proxy.pid.json').read_text())['pid'];stat=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split();ticks=int(stat[11])+int(stat[12]);now=time.time()
        resource={'pid':pid,'rss_kb':int(stat[21])*os.sysconf('SC_PAGE_SIZE')//1024,'threads':int(stat[17])}
        if previous and previous[0]==pid:resource['cpu_percent']=(ticks-previous[1])/os.sysconf('SC_CLK_TCK')/(now-previous[2])*100
        previous=(pid,ticks,now)
    except (OSError,ValueError,KeyError):pass
    row={**SETTINGS,'time':time.time(),'completed_episodes':progress.get('completed_episodes',0),'expected_episodes':171,'active_scientific_workers':progress.get('active_scientific_workers',0),'actual_upstream_calls':health.get('metrics',{}).get('active_upstream_requests',0),'upstream_limit':health.get('upstream_limit'),'responses_per_min':progress.get('responses_per_min',0),'proxy_metrics':health.get('metrics',{}),'proxy_resources':resource}
    with (OUT/'operations/resource_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if (OUT/'final_status.json').exists():break
    time.sleep(60)
