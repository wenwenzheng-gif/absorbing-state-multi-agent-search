from pathlib import Path
import json,time,collections
root=Path(__file__).resolve().parents[2]/'outputs/gpt-6-luna'
print('Elapsed seconds:',round(time.time()-json.loads((root/'preflight.json').read_text())['started']))
for task in ['synthetic','tcas','ode']:
 smoke=root/'_smoke'/task
 sd=json.loads((smoke/'status.json').read_text())
 states=[]; counts=collections.Counter()
 for d in (root/task).glob('seed_*'):
  if (d/'status.json').exists():states.append(json.loads((d/'status.json').read_text())['status'])
  for name in ['responses','transport_errors','episodes']:
   p=d/f'{name}.jsonl'
   if p.exists():counts[name]+=sum(1 for _ in p.open())
 print(task,'smoke:',sd['status'],'smoke responses:',sum(1 for _ in (smoke/'responses.jsonl').open()) if (smoke/'responses.jsonl').exists() else 0,'seeds:',dict(collections.Counter(states)),'counts:',dict(counts))
print('Control:',(root/'control.json').read_text())
print('Supervisor:',(root/'supervisor.log').read_text()[-1500:])
