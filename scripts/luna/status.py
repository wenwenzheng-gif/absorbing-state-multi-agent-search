#!/usr/bin/env python3
"""Compact status for the isolated corrected batch."""
import json
from common import OUT
p=OUT/'progress.json'
if p.exists():
    r=json.loads(p.read_text());print(f'{r["completed_episodes"]}/{r["expected_episodes"]} episodes; {r["active_scientific_workers"]}/9 workers; upstream={r["upstream_limit"]}; {r["responses_per_min"]:.1f} responses/min')
    for job in r['jobs']:print(job['task'],job['seed'],job['status'],f'{job["completed_episodes"]}/{job["expected_episodes"]}',f'API={job["api_requests"]}',f'errors={job["transport_retries"]}')
else:
    p=OUT/'preflight.json'
    print(json.dumps(json.loads(p.read_text()) if p.exists() else {'status':'NOT_STARTED'},indent=2))
