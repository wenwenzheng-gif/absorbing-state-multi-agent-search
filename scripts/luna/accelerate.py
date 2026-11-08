#!/usr/bin/env python3
"""Inspect or adjust only the corrected batch's shared upstream gate."""
import argparse,json
from common import OUT,write
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args()
    c=json.loads((OUT/'control.json').read_text())
    if args.limit is not None:
        assert 1<=args.limit<=108
        c['upstream_limit']=args.limit;write(OUT/'control.json',c)
    print(json.dumps(c,indent=2))
