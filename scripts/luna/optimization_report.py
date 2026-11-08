#!/usr/bin/env python3
"""Display corrected-batch validation/publication metadata only."""
import json
from common import OUT
if __name__=='__main__':
    for name in ['validation_report.json','upload_status.json','publication.json','final_status.json']:
        p=OUT/name
        if p.exists():print(name,json.dumps(json.loads(p.read_text()),indent=2))
