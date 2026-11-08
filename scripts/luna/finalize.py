#!/usr/bin/env python3
"""Finalize only the fully validated corrected nine-job batch."""
from common import OUT, SEEDS, TASKS
from audit import finish
import json
if __name__=='__main__':
    for task in TASKS:
        for seed in SEEDS:
            assert json.loads((OUT/'jobs'/f'{task}_seed{seed}'/'status.json').read_text())['status']=='COMPLETE'
    finish()
