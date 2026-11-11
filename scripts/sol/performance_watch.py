#!/usr/bin/env python3
"""Refresh observed-round forecasts without any model calls."""
import fcntl,subprocess,time
from run import OUT,PYTHON,ROOT
lock=(OUT/'performance_watch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 with (OUT/'diagnostics/performance_watch.log').open('a') as log:
  subprocess.run([str(PYTHON),'scripts/sol/diagnose.py'],cwd=ROOT,stdout=log,stderr=log)
 if (OUT/'completion.json').exists(): break
 time.sleep(300)
