"""Test four additional workers; rollback means no new admissions, never kill workers."""
import time,json,urllib.request,fcntl
from run import OUT,read,write
lock=(OUT/'a_worker_probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def health():
 with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=10) as f:return json.load(f)['metrics']
capacity=read(OUT/'diagnostics/channel_a_capacity.json',{});assert capacity.get('state')=='FINISHED'
baseline=capacity['successful_rps_at_selected'];before=health();start=time.time();v0=read(OUT/'validation.json')['passed'];selected=36
write(OUT/'execution_limits.json',{'workers':40,'max_workers':40,'reason':'One bounded four-worker expansion test; existing workers never terminated.'})
try:
 for _ in range(12):
  time.sleep(15)
  if health()['upstream_429']>before['upstream_429']:break
 after=health();elapsed=time.time()-start;rate=(after['completed_responses']-before['completed_responses'])/elapsed;limited=after['upstream_429']-before['upstream_429']
 if not limited and rate>baseline*1.02:selected=40
 write(OUT/'diagnostics/channel_a_worker_probe.json',{'finished':time.time(),'baseline_workers':36,'trial_workers':40,'baseline_successful_rps':baseline,'trial_successful_rps':rate,'seconds':elapsed,'429s':limited,'validated_experiments':read(OUT/'validation.json')['passed']-v0,'selected_admission_target':selected,'worker_interruptions':0,'note':'Short-window completion counts are noisy; expansion retained only with higher successful API throughput. Rollback drains surplus workers naturally.'})
finally:write(OUT/'execution_limits.json',{'workers':selected,'max_workers':40,'reason':'Measured worker capacity test result; never interrupt surplus live workers.'})
