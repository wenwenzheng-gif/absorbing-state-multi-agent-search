#!/usr/bin/env python3
"""Smooth global upstream starts without changing inference request bodies."""
from pathlib import Path

def main():
    p=Path('/data4/toby/codex_api/server.py');s=p.read_text()
    if 'def pace_upstream_start(' not in s:
     marker='class TrackedUpstreamResponse:'
     s=s.replace(marker,'''PACE_LOCK = threading.Lock()
    PACE_NEXT = 0.0
    
    def upstream_rps():
        try: return max(0.5, float((HERE / "upstream_rps.txt").read_text().strip()))
        except (OSError, ValueError): return 2.5
    
    def pace_upstream_start():
        global PACE_NEXT
        with PACE_LOCK:
            start = max(time.monotonic(), PACE_NEXT)
            PACE_NEXT = start + 1.0 / upstream_rps()
        delay = start - time.monotonic()
        if delay > 0: time.sleep(delay)
    
    '''+marker)
     s=s.replace('''    def __init__(self, response):
            self.response = response''','''    def __init__(self, response, network_started):
            self.response = response
            self.network_started = network_started''')
     s=s.replace('''            metric("upstream_attempts")
                metric("active_upstream_requests")
                return TrackedUpstreamResponse(urllib.request.urlopen(req, timeout=TIMEOUT))''','''            pace_upstream_start()
                metric("upstream_attempts")
                metric("active_upstream_requests")
                network_started = time.monotonic()
                return TrackedUpstreamResponse(urllib.request.urlopen(req, timeout=TIMEOUT), network_started)''')
     s=s.replace('''"metrics": metrics_snapshot()}''','''"metrics": metrics_snapshot(), "upstream_starts_per_second": upstream_rps()}''')
     s=s.replace('''"upstream_latency_s": time.monotonic() - started}''','''"upstream_latency_s": time.monotonic() - started,
                                       "upstream_network_latency_s": time.monotonic() - getattr(resp, "network_started", started)}''')
    compile(s,str(p),'exec');p.write_text(s)

if __name__ == "__main__":
    main()
