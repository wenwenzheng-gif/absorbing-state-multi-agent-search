#!/usr/bin/env python3
"""Track live upstream streams separately from retry backoff and admitted requests."""
from pathlib import Path

def main():
    p=Path('/data4/toby/codex_api/server.py');s=p.read_text()
    if 'class TrackedUpstreamResponse:' not in s:
     s=s.replace('def metric(name):','def metric(name, amount=1):').replace('METRICS.get(name, 0) + 1','METRICS.get(name, 0) + amount')
     marker='# ---------------------------------------------------------------- auth'
     s=s.replace(marker,'''class TrackedUpstreamResponse:
        def __init__(self, response):
            self.response = response
            self.closed = False
        def __getattr__(self, name): return getattr(self.response, name)
        def __iter__(self): return iter(self.response)
        def __enter__(self): return self
        def __exit__(self, *exc): self.close()
        def close(self):
            if not self.closed:
                self.closed = True
                try: self.response.close()
                finally: metric("active_upstream_requests", -1)
    
    '''+marker)
     s=s.replace('''            metric("upstream_attempts")
                return urllib.request.urlopen(req, timeout=TIMEOUT)''','''            metric("upstream_attempts")
                metric("active_upstream_requests")
                return TrackedUpstreamResponse(urllib.request.urlopen(req, timeout=TIMEOUT))''')
     s=s.replace('''        except urllib.error.HTTPError as e:
                if e.code == 429:''','''        except urllib.error.HTTPError as e:
                metric("active_upstream_requests", -1)
                if e.code == 429:''')
     s=s.replace('''        except (urllib.error.URLError, TimeoutError) as e:
                metric("upstream_transport_errors")''','''        except (urllib.error.URLError, TimeoutError) as e:
                metric("active_upstream_requests", -1)
                metric("upstream_transport_errors")''')
     s=s.replace('''                    time.sleep(delay)
                        waited += delay''','''                    metric("upstream_retries")
                        time.sleep(delay)
                        waited += delay''')
     s=s.replace('''    return min(2 ** attempt, 30)
    ''','''    jitter = 0.85 + (time.monotonic_ns() % 3001) / 10000
        return min(2 ** attempt, 30) * jitter
    ''')
    compile(s,str(p),'exec');p.write_text(s)

if __name__ == "__main__":
    main()
