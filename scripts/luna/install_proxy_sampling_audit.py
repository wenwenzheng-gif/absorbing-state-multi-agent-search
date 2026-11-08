#!/usr/bin/env python3
"""Add raw upstream provenance and capacity-error metrics to codex_api."""
from pathlib import Path
import shutil

def main():
    p=Path('/data4/toby/codex_api/server.py');backup=p.with_name('server.py.before-none-audit');shutil.copy2(p,backup) if not backup.exists() else None
    s=p.read_text()
    marker='# ---------------------------------------------------------------- auth'
    if 'METRICS_LOCK =' not in s:
     s=s.replace(marker,'''METRICS_LOCK = threading.Lock()
    METRICS = {key: 0 for key in ("upstream_attempts", "upstream_429", "upstream_5xx", "upstream_transport_errors", "upstream_timeouts", "completed_responses")}
    def metric(name):
        with METRICS_LOCK:
            METRICS[name] = METRICS.get(name, 0) + 1
    
    def metrics_snapshot():
        with METRICS_LOCK:
            return dict(METRICS)
    
    '''+marker)
     s=s.replace('''            return urllib.request.urlopen(req, timeout=TIMEOUT)
            except urllib.error.HTTPError as e:''','''            metric("upstream_attempts")
                return urllib.request.urlopen(req, timeout=TIMEOUT)
            except urllib.error.HTTPError as e:
                if e.code == 429: metric("upstream_429")
                if e.code >= 500: metric("upstream_5xx")''')
     s=s.replace('''            raise
    
    
    def sse_events(resp):''','''            raise
            except (urllib.error.URLError, TimeoutError) as e:
                metric("upstream_transport_errors")
                if isinstance(e, TimeoutError) or "timed out" in str(e).lower(): metric("upstream_timeouts")
                raise
    
    
    def sse_events(resp):''')
     s=s.replace('''"active_requests": SLOTS.active}''','''"active_requests": SLOTS.active, "metrics": metrics_snapshot()}''')
     s=s.replace('''        except (BrokenPipeError, ConnectionResetError):
                pass''','''        except (BrokenPipeError, ConnectionResetError):
                pass
            except (urllib.error.URLError, TimeoutError):
                self._error(502, "upstream transport failed")''')
     s=s.replace('''        with upstream("/responses", body) as resp:
                deltas = chat_deltas(sse_events(resp))''','''        actual = {}
            started = time.monotonic()
            def audited_events(resp):
                for event in sse_events(resp):
                    if event.get("type") == "response.completed":
                        r = event.get("response", {})
                        actual.update({"upstream_response_id": r.get("id"), "actual_model": r.get("model"),
                                       "effective_temperature": r.get("temperature"),
                                       "effective_reasoning_effort": (r.get("reasoning") or {}).get("effort"),
                                       "reasoning_tokens": ((r.get("usage") or {}).get("output_tokens_details") or {}).get("reasoning_tokens"),
                                       "upstream_latency_s": time.monotonic() - started})
                        metric("completed_responses")
                    yield event
            with upstream("/responses", body) as resp:
                deltas = chat_deltas(audited_events(resp))''')
     s=s.replace('''"id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices"''','''"id": cid, "object": "chat.completion", "created": created, "model": actual.get("actual_model") or model,
                "sampling_audit": {**actual, "temperature_requested": req.get("temperature"),
                                   "temperature_forwarded": False, "seed_requested": req.get("seed"),
                                   "seed_forwarded": False, "reasoning_effort_requested": req.get("reasoning_effort")},
                "choices"''')
    compile(s,str(p),'exec');p.write_text(s)

if __name__ == "__main__":
    main()
