#!/usr/bin/env python3
"""Install live capacity reload support without changing model request bodies."""
import pathlib, shutil

def main():
    p=pathlib.Path('/data4/toby/codex_api/server.py');backup=p.with_name('server.py.before-luna-throughput');shutil.copy2(p,backup) if not backup.exists() else None
    s=p.read_text();s=s.replace('SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)', '''class AdaptiveSlots:
        """Reload transport capacity without interrupting active requests."""
        def __init__(self):
            self.condition = threading.Condition()
            self.active = 0
        def limit(self):
            try:
                return max(1, int((HERE / "upstream_limit.txt").read_text().strip()))
            except (OSError, ValueError):
                return MAX_CONCURRENCY
        def __enter__(self):
            with self.condition:
                while self.active >= self.limit():
                    self.condition.wait(timeout=0.25)
                self.active += 1
        def __exit__(self, *exc):
            with self.condition:
                self.active -= 1
                self.condition.notify_all()
    SLOTS = AdaptiveSlots()''')
    s=s.replace('return self._send(200, {"ok": True})','return self._send(200, {"ok": True, "upstream_limit": SLOTS.limit(), "active_requests": SLOTS.active})')
    compile(s,str(p),'exec');p.write_text(s)

if __name__ == "__main__":
    main()
