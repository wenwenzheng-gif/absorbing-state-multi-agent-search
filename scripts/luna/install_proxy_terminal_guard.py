#!/usr/bin/env python3
"""Reject prematurely closed upstream streams instead of inventing a completion."""
from pathlib import Path

def main():
    path=Path('/data4/toby/codex_api/server.py');source=path.read_text()
    marker='        msg = {"role": "assistant", "content": "".join(text) or None}'
    guard='''        if not actual.get("upstream_response_id"):
            metric("unterminated_streams")
            return self._error(502, "upstream stream ended without a terminal response")
'''
    if guard not in source:
        assert source.count(marker)==1
        source=source.replace(marker,guard+marker)
    compile(source,str(path),'exec');path.write_text(source)

if __name__=='__main__':main()
