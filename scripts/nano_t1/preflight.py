#!/usr/bin/env python3
"""Verify original provenance, then send exactly one small TRUE Nano t1 API request."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone

MODEL="gpt-5.4-nano-2026-03-17"
OUT=Path("outputs/gpt-5.4-nano-t1-none-new3")
AUDIT=OUT/"audit.json"

def check():
    report={"requested_model":MODEL,"temperature":1.0,"reasoning_effort":"none",
            "thinking_disabled":True,"endpoint":"https://api.openai.com/v1",
            "started":datetime.now(timezone.utc).isoformat(),"status":"BLOCKED"}
    try:
        provenance=json.loads(AUDIT.read_text())
        assert provenance["valid"] and not provenance["blockers"]
        assert provenance["resolved_original_snapshot"]==MODEL
        assert provenance["checks"]["hf_seed_indices_observed_in_paths"]==list(range(1,7))
        assert provenance["requested_new_seed_indices"]==[7,8,9]
        assert os.environ.get("OPENAI_MODEL")==MODEL
        assert os.environ.get("OPENAI_BASE_URL")=="https://api.openai.com/v1"
        token=os.environ.get("OPENAI_API_KEY")
        if not token:
            raise RuntimeError("No OPENAI_API_KEY injected")
        def api(endpoint,payload=None):
            req=Request("https://api.openai.com/v1"+endpoint,
                 data=json.dumps(payload).encode() if payload is not None else None,
                 headers={"Authorization":"Bearer "+token,
                          "Content-Type":"application/json"},
                 method="POST" if payload is not None else "GET")
            with urlopen(req,timeout=120) as raw: return json.load(raw)
        models=api("/models/"+MODEL)
        if models.get("id")!=MODEL: raise RuntimeError("Exact dated model unavailable")
        request={"model":MODEL,"temperature":1.0,"reasoning_effort":"none",
                 "messages":[{"role":"user","content":"Return a JSON object with ok=true."}],
                 "max_completion_tokens":3072,
                 "response_format":{"type":"json_schema",
                     "json_schema":{"name":"nano_t1_preflight","strict":True,
                     "schema":{"type":"object","additionalProperties":False,
                         "properties":{"ok":{"type":"boolean","enum":[True]}},"required":["ok"]}}}}
        # The final payload is checked at the API request layer, not inferred from config.
        assert request["model"]==MODEL and request["temperature"]==1.0
        assert request["reasoning_effort"]=="none" and "enable_thinking" not in request
        answer=api("/chat/completions",request)
        assert answer.get("model")==MODEL, "API substituted a model"
        assert json.loads(answer["choices"][0]["message"]["content"])=={"ok":True}
        report.update(status="PASS",actual_response_model=answer["model"],
                      finish_reason=answer["choices"][0].get("finish_reason"),
                      usage=answer.get("usage",{}),
                      authenticated=True,
                      completed=datetime.now(timezone.utc).isoformat())
        print("PASS exact dated snapshot, temperature 1.0, reasoning none, structured output")
        return_code=0
    except HTTPError as e:
        report.update(reason="HTTP_ERROR",http_status=e.code)
        print("BLOCKED HTTP",e.code)
        return_code=2
    except Exception as e:
        report.update(reason="PREFLIGHT_ERROR",error_type=type(e).__name__,error=str(e)[:160])
        print("BLOCKED",type(e).__name__,str(e)[:100])
        return_code=2
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"preflight.json").write_text(json.dumps(report,indent=2))
    return return_code
if __name__=="__main__": raise SystemExit(check())
