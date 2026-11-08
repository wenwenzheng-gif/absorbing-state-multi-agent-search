#!/usr/bin/env python3
"""Exact-model, no-secret-log OpenAI API preflight for the Nano experiment."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "gpt-5.4-nano"
OUTPUT = Path("outputs/gpt-5.4-nano/preflight.json")

def write(value):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(OUTPUT)

def api(url, token, body=None):
    headers = {"Authorization": "Bearer " + token}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers, method="POST" if body is not None else "GET")
    with urlopen(req, timeout=90) as response:
        return json.load(response)

def main():
    now = datetime.now(timezone.utc).isoformat()
    report = {"requested_model": MODEL, "reasoning_effort": "medium", "started_at": now, "status": "BLOCKED"}
    token = os.getenv("OPENAI_API_KEY", "")
    if not token:
        report["reason"] = "OPENAI_API_KEY_NOT_INJECTED"
        write(report)
        print("BLOCKED: OPENAI_API_KEY is absent in this runner; value never logged.", flush=True)
        return 78
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if base != "https://api.openai.com/v1":
        report["reason"] = "NON_OFFICIAL_ENDPOINT_REQUIRES_EXPLICIT_VERIFICATION"
        write(report)
        print("BLOCKED: unexpected API endpoint; not attempting an alternate provider.", flush=True)
        return 78
    try:
        metadata = api(base + "/models/" + MODEL, token)
        report["model_endpoint_id"] = metadata.get("id")
        if metadata.get("id") != MODEL:
            raise ValueError("model lookup returned an unexpected model ID")
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Return a JSON object whose ok field is true."}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "preflight", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["ok"], "properties": {"ok": {"type": "boolean", "enum": [True]}}}}},
            "reasoning_effort": "medium",
            "max_completion_tokens": 1024,
        }
        response = api(base + "/chat/completions", token, body)
        resolved = response.get("model")
        if resolved not in (MODEL, "gpt-5.4-nano-2026-03-17"):
            raise ValueError("completion returned another model ID: " + str(resolved))
        choice = response["choices"][0]
        result = json.loads(choice["message"]["content"])
        if result != {"ok": True}:
            raise ValueError("structured response failed parser validation")
        report.update(status="PASS", response_model_id=resolved,
                      finish_reason=choice.get("finish_reason"),
                      token_usage=response.get("usage", {}),
                      completed_at=datetime.now(timezone.utc).isoformat())
        write(report)
        print("PASS: exact requested model", MODEL, "resolved model", resolved, "reasoning medium; structured parser succeeded.", flush=True)
        return 0
    except HTTPError as err:
        report.update(reason="HTTP_ERROR", http_status=err.code, completed_at=datetime.now(timezone.utc).isoformat())
    except (URLError, TimeoutError) as err:
        report.update(reason="TRANSPORT_ERROR", error_type=type(err).__name__, completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as err:
        report.update(reason="COMPATIBILITY_ERROR", error_type=type(err).__name__,
                      error_message=str(err)[:160], completed_at=datetime.now(timezone.utc).isoformat())
    write(report)
    print("BLOCKED:", report["reason"], report.get("http_status", report.get("error_type", "")), flush=True)
    return 1

if __name__ == "__main__":
    sys.exit(main())
