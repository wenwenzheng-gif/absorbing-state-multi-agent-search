"""Small task policies on the existing DecisionContext/LLMBackend API stack."""
from __future__ import annotations

from dataclasses import asdict
import json
import re
import time

from ..policies.base import DecisionContext
from ..policies.openai_policy import LLMBackend
from ..schemas import IncompleteOutput, PolicyFailure
from ..storage import RunStore, iter_jsonl, utc_now
from .config import fingerprint

NEEDLE_PROMPT = """Find the team's single hidden target among atomic integer candidate IDs.
Testing an ID returns TARGET or NOT TARGET and says nothing about other IDs.
Choose exactly batch_size candidate IDs for this round, in one JSON candidates array.
The batch is chosen before any test feedback. All agents choose simultaneously.
The environment automatically sends true results to direct neighbors next round.
Goal: maximize the chance the team tests the target. Avoid duplicate work.
scheduled_candidates is a public disjoint work assignment for this agent and round,
generated independently of the target. You may choose these IDs or any other legal IDs.
Messages contain only past neighbor results. No other agent's private history is visible.
Return only the requested JSON."""

NEEDLE_HIGH_V2_PROMPT = """Find the team's single hidden target among atomic integer candidate IDs.
Testing an ID returns TARGET or NOT TARGET and says nothing about other IDs.
Choose exactly batch_size candidate IDs for this round, in one JSON candidates array.
The batch is chosen before any test feedback. All agents choose simultaneously.
The environment automatically sends true results to direct neighbors next round.
Goal: maximize the chance the team tests the target. Avoid unnecessary duplicate work.
Your scheduled_candidates are a collision-free public work assignment within each
coverage pass, generated independently of the target. Following them is normally
the best default because the team wants broad unique coverage. The first round
assigns every candidate once; the second round provides a second complete coverage
pass, so repetition across rounds is intentional.
Do not replace scheduled candidates with arbitrary low-numbered IDs or another
pattern unless information already observed in the episode gives a concrete
reason to do so. You may legally choose other candidates, but unnecessary
departures can duplicate work and reduce the team's chance of finding the target.
Messages contain only past neighbor results. No other agent's private history is visible.
Return only the requested JSON."""

CHAIN_PROMPT = """Complete a sequential chain. A level requires the exact pointer key returned
by completing its predecessor. Only the currently available level can be submitted.
Choose one action as JSON: level (integer) and key (string). If no pointer is available,
return level 0 and key empty. Copy an available pointer exactly to complete that level.
All agents act simultaneously; at most one level can complete per global round.
New pointers are issued only after the round ends. Neighbor receipts arrive next round.
Goal: complete the final level before the horizon. Never attempt a locked future level.
Return only the requested JSON."""


def messages_and_schema(view: dict) -> tuple[list[dict], dict]:
    if view["task"] == "needle":
        prompt = NEEDLE_HIGH_V2_PROMPT if view.get("policy_variant") == "needle_high_v2" else NEEDLE_PROMPT
        properties = {"candidates": {"type": "array", "minItems": view["batch_size"],
                      "maxItems": view["batch_size"], "items": {"type": "integer"}}}
    else:
        prompt = CHAIN_PROMPT
        properties = {"level": {"type": "integer"}, "key": {"type": "string"}}
    schema = {"type": "object", "additionalProperties": False,
              "required": list(properties), "properties": properties}
    return ([{"role": "system", "content": prompt},
             {"role": "user", "content": json.dumps(view, sort_keys=True)}], schema)


class ReferencePolicy:
    kind = "deterministic"

    async def decide(self, view: dict, context: DecisionContext) -> dict:
        if view["task"] == "needle":
            action = {"candidates": view["scheduled_candidates"][:]}
        else:
            node = view["available_pointer"]
            action = {"level": node.get("level", 0), "key": node.get("key", "")}
        return {"action": action, "api_error": False, "behavior_error": False}


class NanoPolicy:
    """No action repair or deterministic fallback; errors remain observations.

    Persist successful HTTP responses before parsing/validating, and persist
    terminal API failures as decisions. Restart never resamples a saved call.
    An interrupted in-flight request with no saved response is necessarily
    ambiguous at the provider boundary; no client can promise exactly-once
    remote execution without provider-supported idempotency.
    """
    kind = "llm"

    def __init__(self, backend: LLMBackend, store: RunStore):
        self.backend, self.store = backend, store
        (store.path / "calls").mkdir(exist_ok=True)
        (store.path / "decisions").mkdir(exist_ok=True)
        backend.request_sink = store.sink("requests")
        backend.response_sink = self._save_response
        self._http_index = sum(r.get("kind") == "http_started" for r in iter_jsonl(store.path / "events.jsonl"))
        # Transport timing excludes time waiting for the backend semaphore.
        backend._client.event_hooks = {"request": [self._request_hook], "response": [self._response_hook]}

    async def _request_hook(self, request):
        self._http_index += 1
        request.extensions["nc_attempt"] = self._http_index
        request.extensions["nc_started"] = time.monotonic()
        self.store.append("events", {"kind": "http_started", "attempt": self._http_index, "at": utc_now()})

    async def _response_hook(self, response):
        await response.aread()
        self.store.append("events", {"kind": "http_completed",
                          "attempt": response.request.extensions["nc_attempt"],
                          "status_code": response.status_code,
                          "latency_s": time.monotonic() - response.request.extensions["nc_started"]})

    def _save_response(self, record: dict) -> None:
        self.store.write_json(f"calls/{record['cache_key']}.json", record)
        self.store.append("responses", record)

    async def decide(self, view: dict, context: DecisionContext) -> dict:
        messages, schema = messages_and_schema(view)
        digest = fingerprint({"messages": messages, "schema": schema})
        key = self.backend.cache_key(context, batch=0, attempt=0,
                                     schema_name="needle_chain_v1", digest=digest)
        saved = self.store.read_json(f"decisions/{key}.json")
        if saved is not None:
            return saved
        record = self.store.read_json(f"calls/{key}.json")
        error = None
        if record is None:
            try:
                record = await self.backend.complete(messages=messages, schema=schema,
                    schema_name="needle_chain_v1", cache_key=key, metadata=asdict(context))
            except (PolicyFailure, IncompleteOutput) as exc:
                # The provider's error body may echo secrets: never store it.
                record = self.store.read_json(f"calls/{key}.json")
                match = re.search(r"HTTP (\d{3})", str(exc))
                error = {"type": type(exc).__name__, "http_status": int(match[1]) if match else None}
        result = {"action": {}, "api_error": record is None, "behavior_error": False,
                  "cache_key": key, "error": error}
        if record is not None:
            try:
                if record.get("finish_reason") != "stop" or record.get("refusal"):
                    raise ValueError("incomplete or refused")
                action = json.loads(record["content"])
                if not isinstance(action, dict):
                    raise ValueError("not an object")
                result["action"] = action
            except (ValueError, TypeError, KeyError):
                result["behavior_error"] = True
        self.store.write_json(f"decisions/{key}.json", result)
        return result
