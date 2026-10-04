"""Request dialects: api_style="vllm" (default) vs "openai" (hosted OpenAI API)."""

import asyncio
import json
import unittest
from dataclasses import replace

import httpx

from src.agent_system.schemas import PolicyFailure
from src.agent_system.policies.openai_policy import LLMBackend, branch_schema, communication_schema
from src.agent_system.schemas import Hypothesis, LLMConfig, ParentSlot

SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
REPLY = {"id": "r", "model": "m", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}


def sent_body(config: LLMConfig, statuses: tuple[int, ...] = (200,)) -> tuple[dict, int]:
    """Run one complete() against a mock server; return the last request body and the call count."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        status = statuses[min(len(bodies), len(statuses)) - 1]
        return httpx.Response(status, json=REPLY if status == 200 else {"error": "busy"})

    async def run() -> None:
        backend = LLMBackend(config, "key")
        backend._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
        try:
            await backend.complete(messages=[{"role": "user", "content": "hi"}], schema=SCHEMA,
                                   schema_name="s", cache_key="k", metadata={})
        finally:
            await backend.aclose()

    asyncio.run(run())
    return bodies[-1], len(bodies)


class ApiStyleTests(unittest.TestCase):
    base = LLMConfig(model="m", base_url="http://test", max_output_tokens=3072)

    def test_vllm_body_unchanged(self) -> None:
        body, _ = sent_body(self.base)
        self.assertEqual(body["max_tokens"], 3072)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("reasoning_effort", body)

    def test_openai_body(self) -> None:
        config = replace(self.base, api_style="openai", reasoning_effort="none", temperature=None)
        body, _ = sent_body(config)
        self.assertEqual(body["max_completion_tokens"], 3072)
        self.assertEqual(body["reasoning_effort"], "none")
        for key in ("max_tokens", "temperature", "chat_template_kwargs"):
            self.assertNotIn(key, body)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])

    def test_rate_limit_is_retried(self) -> None:
        config = replace(self.base, transport_retries=1)
        original = asyncio.sleep

        async def no_wait(_seconds: float) -> None:
            await original(0)

        asyncio.sleep = no_wait  # skip the real backoff
        try:
            _, calls = sent_body(config, statuses=(429, 200))
        finally:
            asyncio.sleep = original
        self.assertEqual(calls, 2)

    def test_bad_request_is_not_retried(self) -> None:
        config = replace(self.base, transport_retries=8)
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, json={"error": {"message": "Invalid schema"}})

        async def run() -> None:
            backend = LLMBackend(config, "key")
            backend._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
            try:
                await backend.complete(messages=[{"role": "user", "content": "hi"}], schema=SCHEMA,
                                       schema_name="s", cache_key="k", metadata={})
            finally:
                await backend.aclose()

        with self.assertRaisesRegex(PolicyFailure, "HTTP 400"):
            asyncio.run(run())
        self.assertEqual(len(calls), 1)

    def test_empty_communication_schema(self) -> None:
        # vLLM keeps the bare item (unchanged request bodies); OpenAI strict mode
        # needs every object closed with additionalProperties: false.
        self.assertEqual(communication_schema([])["properties"]["reports"]["items"], {"type": "object"})
        item = communication_schema([], strict_objects=True)["properties"]["reports"]["items"]
        self.assertEqual(item["additionalProperties"], False)
        self.assertEqual(communication_schema(["e1"], strict_objects=True), communication_schema(["e1"]))

    def test_branch_schema_parent_without_legal_additions(self) -> None:
        # xgrammar rejects an empty enum (vLLM HTTP 500); a parent with no legal
        # addition gets a plain string item, unreachable given maxItems=0.
        full = ParentSlot(hypothesis=Hypothesis(((0, 0),)), legal_additions=((1, 0), (2, 1)), required=2)
        stuck = ParentSlot(hypothesis=Hypothesis(((0, 1),)), legal_additions=(), required=0)
        updates = branch_schema([full, stuck])["properties"]["updates"]["properties"]
        self.assertEqual(updates[stuck.hypothesis.hid], {"type": "array", "minItems": 0, "maxItems": 0, "items": {"type": "string"}})
        self.assertEqual(updates[full.hypothesis.hid]["items"]["enum"], ["c1v0", "c2v1"])
        self.assertNotIn('"enum": []', json.dumps(branch_schema([full, stuck])))


if __name__ == "__main__":
    unittest.main()
