"""Offline replay of recorded LLM decisions.

The replay backend rebuilds exactly the same prompt and cache key as the live
run.  A missing key means the visible input changed, so replay refuses to
continue rather than silently re-deciding.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from ..schemas import LLMConfig, PolicyFailure
from .openai_policy import LLMBackend, LLMPolicy, Usage


class ReplayBackend:
    """Drop-in stand-in for :class:`LLMBackend` that never touches the network."""

    def __init__(
        self,
        config: LLMConfig,
        records: dict[str, dict],
        source_run_id: str,
        *,
        alternate_run_ids: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.usage = Usage()
        self.responses = records
        self.source_run_id = source_run_id
        # A sharded run wrote its keys under one run id per task seed and the
        # merged directory keeps only the umbrella id, so replaying a merged
        # run has to try the shard ids too.  Nothing else about the key
        # changes, so a hit is still an exact match on the visible input.
        self.alternate_run_ids = tuple(alternate_run_ids)
        self.missing: list[str] = []

    async def aclose(self) -> None:
        return None

    def cache_key(self, context, **kwargs) -> str:
        """Key against the recorded run, not the replay run."""
        original = replace(context, run_id=self.source_run_id)
        key = LLMBackend.cache_key(self, original, **kwargs)
        if key in self.responses:
            return key
        for run_id in self.alternate_run_ids:
            candidate = LLMBackend.cache_key(
                self, replace(context, run_id=run_id), **kwargs
            )
            if candidate in self.responses:
                return candidate
        return key

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        cache_key: str,
        metadata: dict[str, Any],
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self.responses.get(cache_key)
        if record is None:
            self.missing.append(cache_key)
            raise PolicyFailure(
                "no recorded response for this request; the visible input or the "
                f"schedule changed (cache_key={cache_key[:16]}..., {metadata})"
            )
        self.usage.requests += 1
        self.usage.attempts += 1
        self.usage.cache_hits += 1
        usage = record.get("usage") or {}
        self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
        if record.get("refusal"):
            self.usage.refusals += 1
            raise PolicyFailure(f"recorded refusal: {record['refusal']!r}")
        return record


def load_recorded_responses(run_dir: Path) -> dict[str, dict]:
    path = Path(run_dir) / "responses.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no responses.jsonl in {run_dir}")
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records[payload["cache_key"]] = payload
    return records


def recorded_model(records: dict[str, dict]) -> str:
    """The model the recorded run actually answered with.

    Older merged manifests have no ``resolved_model``, and the cache key
    hashes the model, so without this a replay of such a run misses every
    key.  The responses themselves carry the name the server reported.
    """
    for record in records.values():
        name = record.get("model")
        if name:
            return str(name)
    return ""


def shard_run_ids(run_dir: Path) -> tuple[str, ...]:
    """The per-seed run ids a merged sharded run was actually written under."""
    shards = Path(run_dir) / "shards"
    if not shards.is_dir():
        return ()
    ids = []
    for child in sorted(shards.iterdir()):
        manifest = child / "manifest.json"
        if manifest.is_file():
            try:
                ids.append(json.loads(manifest.read_text(encoding="utf-8"))["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return tuple(ids)


class ReplayPolicy(LLMPolicy):
    kind = "replay"
