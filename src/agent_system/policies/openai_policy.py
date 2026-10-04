"""OpenAI-compatible LLM policy.

The backend speaks the ``/v1/chat/completions`` dialect with JSON-schema
structured outputs, which both the OpenAI service and a local vLLM server
implement; the plan's Responses-API sketch is narrowed to this dialect so the
same code path drives the local model.

Hard rules kept here:

* no silent fallback -- an illegal action that survives repair raises
  :class:`PolicyFailure`;
* a repair prompt may only restate the violated constraint, never the truth;
* every attempt, including transport retries, is counted;
* the response cache key carries run/episode/agent/round/phase/slot/attempt,
  the model, the schema version, every prompt switch, the sampling settings
  and a digest of the full visible input, so two independent repeats never
  share a sample.

The prompt gives each agent the context it needs to stop duplicating its
neighbours' work.  Every part of it is an independent switch on
:class:`~src.agent_system.schemas.LLMConfig`; the defaults are the final
experiment setting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
import re
import statistics
import time
from typing import Any, Callable, Sequence

import httpx

from ...scaling_model import derived_rng
from ..schemas import (
    ActionError,
    IncompleteOutput,
    AgentView,
    BranchDecision,
    BudgetPaused,
    CommunicationDecision,
    ExperimentCandidate,
    ExperimentCondition,
    ExperimentDecision,
    Hypothesis,
    LabelRecord,
    LLMConfig,
    OwnExperimentSummary,
    Pair,
    ParentSlot,
    ParentUpdate,
    PolicyFailure,
    PrivateObservation,
    ReportItem,
    pair_id,
)
from .base import PROMPT_DIR, DecisionContext, Policy
from .planner import plan_experiment

SCHEMA_VERSION = "agent_v3_json_schema"
DEFAULT_PARENT_BATCH = 16

REASON_MAX_CHARS = 240

# SplitMix64 domains for the two prompt randomisations.  Both are disjoint from
# every domain in src/scaling_model.py and from the protocol knobs in
# runner.py / communication.py / environments/tcas.py, so switching them on
# consumes no draw any other stream would have taken.
SHUFFLE_DOMAIN = 0x2545F4914F6CDD1D
SEED_DOMAIN = 0x9E3779B97F4A7C15
PHASE_CODES = {"branch": 1, "experiment": 2, "prune": 3, "communication": 4}
COMMUNICATION_TEXT_MAX_CHARS = 500
# The two points at which the runner hands pruning to the model.  The slot
# index distinguishes them in the cache key as well as in the prompt.
PRUNE_STAGES = {0: "receive", 1: "own-experiment"}


def branch_schema(
    batch: Sequence[ParentSlot], *, analysis_max_chars: int | None = None
) -> dict[str, Any]:
    """Build the batch's schema so an illegal branch cannot be decoded.

    The parent set, the per-parent addition count and the legal additions are
    all written into the grammar: ``updates`` is an object keyed by parent id
    with ``additionalProperties: false`` and every key required, each value is
    a fixed-length array, and each element is a component label drawn from
    that parent's own ``legal_additions``.  Constrained decoding therefore
    rules out a wrong count, a missing or invented parent, and an illegal
    component, leaving only duplicates inside one parent for the parser to
    catch.  ``updates`` is declared before ``reason`` so a length-limited
    decode loses the commentary rather than the action.

    ``analysis_max_chars`` puts a bounded free-text field *before* the action
    instead, which is what makes the reasoning genuine rather than post hoc:
    constrained decoding emits the fields in the order they are declared.
    """
    properties = {
        slot.hypothesis.hid: {
            "type": "array",
            "minItems": slot.required,
            "maxItems": slot.required,
            # A parent with no legal addition (required == 0) gets a plain
            # string item: unreachable given maxItems=0, and xgrammar rejects
            # an empty enum outright (vLLM HTTP 500).
            "items": (
                {"type": "string", "enum": [pair_id(pair) for pair in slot.legal_additions]}
                if slot.legal_additions
                else {"type": "string"}
            ),
        }
        for slot in batch
    }
    action = {
        "updates": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(properties),
            "properties": properties,
        },
        "reason": {"type": "string", "maxLength": REASON_MAX_CHARS},
    }
    return _with_analysis(
        ["updates", "reason"], action, analysis_max_chars=analysis_max_chars
    )


CHOICE_SEPARATOR = "|"


def choice_label(pair: Pair, parent_id: str) -> str:
    """One testable ``(component, generating parent)`` pair, as one token run."""
    return f"{pair_id(pair)}{CHOICE_SEPARATOR}{parent_id}"


def experiment_schema(
    candidates: Sequence[ExperimentCandidate],
    library: Sequence[ExperimentCondition],
    with_condition: bool,
    *,
    analysis_max_chars: int | None = None,
) -> dict[str, Any]:
    """Build the slot's schema so an illegal experiment cannot be decoded.

    A component and a parent are legal only *together*: the parent has to be
    one that proposed that component.  Two independent enums cannot express
    that, so the offered combinations are flattened into a single ``choice``
    enum, which makes the joint constraint hold by construction.  The
    condition, when the policy picks one, is an enum over the public library.
    ``choice`` comes first and ``reason`` last so a length-limited decode
    loses the commentary rather than the action.  The enum order follows the
    order of ``candidates`` and ``library``, which ``shuffle_candidates``
    permutes per agent and per round.
    """
    choices = [
        choice_label(candidate.pair, parent_id)
        for candidate in candidates
        for parent_id in candidate.parent_ids
    ]
    properties: dict[str, Any] = {"choice": {"type": "string", "enum": choices}}
    required = ["choice"]
    if with_condition:
        properties["experiment_id"] = {
            "type": "string",
            "enum": [condition.experiment_id for condition in library],
        }
        required.append("experiment_id")
    properties["reason"] = {"type": "string", "maxLength": REASON_MAX_CHARS}
    required.append("reason")
    return _with_analysis(required, properties, analysis_max_chars=analysis_max_chars)


def prune_schema(
    hypothesis_ids: Sequence[str], *, analysis_max_chars: int | None = None
) -> dict[str, Any]:
    """Only ids that were offered can be dropped, and at most all of them."""
    action = {
        "drop": {
            "type": "array",
            "maxItems": len(hypothesis_ids),
            "items": {"type": "string", "enum": list(hypothesis_ids)},
        },
        "reason": {"type": "string", "maxLength": REASON_MAX_CHARS},
    }
    return _with_analysis(
        ["drop", "reason"], action, analysis_max_chars=analysis_max_chars
    )


def communication_schema(
    event_ids: Sequence[str],
    *,
    analysis_max_chars: int | None = None,
    strict_objects: bool = False,
) -> dict[str, Any]:
    """The (optional) extra call under ``communication_mode == "free"``.

    ``reports`` may only name one of this agent's own ``event_ids`` this
    round, at most once each; ``claimed_outcome`` is free to disagree with
    what was actually observed, since misreporting is a legal move.  An
    agent with no experiments this round is offered an empty enum and
    ``maxItems: 0``, so it can still send bare text (or nothing).

    ``strict_objects`` (the OpenAI API) spells that unreachable item as a
    closed empty object: OpenAI's strict mode rejects any object schema
    without ``additionalProperties: false`` with HTTP 400. vLLM keeps the
    bare ``{"type": "object"}`` so its request bodies stay unchanged.
    """
    if event_ids:
        item = {
            "type": "object",
            "additionalProperties": False,
            "required": ["event_id", "claimed_outcome"],
            "properties": {
                "event_id": {"type": "string", "enum": list(event_ids)},
                "claimed_outcome": {"type": "string", "enum": ["positive", "negative"]},
            },
        }
    else:
        # Unreachable given maxItems=0; kept schema-valid without an empty
        # enum, which some structured-output backends reject outright.
        item = {"type": "object"}
        if strict_objects:
            item = {"type": "object", "additionalProperties": False, "properties": {}, "required": []}
    action = {
        "reports": {"type": "array", "maxItems": len(event_ids), "items": item},
        "text": {"type": "string", "maxLength": COMMUNICATION_TEXT_MAX_CHARS},
        "reason": {"type": "string", "maxLength": REASON_MAX_CHARS},
    }
    return _with_analysis(
        ["reports", "text", "reason"], action, analysis_max_chars=analysis_max_chars
    )


def _with_analysis(
    required: Sequence[str],
    properties: dict[str, Any],
    *,
    analysis_max_chars: int | None,
) -> dict[str, Any]:
    """Assemble a schema, optionally with a bounded ``analysis`` field first."""
    if analysis_max_chars is None:
        ordered = dict(properties)
        keys = list(required)
    else:
        ordered = {
            "analysis": {"type": "string", "maxLength": int(analysis_max_chars)},
            **properties,
        }
        keys = ["analysis", *required]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": keys,
        "properties": ordered,
    }


@dataclass
class Usage:
    requests: int = 0
    attempts: int = 0
    transport_retries: int = 0
    repairs: int = 0
    refusals: int = 0
    incomplete: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cache_hits: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "llm_requests": self.requests,
            "llm_attempts": self.attempts,
            "llm_transport_retries": self.transport_retries,
            "llm_repairs": self.repairs,
            "llm_refusals": self.refusals,
            "llm_incomplete": self.incomplete,
            "llm_failures": self.failures,
            "llm_prompt_tokens": self.prompt_tokens,
            "llm_completion_tokens": self.completion_tokens,
            "llm_total_tokens": self.prompt_tokens + self.completion_tokens,
            "llm_latency_s": self.latency_s,
            "llm_cache_hits": self.cache_hits,
        }


def request_seed(context: DecisionContext, *, batch: int, attempt: int) -> int:
    """A deterministic sampling seed for exactly one request.

    The episode id is folded in as well as the episode seed, because paired
    episodes at different degrees deliberately share the seed.
    """
    episode = int(hashlib.sha256(context.episode_id.encode()).hexdigest()[:12], 16)
    rng = derived_rng(
        context.seed,
        SEED_DOMAIN,
        episode,
        context.agent_index,
        context.round_number,
        PHASE_CODES.get(context.phase, 0),
        context.slot,
        batch,
        attempt,
    )
    return rng.next() % (1 << 31)


def shuffled(items: Sequence[Any], rng) -> list[Any]:
    """Fisher-Yates off a SplitMix64 stream: reproducible, agent-specific."""
    result = list(items)
    for position in range(len(result) - 1, 0, -1):
        target = rng.below(position + 1)
        result[position], result[target] = result[target], result[position]
    return result


class LLMBackend:
    """One shared client, concurrency limiter, ledger and resource ceiling."""

    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        *,
        request_sink: Callable[[dict], None] | None = None,
        response_sink: Callable[[dict], None] | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("OPENAI_MODEL / policy.llm.model must be set")
        if not config.base_url:
            raise ValueError("OPENAI_BASE_URL / policy.llm.base_url must be set")
        if not api_key:
            raise ValueError("the API key is empty; set it in the environment file")
        self.config = config
        self.usage = Usage()
        self.started = time.monotonic()
        self.semaphore = asyncio.Semaphore(max(1, config.concurrency))
        self.request_sink = request_sink
        self.response_sink = response_sink
        self.responses: dict[str, dict] = {}
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(config.timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _check_ceilings(self) -> None:
        limits = self.config
        if limits.max_requests is not None and self.usage.requests >= limits.max_requests:
            raise BudgetPaused(f"request ceiling {limits.max_requests} reached")
        total = self.usage.prompt_tokens + self.usage.completion_tokens
        if limits.max_total_tokens is not None and total >= limits.max_total_tokens:
            raise BudgetPaused(f"token ceiling {limits.max_total_tokens} reached")
        if limits.max_wall_seconds is not None:
            if time.monotonic() - self.started >= limits.max_wall_seconds:
                raise BudgetPaused(f"wall-clock ceiling {limits.max_wall_seconds}s reached")

    def cache_key(
        self,
        context: DecisionContext,
        *,
        batch: int,
        attempt: int,
        schema_name: str,
        digest: str,
    ) -> str:
        payload = "|".join(
            [
                context.run_id,
                context.episode_id,
                str(context.agent_index),
                str(context.round_number),
                context.phase,
                str(context.slot),
                str(batch),
                str(attempt),
                self.config.model,
                SCHEMA_VERSION,
                schema_name,
                sampling_payload(self.config, context, batch, attempt),
                digest,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

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
        self._check_ceilings()
        thinking = self.config.enable_thinking or self.config.reasoning == "thinking"
        max_tokens = self.config.max_output_tokens
        if self.config.reasoning == "thinking":
            max_tokens = max(max_tokens, self.config.thinking_max_output_tokens)
        openai_api = self.config.api_style == "openai"
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            ("max_completion_tokens" if openai_api else "max_tokens"): max_tokens,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        if self.config.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        if openai_api:
            if self.config.reasoning_effort is not None:
                body["reasoning_effort"] = self.config.reasoning_effort
        elif not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.config.reasoning == "thinking":
            body["chat_template_kwargs"] = {"enable_thinking": True}
        if extra_body:
            body.update(extra_body)

        if self.request_sink is not None:
            self.request_sink(
                {
                    "cache_key": cache_key,
                    "model": self.config.model,
                    "schema_name": schema_name,
                    "messages": messages,
                    **metadata,
                }
            )

        last_error: Exception | None = None
        for transport_attempt in range(self.config.transport_retries + 1):
            self.usage.attempts += 1
            if transport_attempt:
                self.usage.transport_retries += 1
                # Back off before retrying so a rate limit (HTTP 429) or a
                # transient server error has time to clear.
                await asyncio.sleep(min(60.0, 2.0 ** transport_attempt))
            started = time.monotonic()
            try:
                async with self.semaphore:
                    response = await self._client.post("/chat/completions", json=body)
                elapsed = time.monotonic() - started
                self.usage.latency_s += elapsed
                if response.status_code >= 400:
                    last_error = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:400]}"
                    )
                    # A malformed request (400, 401, 404, 422, ...) fails the
                    # same way every time; only timeouts, conflicts, rate
                    # limits and server errors are worth retrying.
                    if response.status_code < 500 and response.status_code not in (408, 409, 429):
                        break
                    continue
                payload = response.json()
            except (httpx.HTTPError, ValueError) as error:  # transport or decoding
                self.usage.latency_s += time.monotonic() - started
                last_error = error
                continue

            self.usage.requests += 1
            usage = payload.get("usage") or {}
            self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            record = {
                "cache_key": cache_key,
                "response_id": payload.get("id"),
                "model": payload.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "content": message.get("content"),
                "refusal": message.get("refusal"),
                "reasoning": message.get("reasoning"),
                "usage": usage,
                "latency_s": elapsed,
                **metadata,
            }
            self.responses[cache_key] = record
            if self.response_sink is not None:
                self.response_sink(record)
            if message.get("refusal"):
                self.usage.refusals += 1
                raise PolicyFailure(f"model refused: {message['refusal']!r}")
            if choice.get("finish_reason") == "length":
                self.usage.incomplete += 1
                raise IncompleteOutput(
                    "the answer was cut off before the JSON was finished; the "
                    "commentary was too long"
                )
            return record

        self.usage.failures += 1
        # httpx timeout exceptions stringify to "", so name the class as well.
        detail = (
            f"{type(last_error).__name__}: {last_error}" if last_error else "no error recorded"
        )
        raise PolicyFailure(f"transport failed after retries: {detail}")


def sampling_payload(
    config: LLMConfig, context: DecisionContext, batch: int, attempt: int
) -> str:
    """The sampling and prompt settings the cache key hashes.

    Every prompt switch is included whatever its value, so an ablation arm can
    never collide with the default setting.
    """
    payload: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_output_tokens": config.max_output_tokens,
        "enable_thinking": config.enable_thinking,
    }
    payload.update(prompt_key_extras(config))
    # Only a non-default dialect enters the key, so every existing vLLM cache
    # key (and the replay of runs recorded with it) is unchanged.
    if config.api_style != "vllm":
        payload["api_style"] = config.api_style
        payload["reasoning_effort"] = config.reasoning_effort
    if config.per_agent_seed:
        payload["request_seed"] = request_seed(context, batch=batch, attempt=attempt)
    return json.dumps(payload, sort_keys=True)


def prompt_key_extras(config: LLMConfig) -> dict[str, Any]:
    """Every prompt switch, for the cache key and the run manifest."""
    extras: dict[str, Any] = {
        name: getattr(config, name)
        for name in (
            "context_identity",
            "team_objective",
            "evidence_attribution",
            "shuffle_candidates",
            "per_agent_seed",
            "reasoning",
            "physics_observations",
            "communication_mode",
        )
    }
    if config.reasoning == "reason_first":
        extras["reasoning_max_chars"] = config.reasoning_max_chars
    if config.reasoning == "thinking":
        extras["thinking_max_output_tokens"] = config.thinking_max_output_tokens
    return extras


# --------------------------------------------------------------------------
# Prompt text blocks
# --------------------------------------------------------------------------

TEAM_OBJECTIVE_TEXT = """TEAM OBJECTIVE
The team wins when the TEAM has ruled out every wrong hypothesis in the fewest
rounds, not when you personally look clever. Your neighbours are choosing at
this very moment from lists built exactly like yours, so a test that one of
them is likely to run as well buys the team almost nothing: the same label
arrives twice and one round of one agent is wasted.

A test is informative for you when all three hold:
  1. the component appears in many of the hypotheses you still hold, so its
     label settles many of them at once;
  2. you cannot already infer its label from the labels you hold;
  3. its family still has more than one value allowed -- a family already
     narrowed to a single value is determined, and testing it removes nothing.
A positive label ("in the mechanism") in a family with several values allowed
also removes every sibling value of that family at once, so it is worth more
than a negative one.
"""

BRANCH_DEVICE_TEXT = """SYMMETRY BREAKING (arithmetic, not taste)
Two agents that reason identically pick identically, which is the failure to
avoid. So: for each parent, rank its legal_additions by how informative they
are under the three tests above, keeping the listed order for ties, and let k
be the length of that ranking. You are agent {agent_index}. Take the entries
starting at position ({agent_index} mod k), counting from 0 and wrapping
round, until you have the required number. Do not start at position 0 unless
the arithmetic says so.
"""

EXPERIMENT_DEVICE_TEXT = """SYMMETRY BREAKING (arithmetic, not taste)
Two agents that reason identically pick identically, which is the failure to
avoid. So: rank the candidates you consider informative under the three tests
above, keeping the listed order for ties, and let k be the length of that
ranking. You are agent {agent_index}. Take the candidate at position
({agent_index} mod k), counting from 0. Do not take position 0 unless the
arithmetic says so.
"""

# The C0/C1/I1 experiment's two short objective texts.  Neither states any
# search or division-of-labour strategy and neither adds the device
# symmetry breaker: they say only what the agent is trying to improve.
TEAM_OBJECTIVE_SIMPLE_TEXT = (
    "OBJECTIVE\n"
    "Your goal is to improve the team's performance at finding the hidden "
    "mechanism. The team's result depends on what its agents achieve "
    "together.\n"
)
TEAM_OBJECTIVE_INDIVIDUAL_TEXT = (
    "OBJECTIVE\n"
    "Your goal is to improve your own performance at finding the hidden "
    "mechanism. Your result depends on what you achieve yourself.\n"
)

OPENING_LINE_TEAM = "You are one scientist in a team that is identifying a hidden mechanism."
OPENING_LINE_INDIVIDUAL = "You are one scientist identifying a hidden mechanism."

# team_objective in {"simple", "individual"} replaces the old PROTOCOL text
# (below) with one of these two, chosen by communication_mode.  The wording
# is identical between "simple" and "individual" -- only the objective block
# and the opening line differ between C1 and I1.
PROTOCOL_FIXED_NEW_TEXT = (
    "PROTOCOL\n"
    "All {agent_count} agents act concurrently. Each result from an "
    "experiment you run this\n"
    "round is sent automatically and truthfully to your direct neighbours, "
    "arriving at\n"
    "the start of the next round. One hop, never forwarded. Nothing you "
    "decide is\n"
    "visible to anyone else this round.\n\n"
)
PROTOCOL_FREE_NEW_TEXT = (
    "PROTOCOL\n"
    "All {agent_count} agents act concurrently. At the end of a round each "
    "agent can choose\n"
    "which of its own experimental results to report, can add text, or can "
    "send\n"
    "nothing. Reported outcomes may be inaccurate. A message reaches exactly "
    "your\n"
    "{degree} neighbour(s) at the start of the next round and is never "
    "forwarded.\n"
    "Nothing you decide is visible to another agent this round.\n\n"
)


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_allowed_values(view: AgentView) -> str:
    """Allowed values without an empty list anywhere.

    A family down to one value is *determined*, which a one-element JSON list
    would bury, and a family with no value left is named once at the end
    instead of as ``[]``.
    """
    parts: list[str] = []
    exhausted: list[str] = []
    for family, values in sorted(view.allowed_values.items()):
        if not values:
            exhausted.append(f"c{family}")
        elif len(values) == 1:
            parts.append(f"c{family}: determined = v{values[0]}")
        else:
            parts.append(f"c{family}: " + ",".join(f"v{value}" for value in values))
    text = "; ".join(parts) if parts else "nothing left to determine"
    if exhausted:
        text += f"; no value remains for {', '.join(exhausted)}"
    return text


def render_labels(items: Sequence[LabelRecord]) -> str:
    if not items:
        return "none"
    rendered = []
    for item in items:
        where = (
            f"from agent {item.sender}, round {item.round_number}"
            if item.sender is not None
            else f"round {item.round_number}"
        )
        rendered.append(f"{pair_id(item.pair)}={item.outcome} ({where})")
    return "; ".join(rendered)


def deduplicate_observations(
    observations: Sequence[PrivateObservation],
) -> list[PrivateObservation]:
    """One trajectory per experiment_id; the physics prompts repeat them."""
    seen: set[str] = set()
    kept: list[PrivateObservation] = []
    for observation in observations:
        key = observation.experiment_id or observation.observation_id
        if key in seen:
            continue
        seen.add(key)
        kept.append(observation)
    return kept


def summarise_observation(observation: PrivateObservation) -> dict[str, Any]:
    """A few statistics per variable instead of ~100 raw floats per trajectory."""
    payload: dict[str, Any] = {
        "observation_id": observation.observation_id,
        "kind": observation.kind,
        "experiment_id": observation.experiment_id,
        "settings": {k: round(float(v), 4) for k, v in observation.settings.items()},
        "samples": len(observation.times),
    }
    if observation.times:
        payload["t_range"] = [
            round(float(observation.times[0]), 4),
            round(float(observation.times[-1]), 4),
        ]
    for position, name in enumerate(observation.variables):
        series = [float(state[position]) for state in observation.states if len(state) > position]
        if not series:
            continue
        crossings = sum(
            1
            for left, right in zip(series, series[1:])
            if (left < 0) != (right < 0)
        )
        payload[name] = {
            "first": round(series[0], 4),
            "last": round(series[-1], 4),
            "min": round(min(series), 4),
            "max": round(max(series), 4),
            "mean": round(statistics.fmean(series), 4),
            "std": round(statistics.pstdev(series), 4) if len(series) > 1 else 0.0,
            "sign_changes": crossings,
        }
    return payload


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


class LLMPolicy(Policy):
    """One agent's decision maker.  State is local; the backend is shared."""

    uses_llm = True

    def __init__(
        self,
        backend: LLMBackend,
        *,
        kind: str = "llm_full",
        planner: str = "disagreement_v1",
        planner_max_conditions: int | None = None,
        planner_max_parents: int | None = None,
        parent_batch: int = DEFAULT_PARENT_BATCH,
        uniform_fallback_pair: Callable[..., Pair] | None = None,
        behaves_as: str | None = None,
        prune_batch: int = 64,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.backend = backend
        self.planner = planner
        self.planner_max_conditions = planner_max_conditions
        self.planner_max_parents = planner_max_parents
        self.parent_batch = parent_batch
        self.prune_batch = max(1, prune_batch)
        behaviour = behaves_as or kind
        self.behaviour = behaviour
        self.choose_pair = behaviour in ("llm_branch_test", "llm_full", "llm_prune")
        self.choose_condition = behaviour in ("llm_full", "llm_prune")
        # The runner asks for this: when it is set the runner stops calling
        # EvidenceStore.prune at the two prune points and asks the policy.
        self.prunes = behaviour == "llm_prune"
        # Every block that is constant for the run or the agent -- the
        # environment note, the component library and the experiment library
        # -- comes ahead of everything that changes per round, so the server's
        # prefix cache covers most of each prompt.
        self._branch_template = (PROMPT_DIR / "branch.txt").read_text(encoding="utf-8")
        self._experiment_template = (PROMPT_DIR / "experiment.txt").read_text(
            encoding="utf-8"
        )
        self._prune_template = (PROMPT_DIR / "prune.txt").read_text(encoding="utf-8")
        self._communication_template = (PROMPT_DIR / "communication.txt").read_text(
            encoding="utf-8"
        )
        self._uniform_pair = uniform_fallback_pair

    def usage(self) -> dict[str, object]:
        return {}  # aggregated once from the shared backend

    # -- shared plumbing ----------------------------------------------------
    @property
    def _analysis_chars(self) -> int | None:
        config = self.backend.config
        if config.reasoning == "reason_first":
            return config.reasoning_max_chars
        return None

    def _extra_body(
        self, context: DecisionContext, *, batch: int, attempt: int
    ) -> dict[str, Any]:
        if not self.backend.config.per_agent_seed:
            return {}
        return {"seed": request_seed(context, batch=batch, attempt=attempt)}

    async def _complete(self, *, context, batch, attempt, **kwargs) -> dict[str, Any]:
        """Call the backend, passing a body extension only when there is one.

        Not passing an empty extension keeps the plain call signature, which
        matters for the stub backends the tests use.
        """
        extra = self._extra_body(context, batch=batch, attempt=attempt)
        if extra:
            return await self.backend.complete(extra_body=extra, **kwargs)
        return await self.backend.complete(**kwargs)

    def _shuffle_rng(self, context: DecisionContext, *labels: int):
        return derived_rng(
            context.seed,
            SHUFFLE_DOMAIN,
            int(hashlib.sha256(context.episode_id.encode()).hexdigest()[:12], 16),
            context.agent_index,
            context.round_number,
            PHASE_CODES.get(context.phase, 0),
            *labels,
        )

    # -- branch ------------------------------------------------------------
    async def propose(
        self,
        view: AgentView,
        parents: Sequence[ParentSlot],
        context: DecisionContext,
    ) -> BranchDecision:
        ordered = sorted(parents, key=lambda slot: slot.hypothesis.hid)
        batches = [
            ordered[start : start + self.parent_batch]
            for start in range(0, len(ordered), self.parent_batch)
        ]
        updates: list[ParentUpdate] = []
        for batch_index, batch in enumerate(batches):
            updates.extend(await self._branch_batch(view, batch, context, batch_index))
        return BranchDecision(
            updates=tuple(updates), reason=f"llm branch over {len(batches)} batch(es)"
        )

    def _branch_order(
        self, batch: Sequence[ParentSlot], context: DecisionContext, batch_index: int
    ) -> list[ParentSlot]:
        """Permute each parent's legal additions, per agent and per round.

        The permutation only reorders a list whose *set* is fixed by the
        evidence, so it cannot make an illegal action legal; it removes the
        position bias that made every agent pick the same first entry.
        """
        if not self.backend.config.shuffle_candidates:
            return list(batch)
        ordered: list[ParentSlot] = []
        for position, slot in enumerate(batch):
            rng = self._shuffle_rng(context, batch_index, position)
            ordered.append(
                replace(slot, legal_additions=tuple(shuffled(slot.legal_additions, rng)))
            )
        return ordered

    async def _branch_batch(
        self,
        view: AgentView,
        batch: Sequence[ParentSlot],
        context: DecisionContext,
        batch_index: int,
    ) -> list[ParentUpdate]:
        presented = self._branch_order(batch, context, batch_index)
        prompt = self._branch_prompt(view, presented)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        schema = branch_schema(presented, analysis_max_chars=self._analysis_chars)
        complaint: str | None = None
        for attempt in range(self.backend.config.repair_attempts + 1):
            messages = [
                {"role": "system", "content": "You answer with JSON only."},
                {"role": "user", "content": prompt},
            ]
            if complaint is not None:
                messages.append({"role": "user", "content": complaint})
                self.backend.usage.repairs += 1
            key = self.backend.cache_key(
                context,
                batch=batch_index,
                attempt=attempt,
                schema_name="branch_decision",
                digest=digest,
            )
            try:
                record = await self._complete(
                    context=context,
                    batch=batch_index,
                    attempt=attempt,
                    messages=messages,
                    schema=schema,
                    schema_name="branch_decision",
                    cache_key=key,
                    metadata={
                        "agent": context.agent_index,
                        "round": context.round_number,
                        "phase": "branch",
                        "batch": batch_index,
                        "attempt": attempt,
                        "episode": context.episode_id,
                    },
                )
                return _parse_branch(record["content"], presented)
            except ActionError as error:
                complaint = (
                    "Your previous answer broke a hard constraint: "
                    f"{error}. Re-answer for the same parents, using only the "
                    "legal_additions already given, with the exact required count, "
                    "and keep 'reason' under 20 words."
                )
        raise PolicyFailure(f"branch decision still illegal after repair: {complaint}")

    # -- experiment ---------------------------------------------------------
    async def choose_experiment(
        self,
        view: AgentView,
        candidates: Sequence[ExperimentCandidate],
        context: DecisionContext,
    ) -> ExperimentDecision:
        if not self.choose_pair:
            if self._uniform_pair is None:
                raise PolicyFailure(f"{self.kind} needs a uniform pair sampler")
            pair = self._uniform_pair(view, candidates, context)
            chosen = next(item for item in candidates if item.pair == pair)
            return ExperimentDecision(
                pair=chosen.pair,
                parent_id=chosen.parent_ids[0],
                experiment_id=self._plan(view, chosen, context),
                reason="uniform pair draw; llm branch only",
            )

        offered, library = self._experiment_order(view, candidates, context)
        prompt = self._experiment_prompt(view, offered, library)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        with_condition = self.choose_condition and bool(view.experiment_library)
        schema = experiment_schema(
            offered, library, with_condition, analysis_max_chars=self._analysis_chars
        )
        complaint: str | None = None
        for attempt in range(self.backend.config.repair_attempts + 1):
            messages = [
                {"role": "system", "content": "You answer with JSON only."},
                {"role": "user", "content": prompt},
            ]
            if complaint is not None:
                messages.append({"role": "user", "content": complaint})
                self.backend.usage.repairs += 1
            key = self.backend.cache_key(
                context,
                batch=0,
                attempt=attempt,
                schema_name="experiment_decision",
                digest=digest,
            )
            try:
                record = await self._complete(
                    context=context,
                    batch=0,
                    attempt=attempt,
                    messages=messages,
                    schema=schema,
                    schema_name="experiment_decision",
                    cache_key=key,
                    metadata={
                        "agent": context.agent_index,
                        "round": context.round_number,
                        "phase": "experiment",
                        "slot": context.slot,
                        "attempt": attempt,
                        "episode": context.episode_id,
                    },
                )
                decision = _parse_experiment(record["content"], offered)
            except ActionError as error:
                complaint = (
                    "Your previous answer broke a hard constraint: "
                    f"{error}. Re-answer choosing only from the candidate list given, "
                    "and keep 'reason' under 20 words."
                )
                continue
            if self.choose_condition and view.experiment_library:
                allowed = {item.experiment_id for item in view.experiment_library}
                if decision.experiment_id not in allowed:
                    complaint = (
                        "Your previous answer used an experiment_id outside the public "
                        "library. Re-answer with one of the listed experiment_id values."
                    )
                    continue
                return decision
            return ExperimentDecision(
                pair=decision.pair,
                parent_id=decision.parent_id,
                experiment_id=self._plan(
                    view, next(c for c in candidates if c.pair == decision.pair), context
                ),
                reason=decision.reason,
            )
        raise PolicyFailure(f"experiment decision still illegal after repair: {complaint}")

    def _experiment_order(
        self,
        view: AgentView,
        candidates: Sequence[ExperimentCandidate],
        context: DecisionContext,
    ) -> tuple[list[ExperimentCandidate], list[ExperimentCondition]]:
        library = list(view.experiment_library)
        if not self.backend.config.shuffle_candidates:
            return list(candidates), library
        rng = self._shuffle_rng(context, context.slot, 0)
        ordered = [
            replace(
                candidate,
                parent_ids=tuple(
                    shuffled(
                        candidate.parent_ids,
                        self._shuffle_rng(context, context.slot, 1 + index),
                    )
                ),
            )
            for index, candidate in enumerate(candidates)
        ]
        ordered = shuffled(ordered, rng)
        library = shuffled(library, self._shuffle_rng(context, context.slot, 9973))
        return ordered, library

    def _plan(
        self, view: AgentView, candidate: ExperimentCandidate, context: DecisionContext
    ) -> str | None:
        if not view.experiment_library:
            return None
        parents = [
            slot.hypothesis for slot in view.parents if slot.hypothesis.hid in candidate.parent_ids
        ]
        return plan_experiment(
            planner=self.planner,
            pair=candidate.pair,
            parents=parents,
            library=view.experiment_library,
            predictor=self.predictor,
            seed=context.seed,
            round_number=view.round_number,
            agent_index=context.agent_index,
            slot=context.slot,
            max_conditions=self.planner_max_conditions,
            max_parents=self.planner_max_parents,
        )

    # -- prune (llm_prune only) ----------------------------------------------
    async def prune(
        self,
        view: AgentView,
        hypotheses: Sequence[Hypothesis],
        context: DecisionContext,
    ) -> tuple[str, ...]:
        """Ask which hypotheses the new evidence has refuted.

        Returns the ids the model wants dropped.  Whether a drop is applied is
        the runner's decision (``llm_prune_apply``); this method never looks
        at whether a hypothesis really is refuted, so the shadow score it is
        graded against is independent of it.
        """
        if not hypotheses:
            return ()
        batches = [
            list(hypotheses)[start : start + self.prune_batch]
            for start in range(0, len(hypotheses), self.prune_batch)
        ]
        dropped: list[str] = []
        for batch_index, batch in enumerate(batches):
            dropped.extend(await self._prune_batch(view, batch, context, batch_index))
        return tuple(dict.fromkeys(dropped))

    async def _prune_batch(
        self,
        view: AgentView,
        batch: Sequence[Hypothesis],
        context: DecisionContext,
        batch_index: int,
    ) -> list[str]:
        prompt = self._prune_prompt(view, batch, PRUNE_STAGES.get(context.slot, "prune"))
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        ids = [item.hid for item in batch]
        schema = prune_schema(ids, analysis_max_chars=self._analysis_chars)
        complaint: str | None = None
        for attempt in range(self.backend.config.repair_attempts + 1):
            messages = [
                {"role": "system", "content": "You answer with JSON only."},
                {"role": "user", "content": prompt},
            ]
            if complaint is not None:
                messages.append({"role": "user", "content": complaint})
                self.backend.usage.repairs += 1
            key = self.backend.cache_key(
                context,
                batch=batch_index,
                attempt=attempt,
                schema_name="prune_decision",
                digest=digest,
            )
            try:
                record = await self._complete(
                    context=context,
                    batch=batch_index,
                    attempt=attempt,
                    messages=messages,
                    schema=schema,
                    schema_name="prune_decision",
                    cache_key=key,
                    metadata={
                        "agent": context.agent_index,
                        "round": context.round_number,
                        "phase": "prune",
                        "slot": context.slot,
                        "batch": batch_index,
                        "attempt": attempt,
                        "episode": context.episode_id,
                    },
                )
                return _parse_prune(record["content"], ids)
            except ActionError as error:
                complaint = (
                    "Your previous answer broke a hard constraint: "
                    f"{error}. Re-answer with 'drop' containing only ids from the "
                    "list given, and keep 'reason' under 20 words."
                )
        raise PolicyFailure(f"prune decision still illegal after repair: {complaint}")

    # -- communication (communication_mode == "free" only) -------------------
    async def communicate(
        self, view: AgentView, context: DecisionContext
    ) -> CommunicationDecision:
        """The extra, optional call: which of this round's own results (if
        any) to tell neighbours about, truthfully or not, plus optional text.
        """
        own_experiments = view.own_experiments
        ids = [item.event_id for item in own_experiments]
        prompt = self._communication_prompt(view, own_experiments)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        schema = communication_schema(
            ids,
            analysis_max_chars=self._analysis_chars,
            strict_objects=getattr(getattr(self.backend, "config", None), "api_style", "vllm") == "openai",
        )
        complaint: str | None = None
        for attempt in range(self.backend.config.repair_attempts + 1):
            messages = [
                {"role": "system", "content": "You answer with JSON only."},
                {"role": "user", "content": prompt},
            ]
            if complaint is not None:
                messages.append({"role": "user", "content": complaint})
                self.backend.usage.repairs += 1
            key = self.backend.cache_key(
                context,
                batch=0,
                attempt=attempt,
                schema_name="communication_decision",
                digest=digest,
            )
            try:
                record = await self._complete(
                    context=context,
                    batch=0,
                    attempt=attempt,
                    messages=messages,
                    schema=schema,
                    schema_name="communication_decision",
                    cache_key=key,
                    metadata={
                        "agent": context.agent_index,
                        "round": context.round_number,
                        "phase": "communication",
                        "attempt": attempt,
                        "episode": context.episode_id,
                    },
                )
                return _parse_communication(record["content"], ids)
            except ActionError as error:
                complaint = (
                    "Your previous answer broke a hard constraint: "
                    f"{error}. Re-answer with 'reports' naming only event_id values "
                    "from the list given, each at most once, and 'text' at most "
                    "500 characters."
                )
        raise PolicyFailure(f"communication decision still illegal after repair: {complaint}")

    # -- prompt rendering -----------------------------------------------------
    def _branch_prompt(self, view: AgentView, batch: Sequence[ParentSlot]) -> str:
        depth = batch[0].hypothesis.depth if batch else view.round_number
        return self._render(
            self._branch_template,
            opening_line=self._opening_line(),
            environment_notes=view.environment_notes,
            task_size=view.task_size,
            round_number=view.round_number,
            total_rounds=view.total_rounds,
            parent_depth=depth,
            child_depth=depth + 1,
            component_library=_dump(
                [spec.public_dict() for spec in view.component_library]
            ),
            identity_block=self._identity_block(view),
            protocol_block=self._protocol_block(view),
            objective_block=self._objective_block(view, "branch"),
            clock_block=self._clock_block(view),
            messages_block=self._messages_block(view),
            labels_heading=self._labels_heading(),
            labels_block=self._labels_block(view),
            observations_block=self._observations_block(view, "branch"),
            memory_block=self._memory_block(view),
            parents=_dump([_slot_dict(slot) for slot in batch]),
            parent_count=len(batch),
            selection_rule=self._selection_rule("branch"),
            field_order_rule=self._field_order_rule('"updates"'),
        )

    def _experiment_prompt(
        self,
        view: AgentView,
        candidates: Sequence[ExperimentCandidate],
        library: Sequence[ExperimentCondition] | None = None,
    ) -> str:
        library = list(view.experiment_library if library is None else library)
        has_library = bool(view.experiment_library) and self.choose_condition
        condition_help = (
            "You also choose the initial condition the true system is run from, "
            "which fixes the trajectory you will observe."
            if has_library
            else "The observation conditions are fixed for you."
        )
        condition_block = (
            "PUBLIC EXPERIMENT LIBRARY\n"
            + _dump([item.public_dict() for item in library])
            if has_library
            else ""
        )
        condition_rule = (
            "Also pick one experiment_id from the public library." if has_library else ""
        )
        return self._render(
            self._experiment_template,
            opening_line=self._opening_line(),
            environment_notes=view.environment_notes,
            round_number=view.round_number,
            total_rounds=view.total_rounds,
            condition_help=condition_help,
            component_library=_dump(
                [spec.public_dict() for spec in view.component_library]
            ),
            condition_block=condition_block,
            identity_block=self._identity_block(view),
            protocol_block=self._protocol_block(view),
            objective_block=self._objective_block(view, "experiment"),
            clock_block=self._clock_block(view),
            messages_block=self._messages_block(view),
            labels_heading=self._labels_heading(),
            labels_block=self._labels_block(view),
            observations_block=self._observations_block(view, "experiment"),
            memory_block=self._memory_block(view),
            candidates=_dump([item.public_dict() for item in candidates]),
            condition_rule=condition_rule,
            selection_rule=self._selection_rule("experiment"),
            field_order_rule=self._field_order_rule("the action fields"),
        )

    def _prune_prompt(
        self, view: AgentView, batch: Sequence[Hypothesis], stage: str
    ) -> str:
        subset = any(item.configuration is not None for item in view.new_evidence)
        if subset:
            rule = (
                "A verdict here belongs to a whole configuration, not to a single\n"
                "assignment: a configuration that PASSES contains no failure-inducing\n"
                "combination, so every hypothesis whose assignments all appear in a\n"
                "passing configuration is refuted. A configuration that FAILS refutes\n"
                "nothing on its own."
            )
        else:
            rule = (
                'A hypothesis is refuted when it contradicts a label you hold: it\n'
                'contains a component labelled "not in the mechanism", or it gives a\n'
                'family a value different from the one a "in the mechanism" label\n'
                "already fixed for that family. Nothing else refutes a hypothesis; in\n"
                "particular, a hypothesis is not refuted for merely looking unlikely."
            )
        return self._render(
            self._prune_template,
            opening_line=self._opening_line(),
            environment_notes=view.environment_notes,
            refutation_rule=rule,
            identity_block=self._identity_block(view),
            objective_block=self._objective_block(view, "prune"),
            messages_block=self._messages_block(view),
            labels_heading=self._labels_heading(),
            round_number=view.round_number,
            total_rounds=view.total_rounds,
            stage=stage,
            new_evidence=render_labels(view.new_evidence),
            labels_block=self._labels_block(view),
            hypothesis_count=len(batch),
            hypotheses="\n".join(
                f"{item.hid}: " + ", ".join(pair_id(pair) for pair in item.components)
                for item in batch
            ),
            field_order_rule=self._field_order_rule('"drop"'),
        )

    def _communication_prompt(
        self, view: AgentView, own_experiments: Sequence[OwnExperimentSummary]
    ) -> str:
        return self._render(
            self._communication_template,
            opening_line=self._opening_line(),
            environment_notes=view.environment_notes,
            component_library=_dump(
                [spec.public_dict() for spec in view.component_library]
            ),
            identity_block=self._identity_block(view),
            protocol_block=self._protocol_block(view),
            objective_block=self._objective_block(view, "communication"),
            round_number=view.round_number,
            total_rounds=view.total_rounds,
            clock_block=self._clock_block(view),
            messages_block=self._messages_block(view),
            labels_heading=self._labels_heading(),
            labels_block=self._labels_block(view),
            observations_block=self._observations_block(view, "communication"),
            memory_block=self._memory_block(view),
            hypotheses="\n".join(
                f"{item.hid}: " + ", ".join(pair_id(pair) for pair in item.components)
                for item in sorted(view.prune_hypotheses, key=lambda h: h.hid)
            )
            or "none left.",
            own_experiments="\n".join(
                f"{item.event_id}: tested {pair_id(item.pair)}"
                + (f" (configuration {list(item.configuration)})" if item.configuration else "")
                + f" -> you observed {item.outcome}"
                for item in own_experiments
            )
            or "none this round.",
            own_experiment_count=len(own_experiments),
            field_order_rule=self._field_order_rule('"reports"'),
        )

    # -- prompt blocks ---------------------------------------------------------
    @staticmethod
    def _render(template: str, **fields: Any) -> str:
        """Fill a template and close the gaps a switched-off block leaves."""
        text = template.format(**fields)
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    def _identity_block(self, view: AgentView) -> str:
        if not self.backend.config.context_identity:
            return ""
        ids = ", ".join(str(index) for index in view.neighbours) or "none"
        return (
            "YOUR PLACE IN THE TEAM\n"
            f"You are agent {view.agent_index} of {view.agent_count} agents working on "
            f"the same hidden mechanism.\n"
            f"You have {view.degree} direct neighbour(s): {ids}. You never see anyone "
            "else's hypotheses.\n\n"
        )

    def _protocol_block(self, view: AgentView) -> str:
        if not self.backend.config.context_identity:
            return ""
        if self.backend.config.team_objective in ("simple", "individual"):
            template = (
                PROTOCOL_FREE_NEW_TEXT
                if self.backend.config.communication_mode == "free"
                else PROTOCOL_FIXED_NEW_TEXT
            )
            return template.format(agent_count=view.agent_count, degree=view.degree)
        return (
            "PROTOCOL\n"
            f"All {view.agent_count} agents act at the same time, from lists built the "
            "same way as\n"
            "yours, so those lists overlap heavily. A label you produce this round "
            f"reaches\n"
            f"exactly your {view.degree} neighbour(s) at the START OF THE NEXT ROUND -- "
            "not sooner, and\n"
            "never further than one hop, because evidence is not forwarded. You likewise "
            "hold\n"
            "only what your own neighbours produced in the previous round. Nothing you "
            "decide\n"
            "now is visible to anyone else this round.\n\n"
        )

    def _clock_block(self, view: AgentView) -> str:
        if not self.backend.config.context_identity:
            return ""
        remaining = max(0, view.total_rounds - view.round_number)
        return (
            f"{remaining} round(s) remain after this one. Your nominal budget is "
            f"{view.round_budget} experiment(s)\n"
            "this round; the scheduler may hand you more or fewer slots.\n"
        )

    def _objective_block(self, view: AgentView, phase: str) -> str:
        mode = self.backend.config.team_objective
        if mode == "off":
            return ""
        if mode == "simple":
            return TEAM_OBJECTIVE_SIMPLE_TEXT + "\n"
        if mode == "individual":
            return TEAM_OBJECTIVE_INDIVIDUAL_TEXT + "\n"
        text = TEAM_OBJECTIVE_TEXT
        if mode == "device":
            device = BRANCH_DEVICE_TEXT if phase == "branch" else EXPERIMENT_DEVICE_TEXT
            text = text + "\n" + device.format(agent_index=view.agent_index)
        return text + "\n"

    def _opening_line(self) -> str:
        if self.backend.config.team_objective == "individual":
            return OPENING_LINE_INDIVIDUAL
        return OPENING_LINE_TEAM

    def _labels_heading(self) -> str:
        """Heading of the held-labels block.

        Under free communication the block also holds neighbours' claims,
        which may be false, so it is not called "exact" there.
        """
        if self.backend.config.communication_mode == "free":
            return "LABELS YOU HOLD (your own results and neighbours' claims)"
        return "EXACT LABELS YOU HOLD"

    def _messages_block(self, view: AgentView) -> str:
        """Neighbours' free-communication notes, verbatim and clearly labelled.

        Only rendered under ``communication_mode == "free"``; empty for every
        other mode, so a "fixed" prompt is unaffected byte for byte.  Text
        here is shown, never parsed as evidence.
        """
        if self.backend.config.communication_mode != "free":
            return ""
        if not view.received_messages:
            return "MESSAGES FROM NEIGHBOURS (free text; not evidence)\nnone this round.\n\n"
        lines = [
            f"from agent {item.sender} (round {item.round_number}): {item.text}"
            for item in view.received_messages
        ]
        return (
            "MESSAGES FROM NEIGHBOURS (free text; not evidence)\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def _labels_block(self, view: AgentView) -> str:
        allowed = render_allowed_values(view)
        if not self.backend.config.evidence_attribution:
            return (
                f"positive (component is in the mechanism): {_dump_pairs(view.known_positive)}\n"
                f"negative (component is not in the mechanism): {_dump_pairs(view.known_negative)}\n"
                f"values still allowed per family: {allowed}"
            )
        return (
            f"labels you produced yourself: {render_labels(view.own_labels)}\n"
            f"labels a neighbour sent you: {render_labels(view.received_labels)}\n"
            f"values still allowed per family: {allowed}"
        )

    def _memory_block(self, view: AgentView) -> str:
        if not self.backend.config.evidence_attribution:
            return ""
        if view.own_history:
            lines = []
            for memory in view.own_history:
                tested = ", ".join(
                    f"{pair_id(pair)} -> {outcome}"
                    for pair, outcome in zip(memory.tested, memory.outcomes)
                )
                lines.append(f"round {memory.round_number}: {tested or 'no slot'}")
            body = "\n".join(lines)
        else:
            body = "nothing yet; this is your first round."
        return (
            "WHAT YOU YOURSELF DID IN EARLIER ROUNDS\n"
            f"{body}\n"
            "Each call you make is otherwise stateless, so this list is your only "
            "memory.\n\n"
        )

    def _observations_block(self, view: AgentView, phase: str) -> str:
        """Nothing at all when there is nothing to show.

        Printing the header with nothing under it, plus "prefer additions your
        own observations make plausible", would be advice about data the agent
        does not have on a label-only environment such as tcas.
        """
        if not view.private_observations:
            return ""
        kept = deduplicate_observations(view.private_observations)
        if self.backend.config.physics_observations == "summary":
            payload = _dump([summarise_observation(item) for item in kept])
            header = (
                "YOUR PRIVATE OBSERVATIONS (yours alone; summary statistics per "
                "trajectory)"
            )
        else:
            payload = _dump([item.public_dict() for item in kept])
            header = "YOUR PRIVATE OBSERVATIONS (yours alone; raw trajectories)"
        verb = "additions" if phase == "branch" else "tests"
        return (
            f"{header}\n{payload}\n"
            f"Prefer {verb} that these observations make plausible.\n\n"
        )

    def _selection_rule(self, phase: str) -> str:
        mode = self.backend.config.team_objective
        if mode in ("off", "simple", "individual"):
            # Neither new objective prescribes a search or division-of-labour
            # strategy; the old "soft" fallback below does, so it stays out.
            return ""
        noun = "additions" if phase == "branch" else "tests"
        if mode == "device":
            return (
                f"Apply the symmetry-breaking arithmetic above when several {noun} "
                "look equally informative.\n"
            )
        return (
            f"Prefer {noun} that are not already settled by the labels you hold and "
            "that a neighbour reading the same library is unlikely to pick.\n"
        )

    def _field_order_rule(self, action: str) -> str:
        mode = self.backend.config.reasoning
        if mode == "reason_first":
            limit = self.backend.config.reasoning_max_chars
            return (
                f'Write "analysis" FIRST: at most {limit} characters in which you '
                "actually work out\nwhich components are still undetermined, which of "
                f"your hypotheses contain them,\nand what a neighbour would pick. Then "
                f'{action}. Keep "reason" under 20 words.'
            )
        if mode == "thinking":
            return (
                f"Think it through first, then emit only the JSON, with {action} "
                'first and "reason"\nunder 20 words. Keep the thinking short enough '
                "that the JSON is not cut off."
            )
        return f'Put {action} first and keep "reason" under 20 words.'


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _slot_dict(slot: ParentSlot) -> dict[str, Any]:
    """Render a parent with the component labels the schema's enums use."""
    payload = slot.hypothesis.public_dict()
    payload["required_additions"] = slot.required
    payload["legal_additions"] = [pair_id(pair) for pair in slot.legal_additions]
    return payload


def _pair_from_label(label: object) -> Pair:
    text = str(label)
    if text.startswith("c") and "v" in text:
        family, _, variant = text[1:].partition("v")
        if family.isdigit() and variant.isdigit():
            return int(family), int(variant)
    raise ActionError(f"malformed component label {label!r}")


def _dump_pairs(pairs: Sequence[Pair]) -> str:
    return _dump([{"family_id": f, "variant_id": v} for f, v in pairs])


def _loads(content: str | None) -> dict:
    if content is None:
        raise ActionError("the model returned no content")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ActionError(f"output was not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ActionError("output was not a JSON object")
    return payload


def _parse_branch(content: str | None, batch: Sequence[ParentSlot]) -> list[ParentUpdate]:
    """Parse and fully check one batch.

    The same semantic rules the runner enforces are applied here so that a
    violation becomes a repairable complaint instead of an aborted episode.
    """
    payload = _loads(content)
    raw_updates = payload.get("updates")
    if not isinstance(raw_updates, dict):
        raise ActionError("'updates' must be an object keyed by parent_id")
    slots = {slot.hypothesis.hid: slot for slot in batch}
    unknown = sorted(set(raw_updates) - set(slots))
    if unknown:
        raise ActionError(f"parent_id {unknown} is not in this batch")
    missing = sorted(set(slots) - set(raw_updates))
    if missing:
        raise ActionError(f"missing updates for parents {missing}")
    updates: list[ParentUpdate] = []
    for parent_id, slot in sorted(slots.items()):
        raw_additions = raw_updates[parent_id]
        if not isinstance(raw_additions, list):
            raise ActionError(f"'{parent_id}' must map to a list of component labels")
        additions = tuple(_pair_from_label(item) for item in raw_additions)
        if len(additions) != slot.required:
            raise ActionError(
                f"parent {parent_id!r} needs exactly {slot.required} addition(s), "
                f"got {len(additions)}"
            )
        if len(set(additions)) != len(additions):
            raise ActionError(f"parent {parent_id!r} repeats an addition")
        legal = set(slot.legal_additions)
        for pair in additions:
            if pair not in legal:
                raise ActionError(
                    f"addition {pair_id(pair)!r} is not in the legal_additions "
                    f"list of parent {parent_id!r}"
                )
        updates.append(ParentUpdate(parent_id=parent_id, additions=additions))
    return updates


def _parse_experiment(
    content: str | None, candidates: Sequence[ExperimentCandidate]
) -> ExperimentDecision:
    payload = _loads(content)
    offered = {
        choice_label(candidate.pair, parent_id): (candidate.pair, parent_id)
        for candidate in candidates
        for parent_id in candidate.parent_ids
    }
    choice = str(payload.get("choice", ""))
    if choice not in offered:
        raise ActionError(f"{choice!r} is not one of the offered choices")
    pair, parent_id = offered[choice]
    experiment_id = payload.get("experiment_id")
    return ExperimentDecision(
        pair=pair,
        parent_id=parent_id,
        experiment_id=str(experiment_id) if experiment_id is not None else None,
        reason=str(payload.get("reason", ""))[:500],
    )


def _parse_communication(
    content: str | None, offered: Sequence[str]
) -> CommunicationDecision:
    payload = _loads(content)
    raw_reports = payload.get("reports")
    if raw_reports is None:
        raise ActionError("'reports' is missing; return an empty list to report nothing")
    if not isinstance(raw_reports, list):
        raise ActionError("'reports' must be a list of {event_id, claimed_outcome} objects")
    known = set(offered)
    seen: set[str] = set()
    reports: list[ReportItem] = []
    for item in raw_reports:
        if not isinstance(item, dict):
            raise ActionError("each report must be a JSON object")
        event_id = str(item.get("event_id", ""))
        if event_id not in known:
            raise ActionError(f"event_id {event_id!r} was not offered")
        if event_id in seen:
            raise ActionError(f"event_id {event_id!r} was reported twice")
        seen.add(event_id)
        outcome = item.get("claimed_outcome")
        if outcome not in ("positive", "negative"):
            raise ActionError(f"claimed_outcome {outcome!r} must be 'positive' or 'negative'")
        reports.append(ReportItem(event_id=event_id, claimed_outcome=outcome))
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise ActionError("'text' must be a string")
    if len(text) > COMMUNICATION_TEXT_MAX_CHARS:
        raise ActionError(f"'text' must be at most {COMMUNICATION_TEXT_MAX_CHARS} characters")
    return CommunicationDecision(
        reports=tuple(reports), text=text, reason=str(payload.get("reason", ""))[:500]
    )


def _parse_prune(content: str | None, offered: Sequence[str]) -> list[str]:
    payload = _loads(content)
    raw = payload.get("drop")
    if raw is None:
        raise ActionError("'drop' is missing; return an empty list to keep everything")
    if not isinstance(raw, list):
        raise ActionError("'drop' must be a list of hypothesis ids")
    known = set(offered)
    dropped: list[str] = []
    for item in raw:
        text = str(item)
        if text not in known:
            raise ActionError(f"hypothesis id {text!r} was not offered")
        if text not in dropped:
            dropped.append(text)
    return dropped
