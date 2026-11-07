"""Round-barrier runner, automatic truthful Router messages, durable episodes."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

from ..communication import CommunicationMessage, Router
from ..policies.base import DecisionContext
from ..policies.openai_policy import LLMBackend
from ..schemas import LLMConfig
from ..storage import RunStore, iter_jsonl, utc_now
from .config import fingerprint, git_sha, threshold
from .environments import ChainEnvironment, NeedleEnvironment, candidate_order, reference_batch
from .policies import NanoPolicy, ReferencePolicy


def episode_id(condition: dict, kind: str, seed: int, degree: int) -> str:
    return f"{condition['name']}_{kind}_s{seed}_d{degree}"


def _route(router: Router, outgoing: list[list[dict]], round_number: int) -> None:
    for agent, records in enumerate(outgoing):
        if records:
            router.stage_message(agent, CommunicationMessage(sender=agent, round_number=round_number,
                                 text=json.dumps(records, sort_keys=True)))
    router.commit_round()


def api_metrics(store: RunStore) -> dict:
    events = list(iter_jsonl(store.path / "events.jsonl"))
    starts = [r for r in events if r.get("kind") == "http_started"]
    ends = [r for r in events if r.get("kind") == "http_completed"]
    records = [json.loads(p.read_text()) for p in sorted((store.path / "calls").glob("*.json"))]
    decisions = [json.loads(p.read_text()) for p in sorted((store.path / "decisions").glob("*.json"))]
    usage = [r.get("usage", {}) for r in records]
    requests = list(iter_jsonl(store.path / "requests.jsonl"))
    logical_keys = {r["cache_key"] for r in requests}
    ended_ids = {r["attempt"] for r in ends}
    return {"api_attempts": len(starts), "successful_api_calls": len(records),
            "api_failed_attempts": max(0, len(starts) - len(records)),
            "retry_count": max(0, len(starts) - len(logical_keys)),
            "ambiguous_or_transport_interrupted_attempts": sum(r["attempt"] not in ended_ids for r in starts),
            "api_failed_decisions": sum(d["api_error"] for d in decisions),
            "behavior_failed_decisions": sum(d["behavior_error"] for d in decisions),
            "api_service_seconds": sum(r["latency_s"] for r in ends),
            "api_timed_attempts": len(ends),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "cached_prompt_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usage),
            "response_models": sorted({r.get("model", "unknown") for r in records})}


async def run_episode(config: dict, condition: dict, kind: str, seed: int, degree: int,
                      run_dir: Path, env_values: dict | None = None, *, policy_override=None) -> dict:
    eid = episode_id(condition, kind, seed, degree)
    identity = {"protocol": config["protocol_version"], "config_digest": fingerprint(config),
                "condition": condition, "agent_type": kind, "seed": seed, "degree": degree}
    with RunStore(run_dir / "episodes", eid) as store:
        done = store.read_json("episode_result.json")
        if done is not None:
            if done["identity"] != identity:
                raise ValueError("configuration differs from completed episode; choose a new run directory")
            return done
        manifest = store.read_json("manifest.json")
        if manifest is not None and manifest["identity"] != identity:
            raise ValueError("configuration differs from checkpoint")
        if manifest is None:
            manifest = {"identity": identity, "git_commit_sha": git_sha(), "created_at": utc_now(),
                        "model_settings": config["llm"] if kind == "llm" else {}}
            store.write_manifest(manifest)
        start = time.monotonic()
        n, horizon = config["agents"], condition["horizon"]
        graph_seed = seed + config["graph_seed_offset"]
        router = Router(n, degree, graph_seed)
        assert all(len(v) == degree for v in router.graph)
        is_needle = condition["task"] == "needle"
        environment = (NeedleEnvironment(condition["candidates"], seed) if is_needle else
                       ChainEnvironment(condition["chain_length"], n, seed, horizon))
        order = candidate_order(condition["candidates"], seed) if is_needle else []
        state = store.read_json("checkpoint.json")
        if state is None:
            state = {"round_completed": 0, "histories": [[] for _ in range(n)], "messages": [],
                     "actions": [], "last_outgoing": [[] for _ in range(n)], "invalid_actions": 0,
                     "illegal_future_attempts": 0, "api_errors": 0, "behavior_errors": 0,
                     "decisions": 0, "wall_seconds": 0.0, "rounds": []}
        else:
            environment.restore(state["environment"])
            _route(router, state["last_outgoing"], state["round_completed"])
        previous_wall = state["wall_seconds"]
        backend = None
        if policy_override is not None:
            policy = policy_override
        elif kind == "llm":
            values = env_values or {}
            settings = dict(config["llm"])
            settings["base_url"] = values.get("OPENAI_BASE_URL") or settings["base_url"]
            if values.get("OPENAI_MODEL", settings["model"]) != settings["model"]:
                raise ValueError("OPENAI_MODEL disagrees with preregistered exact model")
            key = values.get(settings["api_key_env"], "")
            backend = LLMBackend(LLMConfig(**settings), key)
            policy = NanoPolicy(backend, store)
        else:
            policy = ReferencePolicy()
        try:
            for r in range(state["round_completed"] + 1, horizon + 1):
                # Needle deliberately uses its fixed whole budget even after discovery.
                if not is_needle and environment.success:
                    break
                if not is_needle:
                    environment.begin_round(r)
                views, contexts = [], []
                for agent in range(n):
                    received = []
                    for sender, message in router.message_deliveries(agent):
                        received.extend({"sender": sender, "sent_round": message.round_number,
                                         "record": item} for item in json.loads(message.text))
                    state["messages"].extend({"receiver": agent, "received_round": r, **m} for m in received)
                    if not is_needle:
                        for message in received:
                            pointer = message["record"].get("receipt", {}).get("next_pointer")
                            if pointer:
                                assert environment.receive_pointer(agent, pointer)
                    view = {"task": condition["task"], "agent_id": agent, "num_agents": n,
                            "round": r, "horizon": horizon, "own_history": state["histories"][agent],
                            "neighbor_messages": received}
                    if is_needle:
                        view.update(candidate_id_min=0, candidate_id_max=condition["candidates"] - 1,
                                    batch_size=condition["batch_size"], scheduled_candidates=reference_batch(
                                        order, n, agent, r, condition["batch_size"]))
                    else:
                        view.update(chain_length=condition["chain_length"], available_pointer=environment.observe(agent))
                    views.append(view)
                    contexts.append(DecisionContext(run_id=config["name"] + "_" + config["mode"],
                        episode_id=eid, agent_index=agent, round_number=r, phase=condition["task"], seed=seed))
                # A frozen observation for everyone; no environment mutation inside a policy task.
                decisions = await asyncio.gather(*(policy.decide(v, c) for v, c in zip(views, contexts)))
                outgoing = [[] for _ in range(n)]
                depth_before = 0 if is_needle else environment.depth
                for agent, decision in enumerate(decisions):
                    state["decisions"] += 1
                    state["api_errors"] += int(decision["api_error"])
                    state["behavior_errors"] += int(decision["behavior_error"])
                    action = decision["action"]
                    if is_needle:
                        selections = action.get("candidates", [])
                        if not isinstance(selections, list) or len(selections) != condition["batch_size"]:
                            selections = [None] * condition["batch_size"]
                        for slot, candidate in enumerate(selections):
                            result = environment.verify(candidate, round_number=r)
                            invalid = result == "INVALID_CANDIDATE"
                            event = {"agent_id": agent, "round": r, "slot": slot,
                                     "candidate_tested": candidate, "candidate_result": result,
                                     "invalid": invalid, "unique_candidates_tested": len(environment.tested),
                                     "duplicate_tests": environment.duplicate_tests, "total_tests": environment.total_tests,
                                     "target_found": environment.success}
                            state["invalid_actions"] += int(invalid)
                            state["actions"].append(event)
                            # Only own public verdicts enter communication, never aggregate evaluator metrics.
                            own = {"candidate_tested": candidate, "candidate_result": result, "round": r}
                            state["histories"][agent].append(own)
                            outgoing[agent].append(own)
                    else:
                        level, key = action.get("level"), action.get("key")
                        result = environment.submit(agent, level, key)
                        invalid = result["status"] != "ACCEPTED_PENDING_BARRIER"
                        future = type(level) is int and level > depth_before + 1
                        state["illegal_future_attempts"] += int(future)
                        state["invalid_actions"] += int(invalid)
                        event = {"agent_id": agent, "round": r, "level": level, "key": key,
                                 "response": result, "invalid": invalid, "future_attempt": future}
                        state["actions"].append(event)
                        outgoing[agent].append({"action": action, "response": result, "round": r})
                if not is_needle:
                    receipts = environment.commit_round()
                    for agent in range(n):
                        outgoing[agent][0]["receipt"] = receipts.get(agent, {"status": "NO_PROGRESS"})
                        state["histories"][agent].append(outgoing[agent][0])
                    assert environment.depth <= r
                _route(router, outgoing, r)
                state["last_outgoing"] = outgoing
                state["round_completed"] = r
                state["environment"] = environment.snapshot()
                state["wall_seconds"] = previous_wall + time.monotonic() - start
                state["rounds"].append({"round": r, "depth": None if is_needle else environment.depth,
                                        "unique_coverage": len(environment.tested) if is_needle else None})
                store.checkpoint(state)
            success = environment.success
            result = {"identity": identity, "experiment": config["name"], "episode_id": eid,
                      "condition": condition["name"], "task": condition["task"], "agent_type": kind,
                      "model": config["llm"]["model"] if kind == "llm" else None,
                      "model_settings": config["llm"] if kind == "llm" else {},
                      "task_seed": seed, "seed": seed, "graph_seed": graph_seed,
                      "degree": degree, "num_agents": n, "graph": router.graph,
                      "budget": n * condition["batch_size"] * horizon, "horizon": horizon,
                      "chain_length": condition.get("chain_length"), "candidates": condition.get("candidates"),
                      "status": "completed_with_api_errors" if state["api_errors"] else "ok",
                      "success": success, "target_found": success, "rounds_used": state["round_completed"],
                      "round_of_success": environment.round_of_success if is_needle else (environment.round_number if success else None),
                      "unique_candidates_tested": len(environment.tested) if is_needle else 0,
                      "total_tests": environment.total_tests if is_needle else len(state["actions"]),
                      "duplicate_tests": environment.duplicate_tests if is_needle else 0,
                      "maximum_depth": 0 if is_needle else environment.depth,
                      "depth_after_each_round": [] if is_needle else environment.depth_history,
                      "hidden_target": environment._target if is_needle else None,
                      "hidden_chain_fingerprint": None if is_needle else fingerprint(environment._keys),
                      "threshold_theory": threshold(condition),
                      "git_commit_sha": manifest["git_commit_sha"], "created_at": manifest["created_at"],
                      **{k: state[k] for k in ("actions", "messages", "invalid_actions", "illegal_future_attempts",
                                               "api_errors", "behavior_errors", "decisions", "wall_seconds", "rounds")},
                      **api_metrics(store)}
            result["wall_seconds"] = previous_wall + time.monotonic() - start
            if not is_needle:
                assert condition["horizon"] >= condition["chain_length"] or not success
            store.write_json("episode_result.json", result)
            return result
        finally:
            if backend is not None:
                await backend.aclose()
