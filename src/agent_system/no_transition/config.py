"""Frozen experiment configuration and provenance, with no finite-dc fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "needle_chain_v1"
PROTOCOL_V2 = "needle_chain_v2"
PROTOCOL_FULL_V2 = "needle_chain_full_v2"


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate(config)
    return config


def resolved_llm_settings(config: dict, env_values: dict | None = None) -> dict:
    """Pin the user-requested snapshot without changing frozen task resources."""
    provider = json.loads((ROOT / "configs/needle_chain/provider.json").read_text())
    assert config["llm"]["model"] == provider["family_alias_in_frozen_resource_config"]
    values = env_values or {}
    settings = {**config["llm"], "model": provider["model"]}
    if values.get("OPENAI_MODEL", provider["model"]) != provider["model"]:
        raise ValueError("OPENAI_MODEL disagrees with the user-requested snapshot")
    settings["base_url"] = values.get("OPENAI_BASE_URL") or provider["base_url"]
    return settings


def validate(config: dict) -> None:
    assert config["protocol_version"] in {PROTOCOL, PROTOCOL_V2, PROTOCOL_FULL_V2}
    if config["protocol_version"] == PROTOCOL_V2:
        assert config["mode"] == "pilot", "v2 is pilot-only; a full design needs separate authorization"
    if config["protocol_version"] == PROTOCOL_FULL_V2:
        assert config["mode"] == "full" and config["agents"] == 20
        assert config["degrees"] == [0, 4, 8, 12, 16, 19]
        assert len(config["deterministic_seeds"]) == 200 and len(config["llm_seeds"]) == 12
        assert not config.get("llm_seeds_by_condition")
    n = config["agents"]
    assert isinstance(n, int) and n >= 2
    assert len(config["degrees"]) == len(set(config["degrees"]))
    assert 0 in config["degrees"]
    for degree in config["degrees"]:
        assert isinstance(degree, int) and 0 <= degree < n and n * degree % 2 == 0
    for key in ("deterministic_seeds", "llm_seeds"):
        assert config[key] and len(config[key]) == len(set(config[key]))
        assert all(type(seed) is int and seed >= 0 for seed in config[key])
    assert config["llm"]["model"] == "gpt-5.4-nano", "Do not silently substitute models"
    assert len(config["conditions"]) == len({c["name"] for c in config["conditions"]})
    overrides = config.get("llm_seeds_by_condition", {})
    assert set(overrides) <= {c["name"] for c in config["conditions"]}
    for seeds in overrides.values():
        assert seeds and len(seeds) == len(set(seeds))
        assert all(type(seed) is int and seed >= 0 for seed in seeds)
    for c in config["conditions"]:
        assert c["task"] in {"needle", "chain"}
        assert c["horizon"] > 0 and c["batch_size"] > 0
        if c["task"] == "needle":
            passes = c.get("coverage_passes", 1)
            assert type(passes) is int and passes >= 1
            assert c["candidates"] * passes >= n * c["batch_size"] * c["horizon"]
            if passes > 1:
                assert config["protocol_version"] in {PROTOCOL_V2, PROTOCOL_FULL_V2}
                assert c["name"] == "Needle-High" and c.get("policy_variant") == "needle_high_v2"
                assert (c["candidates"], c["horizon"], c["batch_size"], passes) == (n * 8, 2, 8, 2)
        else:
            assert c["chain_length"] > 0 and c["batch_size"] == 1


def threshold(condition: dict) -> dict:
    """A theoretical label, never a finite-sample estimate or success filter."""
    if condition["task"] == "chain" and condition["horizon"] < condition["chain_length"]:
        return {"dc": "infinity", "interpretation": "insufficient sequential horizon", "source": "theory"}
    return {"dc": 0, "interpretation": "no positive communication threshold", "source": "theory"}


def request_bound(config: dict) -> int:
    return (len(config["degrees"]) * config["agents"]
            * sum(len(episode_seeds(config, c, "llm")) * c["horizon"] for c in config["conditions"]))


def episode_seeds(config: dict, condition: dict, kind: str) -> list[int]:
    if kind == "deterministic":
        return config["deterministic_seeds"]
    return config.get("llm_seeds_by_condition", {}).get(condition["name"], config["llm_seeds"])


def expected_episode_count(config: dict, kind: str | None = None) -> int:
    return len(config["degrees"]) * sum(len(episode_seeds(config, c, k))
        for c in config["conditions"] for k in ([kind] if kind else ["deterministic", "llm"]))


def artifact_label(config: dict) -> str:
    if config["protocol_version"] == PROTOCOL_FULL_V2:
        return "full_v2"
    return "pilot_v2" if config["protocol_version"] == PROTOCOL_V2 else config["mode"]
