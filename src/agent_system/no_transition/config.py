"""Frozen experiment configuration and provenance, with no finite-dc fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "needle_chain_v1"


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate(config)
    return config


def validate(config: dict) -> None:
    assert config["protocol_version"] == PROTOCOL
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
    for c in config["conditions"]:
        assert c["task"] in {"needle", "chain"}
        assert c["horizon"] > 0 and c["batch_size"] > 0
        if c["task"] == "needle":
            assert c["candidates"] >= n * c["batch_size"] * c["horizon"]
        else:
            assert c["chain_length"] > 0 and c["batch_size"] == 1


def threshold(condition: dict) -> dict:
    """A theoretical label, never a finite-sample estimate or success filter."""
    if condition["task"] == "chain" and condition["horizon"] < condition["chain_length"]:
        return {"dc": "infinity", "interpretation": "insufficient sequential horizon", "source": "theory"}
    return {"dc": 0, "interpretation": "no positive communication threshold", "source": "theory"}


def request_bound(config: dict) -> int:
    return (len(config["llm_seeds"]) * len(config["degrees"]) * config["agents"]
            * sum(c["horizon"] for c in config["conditions"]))
