"""Run directory layout, append-only ledgers and checkpoints.

A run directory holds ``manifest.json``, ``events.jsonl``, ``requests.jsonl``,
``responses.jsonl``, ``checkpoint.json`` and ``episode_result.json``.  The
manifest never contains a key, and hidden-evaluation fields are kept under an
explicit ``hidden_`` namespace so a reader can tell them from agent-visible
data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

LEDGERS = ("events", "requests", "responses", "episodes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}

    # -- ledgers ----------------------------------------------------------
    def _handle(self, name: str):
        if name not in LEDGERS:
            raise ValueError(f"unknown ledger {name!r}")
        handle = self._handles.get(name)
        if handle is None:
            handle = (self.path / f"{name}.jsonl").open("a", encoding="utf-8")
            self._handles[name] = handle
        return handle

    def append(self, name: str, payload: dict) -> None:
        handle = self._handle(name)
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()

    def sink(self, name: str):
        def _write(payload: dict) -> None:
            self.append(name, payload)

        return _write

    # -- documents ---------------------------------------------------------
    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    def read_json(self, name: str) -> Any:
        target = self.path / name
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def write_manifest(self, payload: dict) -> Path:
        redacted = _redact(payload)
        redacted.setdefault("created_at", utc_now())
        redacted["run_id"] = self.run_id
        return self.write_json("manifest.json", redacted)

    def checkpoint(self, payload: dict) -> Path:
        return self.write_json("checkpoint.json", {"updated_at": utc_now(), **payload})

    def completed_episodes(self) -> set[str]:
        path = self.path / "episodes.jsonl"
        if not path.exists():
            return set()
        done: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["episode_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def read_episodes(self) -> list[dict]:
        path = self.path / "episodes.jsonl"
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Exact names and suffixes only: a substring rule would also hide harmless
# settings such as ``max_output_tokens``, which the response cache key hashes.
SECRET_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer",
        "password",
        "passwd",
        "secret",
        "token",
    }
)
SECRET_SUFFIXES = ("_api_key", "_apikey", "_secret", "_password", "_auth_token", "_access_token")


def is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SECRET_NAMES or lowered.endswith(SECRET_SUFFIXES)


def _redact(payload: Any) -> Any:
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            if is_secret_key(key):
                result[key] = "<redacted>"
            else:
                result[key] = _redact(value)
        return result
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    return payload


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not Path(path).exists():
        return
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)
