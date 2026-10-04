#!/usr/bin/env python3
"""Command line entry point: ``validate``, ``run``, ``replay`` and ``sweep``."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from ..utils import PACKAGE_ROOT
from .driver import RunDriver, estimate_requests, load_env_file
from .schemas import RunConfig, load_run_config
from .storage import RunStore, utc_now


def read_config(path: Path) -> RunConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_run_config(payload)


def default_run_id(config: RunConfig) -> str:
    stamp = utc_now().replace(":", "").replace("-", "")
    return f"{config.name}_{stamp}"


def _manifest(config: RunConfig, env_values: dict[str, str], run_id: str) -> dict:
    llm = config.policy.llm
    return {
        "run_id": run_id,
        "created_at": utc_now(),
        "config": config.as_dict(),
        "estimate": estimate_requests(config),
        "resolved_model": env_values.get("OPENAI_MODEL", llm.model),
        "resolved_base_url": env_values.get("OPENAI_BASE_URL", llm.base_url),
        "api_key_present": bool(env_values.get(llm.api_key_env)),
        "python": sys.version,
    }


def command_validate(args: argparse.Namespace) -> int:
    config = read_config(args.config)
    env_values = load_env_file(args.env_file) if args.env_file else {}
    estimate = estimate_requests(config)
    report = {
        "status": "ok",
        "name": config.name,
        "protocol_version": config.protocol_version,
        "policy": config.policy.kind,
        "environment": config.environment.kind,
        "rounds": config.effective_rounds,
        "degrees": list(config.degrees),
        "episodes": estimate["episodes"],
        "estimate": estimate,
    }
    if config.policy.kind.startswith("llm_"):
        llm = config.policy.llm
        model = env_values.get("OPENAI_MODEL", llm.model)
        base_url = env_values.get("OPENAI_BASE_URL", llm.base_url)
        key = env_values.get(llm.api_key_env, "")
        problems = []
        if not model:
            problems.append("OPENAI_MODEL is empty")
        if not base_url:
            problems.append("OPENAI_BASE_URL is empty")
        if not key:
            problems.append(f"{llm.api_key_env} is empty")
        report["resolved_model"] = model
        report["resolved_base_url"] = base_url
        report["api_key_present"] = bool(key)
        if problems:
            report["status"] = "error"
            report["problems"] = problems
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


def _run(config: RunConfig, args: argparse.Namespace, *, replay_dir: Path | None = None) -> int:
    env_values = load_env_file(args.env_file) if getattr(args, "env_file", None) else {}
    run_id = getattr(args, "run_id", None) or default_run_id(config)
    root = Path(args.output_root or config.output_root)
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    with RunStore(root, run_id) as store:
        store.write_manifest(_manifest(config, env_values, run_id))
        driver = RunDriver(config, store=store, env_values=env_values, replay_dir=replay_dir)
        summary = asyncio.run(driver.run(resume=getattr(args, "resume", False)))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"run directory: {store.path}")
    return 0 if not summary["statuses"].get("policy_failed") else 1


def command_run(args: argparse.Namespace) -> int:
    return _run(read_config(args.config), args)


def command_sweep(args: argparse.Namespace) -> int:
    config = read_config(args.config)
    if len(config.degrees) < 2:
        print("warning: sweep was given fewer than two degrees", file=sys.stderr)
    return _run(config, args)


def command_replay(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest["config"]
    payload["policy"] = {**payload["policy"], "kind": "replay"}
    payload["name"] = payload["name"] + "_replay"
    config = load_run_config(payload)
    args.env_file = None
    args.output_root = args.output_root or str(run_dir.parent)
    args.run_id = args.run_id or (run_dir.name + "_replay")
    args.resume = False
    return _run(config, args, replay_dir=run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.agent_system.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check a config and estimate scale, no model calls")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--env-file", type=Path)
    validate.set_defaults(handler=command_validate)

    for name, handler, help_text in (
        ("run", command_run, "run the configured episodes"),
        ("sweep", command_sweep, "run every configured degree"),
    ):
        node = sub.add_parser(name, help=help_text)
        node.add_argument("--config", type=Path, required=True)
        node.add_argument("--env-file", type=Path)
        node.add_argument("--run-id")
        node.add_argument("--output-root")
        node.add_argument("--resume", action="store_true")
        node.set_defaults(handler=handler)

    replay = sub.add_parser("replay", help="re-run from recorded responses, offline")
    replay.add_argument("--run-dir", type=Path, required=True)
    replay.add_argument("--run-id")
    replay.add_argument("--output-root")
    replay.set_defaults(handler=command_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
