"""Tests → reference pilot → invariants → GPT pilot → analysis; then stop.

python -m src.agent_system.no_transition.cli --mode pilot
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from ..driver import load_env_file
from ..storage import utc_now
from .analysis import load_episodes, verify_invariants
from .config import ROOT, fingerprint, git_sha, load_config, request_bound
from .report import analyze, write_json
from .runner import episode_id, run_episode


def run_tests() -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "tests/needle_chain",
        "tests/agent_system/test_api_style.py", "tests/agent_system/test_llm_policy.py",
        "tests/agent_system/test_free_communication.py", "tests/agent_system/test_protocol.py",
        "tests/agent_system/test_view_isolation.py", "tests/agent_system/test_parity.py", "-q"], cwd=ROOT).returncode


def terminal_summary(summary: dict, mode: str) -> None:
    print(f"\nNEEDLE / CHAIN {mode.upper()} {'COMPLETE' if summary['complete'] else 'INCOMPLETE'}")
    print("Branch cleanup: three exact obsolete branches confirmed already absent.")
    print("Branch: experiment/needle-chain-no-transition-v1")
    for kind in ("deterministic", "llm"):
        print(f"\n{kind.upper()}")
        for r in summary["rows"]:
            if r["agent_type"] == kind:
                print(f"{r['condition']} d={r['degree']}: {r['successes']}/{r['episodes']} "
                      f"P={r['p_success']:.3f} CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print("\nFinite communication transition: assess diagnostics; absence is not established by a small pilot.")
    print(json.dumps(summary["runtime"], indent=2))
    print("Recommended next action: review pilot. No automatic larger run or merge.")
    print(f"Current source commit: {git_sha()}")
    print("Remote branch: origin/experiment/needle-chain-no-transition-v1")


async def execute(config: dict, run_dir: Path, output: Path, env_values: dict, execution: dict) -> dict:
    for kind in ("deterministic", "llm"):
        seeds = config["deterministic_seeds" if kind == "deterministic" else "llm_seeds"]
        if kind == "llm":
            # Completed cached episodes do not require credentials on resume.
            pending = any(not (run_dir / "episodes" / episode_id(c, kind, s, d) / "episode_result.json").exists()
                          for c in config["conditions"] for s in seeds for d in config["degrees"])
            if pending and not env_values.get(config["llm"]["api_key_env"]):
                execution["blocker"] = "No authenticated OpenAI API setup is available: OPENAI_API_KEY is unset; supply the location of an existing environment file or authorized runner. No GPT calls were made."
                raise RuntimeError("OPENAI_API_KEY is unavailable. Reference results are saved; resume with --env-file PATH. No GPT calls were made.")
            execution.pop("blocker", None)
        stage_start = time.monotonic()
        try:
            # Config order is Needle low/high, then Chain short/sufficient.
            for condition in config["conditions"]:
                for seed in seeds:
                    for degree in config["degrees"]:
                        result = await run_episode(config, condition, kind, seed, degree, run_dir, env_values)
                        print(f"{result['episode_id']}: success={result['success']} depth={result['maximum_depth']} "
                              f"coverage={result['unique_candidates_tested']} api_errors={result['api_errors']}", flush=True)
        finally:
            key = f"{kind}_wall_seconds"
            execution[key] = execution.get(key, 0.0) + time.monotonic() - stage_start
            write_json(run_dir / "execution.json", execution)
        if kind == "deterministic":
            verify_invariants(load_episodes(run_dir), config)
            # Archive/analysis checkpoint before any live model calls.
            analyze(run_dir, output, config, execution)
    return analyze(run_dir, output, config, execution)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute-full", action="store_true", help="explicit launch gate for a subsequent full-run task")
    args = parser.parse_args(argv)
    if args.test:
        return run_tests()
    config = load_config(args.config or ROOT / f"configs/needle_chain/{args.mode}.json")
    if config["mode"] != args.mode:
        parser.error("config mode does not match --mode")
    run_dir = args.run_dir or ROOT / f"runs/needle_chain/{args.mode}"
    output = args.output_dir or ROOT / f"results/needle_chain/{args.mode}"
    if args.validate_only:
        print(json.dumps({"status": "valid", "mode": args.mode, "maximum_llm_requests": request_bound(config),
                          "llm_episodes": len(config['degrees']) * len(config['llm_seeds']) * len(config['conditions']),
                          "executed": False}, indent=2))
        return 0
    if args.analyze_existing:
        saved = run_dir / "config.json"
        if saved.exists() and json.loads(saved.read_text()) != config:
            parser.error("analysis config differs from archived config")
        execution_path = run_dir / "execution.json"
        execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
        terminal_summary(analyze(run_dir, output, config, execution), args.mode)
        return 0
    if args.mode == "full" and not args.execute_full:
        parser.error("Full run was NOT started. A subsequent task must explicitly supply --execute-full.")
    if run_tests():
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    # One process per run directory. A second launch cannot duplicate requests.
    with (run_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process owns this run directory") from None
        config_path = run_dir / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("frozen run config changed; choose a new directory")
        write_json(config_path, config)
        path = run_dir / "execution.json"
        execution = json.loads(path.read_text()) if path.exists() else {
            "started_at": utc_now(), "config_digest": fingerprint(config), "source_sha": git_sha(),
            "python": sys.version, "full_executed": args.mode == "full", "invocations": [],
            "package_versions": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "networkx", "scipy", "matplotlib", "pandas", "httpx", "pytest")}}
        start = time.monotonic()
        execution["invocations"].append({"started_at": utc_now(), "git_commit_sha": git_sha()})
        try:
            summary = asyncio.run(execute(config, run_dir, output, load_env_file(args.env_file), execution))
            execution["status"] = "complete" if summary["complete"] else "incomplete"
        except Exception as exc:
            execution["status"] = "interrupted"
            execution["last_error_type"] = type(exc).__name__
            # No provider exception body is emitted by this runner.
            print(f"Pipeline interrupted: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            duration = time.monotonic() - start
            execution["execution_wall_seconds"] = execution.get("execution_wall_seconds", 0) + duration
            execution["invocations"][-1]["wall_seconds"] = duration
            write_json(path, execution)
            if load_episodes(run_dir):
                analyze(run_dir, output, config, execution)
        terminal_summary(analyze(run_dir, output, config, execution), args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
