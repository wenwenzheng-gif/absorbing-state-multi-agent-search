"""Restore committed completed episodes, then resume exactly the small pilot.

No credential values are read or written here. GitHub injects the existing
secret into the standard provider environment only for the CLI run step.
"""
from __future__ import annotations

import json
from pathlib import Path

from .analysis import load_episodes, verify_invariants
from .config import ROOT, fingerprint, load_config
from .report import write_json


def restore_completed(source: Path, run_dir: Path, config: dict) -> int:
    if config["mode"] != "pilot" or config["agents"] != 6 or config["degrees"] != [0, 2, 5]:
        raise ValueError("hosted entry point is pilot-only")
    if config["llm_seeds"] != [91000, 91001] or len(config["conditions"]) != 4:
        raise ValueError("hosted pilot must retain the exact 24-episode design")
    original_config = json.loads((source / "config.json").read_text())
    if original_config != config:
        raise ValueError("committed resources differ from the pilot config")
    episodes = load_episodes(source)
    verify_invariants(episodes, config)
    restored = 0
    for episode in episodes:
        if episode["identity"]["config_digest"] != fingerprint(config):
            raise ValueError("committed episode config does not match")
        target = run_dir / "episodes" / episode["episode_id"] / "episode_result.json"
        if target.exists():
            if json.loads(target.read_text()) != episode:
                raise ValueError("existing result differs; refusing to overwrite")
        else:
            write_json(target, episode)
            restored += 1
    if not (run_dir / "config.json").exists():
        write_json(run_dir / "config.json", config)
    if not (run_dir / "execution.json").exists():
        execution = json.loads((source / "execution.json").read_text())
        execution["restored_from_committed_archive"] = True
        write_json(run_dir / "execution.json", execution)
    return restored


if __name__ == "__main__":
    config = load_config(ROOT / "configs/needle_chain/pilot.json")
    count = restore_completed(ROOT / "results/needle_chain/pilot", ROOT / "runs/needle_chain/pilot", config)
    print(f"Restored {count} completed episodes; live execution occurs only in the separate pilot CLI step.")
