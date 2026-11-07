"""Read-only verification of the frozen v1 evidence and v2 path isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import ROOT


def verify_frozen_v1(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "docs/needle_chain/v1_frozen_hashes.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen v1 artifact changed: {name}")
    return {"frozen_commit": manifest["frozen_commit"], "status": "unchanged",
            "verified_files": len(manifest["files"])}


def guard_v2_paths(run_dir: Path, output: Path) -> None:
    for path in (run_dir.resolve(), output.resolve()):
        for protected in (ROOT / "runs/needle_chain/pilot", ROOT / "results/needle_chain/pilot"):
            if path == protected or protected in path.parents:
                raise ValueError("v2 cannot read or write its state in a frozen v1 directory")


def verify_protected_pilots(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "docs/needle_chain/protected_pilots_full_v2.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"protected pilot changed: {name}")
    return {"status": "unchanged", "files_verified": len(manifest["files"]),
            "v1_frozen_commit": manifest["v1_frozen_commit"], "v2_frozen_commit": manifest["v2_frozen_commit"]}


def guard_full_paths(run_dir: Path, output: Path) -> None:
    for path in (run_dir.resolve(), output.resolve()):
        for label in ("pilot", "pilot_v2"):
            for kind in ("runs", "results"):
                protected = ROOT / kind / "needle_chain" / label
                if path == protected or protected in path.parents:
                    raise ValueError("full v2 must not use a protected pilot directory")


def verify_full_design(root: Path = ROOT) -> dict:
    path = root / "docs/needle_chain/full_v2_freeze.json"
    manifest = json.loads(path.read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen full design changed: {name}")
    return {"status": "unchanged", "files_verified": len(manifest["sha256"])}
