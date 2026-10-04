#!/usr/bin/env python3
"""Single entry point for tests, quick analysis, and full simulation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.assemble_results import assemble_full, assemble_quick, write_manifest
from src.bootstrap_dc import run_bootstrap
from src.classify_clean import freeze_classification
from src.fit_scaling import run_fit
from src.scan_dc import run_full_scan
from src.utils import PACKAGE_ROOT
from src.validate_reproduction import validate_quick


def run_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=PACKAGE_ROOT,
        check=True,
    )


def run_quick(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    frame = assemble_quick(output_root)
    write_manifest(frame, output_root, "quick")
    run_fit(
        output_root / "data" / "master_resolved_cells.csv",
        PACKAGE_ROOT / "data" / "bootstrap",
        output_root,
    )
    result = validate_quick(output_root)
    print(f"quick reproduction: {result['status']} ({output_root})")


def run_full(output_root: Path, workers: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw"
    run_full_scan(raw_root, workers)
    run_bootstrap(raw_root, output_root)
    freeze_classification(raw_root, output_root)
    frame = assemble_full(output_root, output_root)
    write_manifest(frame, output_root, "full")
    run_fit(
        output_root / "data" / "master_resolved_cells.csv",
        output_root / "bootstrap",
        output_root,
    )
    print(f"full reproduction completed and matched deterministic references ({output_root})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--quick", action="store_true", help="refit and redraw from compact data")
    modes.add_argument("--full", action="store_true", help="rerun all frozen simulations and analysis")
    modes.add_argument("--test", action="store_true", help="run unit and regression tests")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.test:
        run_tests()
    elif args.quick:
        run_quick(args.output or PACKAGE_ROOT / "reproduced" / "quick")
    else:
        run_full(args.output or PACKAGE_ROOT / "reproduced" / "full", args.workers)


if __name__ == "__main__":
    main()

