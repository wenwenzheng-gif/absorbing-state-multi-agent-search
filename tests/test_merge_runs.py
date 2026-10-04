#!/usr/bin/env python3
"""Tests for ``scripts/merge_runs.py`` and the ``--seed-index`` flag of
``scripts/make_configs.py``.

``scripts/`` is not a package, so both modules are imported by putting
``scripts/`` on ``sys.path`` (mirroring how the scripts themselves resolve
``REPO_ROOT``). Everything here runs against synthetic fake config/run
directories in a temp dir -- no real ``configs/agents/`` or
``runs/agents/`` content is read or written.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_configs  # noqa: E402
import merge_runs  # noqa: E402


# --------------------------------------------------------------------------
# make_configs.py --seed-index
# --------------------------------------------------------------------------


class SeedIndexTagTests(unittest.TestCase):
    def test_tag_is_model_short_seed_index(self) -> None:
        self.assertEqual(make_configs.make_tag("Qwen/Qwen3.5-4B", 1), "qwen3.5-4b_seed1")
        self.assertEqual(make_configs.make_tag("Qwen/Qwen3.5-4B", 2), "qwen3.5-4b_seed2")
        self.assertEqual(make_configs.make_tag("Qwen/Qwen3.5-4B", 3), "qwen3.5-4b_seed3")


class SeedIndexGenerationTests(unittest.TestCase):
    def test_single_seed_index_takes_the_ith_reference_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            rc = make_configs.main(
                ["--model", "Qwen/Qwen3.5-4B", "--seed-index", "2", "--out", str(out_dir)]
            )
            self.assertEqual(rc, 0)
            for cell, spec in make_configs.CELL_SPECS.items():
                for policy in make_configs.POLICY_SPECS:
                    payload = json.loads((out_dir / f"{cell}_{policy}.json").read_text())
                    self.assertEqual(payload["task_seeds"], [spec["task_seeds"][1]])
                    self.assertEqual(payload["name"], f"qwen3.5-4b_seed2_{cell}_{policy}")
                    self.assertEqual(payload["output_root"], "runs/agents/qwen3.5-4b_seed2")

    def test_multiple_seed_indices_write_one_folder_each(self) -> None:
        # --seed-index with more than one value and no --out writes under
        # the default BASE_CONFIG_DIR/<tag> (one folder per index) -- redirect
        # BASE_CONFIG_DIR to a tmpdir (pre-seeded with the base cell configs
        # it reads from) instead of touching the repo's real config tree.
        with tempfile.TemporaryDirectory() as tmp:
            configs_root = Path(tmp) / "configs" / "agents"
            configs_root.mkdir(parents=True)
            for spec in make_configs.CELL_SPECS.values():
                base_name = spec["base"]
                (configs_root / base_name).write_text(
                    (make_configs.BASE_CONFIG_DIR / base_name).read_text(), encoding="utf-8"
                )
            original_base_dir = make_configs.BASE_CONFIG_DIR
            make_configs.BASE_CONFIG_DIR = configs_root
            try:
                rc = make_configs.main(
                    ["--model", "Qwen/Qwen3.5-4B", "--seed-index", "1", "2", "3"]
                )
            finally:
                make_configs.BASE_CONFIG_DIR = original_base_dir
            self.assertEqual(rc, 0)
            for seed_index in (1, 2, 3):
                out_dir = configs_root / f"qwen3.5-4b_seed{seed_index}"
                for cell, spec in make_configs.CELL_SPECS.items():
                    for policy in make_configs.POLICY_SPECS:
                        payload = json.loads((out_dir / f"{cell}_{policy}.json").read_text())
                        self.assertEqual(payload["task_seeds"], [spec["task_seeds"][seed_index - 1]])

    def test_each_seed_config_differs_from_seed1_only_in_seed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "seed1"
            out2 = Path(tmp) / "seed2"
            make_configs.main(["--model", "Qwen/Qwen3.5-4B", "--seed-index", "1", "--out", str(out1)])
            make_configs.main(["--model", "Qwen/Qwen3.5-4B", "--seed-index", "2", "--out", str(out2)])
            for cell in make_configs.CELL_SPECS:
                for policy in make_configs.POLICY_SPECS:
                    cfg1 = json.loads((out1 / f"{cell}_{policy}.json").read_text())
                    cfg2 = json.loads((out2 / f"{cell}_{policy}.json").read_text())
                    for key in ("name", "notes", "output_root", "task_seeds"):
                        cfg1.pop(key)
                        cfg2.pop(key)
                    self.assertEqual(cfg1, cfg2)

    def test_seed_index_out_of_range_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                make_configs.main(
                    ["--model", "Qwen/Qwen3.5-4B", "--seed-index", "7", "--out", str(Path(tmp) / "out")]
                )

    def test_zero_seed_index_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                make_configs.main(
                    ["--model", "Qwen/Qwen3.5-4B", "--seed-index", "0", "--out", str(Path(tmp) / "out")]
                )

    def test_out_with_multiple_seed_indices_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                make_configs.main(
                    [
                        "--model",
                        "Qwen/Qwen3.5-4B",
                        "--seed-index",
                        "1",
                        "2",
                        "--out",
                        str(Path(tmp) / "out"),
                    ]
                )


# --------------------------------------------------------------------------
# merge_runs.py
# --------------------------------------------------------------------------

CELLS = merge_runs.CELLS
POLICIES = merge_runs.POLICIES


def _fake_config(tag: str, cell: str, policy: str, task_seeds: list[int], degrees: list[int]) -> dict:
    return {
        "name": f"{tag}_{cell}_{policy}",
        "agents": 4,
        "branching": 2,
        "experiments_per_agent": 1,
        "initial_width": 4,
        "degrees": degrees,
        "task_seeds": task_seeds,
        "repeats": 1,
        "environment": {"kind": cell, "families": 2, "variants": 2, "task_size": 2},
        "policy": {
            "kind": "llm_full",
            "llm": {
                "model": "m",
                "team_objective": "individual" if policy == "I1" else "simple",
                "communication_mode": "fixed" if policy == "C0" else "free",
            },
        },
        "output_root": f"runs/agents/{tag}",
        "notes": f"{policy} note for {cell}",
    }


def _write_source_cell(
    configs_root: Path,
    runs_root: Path,
    tag: str,
    cell: str,
    policy: str,
    task_seeds: list[int],
    degrees: list[int],
    *,
    status: str = "ok",
    n_episodes_override: int | None = None,
    with_junk_shards_dir: bool = False,
    failed_episode_ids: list[str] | None = None,
    failed_status: str = "policy_failed",
    failed_error: str = "boom",
    also_failed_shards: bool = False,
) -> None:
    config = _fake_config(tag, cell, policy, task_seeds, degrees)
    cfg_dir = configs_root / tag
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{cell}_{policy}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_id = f"{tag}_{cell}_{policy}"
    rdir = runs_root / tag / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    episode_ids = [f"d{d}_t{s}_r0" for d in degrees for s in task_seeds]
    if n_episodes_override is not None:
        episode_ids = episode_ids[:n_episodes_override]
    failed_episode_ids = set(failed_episode_ids or [])

    episode_statuses: dict[str, str] = {}
    with (rdir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for eid in episode_ids:
            if eid in failed_episode_ids:
                record = {
                    "episode_id": eid,
                    "status": failed_status,
                    "note": f"episode:{tag}",
                    "error": failed_error,
                }
            else:
                record = {"episode_id": eid, "status": status, "note": f"episode:{tag}"}
            episode_statuses[eid] = record["status"]
            fh.write(json.dumps(record) + "\n")
    for fname in ("events.jsonl", "requests.jsonl", "responses.jsonl"):
        with (rdir / fname).open("w", encoding="utf-8") as fh:
            for eid in episode_ids:
                fh.write(json.dumps({"episode": eid, "note": f"{fname}:{tag}"}) + "\n")

    statuses: dict[str, int] = {}
    for st in episode_statuses.values():
        statuses[st] = statuses.get(st, 0) + 1
    run_summary = {
        "run_id": run_id,
        "failed_shards": (
            [{"run_id": f"{run_id}_shard0", "note": "not ok"}]
            if also_failed_shards
            else []
        ),
        "statuses": statuses,
        "shards": len(degrees) * len(task_seeds),
        "wall_seconds": 10.0,
        "llm_usage": {"llm_requests": len(episode_ids) * 2, "llm_prompt_tokens": len(episode_ids) * 100.0},
        "per_shard_wall_seconds": {f"{run_id}_shard{i}": 1.0 for i in range(len(episode_ids))},
    }
    (rdir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    if with_junk_shards_dir:
        shards_dir = rdir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        (shards_dir / "junk.json").write_text("{}", encoding="utf-8")


def _write_all_nine(
    configs_root: Path,
    runs_root: Path,
    tag: str,
    task_seeds: list[int],
    degrees: list[int] = (0, 5),
    **kwargs,
) -> None:
    for cell in CELLS:
        for policy in POLICIES:
            _write_source_cell(configs_root, runs_root, tag, cell, policy, list(task_seeds), list(degrees), **kwargs)


class MergeRunsSuccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.configs_root = self.root / "configs" / "agents"
        self.runs_root = self.root / "runs" / "agents"
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100], with_junk_shards_dir=True)
        _write_all_nine(self.configs_root, self.runs_root, "tagB", [101, 102], with_junk_shards_dir=True)

    def test_merge_orders_degree_major_then_union_seed(self) -> None:
        merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        out_dir = self.runs_root / "out" / "out_synthetic_C0"
        lines = (out_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        got = [json.loads(line)["episode_id"] for line in lines]
        expected = [
            "d0_t100_r0", "d0_t101_r0", "d0_t102_r0",
            "d5_t100_r0", "d5_t101_r0", "d5_t102_r0",
        ]
        self.assertEqual(got, expected)
        # events/requests/responses follow the same order (grouped by
        # "episode" instead of "episode_id").
        for fname in ("events.jsonl", "requests.jsonl", "responses.jsonl"):
            other_lines = (out_dir / fname).read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["episode"] for line in other_lines], expected)

    def test_union_task_seeds_and_config_in_manifest_only(self) -> None:
        merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        manifest = json.loads(
            (self.runs_root / "out" / "out_synthetic_C0" / "manifest.json").read_text()
        )
        self.assertEqual(manifest["config"]["task_seeds"], [100, 101, 102])
        self.assertEqual(manifest["config"]["name"], "out_synthetic_C0")
        self.assertEqual(manifest["config"]["output_root"], "runs/agents/out")
        sources = {entry["tag"]: entry["task_seeds"] for entry in manifest["merged_from"]}
        self.assertEqual(sources, {"tagA": [100], "tagB": [101, 102]})

        # An out-tag is a pooled analysis view, not a new task seed: it must
        # NOT get its own configs/agents/<out_tag>/ folder. The merged config
        # lives only in manifest.json above.
        self.assertFalse((self.configs_root / "out").exists())

        # manifest["config"] equals tagA's config except
        # name/notes/output_root/task_seeds.
        written_config = dict(manifest["config"])
        base = json.loads((self.configs_root / "tagA" / "synthetic_C0.json").read_text())
        for key in ("name", "notes", "output_root", "task_seeds"):
            base.pop(key)
            written_config.pop(key)
        self.assertEqual(written_config, base)

    def test_run_summary_sums_llm_usage_and_statuses(self) -> None:
        merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        summary = json.loads(
            (self.runs_root / "out" / "out_synthetic_C0" / "run_summary.json").read_text()
        )
        # tagA: 2 episodes (1 seed x 2 degrees) -> llm_requests=4, prompt_tokens=200
        # tagB: 4 episodes (2 seeds x 2 degrees) -> llm_requests=8, prompt_tokens=400
        self.assertEqual(summary["llm_usage"]["llm_requests"], 12)
        self.assertEqual(summary["llm_usage"]["llm_prompt_tokens"], 600.0)
        self.assertEqual(summary["statuses"], {"ok": 6})
        self.assertEqual(summary["failed_shards"], [])
        self.assertEqual(summary["records"]["episodes.jsonl"], 6)

    def test_all_nine_cells_written(self) -> None:
        merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        for cell in CELLS:
            for policy in POLICIES:
                rdir = self.runs_root / "out" / f"out_{cell}_{policy}"
                self.assertTrue((rdir / "episodes.jsonl").exists(), rdir)
                self.assertTrue((rdir / "manifest.json").exists())
                self.assertTrue((rdir / "run_summary.json").exists())
        # No configs/agents/out/ folder at all -- pooled tags are runs-only.
        self.assertFalse((self.configs_root / "out").exists())

    def test_shards_subdir_never_copied_and_sources_untouched(self) -> None:
        before_a = (self.configs_root / "tagA" / "synthetic_C0.json").read_text()
        merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        out_dir = self.runs_root / "out" / "out_synthetic_C0"
        self.assertFalse((out_dir / "shards").exists())
        after_a = (self.configs_root / "tagA" / "synthetic_C0.json").read_text()
        self.assertEqual(before_a, after_a)
        self.assertTrue((self.runs_root / "tagA" / "tagA_synthetic_C0" / "shards" / "junk.json").exists())


class MergeRunsRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.configs_root = self.root / "configs" / "agents"
        self.runs_root = self.root / "runs" / "agents"

    def _assert_nothing_written(self) -> None:
        self.assertFalse((self.runs_root / "out").exists())
        self.assertFalse((self.configs_root / "out").exists())

    def test_refuses_on_missing_run_dir(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        # tagB has configs but no run dirs at all.
        for cell in CELLS:
            for policy in POLICIES:
                cfg = _fake_config("tagB", cell, policy, [200], [0, 5])
                cfg_dir = self.configs_root / "tagB"
                cfg_dir.mkdir(parents=True, exist_ok=True)
                (cfg_dir / f"{cell}_{policy}.json").write_text(json.dumps(cfg), encoding="utf-8")

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("run dir not found", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_missing_run_summary(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        _write_all_nine(self.configs_root, self.runs_root, "tagB", [101, 102])
        (self.runs_root / "tagB" / "tagB_synthetic_C0" / "run_summary.json").unlink()

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("missing", str(ctx.exception))
        self.assertIn("run_summary.json", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_failed_shard(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        _write_all_nine(self.configs_root, self.runs_root, "tagB", [101, 102])
        summary_path = self.runs_root / "tagB" / "tagB_synthetic_C0" / "run_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["failed_shards"] = [{"run_id": "tagB_synthetic_C0_d0_s101", "returncode": 1}]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("failed_shards", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_non_ok_status(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        _write_source_cell(
            self.configs_root, self.runs_root, "tagB", "synthetic", "C0", [101, 102], [0, 5],
            status="policy_failed",
        )
        for cell in CELLS:
            for policy in POLICIES:
                if (cell, policy) == ("synthetic", "C0"):
                    continue
                _write_source_cell(self.configs_root, self.runs_root, "tagB", cell, policy, [101, 102], [0, 5])

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("non-ok episode statuses", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_fewer_episodes_than_planned(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        _write_source_cell(
            self.configs_root, self.runs_root, "tagB", "synthetic", "C0", [101, 102], [0, 5],
            n_episodes_override=2,  # config plans 4 (2 seeds x 2 degrees)
        )
        for cell in CELLS:
            for policy in POLICIES:
                if (cell, policy) == ("synthetic", "C0"):
                    continue
                _write_source_cell(self.configs_root, self.runs_root, "tagB", cell, policy, [101, 102], [0, 5])

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("incomplete", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_config_mismatch(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        _write_all_nine(self.configs_root, self.runs_root, "tagB", [101, 102])
        cfg_path = self.configs_root / "tagB" / "synthetic_C0.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["agents"] = 999  # not in the allowed-diff set
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("differ in field(s)", str(ctx.exception))
        self.assertIn("agents", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_overlapping_seeds(self) -> None:
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        _write_all_nine(self.configs_root, self.runs_root, "tagB", [100, 101])  # 100 overlaps

        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("overlapping seeds", str(ctx.exception))
        self._assert_nothing_written()

    def test_refuses_on_single_from_tag(self) -> None:
        with self.assertRaises(merge_runs.MergeError):
            merge_runs.merge_all("out", ["tagA"], self.configs_root, self.runs_root)

    def test_refuses_on_duplicate_from_tags(self) -> None:
        with self.assertRaises(merge_runs.MergeError):
            merge_runs.merge_all("out", ["tagA", "tagA"], self.configs_root, self.runs_root)


class MergeRunsAllowFailedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.configs_root = self.root / "configs" / "agents"
        self.runs_root = self.root / "runs" / "agents"
        _write_all_nine(self.configs_root, self.runs_root, "tagA", [100])
        # tagB's synthetic_C0 cell has one failed episode (d5_t101_r0, plus a
        # matching failed_shards entry) alongside the rest "ok" -- exactly
        # the qwen3.5-9b_seed3 tcas_I1 shape (episodes still recorded,
        # counts match the plan, just a non-"ok" status).
        _write_source_cell(
            self.configs_root, self.runs_root, "tagB", "synthetic", "C0", [101, 102], [0, 5],
            failed_episode_ids=["d5_t101_r0"],
            failed_status="policy_failed",
            failed_error="x" * 250,  # longer than 200 chars, to check truncation
            also_failed_shards=True,
        )
        for cell in CELLS:
            for policy in POLICIES:
                if (cell, policy) == ("synthetic", "C0"):
                    continue
                _write_source_cell(self.configs_root, self.runs_root, "tagB", cell, policy, [101, 102], [0, 5])

    def test_default_still_refuses(self) -> None:
        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all("out", ["tagA", "tagB"], self.configs_root, self.runs_root)
        self.assertIn("failed_shards", str(ctx.exception))
        self.assertFalse((self.runs_root / "out").exists())

    def test_allow_failed_merges_the_cell(self) -> None:
        plans = merge_runs.merge_all(
            "out", ["tagA", "tagB"], self.configs_root, self.runs_root, allow_failed=True
        )
        self.assertEqual(len(plans), 9)
        out_dir = self.runs_root / "out" / "out_synthetic_C0"
        lines = (out_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        got = {json.loads(line)["episode_id"]: json.loads(line)["status"] for line in lines}
        self.assertEqual(got["d5_t101_r0"], "policy_failed")
        self.assertEqual(got["d0_t100_r0"], "ok")
        # All the *other* 8 cells (no failures) are unaffected.
        for cell in CELLS:
            for policy in POLICIES:
                if (cell, policy) == ("synthetic", "C0"):
                    continue
                other_dir = self.runs_root / "out" / f"out_{cell}_{policy}"
                statuses = {
                    json.loads(line)["status"]
                    for line in (other_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                }
                self.assertEqual(statuses, {"ok"})

    def test_allow_failed_records_failed_episodes_and_real_statuses(self) -> None:
        merge_runs.merge_all(
            "out", ["tagA", "tagB"], self.configs_root, self.runs_root, allow_failed=True
        )
        out_dir = self.runs_root / "out" / "out_synthetic_C0"
        manifest = json.loads((out_dir / "manifest.json").read_text())
        summary = json.loads((out_dir / "run_summary.json").read_text())

        for payload in (manifest, summary):
            failed = payload["failed_episodes"]
            self.assertEqual(len(failed), 1)
            entry = failed[0]
            self.assertEqual(entry["episode_id"], "d5_t101_r0")
            self.assertEqual(entry["d"], 5)
            self.assertEqual(entry["task_seed"], 101)
            self.assertEqual(entry["status"], "policy_failed")
            self.assertEqual(entry["source_tag"], "tagB")
            self.assertEqual(entry["error"], "x" * 200)  # truncated to 200 chars

        # 1 (tagA) + 2 (tagB) x 2 degrees = 6 episodes total: 5 ok, 1 failed.
        self.assertEqual(summary["statuses"], {"ok": 5, "policy_failed": 1})
        # Sources' own failed_shards is not copied into the merged cell.
        self.assertEqual(summary["failed_shards"], [])

    def test_allow_failed_still_refuses_on_fewer_episodes_than_planned(self) -> None:
        _write_source_cell(
            self.configs_root, self.runs_root, "tagB", "physics", "C1", [101, 102], [0, 5],
            n_episodes_override=2,  # config plans 4
        )
        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all(
                "out", ["tagA", "tagB"], self.configs_root, self.runs_root, allow_failed=True
            )
        self.assertIn("incomplete", str(ctx.exception))
        self.assertFalse((self.runs_root / "out").exists())

    def test_allow_failed_still_refuses_on_config_mismatch(self) -> None:
        cfg_path = self.configs_root / "tagB" / "physics_C1.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["agents"] = 999
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all(
                "out", ["tagA", "tagB"], self.configs_root, self.runs_root, allow_failed=True
            )
        self.assertIn("differ in field(s)", str(ctx.exception))
        self.assertFalse((self.runs_root / "out").exists())

    def test_allow_failed_still_refuses_on_overlapping_seeds(self) -> None:
        _write_source_cell(
            self.configs_root, self.runs_root, "tagB", "physics", "C1", [100, 102], [0, 5],
        )
        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all(
                "out", ["tagA", "tagB"], self.configs_root, self.runs_root, allow_failed=True
            )
        self.assertIn("overlapping seeds", str(ctx.exception))
        self.assertFalse((self.runs_root / "out").exists())

    def test_allow_failed_still_refuses_on_missing_run_dir(self) -> None:
        import shutil

        shutil.rmtree(self.runs_root / "tagB" / "tagB_physics_C1")
        with self.assertRaises(merge_runs.MergeError) as ctx:
            merge_runs.merge_all(
                "out", ["tagA", "tagB"], self.configs_root, self.runs_root, allow_failed=True
            )
        self.assertIn("run dir not found", str(ctx.exception))
        self.assertFalse((self.runs_root / "out").exists())

    def test_cli_flag_wires_through(self) -> None:
        rc = merge_runs.main(
            [
                "--out-tag", "out",
                "--from-tags", "tagA", "tagB",
                "--configs-dir", str(self.configs_root),
                "--runs-dir", str(self.runs_root),
                "--allow-failed",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.runs_root / "out" / "out_synthetic_C0" / "episodes.jsonl").exists())

    def test_cli_flag_default_off(self) -> None:
        rc = merge_runs.main(
            [
                "--out-tag", "out",
                "--from-tags", "tagA", "tagB",
                "--configs-dir", str(self.configs_root),
                "--runs-dir", str(self.runs_root),
            ]
        )
        self.assertEqual(rc, 1)
        self.assertFalse((self.runs_root / "out").exists())


if __name__ == "__main__":
    unittest.main()
