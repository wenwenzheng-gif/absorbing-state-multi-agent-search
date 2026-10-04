#!/usr/bin/env python3
"""``--split {seed,degree}``, merge ordering, llm_usage summation, --concurrency.

``shard.py`` makes one process per task seed today (``--split seed``, the
default and unchanged behaviour).  ``--split degree`` makes one process per
(degree, task_seed) pair instead, so a single-seed, multi-degree cell can run
every degree at once against the local vLLM server.  These tests exercise
``shard_configs``, ``merge`` and ``aggregate_shard_summaries`` directly,
without running any episode or spawning any subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.schemas import load_run_config  # noqa: E402
from src.agent_system.shard import (  # noqa: E402
    aggregate_shard_summaries,
    merge,
    shard_configs,
)


def _config(**overrides):
    payload = {
        "name": "shardtest",
        "agents": 8,
        "branching": 2,
        "experiments_per_agent": 1,
        "initial_width": 4,
        "degrees": [0, 2],
        "task_seeds": [11, 22],
        "repeats": 2,
        "environment": {"kind": "synthetic", "families": 4, "variants": 2, "task_size": 3},
        "policy": {"kind": "uniform"},
    }
    payload.update(overrides)
    return load_run_config(payload)


class ShardConfigsSeedModeTests(unittest.TestCase):
    """``--split seed`` (default) must stay byte-identical."""

    def test_default_split_is_seed(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            explicit = shard_configs(config, Path(tmp) / "explicit", split="seed")
            default = shard_configs(config, Path(tmp) / "default")
            self.assertEqual(
                [(seed, run_id) for seed, _p, run_id in explicit],
                [(seed, run_id) for seed, _p, run_id in default],
            )

    def test_one_shard_per_seed_keeping_every_degree(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            shards = shard_configs(config, Path(tmp), split="seed")
            self.assertEqual([run_id for _s, _p, run_id in shards], ["shardtest_s11", "shardtest_s22"])
            for seed, path, run_id in shards:
                body = json.loads(path.read_text())
                self.assertEqual(body["task_seeds"], [seed])
                self.assertEqual(body["degrees"], [0, 2])
                load_run_config(body).validate()

    def test_grn_targets_sliced_by_seed_position(self) -> None:
        targets = ["1:g1", "2:g2", "3:g3"]
        config = _config(
            degrees=[0],
            task_seeds=[1, 2, 3],
            environment={
                "kind": "grn",
                "families": 99,
                "variants": 2,
                "task_size": 3,
                "grn_targets": targets,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            shards = shard_configs(config, Path(tmp), split="seed")
            self.assertEqual(len(shards), 3)
            for position, (seed, path, _run_id) in enumerate(shards):
                body = json.loads(path.read_text())
                self.assertEqual(body["environment"]["grn_targets"], [targets[position]])
                load_run_config(body).validate()


class ShardConfigsDegreeModeTests(unittest.TestCase):
    def test_one_shard_per_degree_seed_pair_degree_major(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            shards = shard_configs(config, Path(tmp), split="degree")
            self.assertEqual(
                [run_id for _s, _p, run_id in shards],
                [
                    "shardtest_d0_s11",
                    "shardtest_d0_s22",
                    "shardtest_d2_s11",
                    "shardtest_d2_s22",
                ],
            )
            expected_bodies = [(0, 11), (0, 22), (2, 11), (2, 22)]
            for (degree, seed), (_s, path, _run_id) in zip(expected_bodies, shards):
                body = json.loads(path.read_text())
                self.assertEqual(body["degrees"], [degree])
                self.assertEqual(body["task_seeds"], [seed])
                self.assertEqual(body["repeats"], 2)
                load_run_config(body).validate()

    def test_grn_targets_sliced_by_seed_position_in_degree_mode(self) -> None:
        targets = ["1:g1", "2:g2"]
        config = _config(
            degrees=[0, 5],
            task_seeds=[7, 8],
            environment={
                "kind": "grn",
                "families": 99,
                "variants": 2,
                "task_size": 3,
                "grn_targets": targets,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            shards = shard_configs(config, Path(tmp), split="degree")
            self.assertEqual(len(shards), 4)
            by_run_id = {run_id: path for _s, path, run_id in shards}
            for run_id, expected_target in (
                ("shardtest_d0_s7", "1:g1"),
                ("shardtest_d0_s8", "2:g2"),
                ("shardtest_d5_s7", "1:g1"),
                ("shardtest_d5_s8", "2:g2"),
            ):
                body = json.loads(by_run_id[run_id].read_text())
                self.assertEqual(body["environment"]["grn_targets"], [expected_target])
                load_run_config(body).validate()

    def test_unknown_split_rejected(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                shard_configs(config, Path(tmp), split="bogus")


class ConcurrencyOverrideTests(unittest.TestCase):
    def test_no_override_keeps_config_value(self) -> None:
        config = _config(policy={"kind": "uniform", "llm": {"concurrency": 7}})
        with tempfile.TemporaryDirectory() as tmp:
            for split in ("seed", "degree"):
                shards = shard_configs(config, Path(tmp) / split, split=split)
                for _s, path, _run_id in shards:
                    body = json.loads(path.read_text())
                    self.assertEqual(body["policy"]["llm"]["concurrency"], 7)

    def test_override_applies_to_every_shard_both_modes(self) -> None:
        config = _config(policy={"kind": "uniform", "llm": {"concurrency": 7}})
        with tempfile.TemporaryDirectory() as tmp:
            for split in ("seed", "degree"):
                shards = shard_configs(config, Path(tmp) / split, split=split, concurrency=40)
                self.assertTrue(shards)
                for _s, path, _run_id in shards:
                    body = json.loads(path.read_text())
                    self.assertEqual(body["policy"]["llm"]["concurrency"], 40)
                    load_run_config(body).validate()
            # the original config is untouched (frozen dataclass, but check
            # anyway that we never mutated the shared payload in place)
            self.assertEqual(config.policy.llm.concurrency, 7)


class MergeOrderingTests(unittest.TestCase):
    """The merged episode order must match ``driver.episode_plan``:

    degree-major, then task seed, then repeat.
    """

    def _write_fake_episodes(self, shard_dir: Path, episode_ids: list[str]) -> None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        with (shard_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for episode_id in episode_ids:
                fh.write(json.dumps({"episode_id": episode_id, "status": "solved"}) + "\n")

    def test_degree_split_merges_in_episode_plan_order(self) -> None:
        config = _config()  # degrees=[0, 2], task_seeds=[11, 22], repeats=2
        with tempfile.TemporaryDirectory() as tmp:
            shard_root = Path(tmp) / "shards"
            merged = Path(tmp) / "merged"
            shards = shard_configs(config, shard_root, split="degree")
            for degree, seed, (_s, _p, run_id) in zip(
                [0, 0, 2, 2], [11, 22, 11, 22], shards
            ):
                episode_ids = [f"d{degree}_t{seed}_r{r}" for r in range(config.repeats)]
                self._write_fake_episodes(shard_root / run_id, episode_ids)

            merge(shard_root, shards, merged)
            lines = (merged / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            got = [json.loads(line)["episode_id"] for line in lines]

            expected = []
            for degree in config.degrees:
                for seed in config.task_seeds:
                    for repeat in range(config.repeats):
                        expected.append(f"d{degree}_t{seed}_r{repeat}")
            self.assertEqual(got, expected)

    def test_seed_split_merge_order_is_unchanged(self) -> None:
        # Regression lock on the pre-existing (seed-major, degree-minor)
        # order: shard_configs/merge for split="seed" must not change.
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            shard_root = Path(tmp) / "shards"
            merged = Path(tmp) / "merged"
            shards = shard_configs(config, shard_root, split="seed")
            for seed, (_s, _p, run_id) in zip([11, 22], shards):
                episode_ids = [
                    f"d{degree}_t{seed}_r{repeat}"
                    for degree in config.degrees
                    for repeat in range(config.repeats)
                ]
                self._write_fake_episodes(shard_root / run_id, episode_ids)

            merge(shard_root, shards, merged)
            lines = (merged / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            got = [json.loads(line)["episode_id"] for line in lines]

            expected = []
            for seed in config.task_seeds:
                for degree in config.degrees:
                    for repeat in range(config.repeats):
                        expected.append(f"d{degree}_t{seed}_r{repeat}")
            self.assertEqual(got, expected)


class AggregateShardSummariesTests(unittest.TestCase):
    def test_sums_numeric_llm_usage_fields_and_collects_wall_seconds(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            shard_root = Path(tmp) / "shards"
            shards = shard_configs(config, shard_root, split="degree")
            usages = [
                {"llm_requests": 10, "llm_prompt_tokens": 100.0, "llm_latency_s": 1.5},
                {"llm_requests": 5, "llm_prompt_tokens": 50.0, "llm_latency_s": 0.5},
                {"llm_requests": 7, "llm_prompt_tokens": 70.0, "llm_latency_s": 0.7},
                {"llm_requests": 3, "llm_prompt_tokens": 30.0, "llm_latency_s": 0.3},
            ]
            wall_seconds = [12.0, 8.0, 9.5, 4.25]
            for (_s, _p, run_id), usage, wall in zip(shards, usages, wall_seconds):
                shard_dir = shard_root / run_id
                shard_dir.mkdir(parents=True, exist_ok=True)
                (shard_dir / "run_summary.json").write_text(
                    json.dumps({"wall_seconds": wall, "llm_usage": usage}),
                    encoding="utf-8",
                )

            report = aggregate_shard_summaries(shard_root, shards)
            self.assertEqual(
                report["llm_usage"],
                {"llm_requests": 25, "llm_prompt_tokens": 250.0, "llm_latency_s": 3.0},
            )
            self.assertEqual(
                report["per_shard_wall_seconds"],
                {run_id: wall for (_s, _p, run_id), wall in zip(shards, wall_seconds)},
            )

    def test_missing_shard_summary_is_skipped_not_fatal(self) -> None:
        config = _config(degrees=[0], task_seeds=[11])
        with tempfile.TemporaryDirectory() as tmp:
            shard_root = Path(tmp) / "shards"
            shards = shard_configs(config, shard_root, split="seed")
            # No run_summary.json is written anywhere -- simulates a failed shard.
            report = aggregate_shard_summaries(shard_root, shards)
            self.assertEqual(report["llm_usage"], {})
            self.assertEqual(report["per_shard_wall_seconds"], {})


if __name__ == "__main__":
    unittest.main()
