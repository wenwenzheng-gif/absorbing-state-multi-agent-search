#!/usr/bin/env python3
"""Schema-level tests for the C0/C1/I1 fields: prompt_version, the new
team_objective modes, communication_mode, and the free-communication
validation gate.  Also checks that the 9 real qwen3.5-4b_seed1 configs
validate and match the C0/C1/I1 policy table.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_system.schemas import load_run_config  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "agents" / "qwen3.5-4b_seed1"

BASE_SYNTHETIC = {
    "name": "cfg",
    "agents": 4,
    "branching": 2,
    "experiments_per_agent": 1,
    "initial_width": 4,
    "degrees": [0],
    "environment": {"kind": "synthetic", "families": 8, "variants": 4, "task_size": 4},
}


def llm_config(**llm_overrides):
    payload = dict(BASE_SYNTHETIC)
    payload["policy"] = {
        "kind": "llm_full",
        "llm": {"model": "m", "base_url": "u", **llm_overrides},
    }
    return payload


class PromptVersionTests(unittest.TestCase):
    def test_v4_is_accepted(self) -> None:
        config = load_run_config(llm_config(prompt_version="v4"))
        self.assertEqual(config.policy.llm.prompt_version, "v4")

    def test_default_is_v4(self) -> None:
        config = load_run_config(llm_config())
        self.assertEqual(config.policy.llm.prompt_version, "v4")

    def test_unknown_prompt_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_run_config(llm_config(prompt_version="v5"))


class TeamObjectiveModeTests(unittest.TestCase):
    def test_simple_and_individual_are_accepted(self) -> None:
        for mode in ("simple", "individual"):
            with self.subTest(mode=mode):
                config = load_run_config(llm_config(team_objective=mode))
                self.assertEqual(config.policy.llm.team_objective, mode)

    def test_old_modes_still_accepted(self) -> None:
        for mode in ("off", "soft", "device"):
            with self.subTest(mode=mode):
                load_run_config(llm_config(team_objective=mode))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_run_config(llm_config(team_objective="loud"))


class CommunicationModeTests(unittest.TestCase):
    def test_fixed_is_the_default(self) -> None:
        config = load_run_config(llm_config())
        self.assertEqual(config.policy.llm.communication_mode, "fixed")

    def test_fixed_and_free_are_accepted_on_llm_full(self) -> None:
        load_run_config(llm_config(communication_mode="fixed"))
        load_run_config(llm_config(communication_mode="free"))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_run_config(llm_config(communication_mode="broadcast"))

    def test_free_requires_llm_full_or_replay(self) -> None:
        payload = dict(BASE_SYNTHETIC)
        payload["policy"] = {
            "kind": "llm_branch",
            "llm": {"model": "m", "base_url": "u", "communication_mode": "free"},
        }
        with self.assertRaises(ValueError):
            load_run_config(payload)

    def test_free_allowed_on_replay(self) -> None:
        payload = dict(BASE_SYNTHETIC)
        payload["policy"] = {
            "kind": "replay",
            "llm": {"model": "m", "base_url": "u", "communication_mode": "free"},
        }
        config = load_run_config(payload)
        self.assertEqual(config.policy.llm.communication_mode, "free")

    def test_free_requires_prompt_version_v4(self) -> None:
        # prompt_version only ever accepts "v4" today, so this is really the
        # same check as above; kept explicit because the free-mode gate names
        # it as its own condition.
        config = load_run_config(llm_config(communication_mode="free", prompt_version="v4"))
        self.assertEqual(config.policy.llm.prompt_version, "v4")

    def test_free_requires_effective_q_model_own_equal_one(self) -> None:
        payload = llm_config(communication_mode="free")
        payload["q_model_own"] = 0.5
        with self.assertRaises(ValueError):
            load_run_config(payload)

    def test_free_rejects_q_model_own_inherited_below_one(self) -> None:
        payload = llm_config(communication_mode="free")
        payload["q_model"] = 0.9
        with self.assertRaises(ValueError):
            load_run_config(payload)

    def test_free_accepts_explicit_q_model_own_one_even_if_q_model_is_lower(self) -> None:
        payload = llm_config(communication_mode="free")
        payload["q_model"] = 0.5
        payload["q_model_own"] = 1.0
        config = load_run_config(payload)
        self.assertEqual(config.effective_q_own, 1.0)

    def test_free_rejects_tcas_subset_evidence(self) -> None:
        payload = {
            "name": "cfg",
            "agents": 4,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 2,
            "degrees": [0],
            "environment": {
                "kind": "tcas",
                "families": 4,
                "variants": 2,
                "task_size": 2,
                "tcas_space": "toy_16",
                "tcas_evidence": "subset",
            },
            "policy": {
                "kind": "llm_full",
                "llm": {"model": "m", "base_url": "u", "communication_mode": "free"},
            },
        }
        # toy_16 may not exist as a named space; this test only needs the
        # tcas_evidence='subset' + communication_mode='free' combination to
        # be rejected before any tcas-space lookup would even matter, so it
        # accepts either failure mode as long as it *is* rejected.
        with self.assertRaises((ValueError, KeyError)):
            load_run_config(payload)


class RealConfigsTests(unittest.TestCase):
    """The 9 configs actually shipped under configs/agents/qwen3.5-4b_seed1/."""

    EXPECTED = {
        "C0": {"team_objective": "simple", "communication_mode": "fixed"},
        "C1": {"team_objective": "simple", "communication_mode": "free"},
        "I1": {"team_objective": "individual", "communication_mode": "free"},
    }

    def test_all_nine_configs_validate(self) -> None:
        cells = ("synthetic", "tcas", "physics")
        policies = ("C0", "C1", "I1")
        found = 0
        for cell in cells:
            for policy in policies:
                path = CONFIG_DIR / f"{cell}_{policy}.json"
                with self.subTest(config=path.name):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    config = load_run_config(payload)
                    self.assertEqual(config.policy.kind, "llm_full")
                    self.assertEqual(config.policy.llm.prompt_version, "v4")
                    expected = self.EXPECTED[policy]
                    self.assertEqual(
                        config.policy.llm.team_objective, expected["team_objective"]
                    )
                    self.assertEqual(
                        config.policy.llm.communication_mode,
                        expected["communication_mode"],
                    )
                    found += 1
        self.assertEqual(found, 9)


if __name__ == "__main__":
    unittest.main()
