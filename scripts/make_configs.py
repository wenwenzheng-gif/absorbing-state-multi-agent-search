#!/usr/bin/env python3
"""Generate the 9 C0/C1/I1 configs (3 cells x 3 policies) for one model/seed-count.

Reproducibly derives ``configs/agents/<tag>/{cell}_{policy}.json`` from:

* one of this repo's existing base configs per cell
  (``configs/agents/llm_toy.json`` / ``llm_tcas.json`` / ``llm_physics.json``),
  which already carry most of the shared LLM settings (``llm_full``,
  temperature 0.2, ``enable_thinking: false``, ``shuffle_candidates``,
  ``per_agent_seed``, ...);
* the reference cell table (N/M/K/T/m/n0/degrees), hard-coded below as
  ``CELL_SPECS``. These values mirror
  the reference configs ``{synthetic,tcas,physics}_{C0,C1,I1}.json``
  exactly (source runs: ``v3_llm_abl_toy_05_abcde`` for synthetic,
  ``v3_llm_tcas_T6b2_v4`` for tcas, ``v3_llm_phys_v16T4b2_v4`` for physics);
* the C0/C1/I1 policy table (``team_objective`` / ``communication_mode``),
  hard-coded below as ``POLICY_SPECS``, also mirroring the reference configs;
* ``--model`` and ``--seed-index`` (one or more 1-based indices into each
  cell's reference task-seed list), which together name one output folder
  per index, ``<model-short>_seed<I>``, e.g. ``qwen3.5-4b_seed1`` for the
  1st task seed, ``qwen3.5-4b_seed2`` for the 2nd. Each such folder holds 9
  configs (one per cell x policy), each with a single-element
  ``task_seeds`` list (the ``I``-th entry, 1-based, of that cell's
  reference seed table). The same script and the same downstream scripts
  (run_experiments.sh, analyze.sh) serve all of these, and also the
  eventual 8B run (``--model <8B id> --seed-index 1 2 3``, one folder per
  seed). ``scripts/merge_runs.py`` pools several such single-seed run
  batches (e.g. seed1 + seed2 + seed3) into one combined *analysis* tag
  (e.g. ``..._seed1-3``) that lives only under ``runs/agents/`` -- it does
  not get its own ``configs/agents/`` folder, since it is a derived,
  reproducible view over the per-seed run data.

Every generated config is required to be field-for-field identical to its
reference config EXCEPT for:
``name``, ``notes``, ``output_root``, ``policy.llm.model``, and
``task_seeds`` (a single-element list: the ``--seed-index``-th entry of the
reference list). This script enforces that every generated config is a
real LLM run (``policy.kind == "llm_full"``): no rule/uniform/adaptive
config is ever produced here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_DIR = REPO_ROOT / "configs" / "agents"

# Cell table, mirroring the reference configs in
# reference configs {synthetic,tcas,physics}_{C0,C1,I1}.json
# exactly. N=agents, M/K/T = environment.families/variants/task_size,
# m=experiments_per_agent, n0=initial_width, b=2 for every cell (branching).
# ``task_seeds`` is the reference cell's full task-seed list; ``--seeds N``
# takes the first N entries of it. The reference configs list 4 seeds per
# cell; entries 5-6 (e.g. 87004, 87005) extend each list consecutively so
# that --seed-index 5 6 can add further independent tasks.
#
# synthetic (source: v3_llm_abl_toy_05_abcde.json): N=40 M=8 K=2 T=4 m=1
#   n0=8, degrees [0,8,16,24,39], seeds 87000-87003.
# tcas (source: v3_llm_tcas_T6b2_v4.json): N=40 M=12 K=10 T=6 m=1 n0=8,
#   degrees [0,6,10,14,18,24,39], seeds 87200-87203, component-level
#   evidence (|V|=46).
# physics (source: v3_llm_phys_v16T4b2_v4.json): N=40 M=16 K=1 T=4 m=1 n0=8,
#   degrees [0,4,8,12,16,24,39], seeds 82240-82243, support library (|V|=16).
CELL_SPECS: dict[str, dict[str, Any]] = {
    "synthetic": {
        "base": "llm_toy.json",
        "top_overrides": {
            "agents": 40,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 8,
        },
        "env_overrides": {
            "kind": "synthetic",
            "families": 8,
            "variants": 2,
            "task_size": 4,
        },
        "env_keys": ("kind", "families", "variants", "task_size"),
        "degrees": [0, 8, 16, 24, 39],
        "task_seeds": [87000, 87001, 87002, 87003, 87004, 87005],
        "notes_source": "v3_llm_abl_toy_05_abcde.json",
    },
    "tcas": {
        "base": "llm_tcas.json",
        "top_overrides": {
            "agents": 40,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 8,
        },
        "env_overrides": {
            "kind": "tcas",
            "families": 12,
            "variants": 10,
            "task_size": 6,
            "tcas_space": "v2_1036800",
            "tcas_evidence": "component",
            "heldout_size": 512,
            "tcas_bank_size": 64,
        },
        "env_keys": (
            "kind", "families", "variants", "task_size",
            "tcas_space", "tcas_evidence", "heldout_size", "tcas_bank_size",
        ),
        "degrees": [0, 6, 10, 14, 18, 24, 39],
        "task_seeds": [87200, 87201, 87202, 87203, 87204, 87205],
        "notes_source": "v3_llm_tcas_T6b2_v4.json",
    },
    "physics": {
        "base": "llm_physics.json",
        "top_overrides": {
            "agents": 40,
            "branching": 2,
            "experiments_per_agent": 1,
            "initial_width": 8,
        },
        "env_overrides": {
            "kind": "physics",
            "families": 16,
            "variants": 1,
            "task_size": 4,
            "physics_mode": "support_v1",
            "initial_observations": 2,
            "heldout_size": 32,
        },
        "env_keys": (
            "kind", "families", "variants", "task_size",
            "physics_mode", "initial_observations", "heldout_size",
        ),
        "degrees": [0, 4, 8, 12, 16, 24, 39],
        "task_seeds": [82240, 82241, 82242, 82243, 82244, 82245],
        "notes_source": "v3_llm_phys_v16T4b2_v4.json",
    },
}

# Policy table: only these two policy.llm fields differ between C0/C1/I1.
POLICY_SPECS: dict[str, dict[str, str]] = {
    "C0": {"team_objective": "simple", "communication_mode": "fixed"},
    "C1": {"team_objective": "simple", "communication_mode": "free"},
    "I1": {"team_objective": "individual", "communication_mode": "free"},
}

POLICY_LABELS = {
    "C0": "team objective, fixed communication",
    "C1": "team objective, free communication",
    "I1": "individual objective, free communication",
}

# Shared llm fields, mirroring the reference configs exactly (superset of
# what the base configs carry: adds prompt_version, drops nothing).
LLM_DEFAULTS: dict[str, Any] = {
    "model": "",
    "base_url": "",
    "api_key_env": "OPENAI_API_KEY",
    "concurrency": 12,
    "timeout_s": 300.0,
    "transport_retries": 3,
    "repair_attempts": 3,
    "max_output_tokens": 3072,
    "temperature": 0.2,
    "enable_thinking": False,
    "prompt_version": "v4",
    "max_requests": 200000,
    "max_wall_seconds": 54000.0,
    "context_identity": True,
    "team_objective": "simple",
    "evidence_attribution": True,
    "shuffle_candidates": True,
    "per_agent_seed": True,
    "communication_mode": "fixed",
}

# ``--api openai``: the hosted OpenAI API (e.g. gpt-6-luna). Its reasoning
# models accept only the default temperature (1), so ``temperature`` is null
# (not sent); ``reasoning_effort="none"`` is the closest match to the local
# runs' enable_thinking=false; more transport retries (with backoff) absorb
# rate limits. Every other llm field stays as in LLM_DEFAULTS.
OPENAI_API_OVERRIDES: dict[str, Any] = {
    "api_style": "openai",
    "reasoning_effort": "none",
    "temperature": None,
    "transport_retries": 8,
}


def model_short(model: str) -> str:
    """Last path segment of the model id, lowercased: 'Qwen/Qwen3.5-4B' -> 'qwen3.5-4b'."""
    return model.rstrip("/").rsplit("/", 1)[-1].lower()


def make_tag(model: str, seed_index: int, suffix: str = "") -> str:
    """``<model-short><suffix>_seed<I>`` for the (1-based) I-th reference task seed."""
    return f"{model_short(model)}{suffix}_seed{seed_index}"


def load_base(cell: str) -> dict[str, Any]:
    path = BASE_CONFIG_DIR / CELL_SPECS[cell]["base"]
    return json.loads(path.read_text(encoding="utf-8"))


def build_config(
    cell: str,
    policy: str,
    *,
    tag: str,
    model: str,
    task_seeds: list[int],
    degrees: list[int],
    api: str = "vllm",
    temperature: float | None | str = "api-default",
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    spec = CELL_SPECS[cell]
    config = copy.deepcopy(load_base(cell))

    if config["policy"].get("kind") != "llm_full":
        raise AssertionError(
            f"{cell}/{policy}: base config policy.kind must be 'llm_full' "
            f"(this run is LLM-only), got {config['policy'].get('kind')!r}"
        )

    config.update(spec["top_overrides"])
    # Rebuild the environment block explicitly to exactly match the
    # reference field set and order (base configs may carry a superset or
    # different ordering; env_keys pins down exactly what the reference has).
    config["environment"] = {key: spec["env_overrides"][key] for key in spec["env_keys"]}
    config["degrees"] = list(degrees)
    config["task_seeds"] = list(task_seeds)
    config["repeats"] = 1
    config["candidate_pool"] = "proposed_allowed_v1"
    config["output_root"] = f"runs/agents/{tag}"

    policy_spec = POLICY_SPECS[policy]
    llm = dict(LLM_DEFAULTS)
    llm["model"] = model
    llm["team_objective"] = policy_spec["team_objective"]
    llm["communication_mode"] = policy_spec["communication_mode"]
    if api == "openai":
        llm.update(OPENAI_API_OVERRIDES)
    if temperature != "api-default":
        llm["temperature"] = temperature
    if reasoning_effort is not None:
        if api != "openai":
            raise ValueError("--reasoning-effort only applies to --api openai")
        llm["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        llm["max_output_tokens"] = max_output_tokens
    if timeout_s is not None:
        llm["timeout_s"] = timeout_s
    config["policy"]["llm"] = llm

    name = f"{tag}_{cell}_{policy}"
    config["name"] = name
    config["notes"] = (
        f"{policy}: {POLICY_LABELS[policy]}. "
        f"Experiment settings copied from {spec['notes_source']}."
    )

    # Enforce: every generated config is a real LLM run. No rule/uniform/
    # adaptive config is ever written by this script.
    assert config["policy"]["kind"] == "llm_full", (
        f"{name}: refusing to write a non-LLM config "
        f"(policy.kind={config['policy']['kind']!r})"
    )
    return config


def write_config(out_dir: Path, cell: str, policy: str, config: dict[str, Any]) -> Path:
    path = out_dir / f"{cell}_{policy}.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="model id to record in policy.llm.model, e.g. Qwen/Qwen3.5-4B",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        nargs="+",
        required=True,
        metavar="I",
        help=(
            "one or more 1-based indices into each cell's reference "
            "task-seed list; one output folder configs/agents/<model-short>"
            "_seed<I> is written per index, each with a single-element "
            "task_seeds=[seeds[I-1]] (e.g. --seed-index 1 2 3 writes "
            "_seed1, _seed2, _seed3, one task seed each)"
        ),
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=None,
        metavar="D",
        help="override the degree grid for all 9 configs (default: per-cell reference degrees)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory, overriding configs/agents/<tag> (only valid with a single --seed-index)",
    )
    parser.add_argument(
        "--api",
        choices=("vllm", "openai"),
        default="vllm",
        help="request dialect: vllm (local server, default) or openai (hosted OpenAI API, see OPENAI_API_OVERRIDES)",
    )
    parser.add_argument(
        "--temperature",
        type=lambda v: v if v == "api-default" else None if v == "default" else float(v),
        default="api-default",
        help="override llm.temperature (a number, or 'default' to send none); "
        "without it vllm uses 0.2 and --api openai sends none (some OpenAI "
        "models, e.g. gpt-6-luna, accept only the default of 1)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=None,
        help="override llm.reasoning_effort (--api openai only; default: none, "
        "from OPENAI_API_OVERRIDES)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="override llm.max_output_tokens (default 3072 from the base configs); "
        "OpenAI counts reasoning tokens against it, so reasoning runs need more",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="override llm.timeout_s (default 300 from the base configs)",
    )
    parser.add_argument(
        "--tag-suffix",
        default="",
        help="appended to <model-short> in the tag, e.g. --tag-suffix=-re-medium gives "
        "gpt-6-luna-re-medium_seed1",
    )
    args = parser.parse_args(argv)

    max_seeds = min(len(spec["task_seeds"]) for spec in CELL_SPECS.values())
    for seed_index in args.seed_index:
        if not (1 <= seed_index <= max_seeds):
            parser.error(
                f"--seed-index entries must be in [1, {max_seeds}] "
                f"(reference seed lists have {max_seeds} entries), got {seed_index}"
            )
    if args.out is not None and len(args.seed_index) != 1:
        parser.error("--out can only be used with a single --seed-index value")

    written = []
    for seed_index in args.seed_index:
        tag = make_tag(args.model, seed_index, args.tag_suffix)
        out_dir = args.out if args.out is not None else BASE_CONFIG_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for cell, spec in CELL_SPECS.items():
            degrees = args.degrees if args.degrees is not None else spec["degrees"]
            task_seeds = [spec["task_seeds"][seed_index - 1]]
            for policy in POLICY_SPECS:
                config = build_config(
                    cell,
                    policy,
                    tag=tag,
                    model=args.model,
                    task_seeds=task_seeds,
                    degrees=degrees,
                    api=args.api,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                    timeout_s=args.timeout,
                )
                path = write_config(out_dir, cell, policy, config)
                written.append(path)
                try:
                    shown_path = path.relative_to(REPO_ROOT)
                except ValueError:
                    shown_path = path  # --out points outside the repo (e.g. a test tmpdir)
                print(
                    f"wrote {shown_path}  "
                    f"(model={args.model}, task_seeds={task_seeds}, degrees={degrees})"
                )

        print(f"tag={tag}: 9 configs written to {out_dir}\n")

    print(f"total: {len(written)} configs written across {len(args.seed_index)} seed-index folder(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
