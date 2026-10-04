"""Builds environments, policies and episodes from a :class:`RunConfig`."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Sequence

from ..scaling_model import ACTION_DOMAIN, derived_rng
from .environments.base import Environment, build_task_instance
from .environments.grn import GRNEnvironment
from .environments.grn_dataset import build_grn_task, grn_task_instance, load_network
from .environments.tcas import TcasEnvironment
from .environments.tcas_dataset import seed_failure_combination, tcas_task_instance
from .environments.physics import PhysicsEnvironment
from .environments.physics_dataset import PhysicsData, build_physics_data
from .environments.synthetic import SyntheticEnvironment
from .evaluation import HiddenEvaluator
from .hypotheses import Encoding
from .policies.adaptive import AdaptivePolicy
from .policies.base import DecisionContext, Policy
from .policies.openai_policy import DEFAULT_PARENT_BATCH, LLMBackend, LLMPolicy
from .policies.replay import (
    ReplayBackend,
    ReplayPolicy,
    load_recorded_responses,
    recorded_model,
    shard_run_ids,
)
from .policies.uniform import UniformPolicy
from .runner import EpisodeRunner
from .schemas import (
    AgentView,
    BudgetPaused,
    EnvironmentFailure,
    ExperimentCandidate,
    Pair,
    PolicyFailure,
    RunConfig,
)
from .storage import RunStore, utc_now

REPEAT_STRIDE = 1_000_003


def load_env_file(path: str | Path | None) -> dict[str, str]:
    """Read ``KEY=value`` lines.  Already-exported variables win."""
    values: dict[str, str] = {}
    if path is not None:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"environment file not found: {target}")
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            value = raw.split("#")[0].strip().strip('"').strip("'")
            values[key.strip()] = value
    values.update({k: v for k, v in os.environ.items() if k in values or k.startswith("OPENAI_")})
    return values


def episode_seed(task_seed: int, repeat: int) -> int:
    return task_seed + repeat * REPEAT_STRIDE


def uniform_pair_sampler(
    view: AgentView, candidates: Sequence[ExperimentCandidate], context: DecisionContext
) -> Pair:
    rng = derived_rng(
        context.seed, ACTION_DOMAIN, view.round_number, context.agent_index, context.slot
    )
    return candidates[rng.below(len(candidates))].pair


def estimate_requests(config: RunConfig, parent_batch: int = DEFAULT_PARENT_BATCH) -> dict:
    """Pruning-free upper bound on LLM calls, per plan section 9.3."""
    rounds = config.effective_rounds
    branch = sum(
        math.ceil(config.initial_width * config.branching ** (r - 1) / parent_batch)
        for r in range(1, rounds + 1)
    ) * config.agents
    experiment = config.agents * config.experiments_per_agent * rounds
    episodes = len(config.degrees) * len(config.task_seeds) * config.repeats
    kind = config.policy.kind
    # llm_prune adds two calls per agent per round -- one after the receive
    # step and one after the own-experiment step -- plus one more whenever a
    # frontier is larger than the batch cap.
    batch = max(1, config.policy.llm_prune_batch)
    prune = 2 * config.agents * sum(
        math.ceil(
            min(
                config.initial_width * config.branching ** (r - 1),
                config.initial_width * config.branching ** (rounds - 1),
            )
            / batch
        )
        for r in range(1, rounds + 1)
    )
    # communication_mode == "free" adds one more call per agent per round,
    # except the last (a message sent there would arrive after the episode
    # ends).  Upper bound: every agent has a neighbour and every call
    # succeeds without repair, same spirit as branch/experiment above.
    communication = (
        config.agents * max(0, rounds - 1)
        if config.policy.llm.communication_mode == "free"
        else 0
    )
    per_episode = 0
    if kind in ("llm_branch", "llm_branch_test", "llm_full", "llm_prune", "replay"):
        per_episode += branch
    if kind in ("llm_branch_test", "llm_full", "llm_prune"):
        per_episode += experiment
    if kind == "llm_prune":
        per_episode += prune
    if kind in ("llm_full", "replay") and config.policy.llm.communication_mode == "free":
        per_episode += communication
    return {
        "episodes": episodes,
        "branch_requests_per_episode_upper_bound": branch,
        "experiment_requests_per_episode_upper_bound": experiment,
        "prune_requests_per_episode_upper_bound": prune if kind == "llm_prune" else 0,
        "communication_requests_per_episode_upper_bound": (
            communication if kind in ("llm_full", "replay") else 0
        ),
        "requests_per_episode_upper_bound": per_episode,
        "requests_total_upper_bound": per_episode * episodes,
        "environment_experiments_per_episode": config.agents
        * config.experiments_per_agent
        * rounds,
        "note": "upper bound without pruning, repairs or transport retries",
    }


class RunDriver:
    def __init__(
        self,
        config: RunConfig,
        *,
        store: RunStore,
        env_values: dict[str, str] | None = None,
        replay_dir: str | Path | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.store = store
        self.env_values = env_values or {}
        self.encoding = Encoding(config.environment.families, config.environment.variants)
        self.encoding.check()
        self._physics_data: dict[int, PhysicsData] = {}
        self._grn_networks: dict[int, object] = {}
        self._grn_by_seed: dict[int, tuple[object, object]] = {}
        self._tcas_by_seed: dict[int, object] = {}
        self._prediction_cache: dict[int, dict] = {}
        self._truth_cache: dict[int, dict] = {}
        self.backend = None
        self.replay_dir = replay_dir
        self.replay_behaviour = None
        kind = config.policy.kind
        if kind == "replay":
            source = Path(replay_dir or config.policy.replay_dir or "")
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            self.replay_behaviour = manifest["config"]["policy"]["kind"]
            records = load_recorded_responses(source)
            # The cache key hashes the model and base URL that were actually
            # used, so replay has to resolve them exactly as the live run did.
            # A merged sharded run predating ``resolved_model`` has neither in
            # its manifest; the responses still carry the model name.
            model = (
                manifest.get("resolved_model")
                or config.policy.llm.model
                or recorded_model(records)
            )
            recorded_llm = type(config.policy.llm)(
                **{
                    **config.policy.llm.__dict__,
                    "model": model,
                    "base_url": manifest.get("resolved_base_url") or config.policy.llm.base_url,
                }
            )
            self.backend = ReplayBackend(
                recorded_llm,
                records,
                manifest["run_id"],
                alternate_run_ids=shard_run_ids(source),
            )
        elif kind.startswith("llm_"):
            llm = config.policy.llm
            model = self.env_values.get("OPENAI_MODEL", llm.model)
            base_url = self.env_values.get("OPENAI_BASE_URL", llm.base_url)
            api_key = self.env_values.get(llm.api_key_env, "")
            resolved = type(llm)(
                **{**llm.__dict__, "model": model or llm.model, "base_url": base_url or llm.base_url}
            )
            self.backend = LLMBackend(
                resolved,
                api_key,
                request_sink=store.sink("requests"),
                response_sink=store.sink("responses"),
            )

    def tcas_task_for(self, task_seed: int):
        """The seeded failure-inducing combination this task seed stands for."""
        cached = self._tcas_by_seed.get(task_seed)
        if cached is None:
            cached = seed_failure_combination(
                strength=self.config.environment.task_size,
                seed=task_seed,
                mask=self.config.environment.tcas_mask,
                space_name=self.config.environment.tcas_space,
            )
            self._tcas_by_seed[task_seed] = cached
        return cached

    def grn_task_for(self, task_seed: int):
        """The (network, target) this task seed stands for, checked against the config.

        The target gene is the task, so ``grn_targets`` runs parallel to
        ``task_seeds``.  The in-degree and the regulator count come from the
        data and must agree with the configured ``task_size`` and ``families``;
        disagreeing is a configuration error, not something to paper over.
        """
        cached = self._grn_by_seed.get(task_seed)
        if cached is not None:
            return cached
        env = self.config.environment
        position = list(self.config.task_seeds).index(task_seed)
        number, _, gene = env.grn_targets[position].partition(":")
        index = int(number)
        network = self._grn_networks.get(index)
        if network is None:
            network = load_network(index)
            self._grn_networks[index] = network
        grn_task = build_grn_task(network, gene)
        if grn_task.task_size != env.task_size:
            raise ValueError(
                f"{env.grn_targets[position]} has in-degree {grn_task.task_size}, "
                f"but task_size is {env.task_size}"
            )
        if len(grn_task.regulators) != env.families:
            raise ValueError(
                f"{env.grn_targets[position]} has {len(grn_task.regulators)} candidate "
                f"regulators, but families is {env.families}"
            )
        self._grn_by_seed[task_seed] = (network, grn_task)
        return network, grn_task

    # -- factories ---------------------------------------------------------
    def build_environment(self, task, seed: int, task_seed: int) -> Environment:
        if self.config.environment.kind == "synthetic":
            return SyntheticEnvironment(task)
        if self.config.environment.kind == "tcas":
            return TcasEnvironment(
                task,
                tcas_task=self.tcas_task_for(task_seed),
                heldout_size=self.config.environment.heldout_size,
                bank_size=self.config.environment.tcas_bank_size,
                evidence_model=self.config.environment.tcas_evidence,
                selection=self.config.environment.tcas_selection,
            )
        if self.config.environment.kind == "grn":
            network, grn_task = self.grn_task_for(task_seed)
            return GRNEnvironment(
                task,
                network=network,
                grn_task=grn_task,
                observations=self.config.environment.initial_observations,
            )
        data = self._physics_data.get(seed)
        if data is None:
            data = build_physics_data(self.config.environment, seed, self.config.agents)
            self._physics_data[seed] = data
            self._prediction_cache[seed] = {}
            self._truth_cache[seed] = {}
        return PhysicsEnvironment(
            task,
            self.config.environment,
            self.config.agents,
            data=data,
            prediction_cache=self._prediction_cache[seed],
            truth_cache=self._truth_cache[seed],
            mode=self.config.environment.physics_mode,
        )

    def build_policy_factory(self) -> Callable[[], Policy]:
        policy = self.config.policy
        planner_kwargs = {
            "planner": policy.planner,
            "planner_max_conditions": policy.planner_max_conditions,
            "planner_max_parents": policy.planner_max_parents,
        }
        kind = policy.kind
        if kind == "uniform":
            return lambda: UniformPolicy(**planner_kwargs)
        if kind == "adaptive":
            return lambda: AdaptivePolicy(beta=policy.beta, **planner_kwargs)
        if kind == "replay":
            return lambda: ReplayPolicy(
                self.backend,
                kind="replay",
                behaves_as=self.replay_behaviour,
                uniform_fallback_pair=uniform_pair_sampler,
                prune_batch=policy.llm_prune_batch,
                **planner_kwargs,
            )
        if kind.startswith("llm_"):
            return lambda: LLMPolicy(
                self.backend,
                kind=kind,
                uniform_fallback_pair=uniform_pair_sampler,
                prune_batch=policy.llm_prune_batch,
                **planner_kwargs,
            )
        raise ValueError(f"unknown policy kind {kind!r}")

    # -- execution -----------------------------------------------------------
    def episode_plan(self) -> list[tuple[str, int, int, int]]:
        plan = []
        for degree in self.config.degrees:
            for task_seed in self.config.task_seeds:
                for repeat in range(self.config.repeats):
                    seed = episode_seed(task_seed, repeat)
                    plan.append((f"d{degree}_t{task_seed}_r{repeat}", degree, task_seed, seed))
        return plan

    async def run(self, *, resume: bool = False) -> dict:
        completed = self.store.completed_episodes() if resume else set()
        plan = self.episode_plan()
        started = time.monotonic()
        statuses: dict[str, int] = {}
        for episode_id, degree, task_seed, seed in plan:
            if episode_id in completed:
                statuses["skipped_completed"] = statuses.get("skipped_completed", 0) + 1
                continue
            row = await self._one_episode(episode_id, degree, task_seed, seed)
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
            self.store.append("episodes", row)
            self.store.checkpoint(
                {
                    "last_episode": episode_id,
                    "completed": len(self.store.completed_episodes()),
                    "planned": len(plan),
                    "statuses": statuses,
                }
            )
            if row["status"] == "budget_paused":
                break
        summary = {
            "run_id": self.store.run_id,
            "finished_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "planned_episodes": len(plan),
            "statuses": statuses,
            "llm_usage": self.backend.usage.as_dict() if self.backend is not None else {},
        }
        self.store.write_json("run_summary.json", summary)
        if self.backend is not None:
            await self.backend.aclose()
        return summary

    async def _one_episode(
        self, episode_id: str, degree: int, task_seed: int, seed: int
    ) -> dict:
        config = self.config
        if config.environment.kind == "tcas":
            task = tcas_task_instance(
                task=self.tcas_task_for(task_seed),
                agents=config.agents,
                seed=seed,
                initial_width=config.initial_width,
            )
        elif config.environment.kind == "grn":
            _network, grn_task = self.grn_task_for(task_seed)
            task = grn_task_instance(
                task=grn_task,
                agents=config.agents,
                seed=seed,
                initial_width=config.initial_width,
            )
        else:
            task = build_task_instance(
                agents=config.agents,
                encoding=self.encoding,
                task_size=config.task_size,
                seed=seed,
                initial_width=config.initial_width,
            )
        environment = self.build_environment(task, seed, task_seed)
        evaluator = HiddenEvaluator(
            task=task,
            environment=environment,
            encoding=self.encoding,
            epsilon=config.epsilon_predictive,
            truthful_scoring=config.policy.llm.communication_mode == "free",
        )
        runner = EpisodeRunner(
            config=config,
            degree=degree,
            seed=seed,
            task=task,
            environment=environment,
            policy_factory=self.build_policy_factory(),
            episode_id=episode_id,
            run_id=self.store.run_id,
            event_sink=self.store.sink("events"),
            evaluator=evaluator,
        )
        started = time.monotonic()
        try:
            result = await runner.run()
        except BudgetPaused as error:
            return _failed_episode(episode_id, degree, task_seed, seed, "budget_paused", error, started)
        except PolicyFailure as error:
            return _failed_episode(episode_id, degree, task_seed, seed, "policy_failed", error, started)
        except EnvironmentFailure as error:
            return _failed_episode(
                episode_id, degree, task_seed, seed, "environment_failed", error, started
            )
        result["task_seed"] = task_seed
        result["wall_seconds"] = time.monotonic() - started
        if self.backend is not None:
            result["llm_usage"] = self.backend.usage.as_dict()
        return result


def _failed_episode(
    episode_id: str, degree: int, task_seed: int, seed: int, status: str, error: Exception, started: float
) -> dict:
    return {
        "episode_id": episode_id,
        "status": status,
        "d": degree,
        "task_seed": task_seed,
        "seed": seed,
        "error": f"{type(error).__name__}: {error}",
        "wall_seconds": time.monotonic() - started,
        "transitions": [],
    }
