"""Public state, action, evidence and configuration schemas for ``agent_v1``.

Everything a policy may observe is defined here.  Hidden-truth objects live in
:mod:`src.agent_system.environments` and :mod:`src.agent_system.evaluation` and
must never be embedded in an :class:`AgentView`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Literal, Mapping, Sequence


Pair = tuple[int, int]
"""An atomic component as ``(family_id, variant_id)``."""

Outcome = Literal["positive", "negative"]


def pair_id(pair: Pair) -> str:
    return f"c{pair[0]}v{pair[1]}"


def hypothesis_id(components: Sequence[Pair]) -> str:
    """Canonical, order-independent identifier for a set of components."""
    return "h_" + "_".join(pair_id(item) for item in sorted(components))


@dataclass(frozen=True)
class ComponentSpec:
    """One public library entry.  Parameters are public; membership is not."""

    family_id: int
    variant_id: int
    name: str
    expression: str
    parameters: Mapping[str, float] = field(default_factory=dict)

    @property
    def pair(self) -> Pair:
        return self.family_id, self.variant_id

    def public_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "variant_id": self.variant_id,
            "name": self.name,
            "expression": self.expression,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ExperimentCondition:
    """One entry of the public experiment library."""

    experiment_id: str
    description: str
    settings: Mapping[str, float] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "settings": dict(self.settings),
        }


@dataclass(frozen=True)
class Hypothesis:
    """A canonical, unordered set of atomic components."""

    components: tuple[Pair, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(sorted(self.components)))

    @property
    def depth(self) -> int:
        return len(self.components)

    @property
    def hid(self) -> str:
        return hypothesis_id(self.components)

    @property
    def families(self) -> frozenset[int]:
        return frozenset(family for family, _ in self.components)

    def public_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.hid,
            "depth": self.depth,
            "components": [{"family_id": f, "variant_id": v} for f, v in self.components],
        }


@dataclass(frozen=True)
class EvidenceRecord:
    """An exact atomic label produced by the environment."""

    event_id: str
    pair: Pair
    outcome: Outcome
    origin_agent: int
    created_round: int
    experiment_id: str | None
    # Set only by environments whose experiment is a complete assignment rather
    # than a single component, such as configuration fault localisation: there
    # the verdict belongs to the whole configuration and pruning is by subset,
    # so the assignment has to travel with the record.
    configuration: tuple[int, ...] | None = None

    @property
    def positive(self) -> bool:
        return self.outcome == "positive"

    def public_dict(self) -> dict[str, Any]:
        payload = {
            "family_id": self.pair[0],
            "variant_id": self.pair[1],
            "outcome": self.outcome,
            "created_round": self.created_round,
        }
        if self.configuration is not None:
            payload["configuration"] = list(self.configuration)
        return payload


@dataclass(frozen=True)
class OwnExperimentSummary:
    """One of the agent's own experiments this round.

    Shown only in the ``communication`` phase, so the agent can decide which
    of its own results (if any) to tell its neighbours about.  The identity
    (``event_id``, ``pair``, ``configuration``) is exactly what the runner
    later attaches to a report the model chooses to send: the model can only
    ever pick an ``event_id`` from this list and a ``claimed_outcome``, never
    alter what was actually tested.
    """

    event_id: str
    pair: Pair
    outcome: Outcome
    experiment_id: str | None = None
    configuration: tuple[int, ...] | None = None

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "component": pair_id(self.pair),
            "outcome": self.outcome,
        }
        if self.experiment_id is not None:
            payload["experiment_id"] = self.experiment_id
        if self.configuration is not None:
            payload["configuration"] = list(self.configuration)
        return payload


@dataclass(frozen=True)
class ReceivedMessage:
    """A free-text note a direct neighbour attached to its last communication.

    Delivered with exactly the same timing as a label: produced at the end of
    one round, arriving at the start of the next, never forwarded.  The text
    itself is never parsed as evidence anywhere in the pipeline.
    """

    sender: int
    round_number: int
    text: str

    def public_dict(self) -> dict[str, Any]:
        return {"sender": self.sender, "round": self.round_number, "text": self.text}


@dataclass(frozen=True)
class ReportItem:
    """One claim in a communication decision: an identity plus a verdict.

    ``claimed_outcome`` is whatever the model chooses to say and may disagree
    with what was actually observed -- lying and omission are both legal
    moves under ``communication_mode == "free"``.
    """

    event_id: str
    claimed_outcome: Outcome


@dataclass(frozen=True)
class CommunicationDecision:
    """The result of the (optional) extra communication call.

    Empty ``reports`` and empty ``text`` means the agent sends nothing this
    round -- a legal, first-class choice, not a failure.
    """

    reports: tuple[ReportItem, ...] = ()
    text: str = ""
    reason: str = ""


@dataclass(frozen=True)
class LabelRecord:
    """One exact label as the *agent* holds it, with its provenance.

    ``source`` is ``"own"`` for a label the agent produced itself and
    ``"received"`` for one a direct neighbour produced and sent.  ``sender``
    is the neighbour that produced it -- legitimate information: a message
    arrives over an edge the agent knows about, and evidence is never
    forwarded, so the producer is always the neighbour.
    """

    pair: Pair
    outcome: Outcome
    source: Literal["own", "received"]
    sender: int | None
    round_number: int
    configuration: tuple[int, ...] | None = None

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "component": pair_id(self.pair),
            "label": self.outcome,
            "source": self.source,
            "round": self.round_number,
        }
        if self.sender is not None:
            payload["sender"] = self.sender
        if self.configuration is not None:
            payload["configuration"] = list(self.configuration)
        return payload


@dataclass(frozen=True)
class RoundMemory:
    """What the agent itself did in one earlier round."""

    round_number: int
    tested: tuple[Pair, ...] = ()
    outcomes: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "tested": [pair_id(pair) for pair in self.tested],
            "labels": list(self.outcomes),
        }


@dataclass(frozen=True)
class PrivateObservation:
    """A trajectory visible only to the agent that produced or received it."""

    observation_id: str
    kind: Literal["initial", "experiment"]
    experiment_id: str | None
    settings: Mapping[str, float]
    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]
    variables: tuple[str, ...] = ("x", "v")

    def public_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "experiment_id": self.experiment_id,
            "settings": dict(self.settings),
            "variables": list(self.variables),
            "samples": [
                [t, *values] for t, values in zip(self.times, self.states)
            ],
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Environment return value.  Only ``evidence`` may leave the owner."""

    evidence: EvidenceRecord
    observation: PrivateObservation | None


@dataclass(frozen=True)
class ExperimentCandidate:
    """A pair that may be tested this round, with its generating parents."""

    pair: Pair
    parent_ids: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.pair[0],
            "variant_id": self.pair[1],
            "parent_ids": list(self.parent_ids),
        }


@dataclass(frozen=True)
class ParentSlot:
    """A parent awaiting exactly one branch update."""

    hypothesis: Hypothesis
    legal_additions: tuple[Pair, ...]
    required: int

    def public_dict(self) -> dict[str, Any]:
        payload = self.hypothesis.public_dict()
        payload["required_additions"] = self.required
        payload["legal_additions"] = [
            {"family_id": f, "variant_id": v} for f, v in self.legal_additions
        ]
        return payload


@dataclass(frozen=True)
class AgentView:
    """The immutable whitelist a policy is allowed to see."""

    phase: Literal["branch", "experiment", "prune", "communication"]
    round_number: int
    total_rounds: int
    task_size: int
    families: int
    variants: int
    branching: int
    component_library: tuple[ComponentSpec, ...]
    known_positive: tuple[Pair, ...]
    known_negative: tuple[Pair, ...]
    allowed_values: Mapping[int, tuple[int, ...]]
    parents: tuple[ParentSlot, ...] = ()
    candidates: tuple[ExperimentCandidate, ...] = ()
    experiment_library: tuple[ExperimentCondition, ...] = ()
    private_observations: tuple[PrivateObservation, ...] = ()
    environment_kind: str = "synthetic"
    environment_notes: str = ""
    # -- team and history context.  Every field below is information the
    # agent legitimately holds: its own place in the graph, the clock, its
    # own budget, the provenance of the labels it was handed and what it
    # itself did before.  Nothing here depends on the hidden truth.
    agent_index: int = -1
    agent_count: int = 0
    degree: int = 0
    neighbours: tuple[int, ...] = ()
    round_budget: int = 0
    own_labels: tuple[LabelRecord, ...] = ()
    received_labels: tuple[LabelRecord, ...] = ()
    own_history: tuple[RoundMemory, ...] = ()
    # -- llm_prune phase only ------------------------------------------
    prune_hypotheses: tuple[Hypothesis, ...] = ()
    new_evidence: tuple[LabelRecord, ...] = ()
    # -- free communication (communication_mode == "free") --------------
    # ``own_experiments``: communication phase only, the agent's own results
    # this round, offered for it to report on (truthfully or not).
    own_experiments: tuple[OwnExperimentSummary, ...] = ()
    # ``received_messages``: every phase of the round after they arrive, so
    # branch and experiment decisions can be informed by neighbours' claims
    # and text exactly the way ``received_labels`` already is.  Text here is
    # never evidence; it is shown, never parsed.
    received_messages: tuple[ReceivedMessage, ...] = ()

    def with_parents(self, parents: Sequence[ParentSlot]) -> "AgentView":
        return replace(self, parents=tuple(parents))

    def public_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "round": self.round_number,
            "total_rounds": self.total_rounds,
            "task_size": self.task_size,
            "families": self.families,
            "variants_per_family": self.variants,
            "branching": self.branching,
            "environment_kind": self.environment_kind,
            "environment_notes": self.environment_notes,
            "component_library": [spec.public_dict() for spec in self.component_library],
            "known_positive": [{"family_id": f, "variant_id": v} for f, v in self.known_positive],
            "known_negative": [{"family_id": f, "variant_id": v} for f, v in self.known_negative],
            "allowed_values": {str(k): list(v) for k, v in sorted(self.allowed_values.items())},
            "parents": [slot.public_dict() for slot in self.parents],
            "candidates": [item.public_dict() for item in self.candidates],
            "experiment_library": [item.public_dict() for item in self.experiment_library],
            "private_observations": [obs.public_dict() for obs in self.private_observations],
            "agent_index": self.agent_index,
            "agent_count": self.agent_count,
            "degree": self.degree,
            "neighbours": list(self.neighbours),
            "round_budget": self.round_budget,
            "own_labels": [item.public_dict() for item in self.own_labels],
            "received_labels": [item.public_dict() for item in self.received_labels],
            "own_history": [item.public_dict() for item in self.own_history],
            "prune_hypotheses": [item.public_dict() for item in self.prune_hypotheses],
            "new_evidence": [item.public_dict() for item in self.new_evidence],
            "own_experiments": [item.public_dict() for item in self.own_experiments],
            "received_messages": [item.public_dict() for item in self.received_messages],
        }

    def digest_payload(self) -> str:
        return json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ParentUpdate:
    parent_id: str
    additions: tuple[Pair, ...]


@dataclass(frozen=True)
class BranchDecision:
    updates: tuple[ParentUpdate, ...]
    reason: str = ""


@dataclass(frozen=True)
class ExperimentDecision:
    pair: Pair
    parent_id: str
    experiment_id: str | None
    reason: str = ""


class ActionError(ValueError):
    """Raised when a decision is schema-valid but scientifically illegal."""


class IncompleteOutput(ActionError):
    """The model stopped before finishing a valid action.  Repairable."""


class PolicyFailure(RuntimeError):
    """Raised when a policy cannot produce a legal action after repair."""


class BudgetPaused(RuntimeError):
    """Raised when a configured request/token/time ceiling is reached."""


class EnvironmentFailure(RuntimeError):
    """Raised when the hidden environment cannot produce a valid observation."""


# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------


CANDIDATE_POOLS = ("proposed_allowed_v1", "surviving_leaf_v1")
BUDGET_MODES = ("redistribute", "hard_cap")
Q_MODEL_DRAW_MODES = ("per_label", "per_message")
RESPAWN_MODES = ("none", "evidence_consistent")
TOPOLOGIES = ("random_regular", "ring_lattice", "erdos_renyi", "watts_strogatz")
TCAS_SELECTIONS = ("uniform", "max_containment")
POLICY_KINDS = (
    "uniform",
    "adaptive",
    "llm_branch",
    "llm_branch_test",
    "llm_full",
    # Everything llm_full chooses, plus the pruning step: the agent is shown
    # the new raw evidence and its own hypotheses and says which to drop.
    "llm_prune",
    "replay",
)
# "simple" / "individual" are the C0/C1/I1 experiment's objective texts: a
# short statement of what the agent is trying to improve, with no prescribed
# search or division-of-labour strategy and no device symmetry breaker.
TEAM_OBJECTIVE_MODES = ("off", "soft", "device", "simple", "individual")
REASONING_MODES = ("none", "reason_first", "thinking")
API_STYLES = ("vllm", "openai")
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
PHYSICS_OBSERVATION_MODES = ("raw", "summary")
LLM_PRUNE_APPLY_MODES = ("validated", "raw")
# The prompt files are versioned so a future rewrite cannot silently change
# what an old recorded run replays against.  Only "v4" -- the prompts in this
# repository -- is accepted today.
PROMPT_VERSIONS = ("v4",)
# "fixed" is the published protocol: an experiment's true result is sent
# automatically to direct neighbours.  "free" replaces that with an explicit,
# optional, possibly-inaccurate communication decision (see LLMConfig).
COMMUNICATION_MODES = ("fixed", "free")


@dataclass(frozen=True)
class EnvironmentConfig:
    kind: Literal["synthetic", "physics", "grn", "tcas"] = "synthetic"
    families: int = 8
    variants: int = 4
    task_size: int = 4
    observation_window: float = 2.0
    observation_samples: int = 33
    observation_decimals: int = 6
    initial_observations: int = 2
    rtol: float = 1e-9
    atol: float = 1e-11
    integrator: str = "DOP853"
    heldout_size: int = 32
    library_x0: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    library_v0: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    library_t0: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    # grn only: one "network:gene" per task_seed, in the same order.  The
    # target gene is the task, so a run covers several targets of the same
    # in-degree; the seed still drives Mode-B initialisation.  families and
    # task_size stay in the config and are checked against the data, because a
    # real target's in-degree is not free.
    grn_targets: tuple[str, ...] = ()
    # tcas only: the seeded failure-inducing combination is drawn per task
    # seed, so the seed is the task.  tcas_mask adds the masking clause that
    # makes a controlled fraction of failing configurations pass.
    tcas_mask: bool = False
    tcas_bank_size: int = 64
    # Which input space, and whether a verdict is about a whole configuration
    # (subset pruning) or about one assignment (component pruning).
    tcas_space: str = "v1_460800"
    tcas_evidence: str = "subset"
    # How the bank member that carries a subset-level test is picked.
    # "uniform" leaves the choice to the policy's planner, which is what every
    # run so far did; "max_containment" is the long paper's
    # x* = argmax_{x in bank, h subset of x} I(x; H_i), i.e. the pattern that
    # also contains the largest number of the agent's surviving hypotheses.
    tcas_selection: str = "uniform"
    # physics only: "factored_v1" is the (family, variant) library where at
    # most one variant of a family is active; "support_v1" is SINDy-style
    # support recovery over a flat library, where the target is any subset.
    physics_mode: str = "factored_v1"

    @property
    def library_size(self) -> int:
        return len(self.library_x0) * len(self.library_v0) * len(self.library_t0)


@dataclass(frozen=True)
class LLMConfig:
    model: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 4
    timeout_s: float = 120.0
    transport_retries: int = 2
    repair_attempts: int = 1
    max_output_tokens: int = 1024
    temperature: float | None = 0.2
    top_p: float | None = None
    enable_thinking: bool = False
    # Request dialect.  "vllm" (default) sends ``max_tokens`` and vLLM's
    # ``chat_template_kwargs``.  "openai" targets the hosted OpenAI API, whose
    # reasoning models reject both: it sends ``max_completion_tokens`` instead,
    # no ``chat_template_kwargs``, and ``reasoning_effort`` when set.
    api_style: str = "vllm"
    reasoning_effort: str | None = None
    # Which prompt template revision to render.  The prompt files in this
    # repository already are v4; the field exists so a config states the
    # revision it was validated against and an unknown value is rejected
    # rather than silently rendered with whatever prompts happen to be on
    # disk.
    prompt_version: str = "v4"
    max_requests: int | None = None
    max_total_tokens: int | None = None
    max_wall_seconds: float | None = None
    structured_output: bool = True
    # -- prompt switches.  The defaults are the final experiment setting:
    # (a)-(e) all on, no extra reasoning, physics observations summarised.
    # Each switch stays independent so it can still be ablated.
    # (a) agent id, N, degree, neighbour ids, round t of R, budget, and the
    #     "everyone chooses at once, messages arrive one round later" note.
    context_identity: bool = True
    # (b) team-level objective and what makes a test informative.  "soft"
    #     states the objective; "device" adds an executable symmetry breaker;
    #     "off" leaves it out.
    team_objective: str = "device"
    # (c) own vs received labels, with sender and round, plus a per-round
    #     memory line of the agent's own earlier choices.
    evidence_attribution: bool = True
    # (d) per-agent, per-round seeded permutation of the candidate lists and
    #     of the enum order in the JSON schema.
    shuffle_candidates: bool = True
    # (e) deterministic per-(episode, agent, round, phase, slot, attempt)
    #     sampling seed in the request body.
    per_agent_seed: bool = True
    # (f) "none" | "reason_first" (a short analysis field decoded BEFORE the
    #     action) | "thinking" (enable_thinking=true, more output tokens).
    reasoning: str = "none"
    reasoning_max_chars: int = 600
    thinking_max_output_tokens: int = 4096
    # (g) physics hygiene: trajectories as a few summary statistics
    #     rather than ~400 raw floats.  "raw" keeps the old rendering.
    physics_observations: str = "summary"
    # (h) "fixed" (default) reproduces agent_v1 exactly: an experiment's true
    #     result is sent automatically and truthfully to direct neighbours.
    #     "free" replaces that with an explicit communication decision after
    #     each round's experiments -- the agent chooses which of its own
    #     results (if any) to report, may misreport them, and may add a short
    #     free-text note; see policies/openai_policy.py and runner.py.
    communication_mode: str = "fixed"


@dataclass(frozen=True)
class PolicyConfig:
    kind: str = "uniform"
    beta: float = 1.0
    planner: Literal["disagreement_v1", "fixed_first", "uniform"] = "disagreement_v1"
    planner_max_conditions: int | None = None
    planner_max_parents: int | None = None
    llm: LLMConfig = field(default_factory=LLMConfig)
    replay_dir: str | None = None
    # llm_prune only.  "validated" applies a drop only when the hypothesis
    # really is refuted by evidence the agent holds, so the model can miss a
    # prune but can never kill a consistent hypothesis (the paper's
    # zero-false-negative oracle).  "raw" applies whatever it says.
    llm_prune_apply: str = "validated"
    llm_prune_batch: int = 64


@dataclass(frozen=True)
class RunConfig:
    """A complete, frozen description of one run or sweep."""

    name: str
    protocol_version: str = "agent_v1"
    agents: int = 12
    branching: int = 2
    experiments_per_agent: int = 1
    initial_width: int = 8
    rounds: int | None = None
    degrees: tuple[int, ...] = (0,)
    task_seeds: tuple[int, ...] = (12345,)
    repeats: int = 1
    candidate_pool: str = "proposed_allowed_v1"
    # -- protocol knobs.  Every default below reproduces agent_v1 exactly ---
    # Probability that an agent acts on a piece of evidence it is handed.  A
    # miss stores nothing and prunes nothing; it never removes a correct
    # hypothesis, so the false-negative rate of the oracle stays 0.
    q_model: float = 1.0
    # Per-channel overrides.  ``None`` inherits ``q_model``.
    q_model_own: float | None = None
    q_model_received: float | None = None
    # How a ``q_model < 1`` miss is drawn on the *received* channel.
    # "per_label" (default, the published behaviour): the duplicate filter
    # runs first and every unique label gets exactly one Bernoulli(q) draw,
    # so an agent has a single chance at a label however many neighbours
    # deliver it and pruning saturates at ``q`` as the degree grows.
    # "per_message": every delivered copy, i.e. every (sender, label), is an
    # independent Bernoulli(q) opportunity and the label is learned if any
    # copy succeeds, which is what the mean-field term
    # ``(1 - m q_overlap q_model)^d`` of the paper assumes.  Draws are
    # domain-separated hashes of (episode seed, round, receiver, sender,
    # label), so they are reproducible and independent of delivery order.
    # At ``q_model = 1`` the two modes are identical and draw nothing.
    q_model_draw: str = "per_label"
    # "redistribute" is the published rule: the round budget is ``N*m`` and an
    # extinct agent's share goes to whoever still has candidates.  "hard_cap"
    # gives each agent at most ``m`` experiments and leaves the rest unused.
    budget_mode: str = "redistribute"
    # Stop the episode at the end of the first round in which some agent
    # holds a complete, evidence-verified hypothesis: a depth-``T`` candidate
    # every component of which carries a positive label, which is exactly the
    # evaluator's ``evidence_verified`` criterion.  Note that ``agent_v1``
    # fixes ``rounds = T - 1`` and a hypothesis gains one component per round,
    # so depth ``T`` is first reached in the *final* round: at the published
    # horizon this knob can only ever fire there, where stopping is a no-op.
    # It truncates an episode only for a run with more rounds than ``T - 1``.
    stop_on_success: bool = False
    # Score the hidden criteria against the frontier of every round, not only
    # the last one.  Changes recorded evaluation fields, hence a flag.
    evaluate_every_round: bool = False
    # "evidence_consistent" re-seeds an agent whose frontier emptied with up
    # to ``n0`` depth-t hypotheses that its own evidence still allows.
    respawn: str = "none"
    topology: str = "random_regular"
    topology_rewire_p: float = 0.0
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    epsilon_predictive: float = 1e-3
    output_root: str = "runs/agents"
    notes: str = ""

    @property
    def task_size(self) -> int:
        return self.environment.task_size

    @property
    def effective_rounds(self) -> int:
        return self.task_size - 1 if self.rounds is None else self.rounds

    @property
    def effective_q_own(self) -> float:
        return self.q_model if self.q_model_own is None else self.q_model_own

    @property
    def effective_q_received(self) -> float:
        return self.q_model if self.q_model_received is None else self.q_model_received

    @property
    def regular_topology(self) -> bool:
        """Whether every node is required to have exactly ``d`` neighbours."""
        return self.topology in ("random_regular", "ring_lattice")

    def validate(self) -> None:
        env = self.environment
        if self.protocol_version != "agent_v1":
            raise ValueError(f"unsupported protocol_version {self.protocol_version!r}")
        if self.candidate_pool not in CANDIDATE_POOLS:
            raise ValueError(f"unknown candidate_pool {self.candidate_pool!r}")
        if self.policy.kind not in POLICY_KINDS:
            raise ValueError(f"unknown policy kind {self.policy.kind!r}")
        if self.effective_rounds != self.task_size - 1:
            raise ValueError("agent_v1 fixes rounds = task_size - 1")
        if not 2 <= env.task_size <= env.families:
            raise ValueError("task_size must satisfy 2 <= T <= M")
        if self.initial_width < 1:
            raise ValueError("n0 must be positive")
        if self.initial_width - 1 > env.families * env.variants - env.task_size:
            raise ValueError("Mode-B initial width is infeasible")
        if self.branching < 1 or self.experiments_per_agent < 0:
            raise ValueError("b must be positive and m nonnegative")
        for name, value in (
            ("q_model", self.q_model),
            ("q_model_own", self.q_model_own),
            ("q_model_received", self.q_model_received),
        ):
            if value is None:
                continue
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
        if self.q_model_draw not in Q_MODEL_DRAW_MODES:
            raise ValueError(f"unknown q_model_draw {self.q_model_draw!r}")
        llm = self.policy.llm
        if llm.team_objective not in TEAM_OBJECTIVE_MODES:
            raise ValueError(f"unknown team_objective {llm.team_objective!r}")
        if llm.prompt_version not in PROMPT_VERSIONS:
            raise ValueError(
                f"unknown prompt_version {llm.prompt_version!r}; only "
                f"{PROMPT_VERSIONS} is known"
            )
        if llm.communication_mode not in COMMUNICATION_MODES:
            raise ValueError(f"unknown communication_mode {llm.communication_mode!r}")
        if llm.communication_mode == "free":
            if self.policy.kind not in ("llm_full", "replay"):
                raise ValueError(
                    "communication_mode='free' needs policy.kind in "
                    f"('llm_full', 'replay'), got {self.policy.kind!r}"
                )
            if llm.prompt_version != "v4":
                raise ValueError(
                    "communication_mode='free' needs prompt_version='v4', got "
                    f"{llm.prompt_version!r}"
                )
            if self.effective_q_own != 1.0:
                raise ValueError(
                    "communication_mode='free' needs effective q_model_own == 1: "
                    "a sender cannot report a verdict it never actually acted on "
                    f"(got {self.effective_q_own!r})"
                )
            if env.kind == "tcas" and env.tcas_evidence == "subset":
                raise ValueError(
                    "communication_mode='free' is only implemented for "
                    "component-level evidence; tcas_evidence='subset' identifies "
                    "a claim by a whole configuration, which this port does not "
                    "yet support"
                )
        if llm.api_style not in API_STYLES:
            raise ValueError(f"unknown api_style {llm.api_style!r}")
        if llm.reasoning_effort is not None:
            if llm.api_style != "openai":
                raise ValueError("reasoning_effort only applies to api_style='openai'")
            if llm.reasoning_effort not in REASONING_EFFORTS:
                raise ValueError(f"unknown reasoning_effort {llm.reasoning_effort!r}")
        if llm.api_style == "openai" and (llm.enable_thinking or llm.reasoning == "thinking"):
            raise ValueError(
                "api_style='openai' has no enable_thinking switch; use reasoning_effort"
            )
        if llm.reasoning not in REASONING_MODES:
            raise ValueError(f"unknown reasoning mode {llm.reasoning!r}")
        if llm.physics_observations not in PHYSICS_OBSERVATION_MODES:
            raise ValueError(
                f"unknown physics_observations {llm.physics_observations!r}"
            )
        if self.policy.llm_prune_apply not in LLM_PRUNE_APPLY_MODES:
            raise ValueError(
                f"unknown llm_prune_apply {self.policy.llm_prune_apply!r}"
            )
        if self.policy.llm_prune_batch < 1:
            raise ValueError("llm_prune_batch must be positive")
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"unknown budget_mode {self.budget_mode!r}")
        if self.respawn not in RESPAWN_MODES:
            raise ValueError(f"unknown respawn mode {self.respawn!r}")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {self.topology!r}")
        if not 0.0 <= self.topology_rewire_p <= 1.0:
            raise ValueError("topology_rewire_p must lie in [0, 1]")
        if self.topology != "watts_strogatz" and self.topology_rewire_p:
            raise ValueError("topology_rewire_p only applies to watts_strogatz")
        for degree in self.degrees:
            if not 0 <= degree < self.agents:
                raise ValueError(f"degree {degree} outside [0, N)")
            # Only an exactly d-regular topology needs N*d even; Erdos-Renyi
            # and a rewired ring realise d on average, not node by node.
            if self.regular_topology and (self.agents * degree) % 2:
                raise ValueError(f"no simple d-regular graph for N={self.agents}, d={degree}")
            if self.topology in ("ring_lattice", "watts_strogatz") and degree % 2:
                raise ValueError(
                    f"a ring lattice links d/2 neighbours on each side, so d must be "
                    f"even; got d={degree}"
                )
            if self.topology in ("ring_lattice", "watts_strogatz") and degree > self.agents - 1:
                raise ValueError(f"degree {degree} exceeds N-1 for a ring lattice")
        if env.kind == "physics" and env.library_size < 1:
            raise ValueError("the physics experiment library must be non-empty")
        if env.kind == "tcas":
            from .environments.tcas_dataset import assignment_count, space

            if env.tcas_evidence not in ("subset", "component"):
                raise ValueError("tcas_evidence must be 'subset' or 'component'")
            if env.tcas_selection not in TCAS_SELECTIONS:
                raise ValueError(f"unknown tcas_selection {env.tcas_selection!r}")
            if env.tcas_selection == "max_containment" and env.tcas_evidence != "subset":
                raise ValueError(
                    "tcas_selection='max_containment' needs tcas_evidence='subset': "
                    "a component-level verdict has no configuration to choose"
                )
            cardinalities = [size for _, size in space(env.tcas_space)]
            if env.families != len(cardinalities) or env.variants != max(cardinalities):
                raise ValueError(
                    f"tcas space {env.tcas_space} needs families={len(cardinalities)} "
                    f"and variants={max(cardinalities)} (the widest parameter)"
                )
            if not 1 <= env.task_size <= len(cardinalities):
                raise ValueError(f"tcas strength must be within 1..{len(cardinalities)}")
            if env.tcas_bank_size < 1:
                raise ValueError("the tcas experiment bank must be non-empty")
            # Mode-B draws distractors from the legal assignments only.
            legal = assignment_count(env.tcas_space)
            if self.initial_width - 1 > legal - env.task_size:
                raise ValueError(
                    f"Mode-B initial width is infeasible over {legal} assignments"
                )
        if env.kind == "physics":
            if env.physics_mode not in ("factored_v1", "support_v1"):
                raise ValueError("physics_mode must be 'factored_v1' or 'support_v1'")
            if env.physics_mode == "support_v1" and env.variants != 1:
                raise ValueError(
                    "support_v1 recovers a subset of a flat term library, so "
                    "variants must be 1"
                )
        if env.kind == "grn":
            if env.variants != 2:
                raise ValueError("a grn component is (regulator, sign), so variants must be 2")
            if len(env.grn_targets) != len(self.task_seeds):
                raise ValueError(
                    f"grn_targets has {len(env.grn_targets)} entries for "
                    f"{len(self.task_seeds)} task seeds; give one target per seed"
                )
            for entry in env.grn_targets:
                network, _, gene = entry.partition(":")
                if not gene or network not in {"1", "2", "3", "4", "5"}:
                    raise ValueError(
                        f"grn target {entry!r} must be 'network:gene' with network in 1..5"
                    )

    def as_dict(self) -> dict[str, Any]:
        return _to_plain(self)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _to_plain(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _build(cls, payload: Mapping[str, Any]):
    fields = cls.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown configuration keys {unknown}")
    kwargs: dict[str, Any] = {}
    for key, raw in payload.items():
        annotation = str(fields[key].type)
        if "tuple" in annotation and isinstance(raw, list):
            kwargs[key] = tuple(raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


def load_run_config(payload: Mapping[str, Any]) -> RunConfig:
    """Build a :class:`RunConfig` from plain JSON, rejecting unknown keys."""
    data = dict(payload)
    environment = _build(EnvironmentConfig, data.pop("environment", {}))
    policy_payload = dict(data.pop("policy", {}))
    llm = _build(LLMConfig, policy_payload.pop("llm", {}))
    policy = _build(PolicyConfig, {**policy_payload, "llm": llm})
    config = _build(RunConfig, {**data, "environment": environment, "policy": policy})
    config.validate()
    return config
