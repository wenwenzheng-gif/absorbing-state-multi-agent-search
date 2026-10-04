"""The tcas configuration space and a seeded failure-inducing combination.

The subject is the ``tcas`` collision-avoidance module of the Siemens suite,
used here only for its parameter space: twelve categorical parameters with
cardinalities 2^7 * 3^2 * 4 * 10^2 = 460800 complete configurations and
7*2 + 2*3 + 4 + 2*10 = 44 atomic parameter-value assignments.

The oracle is **not** the program.  Following the paper, the ground truth is a
seeded failure-inducing combination H* of strength t*, under a conjunctive
model: a configuration fails exactly when it contains every assignment in H*.
An optional masking clause makes a fraction of those configurations pass
anyway, which breaks monotonicity by a controlled amount and is the knob for
the false-negative ablation.

Because a configuration is evaluated by this rule rather than by execution,
nothing has to be compiled or downloaded, and the 460800-row grid never has to
be materialised: the verdict is a containment test.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...scaling_model import ASSIGNMENT_DOMAIN, derived_rng, make_initial_pairs
from ..hypotheses import Encoding
from .base import TaskInstance

# name, number of values.  Order is the SIR declaration order.
#
# Two input spaces are in circulation for this benchmark and the paper has
# used both.  ``v1`` is the Siemens/SIR parameterisation with
# 2^7 * 3^2 * 4 * 10^2 = 460800 configurations and 44 assignments; ``v2``
# follows Kuhn's pseudo-exhaustive account with
# 3 * 2^3 * 3 * 2 * 4 * 10^2 * 3 * 2 * 3 = 1036800 configurations and 46
# assignments.  The search protocol is identical; only the cardinalities and
# hence |V| differ, which is exactly what sets d_c.
SPACES: dict[str, tuple[tuple[str, int], ...]] = {
    "v1_460800": (
        ("Cur_Vertical_Sep", 3),
        ("High_Confidence", 2),
        ("Two_of_Three_Reports_Valid", 2),
        ("Own_Tracked_Alt", 2),
        ("Own_Tracked_Alt_Rate", 2),
        ("Other_Tracked_Alt", 2),
        ("Alt_Layer_Value", 4),
        ("Up_Separation", 10),
        ("Down_Separation", 10),
        ("Other_RAC", 3),
        ("Other_Capability", 2),
        ("Climb_Inhibit", 2),
    ),
    "v2_1036800": (
        ("Cur_Vertical_Sep", 3),
        ("High_Confidence", 2),
        ("Two_of_Three_Reports_Valid", 2),
        ("Own_Tracked_Alt", 2),
        ("Own_Tracked_Alt_Rate", 3),
        ("Other_Tracked_Alt", 2),
        ("Alt_Layer_Value", 4),
        ("Up_Separation", 10),
        ("Down_Separation", 10),
        ("Other_RAC", 3),
        ("Other_Capability", 2),
        ("Climb_Inhibit", 3),
    ),
}
DEFAULT_SPACE = "v1_460800"


def space(name: str = DEFAULT_SPACE) -> tuple[tuple[str, int], ...]:
    try:
        return SPACES[name]
    except KeyError:
        raise ValueError(
            f"unknown tcas space {name!r}; choose from {sorted(SPACES)}"
        ) from None


def space_size(name: str = DEFAULT_SPACE) -> int:
    total = 1
    for _, size in space(name):
        total *= size
    return total


def assignment_count(name: str = DEFAULT_SPACE) -> int:
    return sum(size for _, size in space(name))


PARAMETERS: tuple[tuple[str, int], ...] = SPACES[DEFAULT_SPACE]
PARAMETER_NAMES = tuple(name for name, _ in PARAMETERS)
CARDINALITIES = tuple(size for _, size in PARAMETERS)
MAX_CARDINALITY = max(CARDINALITIES)
CONFIGURATION_COUNT = 1
for _size in CARDINALITIES:
    CONFIGURATION_COUNT *= _size
ASSIGNMENT_COUNT = sum(CARDINALITIES)
LIBRARY_VERSION = "tcas_configuration_v1"


def allowed_masks(space_name: str = DEFAULT_SPACE) -> list[int]:
    """Per-parameter bitmask of the values that actually exist.

    The encoding is padded to the widest parameter so the shared bit-packed
    key can be reused unchanged; these masks are what keep the padding out of
    every legal extension.
    """
    return [(1 << size) - 1 for _, size in space(space_name)]


@dataclass(frozen=True)
class TcasTask:
    """A seeded failure-inducing combination over the tcas parameter space."""

    truth_values: tuple[int, ...]
    mask_parameter: int | None = None
    mask_value: int | None = None
    space_name: str = DEFAULT_SPACE

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(size for _, size in space(self.space_name))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in space(self.space_name))

    @property
    def strength(self) -> int:
        return sum(value >= 0 for value in self.truth_values)

    def fails(self, configuration: tuple[int, ...]) -> bool:
        """Conjunctive guard, optionally masked.

        Every configuration containing H* fails, except that when a masking
        clause is set, those also carrying ``mask_parameter = mask_value``
        pass.  That is the only source of false negatives.
        """
        for parameter, value in enumerate(self.truth_values):
            if value >= 0 and configuration[parameter] != value:
                return False
        if self.mask_parameter is not None:
            return configuration[self.mask_parameter] != self.mask_value
        return True

    def masked_fraction(self) -> float:
        """Share of H*-containing configurations the mask turns into passes."""
        if self.mask_parameter is None:
            return 0.0
        if self.truth_values[self.mask_parameter] >= 0:
            return 1.0 if self.truth_values[self.mask_parameter] == self.mask_value else 0.0
        return 1.0 / self.cardinalities[self.mask_parameter]

    def label(self) -> str:
        names = self.parameter_names
        parts = [
            f"{names[parameter]}={value}"
            for parameter, value in enumerate(self.truth_values)
            if value >= 0
        ]
        text = " and ".join(parts)
        if self.mask_parameter is not None:
            text += f" unless {names[self.mask_parameter]}={self.mask_value}"
        return text


TRUTH_DOMAIN_TCAS = 0x94D049BB133111EB
BANK_DOMAIN = 0x2545F4914F6CDD1D
HELDOUT_DOMAIN = 0x9E3779B97F4A7C15


def seed_failure_combination(
    *, strength: int, seed: int, mask: bool = False, space_name: str = DEFAULT_SPACE
) -> TcasTask:
    """Draw a strength-``t*`` combination, and optionally a masking clause."""
    cardinalities = [size for _, size in space(space_name)]
    count = len(cardinalities)
    if not 1 <= strength <= count:
        raise ValueError(f"strength must be within 1..{count}")
    rng = derived_rng(seed, TRUTH_DOMAIN_TCAS)
    pool = list(range(count))
    values = [-1] * count
    for index in range(strength):
        chosen = index + rng.below(count - index)
        pool[index], pool[chosen] = pool[chosen], pool[index]
        parameter = pool[index]
        values[parameter] = rng.below(cardinalities[parameter])
    mask_parameter = mask_value = None
    if mask:
        free = [p for p in range(count) if values[p] < 0]
        if not free:
            raise ValueError("a masking clause needs a parameter outside the combination")
        mask_parameter = free[rng.below(len(free))]
        mask_value = rng.below(cardinalities[mask_parameter])
    return TcasTask(
        truth_values=tuple(values),
        mask_parameter=mask_parameter,
        mask_value=mask_value,
        space_name=space_name,
    )


def tcas_task_instance(
    *, task: TcasTask, agents: int, seed: int, initial_width: int
) -> TaskInstance:
    """Mode-B initialisation over the legal, unpadded assignments only."""
    cardinalities = task.cardinalities
    widest = max(cardinalities)
    encoding = Encoding(families=len(cardinalities), variants=widest)
    encoding.check()
    legal = [
        parameter * widest + value
        for parameter, size in enumerate(cardinalities)
        for value in range(size)
    ]
    truth_pairs = [
        parameter * widest + value
        for parameter, value in enumerate(task.truth_values)
        if value >= 0
    ]
    if not truth_pairs:
        raise ValueError("a seeded combination must contain at least one assignment")

    order = list(range(len(truth_pairs)))
    assignment_rng = derived_rng(seed, ASSIGNMENT_DOMAIN)
    assignment_rng.shuffle(order)
    primary_groups = [order[index % len(truth_pairs)] for index in range(agents)]
    assignment_rng.shuffle(primary_groups)

    # make_initial_pairs draws distractors from range(pair_count); remap so the
    # padded slots of the narrow parameters can never be drawn.
    compact = make_initial_pairs(
        agents,
        [legal.index(pair) for pair in truth_pairs],
        list(primary_groups),
        len(legal),
        seed,
        initial_width,
    )
    initial = tuple(
        tuple(sorted(encoding.pair(legal[index]) for index in group))
        for group in compact
    )
    return TaskInstance(
        seed=seed,
        encoding=encoding,
        truth_values=tuple(task.truth_values),
        truth_pairs=tuple(encoding.pair(index) for index in truth_pairs),
        primary_groups=tuple(primary_groups),
        initial_pairs=initial,
    )
