"""DREAM4 In Silico Size 100 gene regulatory networks as a search task.

The hidden mechanism is the complete upstream regulation of one target gene:
which of the other 99 genes regulate it, and whether each activates or
inhibits.  A component is therefore ``(regulator, sign)``, with two variants
per family, and the task size is the target's in-degree.

Two separate perturbation experiments of the same network are used, so the
part an agent can query and the part the evaluator scores are disjoint:

* **knockouts** define the component labels, which is the gene intervention
  the paper gives the rule agent;
* **knockdowns** are read by nothing in the search path.  They are kept
  because they are the natural hold-out, but no predictive criterion is
  derived from them: on this data the magnitude of an indirect effect
  overlaps almost completely with that of a direct one (at any threshold
  either most non-parents look like parents or most parents do not), so a
  thresholded hold-out error would not be well posed.  ``GRNEnvironment``
  reports ``supports_prediction() -> False`` and the evaluator falls back to
  the symbolic proxy it already uses for the synthetic environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from ...scaling_model import ASSIGNMENT_DOMAIN, derived_rng, make_initial_pairs
from ...utils import PACKAGE_ROOT
from ..hypotheses import Encoding
from .base import TaskInstance

DREAM4_ROOT = PACKAGE_ROOT / "data" / "external" / "dream4"
NETWORKS = (1, 2, 3, 4, 5)
ACTIVATION = 0
INHIBITION = 1
SIGN_NAMES = {ACTIVATION: "activates", INHIBITION: "inhibits"}
VARIANTS = 2


def _matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    header = [name.strip().strip('"') for name in lines[0].split("\t")]
    rows = [[float(value) for value in line.split("\t")] for line in lines[1:]]
    return header, rows


@dataclass(frozen=True)
class Network:
    """One DREAM4 network: its genes, signed edges and perturbation data."""

    index: int
    genes: tuple[str, ...]
    wildtype: tuple[float, ...]
    knockout: tuple[tuple[float, ...], ...]
    knockdown: tuple[tuple[float, ...], ...]
    parents: dict[str, tuple[tuple[str, int], ...]]

    def position(self, gene: str) -> int:
        return self.genes.index(gene)

    def targets_with_in_degree(self, size: int) -> tuple[str, ...]:
        return tuple(
            gene
            for gene in self.genes
            if len(self.parents.get(gene, ())) == size
        )


def load_network(index: int, root: Path | None = None) -> Network:
    """Read one network and derive each edge's sign from the knockout data."""
    root = Path(root or DREAM4_ROOT)
    missing = [
        name
        for name in (
            f"goldstandard_{index}.tsv",
            f"knockouts_{index}.tsv",
            f"knockdowns_{index}.tsv",
            f"wildtype_{index}.tsv",
        )
        if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"DREAM4 network {index} is incomplete under {root}: missing {missing}. "
            "Place the DREAM4 In Silico Size 100 files there."
        )
    genes, knockout = _matrix(root / f"knockouts_{index}.tsv")
    _, knockdown = _matrix(root / f"knockdowns_{index}.tsv")
    _, wildtype_rows = _matrix(root / f"wildtype_{index}.tsv")
    wildtype = wildtype_rows[0]
    position = {gene: slot for slot, gene in enumerate(genes)}

    edges: dict[str, list[tuple[str, int]]] = {}
    for line in (root / f"goldstandard_{index}.tsv").read_text(encoding="utf-8").strip().split("\n"):
        regulator, target, present = line.split("\t")
        if present.strip() != "1":
            continue
        # Removing an activator lowers the target; removing an inhibitor
        # raises it.  The knockout row of the regulator against the wild type
        # therefore gives the sign.
        delta = knockout[position[regulator]][position[target]] - wildtype[position[target]]
        sign = INHIBITION if delta > 0.0 else ACTIVATION
        edges.setdefault(target, []).append((regulator, sign))

    return Network(
        index=index,
        genes=tuple(genes),
        wildtype=tuple(wildtype),
        knockout=tuple(tuple(row) for row in knockout),
        knockdown=tuple(tuple(row) for row in knockdown),
        parents={target: tuple(sorted(items)) for target, items in edges.items()},
    )


@dataclass(frozen=True)
class GRNTask:
    """A target gene, its candidate regulators and the true signed parents."""

    network: int
    target: str
    regulators: tuple[str, ...]
    truth_values: tuple[int, ...]

    @property
    def task_size(self) -> int:
        return sum(value >= 0 for value in self.truth_values)

    def label(self) -> str:
        parts = [
            f"{self.regulators[family]} {SIGN_NAMES[value]}"
            for family, value in enumerate(self.truth_values)
            if value >= 0
        ]
        return f"net{self.network}:{self.target} <- " + ", ".join(parts)


def build_grn_task(network: Network, target: str) -> GRNTask:
    regulators = tuple(gene for gene in network.genes if gene != target)
    slot = {gene: index for index, gene in enumerate(regulators)}
    truth = [-1] * len(regulators)
    for regulator, sign in network.parents.get(target, ()):
        truth[slot[regulator]] = sign
    return GRNTask(
        network=network.index,
        target=target,
        regulators=regulators,
        truth_values=tuple(truth),
    )


def grn_task_instance(
    *, task: GRNTask, agents: int, seed: int, initial_width: int
) -> TaskInstance:
    """Wrap a GRN task in the shared Mode-B initialisation.

    The truth comes from the data rather than from ``make_truth``, but the
    per-agent initial frontier is drawn by the same ``make_initial_pairs``
    stream the other environments use, so a task is identical across degrees
    and policies.
    """
    encoding = Encoding(families=len(task.regulators), variants=VARIANTS)
    encoding.check()
    truth_pairs = [
        family * VARIANTS + value
        for family, value in enumerate(task.truth_values)
        if value >= 0
    ]
    if not truth_pairs:
        raise ValueError(f"target {task.target!r} in network {task.network} has no parents")

    # Same stream and shape as make_truth's second half, so every agent still
    # starts from exactly one true component.
    order = list(range(len(truth_pairs)))
    assignment_rng = derived_rng(seed, ASSIGNMENT_DOMAIN)
    assignment_rng.shuffle(order)
    primary_groups = [order[index % len(truth_pairs)] for index in range(agents)]
    assignment_rng.shuffle(primary_groups)

    initial = make_initial_pairs(
        agents,
        list(truth_pairs),
        list(primary_groups),
        encoding.pair_count,
        seed,
        initial_width,
    )
    return TaskInstance(
        seed=seed,
        encoding=encoding,
        truth_values=tuple(task.truth_values),
        truth_pairs=tuple(encoding.pair(index) for index in truth_pairs),
        primary_groups=tuple(primary_groups),
        initial_pairs=tuple(
            tuple(encoding.pair(index) for index in group) for group in initial
        ),
    )


def hypothesis_space_bits(regulators: int, task_size: int) -> float:
    """``H(K) = log2[C(regulators, K) * 2^K]``, the paper's complexity axis."""
    return math.log2(math.comb(regulators, task_size)) + task_size


def select_targets(
    task_size: int,
    limit: int | None = None,
    networks: Sequence[int] = NETWORKS,
    root: Path | None = None,
) -> list[str]:
    """``network:gene`` for every target of the given in-degree, in order."""
    chosen: list[str] = []
    for index in networks:
        network = load_network(index, root)
        for target in network.targets_with_in_degree(task_size):
            chosen.append(f"{index}:{target}")
            if limit is not None and len(chosen) >= limit:
                return chosen
    return chosen


def main(argv: list[str] | None = None) -> int:
    """Print the available targets per in-degree, to write a config from."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--task-size", type=int, help="only this in-degree")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    sizes = [args.task_size] if args.task_size else list(range(1, 9))
    report = {}
    for size in sizes:
        targets = select_targets(size, args.limit)
        if not targets:
            continue
        example = load_network(int(targets[0].split(":")[0]))
        report[str(size)] = {
            "targets": targets,
            "count": len(targets),
            "H_bits": round(hypothesis_space_bits(len(example.genes) - 1, size), 3),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
