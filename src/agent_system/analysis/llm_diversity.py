#!/usr/bin/env python3
"""How much the LLM agents differ from one another, and from a rule policy.

The mechanism behind the LLM runs' weak effective degree is not that the model
reasons badly about any single decision; it is that all ``N`` agents are given
the same prompt with the same candidate order and no identity, so they make
the *same* decision.  Evidence that arrives from a neighbour is then evidence
the receiver already had.  That is a property of the action stream, and it can
be measured without re-running anything:

* collisions -- distinct tested components per test in a round, and the
  largest number of agents on one component;
* entropy of the tested-component distribution across agents, and of the
  chosen ``experiment_id`` where the environment offers a library of them;
* position bias -- where in the offered candidate list the chosen option sat,
  against the uniform baseline implied by the list lengths actually shown;
* a heuristic the prompt never asked for: preferring a family whose allowed
  set has collapsed to a single value, measured against how often such a
  candidate was on offer.

``--paired-rule-run`` adds the matched rule run (same tasks, degrees and seeds)
so every number has its control.  Collisions and entropies come from
``events.jsonl`` and so work for both policies; position bias and the
singleton heuristic need ``requests.jsonl`` and exist only for LLM runs.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import re
from typing import Sequence

from ..storage import iter_jsonl
from .run_cell import cell_parameters, matplotlib_pyplot, save_figure
from .summarize import write_csv

OFFER_PATTERN = re.compile(r"CANDIDATES YOU MAY TEST THIS SLOT\s*\n(\[.*?\])\s*\n", re.S)
ALLOWED_PATTERN = re.compile(r"values still allowed per family:\s*(\{.*?\})\s*\n")
COMPONENT_PATTERN = re.compile(r"^c(\d+)v(\d+)$")

DEGREE_COLUMNS = (
    "run", "policy", "d", "rounds_observed", "tests", "distinct_components",
    "distinct_per_test", "mean_max_multiplicity", "mean_entropy_bits",
    "mean_normalised_entropy", "experiment_id_tests", "distinct_experiment_ids",
    "experiment_id_entropy_bits", "experiment_id_top2_share",
)
POSITION_COLUMNS = (
    "run", "d", "decisions", "P_first", "uniform_baseline_P_first",
    "mean_position", "mean_relative_position", "mean_offer_size",
    "singleton_family_chosen", "singleton_family_available",
)


def _entropy_bits(counts: Sequence[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def episode_degrees(run_dir: Path) -> dict[str, int]:
    return {
        episode["episode_id"]: int(episode["d"])
        for episode in iter_jsonl(Path(run_dir) / "episodes.jsonl")
    }


def collision_table(run_dir: Path) -> list[dict]:
    """Per-degree collision and entropy statistics from the experiment events."""
    run_dir = Path(run_dir)
    degrees = episode_degrees(run_dir)
    cell = cell_parameters(run_dir) or {}
    per_round: dict[tuple[str, int], list] = collections.defaultdict(list)
    experiment_ids: dict[tuple[str, int], list] = collections.defaultdict(list)
    for event in iter_jsonl(run_dir / "events.jsonl"):
        if event.get("kind") != "experiment":
            continue
        key = (event.get("episode"), event.get("round"))
        pair = event.get("pair")
        if pair is not None:
            per_round[key].append(tuple(pair))
        if event.get("experiment_id") is not None:
            experiment_ids[key].append(event["experiment_id"])

    grouped: dict[int, dict] = {}
    for (episode, _round), pairs in per_round.items():
        degree = degrees.get(episode)
        if degree is None:
            continue
        counts = collections.Counter(pairs)
        entry = grouped.setdefault(
            degree,
            {"rounds": 0, "tests": 0, "distinct": 0, "max_mult": [], "entropy": [],
             "normalised": [], "experiment_ids": collections.Counter(),
             "experiment_tests": 0},
        )
        entry["rounds"] += 1
        entry["tests"] += len(pairs)
        entry["distinct"] += len(counts)
        entry["max_mult"].append(max(counts.values()))
        bits = _entropy_bits(list(counts.values()))
        if bits is not None:
            entry["entropy"].append(bits)
            ceiling = math.log2(len(pairs)) if len(pairs) > 1 else None
            if ceiling:
                entry["normalised"].append(bits / ceiling)
    for (episode, _round), ids in experiment_ids.items():
        degree = degrees.get(episode)
        if degree is None or degree not in grouped:
            continue
        grouped[degree]["experiment_ids"].update(ids)
        grouped[degree]["experiment_tests"] += len(ids)

    rows = []
    for degree, entry in sorted(grouped.items()):
        counter = entry["experiment_ids"]
        top2 = sum(count for _, count in counter.most_common(2))
        rows.append(
            {
                "run": run_dir.name,
                "policy": cell.get("policy"),
                "d": degree,
                "rounds_observed": entry["rounds"],
                "tests": entry["tests"],
                "distinct_components": entry["distinct"],
                "distinct_per_test": entry["distinct"] / entry["tests"] if entry["tests"] else None,
                "mean_max_multiplicity": (
                    sum(entry["max_mult"]) / len(entry["max_mult"]) if entry["max_mult"] else None
                ),
                "mean_entropy_bits": (
                    sum(entry["entropy"]) / len(entry["entropy"]) if entry["entropy"] else None
                ),
                "mean_normalised_entropy": (
                    sum(entry["normalised"]) / len(entry["normalised"])
                    if entry["normalised"] else None
                ),
                "experiment_id_tests": entry["experiment_tests"],
                "distinct_experiment_ids": len(counter),
                "experiment_id_entropy_bits": _entropy_bits(list(counter.values())),
                "experiment_id_top2_share": (
                    top2 / entry["experiment_tests"] if entry["experiment_tests"] else None
                ),
            }
        )
    return rows


def _overall(rows: Sequence[dict], run_dir: Path) -> dict:
    tests = sum(row["tests"] for row in rows)
    distinct = sum(row["distinct_components"] for row in rows)
    weights = [(row["mean_max_multiplicity"], row["rounds_observed"]) for row in rows
               if row["mean_max_multiplicity"] is not None]
    rounds = sum(weight for _, weight in weights)
    experiment_tests = sum(row["experiment_id_tests"] for row in rows)
    counter: collections.Counter = collections.Counter()
    for event in iter_jsonl(Path(run_dir) / "events.jsonl"):
        if event.get("kind") == "experiment" and event.get("experiment_id") is not None:
            counter[event["experiment_id"]] += 1
    top2 = sum(count for _, count in counter.most_common(2))
    return {
        "run": Path(run_dir).name,
        "tests": tests,
        "distinct_per_test": distinct / tests if tests else None,
        "mean_max_multiplicity": (
            sum(value * weight for value, weight in weights) / rounds if rounds else None
        ),
        "experiment_id_tests": experiment_tests,
        "distinct_experiment_ids": len(counter),
        "experiment_id_entropy_bits": _entropy_bits(list(counter.values())),
        "experiment_id_top2_share": top2 / experiment_tests if experiment_tests else None,
        "experiment_id_top": counter.most_common(5),
    }


def _chosen_component(content: str) -> tuple[int, int] | None:
    """The tested component, from either response shape the schemas produce."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if isinstance(payload.get("family_id"), int) and isinstance(payload.get("variant_id"), int):
        return payload["family_id"], payload["variant_id"]
    choice = payload.get("choice")
    if isinstance(choice, str):
        match = COMPONENT_PATTERN.match(choice.split("|")[0].strip())
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def position_table(run_dir: Path) -> tuple[list[dict], dict, list[dict]]:
    """Where the chosen candidate sat in the list the model was shown."""
    run_dir = Path(run_dir)
    requests_path = run_dir / "requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    if not requests_path.is_file() or not responses_path.is_file():
        return [], {"status": "no_llm_transcript"}, []
    prompts = {
        request["cache_key"]: request["messages"][-1]["content"]
        for request in iter_jsonl(requests_path)
        if request.get("schema_name") == "experiment_decision" and request.get("messages")
    }
    degrees = episode_degrees(run_dir)
    per_degree: dict[int, dict] = {}
    histogram: collections.Counter = collections.Counter()
    unparsed = 0
    for response in iter_jsonl(responses_path):
        prompt = prompts.get(response.get("cache_key"))
        if prompt is None:
            continue
        offered = OFFER_PATTERN.search(prompt)
        component = _chosen_component(response.get("content") or "")
        if offered is None or component is None:
            unparsed += 1
            continue
        try:
            options = json.loads(offered.group(1))
        except ValueError:
            unparsed += 1
            continue
        index = next(
            (i for i, option in enumerate(options)
             if (option.get("family_id"), option.get("variant_id")) == component),
            None,
        )
        if index is None or not options:
            unparsed += 1
            continue
        degree = degrees.get(response.get("episode"))
        entry = per_degree.setdefault(
            degree,
            {"decisions": 0, "first": 0, "baseline": 0.0, "position": 0, "relative": 0.0,
             "offer": 0, "singleton_chosen": 0, "singleton_available": 0.0,
             "singleton_decisions": 0},
        )
        entry["decisions"] += 1
        entry["first"] += int(index == 0)
        entry["baseline"] += 1.0 / len(options)
        entry["position"] += index
        entry["relative"] += index / (len(options) - 1) if len(options) > 1 else 0.0
        entry["offer"] += len(options)
        histogram[index] += 1
        allowed = ALLOWED_PATTERN.search(prompt)
        if allowed:
            try:
                values = json.loads(allowed.group(1))
            except ValueError:
                values = None
            if values:
                singletons = {int(key) for key, item in values.items() if len(item) == 1}
                entry["singleton_decisions"] += 1
                entry["singleton_chosen"] += int(component[0] in singletons)
                entry["singleton_available"] += sum(
                    1 for option in options if option.get("family_id") in singletons
                ) / len(options)

    rows = []
    for degree, entry in sorted(per_degree.items(), key=lambda item: (item[0] is None, item[0])):
        count = entry["decisions"]
        singles = entry["singleton_decisions"]
        rows.append(
            {
                "run": run_dir.name,
                "d": degree,
                "decisions": count,
                "P_first": entry["first"] / count,
                "uniform_baseline_P_first": entry["baseline"] / count,
                "mean_position": entry["position"] / count,
                "mean_relative_position": entry["relative"] / count,
                "mean_offer_size": entry["offer"] / count,
                "singleton_family_chosen": entry["singleton_chosen"] / singles if singles else None,
                "singleton_family_available": (
                    entry["singleton_available"] / singles if singles else None
                ),
            }
        )
    totals = {key: sum(entry[key] for entry in per_degree.values())
              for key in ("decisions", "first", "position", "singleton_chosen",
                          "singleton_decisions", "offer")}
    totals["baseline"] = sum(entry["baseline"] for entry in per_degree.values())
    totals["singleton_available"] = sum(
        entry["singleton_available"] for entry in per_degree.values()
    )
    count = totals["decisions"]
    overall = {
        "status": "ok" if count else "no_parsable_decisions",
        "decisions": count,
        "unparsed": unparsed,
        "P_first": totals["first"] / count if count else None,
        "uniform_baseline_P_first": totals["baseline"] / count if count else None,
        "mean_position": totals["position"] / count if count else None,
        "mean_offer_size": totals["offer"] / count if count else None,
        "singleton_family_chosen": (
            totals["singleton_chosen"] / totals["singleton_decisions"]
            if totals["singleton_decisions"] else None
        ),
        "singleton_family_available": (
            totals["singleton_available"] / totals["singleton_decisions"]
            if totals["singleton_decisions"] else None
        ),
    }
    histogram_rows = [
        {"position": position, "decisions": count_at, "share": count_at / count}
        for position, count_at in sorted(histogram.items())
    ] if count else []
    return rows, overall, histogram_rows


def _render(
    llm_rows: Sequence[dict],
    rule_rows: Sequence[dict],
    histogram: Sequence[dict],
    figures_dir: Path,
    title: str,
) -> list[str]:
    pyplot = matplotlib_pyplot()
    if pyplot is None or not llm_rows:
        return []
    figure, axes = pyplot.subplots(1, 2, figsize=(7.4, 3.1), dpi=160)
    left, right = axes
    for rows, label in ((llm_rows, "LLM"), (rule_rows, "paired rule")):
        if not rows:
            continue
        left.plot([r["d"] for r in rows], [r["distinct_per_test"] for r in rows],
                  marker="o", markersize=3.5, linewidth=1.1, label=label)
    left.set_xlabel("degree d")
    left.set_ylabel("distinct components / tests")
    left.set_ylim(0, 1.02)
    left.set_title("action diversity", fontsize=9)
    left.legend(fontsize=6.5, frameon=False)
    if histogram:
        right.bar([h["position"] for h in histogram], [h["share"] for h in histogram],
                  width=0.7, label="chosen")
        right.set_xlabel("position in the offered list")
        right.set_ylabel("share of decisions")
        right.set_title("position bias", fontsize=9)
        right.legend(fontsize=6.5, frameon=False)
    else:
        right.axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    written = save_figure(figure, figures_dir, "llm_diversity")
    pyplot.close(figure)
    return written


def report(run_dir: Path, paired_rule_run: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    llm_rows = collision_table(run_dir)
    rule_rows = collision_table(paired_rule_run) if paired_rule_run else []
    positions, position_overall, histogram = position_table(run_dir)

    analysis_dir = run_dir / "analysis"
    write_csv(
        analysis_dir / "llm_diversity_by_degree.csv", [*llm_rows, *rule_rows], DEGREE_COLUMNS
    )
    write_csv(analysis_dir / "llm_diversity_positions.csv", positions, POSITION_COLUMNS)
    if histogram:
        write_csv(
            analysis_dir / "llm_diversity_position_histogram.csv", histogram,
            ("position", "decisions", "share"),
        )
    figures = _render(llm_rows, rule_rows, histogram, run_dir / "figures", run_dir.name)

    summary = {
        "run_dir": str(run_dir),
        "paired_rule_run": str(paired_rule_run) if paired_rule_run else None,
        "llm": _overall(llm_rows, run_dir),
        "rule": _overall(rule_rows, paired_rule_run) if paired_rule_run else None,
        "by_degree": {"llm": llm_rows, "rule": rule_rows},
        "position_bias": position_overall,
        "position_by_degree": positions,
        "figures": figures,
        "note": (
            "distinct_per_test pools over (episode, round): the number of "
            "distinct tested components divided by the number of tests. "
            "Position bias and the singleton-family heuristic are parsed out of "
            "the experiment prompt; a prompt that renames the offered-candidate "
            "block reports no parsable decisions rather than a wrong number."
        ),
    }
    (analysis_dir / "llm_diversity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--paired-rule-run", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.run_dir, args.paired_rule_run), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
