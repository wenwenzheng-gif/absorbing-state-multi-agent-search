"""PNG/PDF figures backed by exported numerical tables; no sigmoid curves."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.summarize import write_csv

STYLES = {"deterministic": ("#17629C", "o", "Deterministic"),
          "llm": ("#C45118", "s", "GPT-5.4 Nano")}


def plot_all(rows: list[dict], episodes: list[dict], output: Path, mode: str) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 220, "pdf.fonttype": 42, "axes.labelsize": 10})
    files = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            files.append(str(path))
        plt.close(fig)

    for task, names, title in (
        ("needle", ["Needle-Low", "Needle-High"], "Needle Search"),
        ("chain", ["Chain-Short-Horizon", "Chain-Sufficient-Horizon"], "Chain Search"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True, layout="constrained")
        for ax, condition in zip(axes, names):
            for kind, (color, marker, label) in STYLES.items():
                sub = sorted([r for r in rows if r["condition"] == condition and r["agent_type"] == kind], key=lambda r: r["degree"])
                if not sub:
                    continue
                x, y = [r["degree"] for r in sub], [r["p_success"] for r in sub]
                error = [[max(0, r["p_success"] - r["ci_low"]) for r in sub],
                         [max(0, r["ci_high"] - r["p_success"]) for r in sub]]
                ax.errorbar(x, y, yerr=error, color=color, marker=marker, capsize=4,
                            linestyle="-" if kind == "deterministic" else "--", linewidth=1.5,
                            markerfacecolor="white" if kind == "llm" else color,
                            label=f"{label} (n={sub[0]['episodes']}/degree)")
                ax.set_xticks(x)
            ax.set_title(condition)
            ax.set_ylim(-.06, 1.06)
            ax.set_xlabel("Communication degree d")
            ax.grid(axis="y", alpha=.2)
        legend = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        fig.legend(list(legend.values()), list(legend), fontsize=9, loc="outside lower center", ncol=2)
        axes[0].set_ylabel("Success probability (Wilson 95% CI)")
        fig.suptitle(f"{title} · {mode.capitalize()}", fontweight="bold")
        save(fig, f"figure_{1 if task == 'needle' else 2}_{task}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for condition, marker in (("Needle-Low", "o"), ("Needle-High", "s")):
        for kind, (color, _, label) in STYLES.items():
            sub = sorted([r for r in rows if r["condition"] == condition and r["agent_type"] == kind], key=lambda r: r["degree"])
            if not sub:
                continue
            for ax, metric in zip(axes[0], ("coverage_fraction", "duplicate_fraction")):
                ax.plot([r["degree"] for r in sub], [r[metric] for r in sub], marker=marker, color=color,
                        linestyle="-" if kind == "deterministic" else "--", label=f"{condition} / {label}")
                ax.set_xlabel("Communication degree d")
                ax.set_ylim(-.04, 1.04)
                ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Unique coverage / S")
    axes[0, 1].set_ylabel("Duplicate tests / valid tests")
    axes[0, 0].legend(fontsize=7)
    depth_rows = []
    degree_styles = ["-", "--", ":", "-."]
    for ax, condition in zip(axes[1], ("Chain-Short-Horizon", "Chain-Sufficient-Horizon")):
        groups = defaultdict(list)
        for e in episodes:
            if e["condition"] == condition:
                groups[e["agent_type"], e["degree"]].append(e)
        degrees = sorted({key[1] for key in groups})
        for (kind, degree), es in sorted(groups.items()):
            horizon = es[0]["horizon"]
            # Completion is absorbing; display completed depth for remaining horizon.
            depths = [[0] + e["depth_after_each_round"] + [e["maximum_depth"]] * (horizon - len(e["depth_after_each_round"])) for e in es]
            average = np.mean(depths, axis=0)
            color, marker, label = STYLES[kind]
            ax.plot(range(horizon + 1), average, color=color, marker=marker, markersize=3,
                    linestyle=degree_styles[degrees.index(degree) % len(degree_styles)], label=f"{label}, d={degree}")
            depth_rows.extend({"condition": condition, "agent_type": kind, "degree": degree,
                               "round": r, "mean_depth": float(value), "episodes": len(es)} for r, value in enumerate(average))
        ax.set_title(condition)
        ax.set_xlabel("Global round")
        ax.set_ylabel("Maximum completed depth")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(f"Coverage, duplication and sequential depth · {mode.capitalize()}", fontweight="bold")
    save(fig, "figure_3_mechanisms")
    write_csv(output.parent / "depth_by_round.csv", depth_rows)
    return files
