#!/usr/bin/env python3
"""Fit clean-regime scaling relations and generate the six final figures.

The full five-exponent design is fitted only when it is identifiable.  If all
THEORY-CLEAN points share one branching factor, alpha_b is reported as missing
rather than being fixed or inferred from redundant cells.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .utils import exact_analytic_dc


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "data" / "master_resolved_cells.csv"
BOOTSTRAP_INPUT = ROOT / "data" / "bootstrap"
OUTPUT_ROOT = ROOT / "reproduced" / "quick"
ANALYSIS = OUTPUT_ROOT / "data"
FIGURES = OUTPUT_ROOT / "figures"
BOOT = ANALYSIS / "bootstrap"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 2026091704

CLASS_COLORS = {
    "THEORY-CLEAN": "#0072B2",
    "SUPPLY-CLEAN-BUT-REDUNDANT": "#E69F00",
    "SUPPLY-DEPLETED": "#CC79A7",
}
CLASS_MARKERS = {
    "THEORY-CLEAN": "o",
    "SUPPLY-CLEAN-BUT-REDUNDANT": "s",
    "SUPPLY-DEPLETED": "X",
}
BASELINE_COLORS = {"A": "#0072B2", "B": "#009E73", "C": "#D55E00"}
BASELINE_MARKERS = {"A": "o", "B": "s", "C": "^"}


def r2(y: np.ndarray, fitted: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    return math.nan if denom <= 0 else 1.0 - float(np.sum((y - fitted) ** 2)) / denom


def qci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


class BootstrapStore:
    def __init__(self) -> None:
        self.archives = {
            "v2": np.load(BOOTSTRAP_INPUT / "v2_threshold_bootstrap_draws.npz"),
            "v3_width": np.load(BOOTSTRAP_INPUT / "v3_width_threshold_bootstrap_draws.npz"),
            "logb": np.load(BOOTSTRAP_INPUT / "logb_threshold_bootstrap_draws.npz"),
        }

    def get(self, row: pd.Series) -> np.ndarray:
        source, key = str(row.bootstrap_source), str(row.bootstrap_key)
        if source not in self.archives or key not in self.archives[source].files:
            raise KeyError(f"missing bootstrap draws for {row.cell_uid}: {source}/{key}")
        values = np.asarray(self.archives[source][key], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) < BOOTSTRAP_DRAWS:
            raise ValueError(f"too few bootstrap draws for {row.cell_uid}: {len(values)}")
        return values[:BOOTSTRAP_DRAWS]


def design_matrix(frame: pd.DataFrame, include_b: bool) -> tuple[np.ndarray, list[str]]:
    columns = [
        np.ones(len(frame)),
        np.log(frame.M.to_numpy(float) / 16.0),
        np.log(frame.K.to_numpy(float) / 4.0),
    ]
    names = ["log_C_ref", "alpha_M", "alpha_K"]
    if include_b:
        columns.append(np.log(np.log(frame.b.to_numpy(float)) / math.log(2.0)))
        names.append("alpha_b")
    columns.extend(
        [
            np.log(frame.m.to_numpy(float)),
            np.log(frame["T"].to_numpy(float) / 4.0),
        ]
    )
    names.extend(["alpha_m", "alpha_T"])
    return np.column_stack(columns), names


def log_sigma(frame: pd.DataFrame) -> np.ndarray:
    low = frame.d_c_CI_low.to_numpy(float)
    high = frame.d_c_CI_high.to_numpy(float)
    sigma = (np.log(high) - np.log(low)) / (2.0 * 1.959963984540054)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("global-fit rows require finite positive bootstrap uncertainty")
    return sigma


def weighted_fit(X: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sw = 1.0 / sigma
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def fit_global(frame: pd.DataFrame, store: BootstrapStore) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    subset = frame.loc[frame.primary_global_fit].copy().reset_index(drop=True)
    if not subset.theory_clean.all():
        raise ValueError("global fit includes a non-THEORY-CLEAN cell")
    include_b = subset.b.nunique() >= 2
    X, names = design_matrix(subset, include_b)
    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise ValueError(f"global design rank {rank} < {X.shape[1]}")
    y = np.log(subset.d_c_data.to_numpy(float))
    sigma = log_sigma(subset)
    beta = weighted_fit(X, y, sigma)
    fitted_log = X @ beta
    fitted = np.exp(fitted_log)
    rel = fitted / subset.d_c_data.to_numpy(float) - 1.0

    draw_matrix = np.column_stack([store.get(row) for _, row in subset.iterrows()])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_betas: list[np.ndarray] = []
    attempts = 0
    while len(bootstrap_betas) < BOOTSTRAP_DRAWS and attempts < BOOTSTRAP_DRAWS * 8:
        attempts += 1
        indices = rng.integers(0, len(subset), size=len(subset))
        Xb = X[indices]
        if np.linalg.matrix_rank(Xb) != X.shape[1]:
            continue
        draw_index = int(rng.integers(0, BOOTSTRAP_DRAWS))
        yb = np.log(draw_matrix[draw_index, indices])
        try:
            bb = weighted_fit(Xb, yb, sigma[indices])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(bb)):
            bootstrap_betas.append(bb)
    if len(bootstrap_betas) < BOOTSTRAP_DRAWS:
        raise RuntimeError(f"only {len(bootstrap_betas)} valid global bootstrap fits")
    boot = np.asarray(bootstrap_betas)
    BOOT.mkdir(parents=True, exist_ok=True)
    global_draws = {}
    for index, name in enumerate(names):
        key = "C_ref" if name == "log_C_ref" else name
        global_draws[key] = np.exp(boot[:, index]) if name == "log_C_ref" else boot[:, index]
    np.savez_compressed(BOOT / "global_parameter_bootstrap_draws.npz", **global_draws)

    rows = []
    for index, name in enumerate(names):
        values = np.exp(boot[:, index]) if name == "log_C_ref" else boot[:, index]
        point = math.exp(beta[index]) if name == "log_C_ref" else beta[index]
        low, high = qci(values)
        rows.append(
            {
                "parameter": "C_ref" if name == "log_C_ref" else name,
                "estimate": float(point),
                "ci_low": low,
                "ci_high": high,
                "reference": (
                    1.0 if name in {"alpha_M", "alpha_K", "alpha_b"}
                    else -1.0 if name in {"alpha_m", "alpha_T"}
                    else math.nan
                ),
                "identifiable": True,
                "note": "C_ref is normalized at M=16,K=4,b=2,m=1,T=4" if name == "log_C_ref" else "",
            }
        )
    if not include_b:
        rows.insert(
            3,
            {
                "parameter": "alpha_b",
                "estimate": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "reference": 1.0,
                "identifiable": False,
                "note": "all primary THEORY-CLEAN cells have b=2",
            },
        )
    params = pd.DataFrame(rows)

    predictions = subset[
        ["cell_uid", "source_cell_id", "experiment_family", "M", "K", "T", "b", "m", "n0", "d_c_data"]
    ].copy()
    predictions["d_c_fit"] = fitted
    predictions["fit_relative_error"] = rel
    predictions["absolute_fit_relative_error"] = np.abs(rel)

    metrics = {
        "fit_scope": "THEORY-CLEAN primary cells only",
        "n_cells": int(len(subset)),
        "normalized_formula": "d_c=C_ref*(M/16)^alpha_M*(K/4)^alpha_K*(ln(b)/ln(2))^alpha_b*m^alpha_m*(T/4)^alpha_T",
        "alpha_b_identifiable": include_b,
        "design_rank": rank,
        "design_columns": X.shape[1],
        "R2_log": r2(y, fitted_log),
        "RMSE_log": float(np.sqrt(np.mean((y - fitted_log) ** 2))),
        "median_absolute_relative_prediction_error": float(np.median(np.abs(rel))),
        "mean_absolute_relative_prediction_error": float(np.mean(np.abs(rel))),
        "max_absolute_relative_prediction_error": float(np.max(np.abs(rel))),
        "bootstrap_method": "hierarchical cell resampling plus within-cell paired-threshold bootstrap",
        "bootstrap_valid_draws": len(bootstrap_betas),
        "bootstrap_attempts": attempts,
        "cells": subset.cell_uid.tolist(),
    }
    return params, predictions, metrics


def ordinary_power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    alpha, log_A = np.polyfit(np.log(x), np.log(y), 1)
    fitted = log_A + alpha * np.log(x)
    return float(math.exp(log_A)), float(alpha), r2(np.log(y), fitted)


def fit_one_factor(frame: pd.DataFrame, store: BootstrapStore) -> pd.DataFrame:
    definitions = {
        "M": ("M", {"v2::M12_N160", "v2::base_N160_T4", "v2::M20_N160", "v2::M24_N160"}, 1.0),
        "K": ("K", {"v2::K3_N160", "v2::base_N160_T4", "v2::K5_N160", "v2::K6_N160"}, 1.0),
        "m": ("m", {"v2::base_N160_T4", "v2::m2_N160", "v2::m3_N160"}, -1.0),
        "T_n0_32": ("T", {"v3_width::b2_T4_n0_32", "v3_width::b2_T5_n0_32", "v2::T6_n0_32"}, -1.0),
    }
    rows: list[dict] = []
    archived_draws: dict[str, np.ndarray] = {}
    for name, (variable, ids, reference) in definitions.items():
        subset = frame.loc[frame.cell_uid.isin(ids)].sort_values(variable)
        if len(subset) != len(ids) or not subset.theory_clean.all():
            raise ValueError(f"invalid controlled one-factor set: {name}")
        x = subset[variable].to_numpy(float)
        y = subset.d_c_data.to_numpy(float)
        A, alpha, fit_r2 = ordinary_power_fit(x, y)
        draws = np.column_stack([store.get(row) for _, row in subset.iterrows()])
        draw_A = np.empty(BOOTSTRAP_DRAWS)
        draw_alpha = np.empty(BOOTSTRAP_DRAWS)
        for i in range(BOOTSTRAP_DRAWS):
            draw_A[i], draw_alpha[i], _ = ordinary_power_fit(x, draws[i])
        A_low, A_high = qci(draw_A)
        alpha_low, alpha_high = qci(draw_alpha)
        archived_draws[f"{name}__A"] = draw_A
        archived_draws[f"{name}__exponent"] = draw_alpha
        rows.append(
            {
                "fit_name": name,
                "fit_role": "primary controlled THEORY-CLEAN one-factor scan",
                "variable": variable,
                "n_cells": len(subset),
                "levels": ";".join(str(value) for value in x),
                "A": A,
                "A_ci_low": A_low,
                "A_ci_high": A_high,
                "exponent": alpha,
                "exponent_ci_low": alpha_low,
                "exponent_ci_high": alpha_high,
                "R2_log": fit_r2,
                "theory_reference": reference,
                "identifiable": True,
                "cells": ";".join(subset.cell_uid),
            }
        )

    # A clean b fit needs multiple distinct clean b levels within one matched baseline.
    for baseline in ("A", "B", "C"):
        family = f"matched_logb_{baseline}"
        all_cells = frame.loc[frame.experiment_family == family].sort_values("b")
        clean = all_cells.loc[all_cells.theory_clean]
        distinct = clean.b.nunique()
        clean_record = {
            "fit_name": f"b_clean_{baseline}",
            "fit_role": "primary THEORY-CLEAN matched-baseline b scan",
            "variable": "ln_b",
            "n_cells": len(clean),
            "levels": ";".join(str(int(value)) for value in clean.b),
            "A": math.nan, "A_ci_low": math.nan, "A_ci_high": math.nan,
            "exponent": math.nan, "exponent_ci_low": math.nan, "exponent_ci_high": math.nan,
            "R2_log": math.nan,
            "theory_reference": 1.0,
            "identifiable": distinct >= 3,
            "cells": ";".join(clean.cell_uid),
            "reason": "fewer than three distinct THEORY-CLEAN b levels" if distinct < 3 else "",
        }
        if distinct >= 3:
            clean_x = np.log(clean.b.to_numpy(float))
            clean_y = clean.d_c_data.to_numpy(float)
            clean_A, clean_alpha, clean_r2 = ordinary_power_fit(clean_x, clean_y)
            clean_draws = np.column_stack([store.get(row) for _, row in clean.iterrows()])
            clean_alpha_draws = np.asarray(
                [ordinary_power_fit(clean_x, draw)[1] for draw in clean_draws]
            )
            clean_low, clean_high = qci(clean_alpha_draws)
            clean_record.update(
                {
                    "A": clean_A,
                    "exponent": clean_alpha,
                    "exponent_ci_low": clean_low,
                    "exponent_ci_high": clean_high,
                    "R2_log": clean_r2,
                }
            )
        rows.append(clean_record)

        # Diagnostic all-cell fit is deliberately separate and includes redundant cells.
        x = np.log(all_cells.b.to_numpy(float))
        y = all_cells.d_c_data.to_numpy(float)
        A, alpha, fit_r2 = ordinary_power_fit(x, y)
        draws = np.column_stack([store.get(row) for _, row in all_cells.iterrows()])
        alpha_draws = np.asarray([ordinary_power_fit(x, draw)[1] for draw in draws])
        alpha_low, alpha_high = qci(alpha_draws)
        slope0 = float(np.dot(x, y) / np.dot(x, x))
        free_slope, free_intercept = np.polyfit(x, y, 1)
        slope0_draws = np.asarray([np.dot(x, draw) / np.dot(x, x) for draw in draws])
        free = np.asarray([np.polyfit(x, draw, 1) for draw in draws])
        slope0_low, slope0_high = qci(slope0_draws)
        c0_low, c0_high = qci(free[:, 1])
        archived_draws[f"b_descriptive_{baseline}__exponent"] = alpha_draws
        archived_draws[f"b_descriptive_{baseline}__through_origin_A"] = slope0_draws
        archived_draws[f"b_descriptive_{baseline}__free_intercept"] = free[:, 1]
        rows.append(
            {
                "fit_name": f"b_descriptive_{baseline}",
                "fit_role": "diagnostic only; includes redundancy-rejected cells",
                "variable": "ln_b",
                "n_cells": len(all_cells),
                "levels": ";".join(str(int(value)) for value in all_cells.b),
                "A": A,
                "exponent": alpha,
                "exponent_ci_low": alpha_low,
                "exponent_ci_high": alpha_high,
                "R2_log": fit_r2,
                "theory_reference": 1.0,
                "identifiable": True,
                "through_origin_slope": slope0,
                "through_origin_slope_ci_low": slope0_low,
                "through_origin_slope_ci_high": slope0_high,
                "through_origin_R2_centered": r2(y, slope0 * x),
                "free_intercept": float(free_intercept),
                "free_intercept_ci_low": c0_low,
                "free_intercept_ci_high": c0_high,
                "free_intercept_slope": float(free_slope),
                "free_intercept_R2": r2(y, free_intercept + free_slope * x),
                "cells": ";".join(all_cells.cell_uid),
                "reason": "not admissible as a THEORY-CLEAN exponent test",
            }
        )
    BOOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BOOT / "one_factor_bootstrap_draws.npz", **archived_draws)
    return pd.DataFrame(rows)


def theory_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = {
        "all_resolved": np.ones(len(frame), dtype=bool),
        "supply_clean": frame.supply_clean.to_numpy(bool),
        "theory_clean": frame.theory_clean.to_numpy(bool),
    }
    rows = []
    for name, mask in groups.items():
        subset = frame.loc[mask]
        abs_err = np.abs(subset.relative_error.to_numpy(float))
        gamma, log_A = np.polyfit(np.log(subset.d_c_analytic), np.log(subset.d_c_data), 1)
        pred = log_A + gamma * np.log(subset.d_c_analytic)
        rows.append(
            {
                "group": name,
                "n_cells": len(subset),
                "median_absolute_relative_error": float(np.median(abs_err)),
                "mean_absolute_relative_error": float(np.mean(abs_err)),
                "max_absolute_relative_error": float(np.max(abs_err)),
                "median_data_over_theory": float(np.median(subset.data_over_theory)),
                "descriptive_A": float(math.exp(log_A)),
                "descriptive_gamma": float(gamma),
                "descriptive_R2_log": r2(np.log(subset.d_c_data), pred),
            }
        )
    effective = frame.loc[np.isfinite(frame.d_c_effective)].copy()
    effective["analytic_absolute_relative_error"] = np.abs(effective.relative_error)
    effective["effective_absolute_relative_error"] = np.abs(effective.effective_relative_error)
    effective_summary = {
        "n_cells": len(effective),
        "analytic_median_absolute_relative_error": float(effective.analytic_absolute_relative_error.median()),
        "analytic_mean_absolute_relative_error": float(effective.analytic_absolute_relative_error.mean()),
        "effective_median_absolute_relative_error": float(effective.effective_absolute_relative_error.median()),
        "effective_mean_absolute_relative_error": float(effective.effective_absolute_relative_error.mean()),
        "effective_max_absolute_relative_error": float(effective.effective_absolute_relative_error.max()),
    }
    return pd.DataFrame(rows), effective, effective_summary


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D8D8D8", linewidth=0.7, alpha=0.65)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    png_tmp = FIGURES / f".{stem}.png.tmp"
    pdf_tmp = FIGURES / f".{stem}.pdf.tmp"
    fig.savefig(png_tmp, format="png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_tmp, format="pdf", bbox_inches="tight")
    os.replace(png_tmp, png)
    os.replace(pdf_tmp, pdf)
    plt.close(fig)


def plot_point(ax: plt.Axes, row, x: float, y: float, label: str | None = None) -> None:
    lower = y - float(row.d_c_CI_low) if pd.notna(row.d_c_CI_low) else 0.0
    upper = float(row.d_c_CI_high) - y if pd.notna(row.d_c_CI_high) else 0.0
    ax.errorbar(
        x, y, yerr=[[lower], [upper]] if lower or upper else None,
        marker=CLASS_MARKERS[row.final_classification], markersize=5.8,
        markerfacecolor=CLASS_COLORS[row.final_classification],
        markeredgecolor="white", markeredgewidth=0.55,
        color=CLASS_COLORS[row.final_classification], capsize=2.1,
        linestyle="none", alpha=0.88, label=label,
    )


def make_figure1(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for ax, log_scale in zip(axes, (False, True)):
        for row in frame.itertuples(index=False):
            plot_point(ax, row, row.d_c_analytic, row.d_c_data)
        lo = min(frame.d_c_analytic.min(), frame.d_c_data.min()) * 0.93
        hi = max(frame.d_c_analytic.max(), frame.d_c_data.max()) * 1.06
        ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1.15)
        if log_scale:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title("Log–log view")
        else:
            ax.set_title("Linear view")
        ax.set_xlabel(r"Exact analytic $d_c$")
        ax.set_ylabel(r"Measured $d_c$")
        style_axes(ax)
    handles = [
        Line2D([0], [0], marker=CLASS_MARKERS[key], color="none",
               markerfacecolor=color, markeredgecolor="white", markersize=7, label=key)
        for key, color in CLASS_COLORS.items()
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle("Measured critical degree versus exact finite-T mean field", fontweight="bold")
    save_figure(fig, "data_vs_theory")


def make_figure2(predictions: pd.DataFrame, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    ax.scatter(predictions.d_c_fit, predictions.d_c_data, s=48, color="#0072B2", edgecolor="white", linewidth=0.6)
    lo = min(predictions.d_c_fit.min(), predictions.d_c_data.min()) * 0.94
    hi = max(predictions.d_c_fit.max(), predictions.d_c_data.max()) * 1.06
    ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1.2)
    for row in predictions.itertuples(index=False):
        ax.annotate(row.source_cell_id, (row.d_c_fit, row.d_c_data), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel(r"Empirical separable-fit $d_c$")
    ax.set_ylabel(r"Measured $d_c$")
    ax.set_title("Primary THEORY-CLEAN fit (b=2)")
    ax.text(0.03, 0.96, f"$R^2_{{log}}$={metrics['R2_log']:.3f}\nMAPE={100*metrics['mean_absolute_relative_prediction_error']:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9)
    style_axes(ax)
    save_figure(fig, "empirical_fit_vs_data")


def scan_subset(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    memberships = frame.primary_one_factor_scans.fillna("").str.split(";")
    return frame.loc[memberships.apply(lambda values: name in values)]


def make_figure3(frame: pd.DataFrame, fits: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.8))
    specs = [("M", "M", "M"), ("K", "K", "K"), ("m", "m", "m"), ("T_n0_32", "T", "T")]
    for ax, (fit_name, variable, title) in zip(axes[:4], specs):
        subset = scan_subset(frame, fit_name).sort_values(variable)
        fit = fits.loc[fits.fit_name == fit_name].iloc[0]
        x = subset[variable].to_numpy(float)
        for row in subset.itertuples(index=False):
            plot_point(ax, row, float(getattr(row, variable)), row.d_c_data)
        grid = np.geomspace(x.min(), x.max(), 120)
        ax.plot(grid, float(fit.A) * grid ** float(fit.exponent), color="#0072B2", linewidth=1.5, label="empirical")
        template = subset.iloc[0]
        theory_grid = np.unique(x.astype(int)) if variable == "T" else grid
        theory = []
        for value in theory_grid:
            pars = {key: int(template[key]) for key in ["M", "K", "T", "b", "m"]}
            pars[variable] = int(value) if variable == "T" else float(value)
            terms = [1.0 - pars["m"] * t / (pars["M"] * pars["K"]) for t in range(1, pars["T"])]
            theory.append(-(pars["T"] - 1) * math.log(pars["b"]) / sum(math.log(v) for v in terms))
        ax.plot(theory_grid, theory, "--", color="#333333", linewidth=1.2, label="exact theory")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
        ax.minorticks_off()
        ax.set_xlabel(title); ax.set_ylabel(r"$d_c$")
        ax.set_title(fr"$\alpha$={fit.exponent:.3f}")
        style_axes(ax)
    ax = axes[4]
    logb = frame.loc[frame.experiment_family.str.startswith("matched_logb_")].copy()
    theory_x = np.linspace(math.log(2), math.log(5), 160)
    coefficient = float(logb.iloc[0].d_c_analytic / math.log(logb.iloc[0].b))
    ax.plot(theory_x, coefficient * theory_x, "--", color="#333333", linewidth=1.2, label="exact theory")
    for family, subset in logb.groupby("experiment_family"):
        baseline = family.rsplit("_", 1)[-1]
        subset = subset.sort_values("b")
        ax.plot(np.log(subset.b), subset.d_c_data, marker=BASELINE_MARKERS[baseline], color=BASELINE_COLORS[baseline], linewidth=1.1, label=baseline)
    ax.set_xlabel(r"$\ln b$"); ax.set_ylabel(r"$d_c$")
    ax.set_title(r"$\alpha_b$ not cleanly identified")
    style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7)
    axes[4].legend(frameon=False, fontsize=7)
    fig.suptitle("Controlled one-factor scaling", fontweight="bold")
    save_figure(fig, "one_factor_scaling")


def make_figure4(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.7), sharey=True)
    specs = [
        ("M", "M", scan_subset(frame, "M")),
        ("K", "K", scan_subset(frame, "K")),
        ("m", "m", scan_subset(frame, "m")),
        ("T", "T", scan_subset(frame, "T_n0_32")),
        ("b", "b", frame.loc[frame.experiment_family.str.startswith("matched_logb_")]),
    ]
    for ax, (title, variable, subset) in zip(axes, specs):
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
        for row in subset.itertuples(index=False):
            ax.scatter(float(getattr(row, variable)), row.data_over_theory,
                       marker=CLASS_MARKERS[row.final_classification],
                       color=CLASS_COLORS[row.final_classification], s=35, alpha=0.88)
        ax.set_xlabel(title); ax.set_title(title)
        style_axes(ax, "y")
    axes[0].set_ylabel(r"$d_c^{data}/d_c^{analytic}$")
    fig.suptitle("Systematic deviations from exact finite-T theory", fontweight="bold")
    save_figure(fig, "theory_ratio")


def make_figure5(frame: pd.DataFrame, fits: pd.DataFrame) -> None:
    subset = frame.loc[frame.experiment_family.str.startswith("matched_logb_")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    coefficient = float(subset.iloc[0].d_c_analytic / math.log(subset.iloc[0].b))
    xx = np.linspace(math.log(2), math.log(5), 160)
    axes[0].plot(xx, coefficient * xx, "--", color="#222222", linewidth=1.4, label="exact theory")
    axes[1].axhline(coefficient, color="#222222", linestyle="--", linewidth=1.4, label="exact theory")
    for family, group in subset.groupby("experiment_family"):
        baseline = family.rsplit("_", 1)[-1]
        group = group.sort_values("b")
        x = np.log(group.b.to_numpy(float))
        axes[0].plot(x, group.d_c_data, marker=BASELINE_MARKERS[baseline], color=BASELINE_COLORS[baseline], linewidth=1.25, label=f"baseline {baseline}")
        axes[1].plot(group.b, group.d_c_data / x, marker=BASELINE_MARKERS[baseline], color=BASELINE_COLORS[baseline], linewidth=1.25, label=f"baseline {baseline}")
        fit = fits.loc[fits.fit_name == f"b_descriptive_{baseline}"].iloc[0]
        axes[0].plot(xx, float(fit.through_origin_slope) * xx, color=BASELINE_COLORS[baseline], alpha=0.35, linewidth=2.4)
    axes[0].set_xlabel(r"$\ln b$"); axes[0].set_ylabel(r"Measured $d_c$")
    axes[0].set_title("Matched-baseline curves")
    axes[1].set_xlabel("b"); axes[1].set_ylabel(r"$d_c/\ln b$")
    axes[1].set_xticks([2, 3, 4, 5]); axes[1].set_title("Coefficient stability")
    for ax in axes: style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Matched $MK/m=64$ test of the $\ln b$ dependence", fontweight="bold")
    save_figure(fig, "logb_validation")


def make_figure6(frame: pd.DataFrame) -> None:
    subset = frame.loc[
        frame.experiment_family.str.startswith("matched_logb_") |
        frame.experiment_family.str.contains("b_width") |
        (frame.experiment_family == "b_scan_v2")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    for ax, variable, label in [
        (axes[0], "duplicate_evidence_fraction", "Duplicate-evidence fraction"),
        (axes[1], "evidence_excess_Jaccard", "Evidence excess Jaccard"),
    ]:
        for classification, group in subset.groupby("final_classification"):
            ax.scatter(group[variable], 100.0 * group.relative_error,
                       marker=CLASS_MARKERS[classification], color=CLASS_COLORS[classification],
                       s=42, alpha=0.85, label=classification)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlabel(label); style_axes(ax)
    axes[0].axvline(0.20, color="#777777", linestyle=":", linewidth=1.1)
    axes[1].axvline(0.30, color="#777777", linestyle=":", linewidth=1.1)
    axes[0].set_ylabel("Relative theory error (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Branching-factor theory error and evidence redundancy", fontweight="bold")
    save_figure(fig, "redundancy_diagnostic")


def run_fit(
    master_path: Path = MASTER_PATH,
    bootstrap_input: Path = BOOTSTRAP_INPUT,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    global MASTER_PATH, BOOTSTRAP_INPUT, OUTPUT_ROOT, ANALYSIS, FIGURES, BOOT
    MASTER_PATH = Path(master_path)
    BOOTSTRAP_INPUT = Path(bootstrap_input)
    OUTPUT_ROOT = Path(output_root)
    ANALYSIS = OUTPUT_ROOT / "data"
    FIGURES = OUTPUT_ROOT / "figures"
    BOOT = ANALYSIS / "bootstrap"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MASTER_PATH)
    computed_theory = np.asarray(
        [
            exact_analytic_dc(int(row.M), int(row.K), int(row.T), int(row.b), int(row.m))
            for row in frame.itertuples(index=False)
        ]
    )
    if not np.allclose(frame.d_c_analytic.to_numpy(float), computed_theory, rtol=0.0, atol=2e-12):
        raise RuntimeError("master table does not match the exact finite-T analytic formula")
    frame["d_c_analytic"] = computed_theory
    frame["data_over_theory"] = frame.d_c_data / frame.d_c_analytic
    frame["relative_error"] = frame.data_over_theory - 1.0
    for col in ["supply_clean", "independence_clean", "theory_clean", "primary_global_fit"]:
        frame[col] = frame[col].astype(str).str.lower().eq("true")
    store = BootstrapStore()
    global_params, predictions, global_metrics = fit_global(frame, store)
    one_factor = fit_one_factor(frame, store)
    summaries, effective, effective_summary = theory_summaries(frame)

    global_params.to_csv(ANALYSIS / "global_fit_parameters.csv", index=False)
    predictions.to_csv(ANALYSIS / "global_fit_predictions.csv", index=False)
    one_factor.to_csv(ANALYSIS / "one_factor_fits.csv", index=False)
    summaries.to_csv(ANALYSIS / "theory_error_summary.csv", index=False)
    effective.to_csv(ANALYSIS / "measured_effective_comparison.csv", index=False)
    (ANALYSIS / "global_fit_metrics.json").write_text(json.dumps(global_metrics, indent=2) + "\n")
    (ANALYSIS / "measured_effective_summary.json").write_text(json.dumps(effective_summary, indent=2) + "\n")

    make_figure1(frame)
    make_figure2(predictions, global_metrics)
    make_figure3(frame, one_factor)
    make_figure4(frame)
    make_figure5(frame, one_factor)
    make_figure6(frame)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "simulator_called": False,
        "global_fit": global_metrics,
        "theory_error_groups": summaries.to_dict(orient="records"),
        "measured_effective": effective_summary,
        "figures": sorted(path.name for path in FIGURES.glob("*.png")),
    }
    (ANALYSIS / "fit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("final",), default="final")
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--bootstrap-dir", type=Path, default=BOOTSTRAP_INPUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run_fit(args.master, args.bootstrap_dir, args.output_root)


if __name__ == "__main__":
    main()
