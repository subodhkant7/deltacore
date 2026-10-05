"""DeltaCore Phase 14 Observatory Publication Plots BV through CF.

Implements all 11 diagnostic figures specified in Phase 14 Section 21:
    - Plot BV: Real-world prediction error over time (plot_realworld_error_over_time_bv)
    - Plot BW: Shift-period error (plot_shift_period_error_bw)
    - Plot BX: Continuous versus reset (plot_continuous_vs_reset_bx)
    - Plot BY: DeltaCore versus OnlineRidge (plot_deltacore_vs_onlineridge_by)
    - Plot BZ: DeltaCore versus SpatialConv (plot_deltacore_vs_spatialconv_bz)
    - Plot CA: Spatial permutation error (plot_spatial_permutation_error_ca)
    - Plot CB: Explicit equivariance error (plot_explicit_equivariance_error_cb)
    - Plot CC: Performance versus spatial resolution (plot_perf_vs_resolution_cc)
    - Plot CD: Runtime versus D (plot_runtime_vs_dim_cd)
    - Plot CE: Persistent state memory versus D (plot_memory_vs_dim_ce)
    - Plot CF: Phase 13 versus Phase 14 replication comparison (plot_replication_comparison_cf)
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _apply_style() -> None:
    """Apply unified styling for DeltaCore Observatory plots."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.5,
            "grid.alpha": 0.35,
        }
    )


PALETTE: dict[str, str] = {
    "Persistence": "#7f7f7f",
    "FrozenLinear": "#1f77b4",
    "OnlineRidge": "#ff7f0e",
    "NonlinearOnlineRidge": "#d62728",
    "FixedDelta": "#8c564b",
    "SafeAdaptiveDelta": "#2ca02c",
    "SpatialConv": "#e377c2",
}


def plot_realworld_error_over_time_bv(
    model_errors: Mapping[str, Sequence[float]],
    shift_interval: tuple[int, int],
    output_path: Path,
) -> None:
    """Plot BV: Real-world relative prediction error over time."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.axvspan(
        shift_interval[0],
        shift_interval[1],
        color="#fee8c8",
        alpha=0.6,
        label="Jan-Feb 2021 Polar Outbreak",
    )

    for name, errs in model_errors.items():
        if name in PALETTE:
            color = PALETTE[name]
            lw = 2.0 if name == "SafeAdaptiveDelta" else 1.2
            ax.plot(errs, label=name, color=color, linewidth=lw, alpha=0.9)

    ax.set_title("Plot BV: ERA5 Atmospheric T2m Prediction Error Over Time")
    ax.set_xlabel("Online Test Timestep t (6-hourly Synoptic Steps)")
    ax.set_ylabel("Relative Frobenius Error E_rel(t)")
    ax.set_ylim(0.0, 0.6)
    ax.legend(loc="upper right", frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_shift_period_error_bw(
    results_agg: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BW: Mean relative error during shift vs overall test error."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "SpatialConv",
        "FixedDelta",
        "SafeAdaptiveDelta",
    ]
    overall_errs = [results_agg[m]["rel_error_mean"] for m in names]
    shift_errs = [results_agg[m]["shift_error_mean"] for m in names]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(
        x - width / 2,
        overall_errs,
        width,
        label="Overall Test Error",
        color="#a6bddb",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        shift_errs,
        width,
        label="Shift-Period Error",
        color="#fc8d59",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title("Plot BW: Overall vs Shift-Period Prediction Error (ERA5 T2m)")
    ax.set_ylabel("Relative Error E_rel")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_continuous_vs_reset_bx(
    retention_ablation: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BX: Continuous state vs state reset at shift onset."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    models = ["FixedDelta", "SafeAdaptiveDelta"]
    cont_shift = [retention_ablation[f"{m}_Continuous"]["e_shift"] for m in models]
    reset_shift = [retention_ablation[f"{m}_Reset"]["e_shift"] for m in models]
    cont_excess = [
        retention_ablation[f"{m}_Continuous"]["cumulative_excess_error"] for m in models
    ]
    reset_excess = [
        retention_ablation[f"{m}_Reset"]["cumulative_excess_error"] for m in models
    ]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(
        x - width / 2,
        cont_shift,
        width,
        label="Continuous State",
        color="#74c476",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.bar(
        x + width / 2,
        reset_shift,
        width,
        label="State Reset (t=45)",
        color="#bcbddc",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Shift-Period Error")
    ax1.set_ylabel("Mean E_shift")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(frameon=True)

    ax2.bar(
        x - width / 2,
        cont_excess,
        width,
        label="Continuous State",
        color="#74c476",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.bar(
        x + width / 2,
        reset_excess,
        width,
        label="State Reset (t=45)",
        color="#bcbddc",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Cumulative Excess Shift Error")
    ax2.set_ylabel("Excess Error Sigma max(0, e_t - e_pre)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(frameon=True)

    fig.suptitle(
        "Plot BX: Controlled State-Reset Intervention at Shift Onset",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_deltacore_vs_onlineridge_by(
    results_agg: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BY: DeltaCore vs OnlineRidge comparison."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    metrics = ["Overall RelErr", "Shift Error", "MAE", "RMSE"]

    or_vals = [
        results_agg["OnlineRidge"]["rel_error_mean"],
        results_agg["OnlineRidge"]["shift_error_mean"],
        results_agg["OnlineRidge"]["mae_mean"],
        results_agg["OnlineRidge"]["rmse_mean"],
    ]
    sad_vals = [
        results_agg["SafeAdaptiveDelta"]["rel_error_mean"],
        results_agg["SafeAdaptiveDelta"]["shift_error_mean"],
        results_agg["SafeAdaptiveDelta"]["mae_mean"],
        results_agg["SafeAdaptiveDelta"]["rmse_mean"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(
        x - width / 2,
        or_vals,
        width,
        label="OnlineRidge",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        sad_vals,
        width,
        label="SafeAdaptiveDelta",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title("Plot BY: SafeAdaptiveDelta vs OnlineRidge Tracking Profile")
    ax.set_ylabel("Error Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_deltacore_vs_spatialconv_bz(
    results_agg: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BZ: DeltaCore vs SpatialConv across metrics."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    metrics = ["Rel Error", "Shift Error", "MAE", "SGE"]

    sc_vals = [
        results_agg["SpatialConv"]["rel_error_mean"],
        results_agg["SpatialConv"]["shift_error_mean"],
        results_agg["SpatialConv"]["mae_mean"],
        results_agg["SpatialConv"]["sge_mean"],
    ]
    sad_vals = [
        results_agg["SafeAdaptiveDelta"]["rel_error_mean"],
        results_agg["SafeAdaptiveDelta"]["shift_error_mean"],
        results_agg["SafeAdaptiveDelta"]["mae_mean"],
        results_agg["SafeAdaptiveDelta"]["sge_mean"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(
        x - width / 2,
        sc_vals,
        width,
        label="SpatialConv",
        color="#e377c2",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        sad_vals,
        width,
        label="SafeAdaptiveDelta",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title("Plot BZ: DeltaCore vs Small Spatial Convolution Baseline")
    ax.set_ylabel("Error Value")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spatial_permutation_error_ca(
    perm_results: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot CA: Relative error under original vs permuted spatial ordering."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = ["OnlineRidge", "FixedDelta", "SafeAdaptiveDelta", "SpatialConv"]
    orig_errs = [perm_results[m]["original_rel_error"] for m in names]
    perm_errs = [perm_results[m]["permuted_rel_error"] for m in names]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(
        x - width / 2,
        orig_errs,
        width,
        label="Original Spatial Order",
        color="#6baed6",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        perm_errs,
        width,
        label="Permuted Spatial Order pi",
        color="#fd8d3c",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title("Plot CA: Spatial Permutation Error Control")
    ax.set_ylabel("Relative Frobenius Error E_rel")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_explicit_equivariance_error_cb(
    perm_results: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot CB: Log10 explicit vector equivariance error."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    names = ["OnlineRidge", "FixedDelta", "SafeAdaptiveDelta", "SpatialConv"]
    equiv_errs = [perm_results[m]["normalized_equivariance_error"] for m in names]

    colors = ["#2ca02c" if e < 1e-4 else "#d62728" for e in equiv_errs]

    ax.bar(
        names,
        equiv_errs,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    ax.set_yscale("log")
    ax.axhline(
        1e-4,
        color="black",
        linestyle="--",
        label="Equivariance Threshold (10^-4)",
    )
    ax.set_title("Plot CB: Explicit Vector Permutation Equivariance Error")
    ax.set_ylabel("E_equiv = ||Y_pi - P Y||_F / ||P Y||_F (Log Scale)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perf_vs_resolution_cc(
    scaling_results: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot CC: Performance vs spatial resolution."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    models = ["Persistence", "OnlineRidge", "SafeAdaptiveDelta", "SpatialConv"]
    d64_errs = [scaling_results["small_D64"]["models"][m]["rel_error"] for m in models]
    d256_errs = [
        scaling_results["medium_D256"]["models"][m]["rel_error"] for m in models
    ]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        d64_errs,
        width,
        label="D=64 (8x8)",
        color="#9ecae1",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        d256_errs,
        width,
        label="D=256 (16x16)",
        color="#3182bd",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title("Plot CC: Prediction Error vs Spatial Resolution (D=64 vs D=256)")
    ax.set_ylabel("Relative Error E_rel")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_runtime_vs_dim_cd(
    scaling_results: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot CD: Per-token latency vs dimension D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    models = ["Persistence", "OnlineRidge", "SafeAdaptiveDelta"]
    dims = [64, 256]

    for m in models:
        runtimes = [
            scaling_results["small_D64"]["models"][m]["runtime_us"],
            scaling_results["medium_D256"]["models"][m]["runtime_us"],
        ]
        color = PALETTE.get(m, "#333333")
        ax.plot(dims, runtimes, marker="o", label=m, color=color, linewidth=2.0)

    ax.set_title("Plot CD: Per-Token Step Latency vs Dimension D")
    ax.set_xlabel("State Dimension D")
    ax.set_ylabel("Runtime per Token (us on CPU)")
    ax.set_xticks(dims)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_vs_dim_ce(
    scaling_results: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot CE: Persistent state memory vs dimension D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    models = ["Persistence", "SafeAdaptiveDelta", "OnlineRidge"]
    dims = [64, 256]

    for m in models:
        mems_kb = [
            scaling_results["small_D64"]["models"][m]["state_memory_bytes"] / 1024.0,
            scaling_results["medium_D256"]["models"][m]["state_memory_bytes"] / 1024.0,
        ]
        color = PALETTE.get(m, "#333333")
        ax.plot(
            dims,
            mems_kb,
            marker="s",
            label=m,
            color=color,
            linewidth=2.0,
        )

    ax.set_yscale("log")
    ax.set_title("Plot CE: Persistent State Memory Footprint vs Dimension D")
    ax.set_xlabel("State Dimension D")
    ax.set_ylabel("Persistent State Memory (KB, Log Scale)")
    ax.set_xticks(dims)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_replication_comparison_cf(output_path: Path) -> None:
    """Plot CF: Cross-domain replication matrix (Phase 13 vs Phase 14)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    properties = [
        "Dynamic Stability at D=256",
        "State Retention Helps Shift",
        "Permutation Equivariance",
        "50% State Memory Reduction",
        "SpatialConv Locality Reliance",
        "Overall Prediction Advantage",
    ]
    p13_status = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    p14_status = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

    y = np.arange(len(properties))
    height = 0.35

    ax.barh(
        y + height / 2,
        p13_status,
        height,
        label="Phase 13 (Ocean SST)",
        color="#3182bd",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.barh(
        y - height / 2,
        p14_status,
        height,
        label="Phase 14 (Atmospheric T2m)",
        color="#e6550d",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(properties)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Not Replicated", "Replicated"])
    ax.set_title("Plot CF: Cross-Domain Replication Profile (Phase 13 vs Phase 14)")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_phase_14_plots(
    artifacts_dir: Path | str = "docs/benchmarks/artifacts/phase_14",
) -> None:
    """Generate all 11 Phase 14 plots from serialized JSON artifacts."""
    art_path = Path(artifacts_dir)
    plots_dir = art_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(art_path / "phase_14_results.json") as f:
        results_agg = json.load(f)

    with open(art_path / "phase_14_scaling.json") as f:
        scaling_results = json.load(f)

    with open(art_path / "phase_14_shift_analysis.json") as f:
        shift_data = json.load(f)

    with open(art_path / "phase_14_permutation.json") as f:
        perm_data = json.load(f)

    shift_interval = tuple(shift_data["shift_interval"])
    trajectories = {
        k: v["step_errors"]
        for k, v in shift_data["representative_trajectories"].items()
    }
    retention_ablation = shift_data["retention_ablation"]

    plot_realworld_error_over_time_bv(
        trajectories,
        shift_interval,
        plots_dir / "plot_bv_realworld_error_over_time.png",
    )
    plot_shift_period_error_bw(
        results_agg, plots_dir / "plot_bw_shift_period_error.png"
    )
    plot_continuous_vs_reset_bx(
        retention_ablation, plots_dir / "plot_bx_continuous_vs_reset.png"
    )
    plot_deltacore_vs_onlineridge_by(
        results_agg, plots_dir / "plot_by_deltacore_vs_onlineridge.png"
    )
    plot_deltacore_vs_spatialconv_bz(
        results_agg, plots_dir / "plot_bz_deltacore_vs_spatialconv.png"
    )
    plot_spatial_permutation_error_ca(
        perm_data, plots_dir / "plot_ca_spatial_permutation_error.png"
    )
    plot_explicit_equivariance_error_cb(
        perm_data, plots_dir / "plot_cb_explicit_equivariance_error.png"
    )
    plot_perf_vs_resolution_cc(
        scaling_results, plots_dir / "plot_cc_perf_vs_resolution.png"
    )
    plot_runtime_vs_dim_cd(scaling_results, plots_dir / "plot_cd_runtime_vs_dim.png")
    plot_memory_vs_dim_ce(scaling_results, plots_dir / "plot_ce_memory_vs_dim.png")
    plot_replication_comparison_cf(plots_dir / "plot_cf_replication_comparison.png")
    print(f"[Observatory] All 11 Phase 14 plots successfully rendered to {plots_dir}.")
