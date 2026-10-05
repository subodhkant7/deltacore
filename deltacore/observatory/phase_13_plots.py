"""DeltaCore Phase 13 Observatory Publication Plots BK through BU.

Implements all 11 diagnostic figures specified in Phase 13 Section 24:
    - Plot BK: Real-world prediction error over time (plot_realworld_error_over_time_bk)
    - Plot BL: Shift-period error and recovery (plot_shift_recovery_bl)
    - Plot BM: State norm trajectory (plot_state_norm_trajectory_bm)
    - Plot BN: State update energy (plot_state_update_energy_bn)
    - Plot BO: Continuous vs reset (plot_continuous_vs_reset_bo)
    - Plot BP: SafeAdaptiveDelta vs FixedDelta (plot_safe_vs_fixed_delta_bp)
    - Plot BQ: Original vs permuted spatial ordering (plot_spatial_permutation_control_bq)
    - Plot BR: Performance versus spatial resolution (plot_perf_vs_resolution_br)
    - Plot BS: Runtime versus D (plot_runtime_vs_dim_bs)
    - Plot BT: State memory versus D (plot_memory_vs_dim_bt)
    - Plot BU: DeltaCore vs small spatial Conv (plot_deltacore_vs_conv_bu)
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Ensure Matplotlib uses local writable directory inside sandbox
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
import matplotlib.ticker as ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    "GRU": "#9467bd",
    "FixedDelta": "#8c564b",
    "SafeAdaptiveDelta": "#2ca02c",
    "SelectiveRetention": "#17becf",
    "SpatialConv": "#e377c2",
}


def plot_realworld_error_over_time_bk(
    model_errors: Mapping[str, Sequence[float]],
    shift_interval: tuple[int, int],
    output_path: Path,
) -> None:
    """Plot BK: Real-world prediction error over time with El Niño transition shading."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.5))

    ax.axvspan(
        shift_interval[0],
        shift_interval[1],
        alpha=0.15,
        color="#ff7f0e",
        label="2015 El Niño Shift Episode",
    )

    for name, errs in model_errors.items():
        color = PALETTE.get(name, "#333333")
        is_highlight = name in ("SafeAdaptiveDelta", "OnlineRidge", "Persistence")
        ax.plot(
            errs,
            label=name,
            color=color,
            lw=2.2 if is_highlight else 1.2,
            alpha=0.95 if is_highlight else 0.55,
        )

    ax.set_title(
        r"Plot BK: NOAA OISST Pacific SST Online Next-Step Relative Error $E_{\mathrm{rel}}(t)$"
    )
    ax.set_xlabel("Test Timestep (Days since Split Boundary)")
    ax.set_ylabel(r"Relative Error $E_{\mathrm{rel}}$")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_shift_recovery_bl(
    shift_data: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    """Plot BL: Shift period relative error vs. recovery time."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    names = list(shift_data.keys())
    shift_errs = [shift_data[m]["shift_error_mean"] for m in names]
    recoveries = [shift_data[m]["first_passage_recovery_mean"] for m in names]
    colors = [PALETTE.get(m, "#333333") for m in names]

    for i, name in enumerate(names):
        ax.scatter(
            recoveries[i],
            shift_errs[i],
            color=colors[i],
            s=90,
            zorder=3,
            label=name,
        )
        ax.annotate(
            name,
            (recoveries[i], shift_errs[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
        )

    ax.set_title("Plot BL: Distribution Shift Recovery vs. Shift Error")
    ax.set_xlabel("First-Passage Recovery Time (Steps)")
    ax.set_ylabel("Mean Shift Relative Error")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_state_norm_trajectory_bm(
    state_trajectories: Mapping[str, Sequence[float]],
    shift_interval: tuple[int, int],
    output_path: Path,
) -> None:
    """Plot BM: State norm ||M_t||_F trajectory through distribution shift."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    ax.axvspan(
        shift_interval[0],
        shift_interval[1],
        alpha=0.15,
        color="#ff7f0e",
        label="El Niño Shift Episode",
    )

    for name, norms in state_trajectories.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(norms, label=name, color=color, lw=1.8)

    ax.set_title(r"Plot BM: Adaptive State Norm Trajectory $\|M_t\|_F$")
    ax.set_xlabel("Test Timestep (Days)")
    ax.set_ylabel(r"Frobenius Norm $\|M_t\|_F$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_state_update_energy_bn(
    update_trajectories: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Plot BN: Cumulative state update energy over time."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    for name, upds in update_trajectories.items():
        color = PALETTE.get(name, "#333333")
        cum_energy = np.cumsum([u**2 for u in upds])
        ax.plot(cum_energy, label=name, color=color, lw=1.8)

    ax.set_title(
        r"Plot BN: Cumulative Adaptation Energy $\sum_{\tau=1}^t \|\Delta M_\tau\|_F^2$"
    )
    ax.set_xlabel("Test Timestep (Days)")
    ax.set_ylabel("Cumulative Energy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_continuous_vs_reset_bo(
    ablation_results: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    """Plot BO: Continuous state vs. State reset control comparison."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    variants = [
        "SafeAdaptiveDelta",
        "FixedDelta",
        "SelectiveRetention",
    ]
    cont_errs = [ablation_results[f"{v}_Continuous"]["e_shift"] for v in variants]
    reset_errs = [ablation_results[f"{v}_Reset"]["e_shift"] for v in variants]

    x = np.arange(len(variants))
    w = 0.35

    ax.bar(
        x - w / 2,
        cont_errs,
        width=w,
        label="Continuous Adaptive State",
        color="#2ca02c",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        reset_errs,
        width=w,
        label="State Reset at Shift Onset",
        color="#d62728",
        alpha=0.85,
    )

    ax.set_title("Plot BO: Continuous State vs. State Reset at El Niño Shift Boundary")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Shift Period Relative Error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_safe_vs_fixed_delta_bp(
    safe_errors: Sequence[float],
    fixed_errors: Sequence[float],
    shift_interval: tuple[int, int],
    output_path: Path,
) -> None:
    """Plot BP: SafeAdaptiveDelta vs. FixedDelta error trajectories."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    ax.axvspan(
        shift_interval[0],
        shift_interval[1],
        alpha=0.15,
        color="#ff7f0e",
        label="El Niño Shift Episode",
    )

    ax.plot(
        safe_errors,
        label="SafeAdaptiveDelta (Contractive Lyapunov Clamping)",
        color="#2ca02c",
        lw=2.0,
    )
    ax.plot(
        fixed_errors,
        label="FixedDelta (Fixed Step Size)",
        color="#8c564b",
        lw=1.6,
        linestyle="--",
    )

    ax.set_title("Plot BP: SafeAdaptiveDelta vs. FixedDelta Real SST Tracking")
    ax.set_xlabel("Test Timestep (Days)")
    ax.set_ylabel(r"Relative Error $E_{\mathrm{rel}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spatial_permutation_control_bq(
    perm_results: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Plot BQ: Original vs. Permuted spatial ordering (Permutation Invariance Test)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    models = list(perm_results.keys())
    orig_errs = [perm_results[m]["original_rel_error"] for m in models]
    perm_errs = [perm_results[m]["permuted_rel_error"] for m in models]

    x = np.arange(len(models))
    w = 0.35

    ax.bar(
        x - w / 2,
        orig_errs,
        width=w,
        label="Original 2D Spatial Grid",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        perm_errs,
        width=w,
        label="Fixed Spatial Permutation (Locality Destroyed)",
        color="#ff7f0e",
        alpha=0.85,
    )

    ax.set_title(
        "Plot BQ: Spatial Permutation Invariance Control (Causal Check for 2D Locality)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean Relative Error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perf_vs_resolution_br(
    scaling_data: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BR: Performance vs. Spatial Resolution (D=64 vs D=256)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    models = ["Persistence", "OnlineRidge", "SafeAdaptiveDelta", "SpatialConv"]
    resolutions = ["small_D64", "medium_D256"]

    for m in models:
        errs = [scaling_data[res]["models"][m]["rel_error"] for res in resolutions]
        color = PALETTE.get(m, "#333333")
        ax.plot([64, 256], errs, marker="o", label=m, color=color, lw=1.8)

    ax.set_title(
        r"Plot BR: Spatio-Temporal Prediction Error vs. Resolution ($D = HWC$)"
    )
    ax.set_xlabel("Spatial Dimension D")
    ax.set_ylabel("Mean Relative Error")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_runtime_vs_dim_bs(
    scaling_data: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BS: Latency per token (microseconds) vs. Dimension D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    models = ["Persistence", "OnlineRidge", "SafeAdaptiveDelta", "SpatialConv"]
    resolutions = ["small_D64", "medium_D256"]

    for m in models:
        runtimes = [scaling_data[res]["models"][m]["runtime_us"] for res in resolutions]
        color = PALETTE.get(m, "#333333")
        ax.plot(
            [64, 256],
            runtimes,
            marker="s",
            label=m,
            color=color,
            lw=1.8,
        )

    ax.set_title(r"Plot BS: Online Step Latency vs. Dimension $D$")
    ax.set_xlabel("Spatial Dimension D")
    ax.set_ylabel(r"Latency per Token ($\mu\mathrm{s}$)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_vs_dim_bt(
    scaling_data: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Plot BT: Persistent state memory (bytes) vs. Dimension D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    models = ["Persistence", "OnlineRidge", "SafeAdaptiveDelta"]
    resolutions = ["small_D64", "medium_D256"]

    for m in models:
        mems = [
            scaling_data[res]["models"][m]["state_memory_bytes"] for res in resolutions
        ]
        color = PALETTE.get(m, "#333333")
        ax.plot([64, 256], mems, marker="^", label=m, color=color, lw=1.8)

    ax.set_title(r"Plot BT: Persistent State Memory Footprint vs. Dimension $D$")
    ax.set_xlabel("Spatial Dimension D")
    ax.set_ylabel("State Memory (Bytes)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([64, 256])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_deltacore_vs_conv_bu(
    fac_scores: Mapping[str, float],
    sge_scores: Mapping[str, float],
    output_path: Path,
) -> None:
    """Plot BU: Field Anomaly Correlation vs. Spatial Gradient Error."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    models = list(fac_scores.keys())
    for m in models:
        fac = fac_scores[m]
        sge = sge_scores[m]
        color = PALETTE.get(m, "#333333")
        ax.scatter(sge, fac, color=color, s=90, label=m, zorder=3)
        ax.annotate(
            m,
            (sge, fac),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
        )

    ax.set_title("Plot BU: Field Anomaly Correlation vs. Spatial Gradient Error")
    ax.set_xlabel("Spatial Gradient Error SGE (Lower is Better)")
    ax.set_ylabel("Field Anomaly Correlation FAC (Higher is Better)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_phase_13_plots(
    results_dir: Path | str, output_dir: Path | str
) -> dict[str, Path]:
    """Generate all Phase 13 Observatory plots BK through BU from saved artifacts."""
    r_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(r_path / "phase_13_results.json") as f:
        results = json.load(f)
    with open(r_path / "phase_13_scaling.json") as f:
        scaling = json.load(f)
    with open(r_path / "phase_13_shift_analysis.json") as f:
        shift_analysis = json.load(f)

    shift_interval = tuple(shift_analysis["shift_interval"])
    rep_trajs = shift_analysis["representative_trajectories"]

    plot_paths: dict[str, Path] = {}

    # Plot BK
    err_dict = {m: rep_trajs[m]["step_errors"] for m in rep_trajs}
    p_bk = out_path / "plot_bk_realworld_error_over_time.png"
    plot_realworld_error_over_time_bk(err_dict, shift_interval, p_bk)
    plot_paths["plot_bk"] = p_bk

    # Plot BL
    p_bl = out_path / "plot_bl_shift_recovery.png"
    plot_shift_recovery_bl(results, p_bl)
    plot_paths["plot_bl"] = p_bl

    # Plot BM
    norm_dict = {
        m: rep_trajs[m]["state_norms"]
        for m in rep_trajs
        if m in ("SafeAdaptiveDelta", "FixedDelta", "OnlineRidge")
    }
    p_bm = out_path / "plot_bm_state_norm_trajectory.png"
    plot_state_norm_trajectory_bm(norm_dict, shift_interval, p_bm)
    plot_paths["plot_bm"] = p_bm

    # Plot BN
    upd_dict = {
        m: rep_trajs[m]["update_norms"]
        for m in rep_trajs
        if m in ("SafeAdaptiveDelta", "FixedDelta", "OnlineRidge")
    }
    p_bn = out_path / "plot_bn_state_update_energy.png"
    plot_state_update_energy_bn(upd_dict, p_bn)
    plot_paths["plot_bn"] = p_bn

    # Plot BO
    p_bo = out_path / "plot_bo_continuous_vs_reset.png"
    plot_continuous_vs_reset_bo(shift_analysis["retention_ablation"], p_bo)
    plot_paths["plot_bo"] = p_bo

    # Plot BP
    p_bp = out_path / "plot_bp_safe_vs_fixed_delta.png"
    plot_safe_vs_fixed_delta_bp(
        rep_trajs["SafeAdaptiveDelta"]["step_errors"],
        rep_trajs["FixedDelta"]["step_errors"],
        shift_interval,
        p_bp,
    )
    plot_paths["plot_bp"] = p_bp

    # Plot BQ
    p_bq = out_path / "plot_bq_spatial_permutation_control.png"
    plot_spatial_permutation_control_bq(
        shift_analysis["spatial_permutation_control"], p_bq
    )
    plot_paths["plot_bq"] = p_bq

    # Plot BR
    p_br = out_path / "plot_br_perf_vs_resolution.png"
    plot_perf_vs_resolution_br(scaling, p_br)
    plot_paths["plot_br"] = p_br

    # Plot BS
    p_bs = out_path / "plot_bs_runtime_vs_dim.png"
    plot_runtime_vs_dim_bs(scaling, p_bs)
    plot_paths["plot_bs"] = p_bs

    # Plot BT
    p_bt = out_path / "plot_bt_memory_vs_dim.png"
    plot_memory_vs_dim_bt(scaling, p_bt)
    plot_paths["plot_bt"] = p_bt

    # Plot BU
    p_bu = out_path / "plot_bu_deltacore_vs_conv.png"
    facs = {m: results[m]["fac_mean"] for m in results}
    sges = {m: results[m]["sge_mean"] for m in results}
    plot_deltacore_vs_conv_bu(facs, sges, p_bu)
    plot_paths["plot_bu"] = p_bu

    return plot_paths
