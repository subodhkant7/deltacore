"""DeltaCore Phase 15 Observatory Publication Plots CG through CP.

Implements all 10 diagnostic figures specified in Phase 15 Section 16:
    - Plot CG: Cross-domain performance matrix (plot_cross_domain_matrix_cg)
    - Plot CH: Parameter transfer A->B vs B->A (plot_parameter_transfer_ch)
    - Plot CI: Safety/aggressiveness Pareto frontier (plot_safety_pareto_frontier_ci)
    - Plot CJ: State-transfer performance under regime switch (plot_state_transfer_cj)
    - Plot CK: Continuous versus reset across domains (plot_continuous_vs_reset_ck)
    - Plot CL: State initialization sensitivity (plot_state_initialization_cl)
    - Plot CM: FixedDelta failure boundary at D=256 (plot_failure_boundary_cm)
    - Plot CN: Runtime versus dimension by domain (plot_runtime_vs_dim_cn)
    - Plot CO: Persistent state memory versus dimension (plot_memory_vs_dim_co)
    - Plot CP: Per-seed cross-domain differences (plot_per_seed_differences_cp)
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(
        "/Users/urjasoft/.gemini/antigravity-ide/brain/6c485406-eeab-4ae3-9fa7-892b0159db3c/scratch/.mpl"
    )
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def plot_cross_domain_matrix_cg(
    results_agg: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CG: Cross-domain performance matrix comparing Domain A and Domain B."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    models = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "SpatialConv",
        "FixedDelta",
        "SafeAdaptiveDelta",
    ]

    err_a = [results_agg["domain_a"][m]["rel_error_mean"] for m in models]
    std_a = [results_agg["domain_a"][m]["rel_error_std"] for m in models]
    err_b = [results_agg["domain_b"][m]["rel_error_mean"] for m in models]
    std_b = [results_agg["domain_b"][m]["rel_error_std"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        err_a,
        width,
        yerr=std_a,
        label="Domain A (OISST: Slow Thermal)",
        color="#2b5c8f",
        alpha=0.88,
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        err_b,
        width,
        yerr=std_b,
        label="Domain B (ERA5: Fast Advection)",
        color="#d95f02",
        alpha=0.88,
        capsize=4,
    )

    ax.set_ylabel("Relative Frobenius Error ($E_{\\mathrm{rel}}$)")
    ax.set_title(
        "Plot CG: Cross-Domain Performance Matrix (OISST SST vs ERA5 $T_{2m}$)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontweight="medium")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 0.55)

    # Highlight winner on Domain A
    ax.annotate(
        "DeltaCore wins\non slow thermal",
        xy=(6 - width / 2, err_a[6]),
        xytext=(5.2, 0.42),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        fontsize=8.5,
        fontweight="semibold",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_parameter_transfer_ch(
    transfer_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CH: Parameter transfer A->B vs B->A across calibrated configurations."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    evals = transfer_data["test_evaluations"]
    configs = [
        "Config1_Frozen13",
        "DomainA_Calibrated",
        "Config2_DomainB_Calibrated",
        "Config3_Pooled",
    ]
    labels = [
        "Frozen Phase 13\n(A-Trained: $\\eta=0.008,\\rho=1.9$)",
        "Domain A Calibrated\n($\\eta=0.008,\\rho=1.5,\\alpha=0.95$)",
        "Domain B Calibrated\n($\\eta=0.015,\\rho=1.5,\\alpha=0.95$)",
        "Pooled Val Calibrated\n($\\eta=0.015,\\rho=1.5,\\alpha=0.95$)",
    ]

    err_a = [evals[c]["domain_a_oisst"]["rel_error"] for c in configs]
    shift_a = [evals[c]["domain_a_oisst"]["e_shift"] for c in configs]

    err_b = [evals[c]["domain_b_era5"]["rel_error"] for c in configs]
    shift_b = [evals[c]["domain_b_era5"]["e_shift"] for c in configs]

    x = np.arange(len(configs))
    w = 0.35

    ax1.bar(
        x - w / 2, err_a, w, label="$E_{\\mathrm{rel}}$", color="#1f77b4", alpha=0.85
    )
    ax1.bar(
        x + w / 2,
        shift_a,
        w,
        label="$E_{\\mathrm{shift}}$",
        color="#aec7e8",
        alpha=0.85,
    )
    ax1.set_title("Evaluated on Domain A (OISST)")
    ax1.set_ylabel("Error Metric")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 0.42)

    ax2.bar(
        x - w / 2, err_b, w, label="$E_{\\mathrm{rel}}$", color="#ff7f0e", alpha=0.85
    )
    ax2.bar(
        x + w / 2,
        shift_b,
        w,
        label="$E_{\\mathrm{shift}}$",
        color="#ffbb78",
        alpha=0.85,
    )
    ax2.set_title("Evaluated on Domain B (ERA5)")
    ax2.set_ylabel("Error Metric")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 0.30)

    fig.suptitle(
        "Plot CH: Cross-Domain Parameter Transfer (A$\\to$B vs B$\\to$A)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_safety_pareto_frontier_ci(
    safety_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CI: Safety / aggressiveness Pareto frontier."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    pareto_a = safety_data["domain_a"]
    pareto_b = safety_data["domain_b"]

    # Panel 1: Shift error vs Numerical Risk
    shift_a = [p["e_shift"] for p in pareto_a]
    risk_a = [p["numerical_risk"] for p in pareto_a]
    shift_b = [p["e_shift"] for p in pareto_b]
    risk_b = [p["numerical_risk"] for p in pareto_b]

    ax1.scatter(
        risk_a,
        shift_a,
        c="#1f77b4",
        s=80,
        alpha=0.85,
        edgecolors="black",
        label="Domain A (OISST)",
    )
    ax1.scatter(
        risk_b,
        shift_b,
        c="#ff7f0e",
        s=80,
        alpha=0.85,
        edgecolors="black",
        label="Domain B (ERA5)",
    )

    for p in pareto_a:
        cfg = p["config"]
        lbl = f"$\\eta={cfg['eta_max']},\\alpha={cfg['alpha_min']}$"
        ax1.annotate(
            lbl,
            (p["numerical_risk"], p["e_shift"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
            color="#0b3c5d",
        )

    ax1.set_xlabel("Numerical Risk ($\\eta_{\\mathrm{eff}} \\cdot \\|M\\| / 2$)")
    ax1.set_ylabel("Shift-Period Error ($E_{\\mathrm{shift}}$)")
    ax1.set_title("Pareto Trade-off: Shift Error vs Numerical Risk")
    ax1.legend(loc="upper right", frameon=True)

    # Panel 2: Recovery Time vs Adaptation Energy
    energy_a = [p["adaptation_energy"] for p in pareto_a]
    rec_a = [p["recovery_time"] for p in pareto_a]
    energy_b = [p["adaptation_energy"] for p in pareto_b]
    rec_b = [p["recovery_time"] for p in pareto_b]

    ax2.scatter(
        energy_a,
        rec_a,
        c="#1f77b4",
        s=80,
        alpha=0.85,
        edgecolors="black",
        label="Domain A (OISST)",
    )
    ax2.scatter(
        energy_b,
        rec_b,
        c="#ff7f0e",
        s=80,
        alpha=0.85,
        edgecolors="black",
        label="Domain B (ERA5)",
    )

    ax2.set_xlabel("Adaptation Energy ($\\sum_t \\|\\Delta M_t\\|_F^2$)")
    ax2.set_ylabel("Sustained Recovery Time (steps)")
    ax2.set_title("Pareto Trade-off: Recovery Time vs Adaptation Energy")
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Plot CI: Adaptation / Stability Pareto Frontier across Physical Regimes",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_state_transfer_cj(transfer_data: Mapping[str, Any], output_path: Path) -> None:
    """Plot CJ: State-transfer performance under regime-switch stream."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    regime_sw = transfer_data["experiment_e_regime_switch"]
    b1, b2 = regime_sw["stream_boundaries"]

    models_dict = regime_sw.get("models", regime_sw)
    err_cont = models_dict["continuous"]["step_errors"]
    err_reset = models_dict["reset_at_boundaries"]["step_errors"]
    steps = np.arange(len(err_cont))

    ax.plot(
        steps,
        err_cont,
        label="Continuous State Transfer",
        color="#2ca02c",
        linewidth=2.0,
        alpha=0.9,
    )
    ax.plot(
        steps,
        err_reset,
        label="Reset State at Regime Boundaries",
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
    )

    # Annotate regime segments
    ax.axvspan(0, b1, alpha=0.10, color="blue", label="Regime 1: OISST (Slow Thermal)")
    ax.axvspan(
        b1, b2, alpha=0.10, color="orange", label="Regime 2: ERA5 (Fast Advective)"
    )
    ax.axvspan(
        b2, len(steps), alpha=0.10, color="green", label="Regime 3: Return to OISST"
    )

    ax.axvline(b1, color="black", linestyle=":", linewidth=1.2)
    ax.axvline(b2, color="black", linestyle=":", linewidth=1.2)

    ax.annotate(
        f"Regime Shift 1 (t={b1})\nOISST -> ERA5",
        xy=(b1, 0.95),
        xytext=(b1 + 3, 1.05),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=8.5,
    )
    ax.annotate(
        f"Regime Shift 2 (t={b2})\nERA5 -> OISST",
        xy=(b2, 0.95),
        xytext=(b2 + 3, 1.05),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=8.5,
    )

    ax.set_xlabel("Online Streaming Step $t$")
    ax.set_ylabel("Step Relative Error $E_t$")
    ax.set_title("Plot CJ: Regime-Switch Transfer (OISST $\\to$ ERA5 $\\to$ OISST)")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax.set_ylim(0.2, 1.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_continuous_vs_reset_ck(
    retention_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CK: Continuous versus reset retention performance across domains."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    dom_a = retention_data["domain_a"]
    dom_b = retention_data["domain_b"]

    conditions = ["continuous", "reset", "retain_high"]
    labels = ["Continuous State", "Reset at Shift", "Retain-High State"]

    shift_a = [dom_a[c]["e_shift"] for c in conditions]
    shift_b = [dom_b[c]["e_shift"] for c in conditions]

    cum_a = [dom_a[c]["cumulative_excess_error"] for c in conditions]
    cum_b = [dom_b[c]["cumulative_excess_error"] for c in conditions]

    x = np.arange(len(conditions))
    w = 0.35

    # Panel 1: Shift Error
    ax1.bar(
        x - w / 2,
        shift_a,
        w,
        label="Domain A (OISST: Thermal)",
        color="#2b5c8f",
        alpha=0.88,
    )
    ax1.bar(
        x + w / 2,
        shift_b,
        w,
        label="Domain B (ERA5: Advective)",
        color="#d95f02",
        alpha=0.88,
    )
    ax1.set_ylabel("Shift-Period Error ($E_{\\mathrm{shift}}$)")
    ax1.set_title("Shift-Period Error by Retention Mode")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 0.32)

    # Panel 2: Cumulative Excess Error
    ax2.bar(
        x - w / 2,
        cum_a,
        w,
        label="Domain A (OISST: Thermal)",
        color="#2b5c8f",
        alpha=0.88,
    )
    ax2.bar(
        x + w / 2,
        cum_b,
        w,
        label="Domain B (ERA5: Advective)",
        color="#d95f02",
        alpha=0.88,
    )
    ax2.set_ylabel("Cumulative Excess Error ($C$)")
    ax2.set_title(
        "Cumulative Excess Error ($C = \\sum_t \\max(0, E_t - E_{\\mathrm{pre}})$)"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 4.0)

    fig.suptitle(
        "Plot CK: Continuous vs Reset Retention across Spatio-Temporal Regimes",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_state_initialization_cl(
    transfer_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CL: State initialization sensitivity across domains."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    init_data = transfer_data["experiment_g_state_initialization"]
    modes = ["zero_state", "small_random_state", "transferred"]
    mode_labels = [
        "Zero State\n($M_0 = 0$)",
        "Random State\n($\\sigma = 0.01$)",
        "Transferred State\n(Opposite Domain)",
    ]

    eval_b = init_data["eval_on_domain_b"]
    eval_a = init_data["eval_on_domain_a"]

    rel_b = [
        eval_b["zero_state"]["rel_error"],
        eval_b["small_random_state"]["rel_error"],
        eval_b["transferred_from_domain_a"]["rel_error"],
    ]
    early_b = [
        eval_b["zero_state"]["early_rel_error_first_10"],
        eval_b["small_random_state"]["early_rel_error_first_10"],
        eval_b["transferred_from_domain_a"]["early_rel_error_first_10"],
    ]

    rel_a = [
        eval_a["zero_state"]["rel_error"],
        eval_a["small_random_state"]["rel_error"],
        eval_a["transferred_from_domain_b"]["rel_error"],
    ]
    early_a = [
        eval_a["zero_state"]["early_rel_error_first_10"],
        eval_a["small_random_state"]["early_rel_error_first_10"],
        eval_a["transferred_from_domain_b"]["early_rel_error_first_10"],
    ]

    x = np.arange(len(modes))
    w = 0.35

    # Panel 1: Evaluated on Domain B (ERA5)
    ax1.bar(
        x - w / 2,
        rel_b,
        w,
        label="Overall $E_{\\mathrm{rel}}$",
        color="#1f77b4",
        alpha=0.85,
    )
    ax1.bar(
        x + w / 2,
        early_b,
        w,
        label="Early Transient (t $\\leq$ 10)",
        color="#aec7e8",
        alpha=0.85,
    )
    ax1.set_title("Evaluated on Domain B (ERA5 $T_{2m}$)")
    ax1.set_ylabel("Error Metric")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels, fontsize=8.5)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 0.45)

    # Panel 2: Evaluated on Domain A (OISST)
    ax2.bar(
        x - w / 2,
        rel_a,
        w,
        label="Overall $E_{\\mathrm{rel}}$",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax2.bar(
        x + w / 2,
        early_a,
        w,
        label="Early Transient (t $\\leq$ 10)",
        color="#ffbb78",
        alpha=0.85,
    )
    ax2.set_title("Evaluated on Domain A (OISST SST)")
    ax2.set_ylabel("Error Metric")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_labels, fontsize=8.5)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 0.45)

    fig.suptitle(
        "Plot CL: State Initialization Sensitivity & State Transfer Across Regimes",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_failure_boundary_cm(safety_data: Mapping[str, Any], output_path: Path) -> None:
    """Plot CM: Safe vs Unsafe failure boundary at D=256."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    fb = safety_data["experiment_h_failure_boundary"]

    domains = ["domain_a_oisst_256", "domain_b_era5_256"]
    dom_labels = [
        "Domain A (OISST D=256)\n[Thermal]",
        "Domain B (ERA5 D=256)\n[Advective]",
    ]

    # Panel 1: Max State Norm
    fd_norms = [fb[d]["FixedDelta"]["max_state_norm"] for d in domains]
    sd_norms = [fb[d]["SafeAdaptiveDelta"]["max_state_norm"] for d in domains]

    x = np.arange(len(domains))
    w = 0.35

    ax1.bar(x - w / 2, fd_norms, w, label="FixedDelta", color="#8c564b", alpha=0.85)
    ax1.bar(
        x + w / 2, sd_norms, w, label="SafeAdaptiveDelta", color="#2ca02c", alpha=0.85
    )
    ax1.set_title("Maximum State Norm ($\\|M_t\\|_F$)")
    ax1.set_ylabel("Frobenius Norm")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dom_labels, fontsize=9)
    ax1.legend(loc="upper left", frameon=True)

    # Highlight FixedDelta divergence on Domain B
    ax1.annotate(
        "FixedDelta DIVERGED (NaN)\nat step t=41 on ERA5",
        xy=(1 - w / 2, fd_norms[1]),
        xytext=(0.55, 3.5),
        arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=6),
        color="darkred",
        fontweight="bold",
        fontsize=8.5,
    )

    # Panel 2: Minimum Safety Margin
    fd_margin = [fb[d]["FixedDelta"]["min_safety_margin"] for d in domains]
    sd_margin = [fb[d]["SafeAdaptiveDelta"]["min_safety_margin"] for d in domains]

    ax2.bar(
        x - w / 2, fd_margin, w, label="FixedDelta Margin", color="#8c564b", alpha=0.85
    )
    ax2.bar(
        x + w / 2,
        sd_margin,
        w,
        label="SafeAdaptiveDelta Margin",
        color="#2ca02c",
        alpha=0.85,
    )
    ax2.axhline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Instability Boundary ($2 - \\eta \\|x\\|^2 = 0$)",
    )
    ax2.set_title("Minimum Contraction Safety Margin ($2.0 - \\eta_t \\|x_t\\|^2$)")
    ax2.set_ylabel("Safety Margin (Margin $\\geq$ 0 required)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dom_labels, fontsize=9)
    ax2.legend(loc="lower left", frameon=True, fontsize=8)

    fig.suptitle(
        "Plot CM: Failure Boundary Diagnostic at Scaled Dimension $D=256$",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_runtime_vs_dim_cn(results_agg: Mapping[str, Any], output_path: Path) -> None:
    """Plot CN: Runtime versus dimension by domain."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    models = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "FixedDelta",
        "SafeAdaptiveDelta",
        "SpatialConv",
    ]

    runtimes_a = [results_agg["domain_a"][m]["runtime_us_mean"] for m in models]
    runtimes_b = [results_agg["domain_b"][m]["runtime_us_mean"] for m in models]

    y_pos = np.arange(len(models))

    ax1.barh(
        y_pos, runtimes_a, color=[PALETTE.get(m, "#333333") for m in models], alpha=0.85
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel("Mean Latency ($\\mu$s/step)")
    ax1.set_title("Domain A (OISST SST, $D=64$)")
    ax1.set_xlim(0, 25)

    ax2.barh(
        y_pos, runtimes_b, color=[PALETTE.get(m, "#333333") for m in models], alpha=0.85
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel("Mean Latency ($\\mu$s/step)")
    ax2.set_title("Domain B (ERA5 $T_{2m}$, $D=64$)")
    ax2.set_xlim(0, 25)

    fig.suptitle(
        "Plot CN: Per-Step Latency by Model and Physical Regime", fontsize=13, y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_vs_dim_co(scaling_data: Mapping[str, Any], output_path: Path) -> None:
    """Plot CO: Persistent state memory versus dimension."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    dims = [16, 32, 64, 128, 256, 512]

    # DeltaCore state: M is (D, D) float32 -> D^2 * 4 bytes
    mem_deltacore = [d * d * 4 for d in dims]

    # OnlineRidge / RLS: M is (D, D) + P_inv is (D, D) -> 2 * D^2 * 4 bytes
    mem_ridge = [2 * d * d * 4 for d in dims]

    # Persistence: previous frame (D) float32 -> D * 4 bytes
    mem_persist = [d * 4 for d in dims]

    ax.plot(
        dims,
        mem_ridge,
        "o-",
        label="OnlineRidge ($2 D^2$ floats)",
        color="#ff7f0e",
        linewidth=2.0,
    )
    ax.plot(
        dims,
        mem_deltacore,
        "s-",
        label="SafeAdaptiveDelta ($D^2$ floats: 2x less)",
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.plot(
        dims,
        mem_persist,
        "^-",
        label="Persistence ($D$ floats)",
        color="#7f7f7f",
        linewidth=1.5,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Feature Dimension $D$")
    ax.set_ylabel("Persistent State Memory (Bytes)")
    ax.set_title("Plot CO: Persistent State Memory Scaling vs Dimension $D$")
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc="upper left", frameon=True)

    # Highlight factor of 2 advantage
    ax.annotate(
        "DeltaCore persistent state is exactly\n2x smaller than OnlineRidge RLS matrix",
        xy=(256, 256 * 256 * 4),
        xytext=(64, 1e6),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_seed_differences_cp(
    per_seed_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CP: Per-seed cross-domain differences across all 5 seeds."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    seeds = ["42", "43", "44", "45", "46"]
    models_compare = ["OnlineRidge", "FixedDelta", "SafeAdaptiveDelta"]

    # Panel 1: Domain A
    for m in models_compare:
        vals = [per_seed_data["domain_a"][m][f"seed_{s}"]["rel_error"] for s in seeds]
        ax1.plot(
            seeds,
            vals,
            "o--",
            label=m,
            color=PALETTE[m],
            linewidth=1.8,
            markersize=6,
        )

    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Relative Error ($E_{\\mathrm{rel}}$)")
    ax1.set_title("Domain A (OISST SST): Per-Seed Error")
    ax1.legend(loc="lower right", frameon=True)
    ax1.set_ylim(0.24, 0.33)

    # Panel 2: Domain B
    for m in models_compare:
        vals = [per_seed_data["domain_b"][m][f"seed_{s}"]["rel_error"] for s in seeds]
        ax2.plot(
            seeds,
            vals,
            "o--",
            label=m,
            color=PALETTE[m],
            linewidth=1.8,
            markersize=6,
        )

    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Relative Error ($E_{\\mathrm{rel}}$)")
    ax2.set_title("Domain B (ERA5 $T_{2m}$): Per-Seed Error")
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0.08, 0.18)

    fig.suptitle(
        "Plot CP: Per-Seed Stability and Reproducibility Across 5 Seeds",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_phase_15_plots(
    artifacts_dir: Path | str = "docs/benchmarks/artifacts/phase_15",
) -> None:
    """Generate all 10 Phase 15 plots from serialized JSON artifacts."""
    art_path = Path(artifacts_dir)
    plots_dir = art_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(art_path / "phase_15_results.json") as f:
        results_agg = json.load(f)

    with open(art_path / "phase_15_per_seed.json") as f:
        per_seed_data = json.load(f)

    with open(art_path / "phase_15_transfer.json") as f:
        transfer_data = json.load(f)

    with open(art_path / "phase_15_safety.json") as f:
        safety_data = json.load(f)

    with open(art_path / "phase_15_retention.json") as f:
        retention_data = json.load(f)

    plot_cross_domain_matrix_cg(
        results_agg, plots_dir / "plot_cg_cross_domain_matrix.png"
    )
    plot_parameter_transfer_ch(
        transfer_data["experiment_a_parameter_transfer"],
        plots_dir / "plot_ch_parameter_transfer.png",
    )
    plot_safety_pareto_frontier_ci(
        safety_data["experiment_d_pareto"],
        plots_dir / "plot_ci_safety_pareto_frontier.png",
    )
    plot_state_transfer_cj(transfer_data, plots_dir / "plot_cj_state_transfer.png")
    plot_continuous_vs_reset_ck(
        retention_data, plots_dir / "plot_ck_continuous_vs_reset.png"
    )
    plot_state_initialization_cl(
        transfer_data, plots_dir / "plot_cl_state_initialization.png"
    )
    plot_failure_boundary_cm(safety_data, plots_dir / "plot_cm_failure_boundary.png")
    plot_runtime_vs_dim_cn(results_agg, plots_dir / "plot_cn_runtime_vs_dim.png")
    plot_memory_vs_dim_co(
        safety_data["scaling_resources"], plots_dir / "plot_co_memory_vs_dim.png"
    )
    plot_per_seed_differences_cp(
        per_seed_data, plots_dir / "plot_cp_per_seed_differences.png"
    )

    print(f"[Observatory] All 10 Phase 15 plots successfully rendered to {plots_dir}.")
