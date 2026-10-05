"""DeltaCore Phase 17 Observatory Publication Plots DC through DP.

Implements all 14 diagnostic figures specified in Phase 17 Section 25:
    - Plot DC: Stationary classification performance (plot_dc_stationary_perf)
    - Plot DD: Post-shift accuracy over time (plot_dd_post_shift_acc_vs_time)
    - Plot DE: First-passage / sustained recovery (plot_de_recovery_curves)
    - Plot DF: Continuous versus reset (plot_df_continuous_vs_reset)
    - Plot DG: Retention-mode comparison (plot_dg_retention_modes)
    - Plot DH: Covariate shift robustness (plot_dh_covariate_shift)
    - Plot DI: Decision-boundary shift robustness (plot_di_decision_boundary_shift)
    - Plot DJ: Accuracy versus dimensionality (plot_dj_acc_vs_dim)
    - Plot DK: Runtime versus dimensionality (plot_dk_runtime_vs_dim)
    - Plot DL: Persistent state memory versus dimensionality (plot_dl_memory_vs_dim)
    - Plot DM: Adaptive-state norms (plot_dm_state_norm_trajectory)
    - Plot DN: Adaptation energy (plot_dn_adaptation_energy)
    - Plot DO: Label-shuffle negative control (plot_do_label_shuffle_control)
    - Plot DP: Feature-permutation control (plot_dp_feature_permutation_control)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(
        "/Users/urjasoft/.gemini/antigravity-ide/brain/6c485406-eeab-4ae3-9fa7-892b0159db3c/scratch/.mpl"
    )
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib  # noqa: E402

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
            "lines.linewidth": 1.7,
            "grid.alpha": 0.35,
        }
    )


PALETTE: dict[str, str] = {
    "FrozenLinear": "#1f77b4",
    "SmallMLP": "#aec7e8",
    "OnlineLogisticRegression": "#ff7f0e",
    "OnlineRidge": "#d62728",
    "OnlineMulticlassLinear": "#9467bd",
    "GRU": "#8c564b",
    "LSTM": "#e377c2",
    "FixedDelta": "#bcbd22",
    "SafeAdaptiveDelta": "#2ca02c",
    "AdaptiveStateOFF": "#7f7f7f",
}


def plot_dc_stationary_perf(results: dict[str, Any], output_path: Path) -> None:
    """Plot DC: Stationary classification performance (useful-prediction gate)."""
    _apply_style()
    models = [m for m in PALETTE if m in results]
    means = [results[m]["stat_acc_mean"] for m in models]
    stds = [results[m]["stat_acc_std"] for m in models]
    colors = [PALETTE[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        models,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(
        1.0 / 6.0,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=r"Chance Level ($1/K=0.167$)",
    )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Stationary Classification Accuracy")
    ax.set_title(
        "Plot DC: Stationary Classification Performance Across Model Matrix ($D=32, K=6$)"
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.legend(loc="lower right")

    for bar, val in zip(bars, means, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dd_post_shift_acc_vs_time(
    retention_results: dict[str, Any], output_path: Path
) -> None:
    """Plot DD: Rolling classification accuracy over time showing regime transitions."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bounds = retention_results.get("regime_bounds", [120, 240, 360])

    # Plot rolling accuracy curves
    cont = retention_results["continuous"]["rolling_acc"]
    reset = retention_results["reset_at_bounds"]["rolling_acc"]
    high = retention_results["fixed_high_retention"]["rolling_acc"]
    low = retention_results["fixed_low_retention"]["rolling_acc"]

    t = np.arange(len(cont))
    ax.plot(
        t, cont, label="SafeAdaptiveDelta (Continuous)", color="#2ca02c", linewidth=2.0
    )
    ax.plot(
        t,
        reset,
        label="SafeAdaptiveDelta (Reset-at-Shift)",
        color="#17becf",
        linestyle="--",
        linewidth=1.8,
    )
    ax.plot(
        t, high, label=r"FixedDelta ($\alpha=0.99$)", color="#bcbd22", linewidth=1.5
    )
    ax.plot(
        t,
        low,
        label=r"FixedDelta ($\alpha=0.70$)",
        color="#e377c2",
        linestyle=":",
        linewidth=1.5,
    )

    reg_names = [
        "Regime A\n(Direction)",
        "Regime B\n(Covariance)",
        "Regime C\n(Boundary)",
        "Regime A\n(Return)",
    ]
    reg_starts = [0] + bounds
    reg_ends = bounds + [len(cont)]

    for b in bounds:
        ax.axvline(b, color="black", linestyle="--", alpha=0.5)

    for i in range(len(reg_names)):
        mid = (reg_starts[i] + reg_ends[i]) / 2.0
        ax.text(
            mid,
            0.25,
            reg_names[i],
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="gray"),
        )

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Online Stream Step $t$")
    ax.set_ylabel("Rolling Window Accuracy (window=15)")
    ax.set_title(
        "Plot DD: Post-Shift Online Accuracy Over Time Across Non-Stationary Regimes ($A \\to B \\to C \\to A$)"
    )
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_de_recovery_curves(shift_results: dict[str, Any], output_path: Path) -> None:
    """Plot DE: First-passage and sustained recovery comparison."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sev = "moderate"
    db_models = shift_results["decision_boundary"][sev]
    models = list(db_models.keys())

    sust = [db_models[m]["sust_recovery"] for m in models]
    post_acc = [db_models[m]["post_shift_acc"] for m in models]
    colors = [PALETTE.get(m, "#333333") for m in models]

    ax1.bar(models, sust, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel(r"Steps to Sustained Recovery ($K=10$ accurate steps)")
    ax1.set_title(f"Sustained Recovery (Decision-Boundary Shift, {sev.title()})")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=25, ha="right")

    ax2.bar(
        models, post_acc, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8
    )
    ax2.set_ylabel("Immediate Post-Shift Accuracy (first 20 steps)")
    ax2.set_title(f"Immediate Post-Shift Accuracy ({sev.title()})")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=25, ha="right")

    plt.suptitle(
        "Plot DE: First-Passage and Sustained Recovery Under Non-Stationary Shift",
        fontsize=13,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_df_continuous_vs_reset(
    retention_results: dict[str, Any], output_path: Path
) -> None:
    """Plot DF: Continuous state versus state reset across regimes in Task D."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cont_acc = retention_results["continuous"]["accuracy"]
    reset_acc = retention_results["reset_at_bounds"]["accuracy"]
    cont_loss = retention_results["continuous"]["excess_loss"]
    reset_loss = retention_results["reset_at_bounds"]["excess_loss"]

    modes = ["Continuous State", "Reset-at-Shift"]
    accs = [cont_acc, reset_acc]
    losses = [cont_loss, reset_loss]
    colors = ["#2ca02c", "#17becf"]

    ax1.bar(
        modes,
        accs,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    ax1.set_ylabel("Overall Stream Accuracy")
    ax1.set_title("Stream Accuracy: Continuous vs Reset")
    ax1.set_ylim(0.8, 1.02)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)

    ax2.bar(
        modes,
        losses,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    ax2.set_ylabel("Cumulative Excess Classification Loss")
    ax2.set_title("Excess Loss: Continuous vs Reset")
    for i, v in enumerate(losses):
        ax2.text(i, v + 2.0, f"{v:.1f}", ha="center", fontsize=10)

    plt.suptitle(
        "Plot DF: Internal Adaptive-State Reset Ablation Under $A \\to B \\to C \\to A$",
        fontsize=13,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dg_retention_modes(
    retention_results: dict[str, Any], output_path: Path
) -> None:
    """Plot DG: Retention-mode comparison (fixed-high, fixed-low, adaptive)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    modes = ["Fixed-High (α=0.99)", "Fixed-Low (α=0.70)", "Safe Adaptive (α_t)"]
    accs = [
        retention_results["fixed_high_retention"]["accuracy"],
        retention_results["fixed_low_retention"]["accuracy"],
        retention_results["adaptive_retention"]["accuracy"],
    ]
    losses = [
        retention_results["fixed_high_retention"]["excess_loss"],
        retention_results["fixed_low_retention"]["excess_loss"],
        retention_results["adaptive_retention"]["excess_loss"],
    ]
    colors = ["#bcbd22", "#e377c2", "#2ca02c"]

    bars = ax.bar(
        modes,
        accs,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    ax.set_ylabel("Overall Stream Accuracy")
    ax.set_ylim(0.8, 1.02)
    ax.set_title("Plot DG: Retention-Mode Comparison Under Non-Stationary Shift Stream")

    for bar, a, l_val in zip(bars, accs, losses, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"Acc: {a:.3f}\nExcess L: {l_val:.1f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dh_covariate_shift(shift_results: dict[str, Any], output_path: Path) -> None:
    """Plot DH: Covariate shift robustness across mild, moderate, severe severities."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    severities = ["mild", "moderate", "severe"]
    models = [
        "FrozenLinear",
        "OnlineLogisticRegression",
        "OnlineRidge",
        "SafeAdaptiveDelta",
    ]

    for m in models:
        accs = [shift_results["covariate"][s][m]["accuracy"] for s in severities]
        ax.plot(
            severities,
            accs,
            marker="o",
            label=m,
            color=PALETTE.get(m, "#333333"),
            linewidth=2.0,
        )

    ax.set_ylabel("Stream Classification Accuracy")
    ax.set_xlabel("Covariate Shift Severity")
    ax.set_ylim(0.9, 1.02)
    ax.set_title(
        "Plot DH: Covariate Shift Robustness Across Shift Severities ($A \\to B \\to A$)"
    )
    ax.legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_di_decision_boundary_shift(
    shift_results: dict[str, Any], output_path: Path
) -> None:
    """Plot DI: Decision-boundary shift robustness across mild, moderate, severe rotations."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    severities = ["mild", "moderate", "severe"]
    models = [
        "FrozenLinear",
        "OnlineLogisticRegression",
        "OnlineRidge",
        "SafeAdaptiveDelta",
    ]

    for m in models:
        accs = [
            shift_results["decision_boundary"][s][m]["accuracy"] for s in severities
        ]
        losses = [
            shift_results["decision_boundary"][s][m]["excess_loss"] for s in severities
        ]
        ax1.plot(
            severities,
            accs,
            marker="s",
            label=m,
            color=PALETTE.get(m, "#333333"),
            linewidth=2.0,
        )
        ax2.plot(
            severities,
            losses,
            marker="^",
            label=m,
            color=PALETTE.get(m, "#333333"),
            linewidth=2.0,
        )

    ax1.set_ylabel("Stream Accuracy")
    ax1.set_xlabel("Boundary Shift Severity")
    ax1.set_ylim(0.6, 1.02)
    ax1.set_title("Accuracy vs Boundary Shift Severity")
    ax1.legend(loc="lower left")

    ax2.set_ylabel("Cumulative Excess Loss")
    ax2.set_xlabel("Boundary Shift Severity")
    ax2.set_title("Cumulative Excess Loss vs Severity")
    ax2.legend(loc="upper left")

    plt.suptitle(
        "Plot DI: Decision-Boundary Shift Robustness Across Severities ($A \\to C \\to A$)",
        fontsize=13,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dj_acc_vs_dim(scaling_results: dict[str, Any], output_path: Path) -> None:
    """Plot DJ: Accuracy versus feature dimension D in {32, 64, 128, 256}."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    dims = scaling_results["dimensions"]
    models = scaling_results["models"]

    for m_name, entries in models.items():
        accs = [e["accuracy"] for e in entries]
        ax.plot(
            dims,
            accs,
            marker="o",
            label=m_name,
            color=PALETTE.get(m_name, "#333333"),
            linewidth=2.0,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_ylabel("Online Non-Stationary Accuracy")
    ax.set_xlabel("Feature Dimension $D$")
    ax.set_ylim(0.8, 1.02)
    ax.set_title(
        "Plot DJ: Classification Accuracy Versus Dimensionality ($D \\in \\{32, 64, 128, 256\\}$)"
    )
    ax.legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dk_runtime_vs_dim(scaling_results: dict[str, Any], output_path: Path) -> None:
    """Plot DK: Per-sample latency versus dimensionality D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    dims = scaling_results["dimensions"]
    models = scaling_results["models"]

    for m_name, entries in models.items():
        lats = [e["latency_us"] for e in entries]
        ax.plot(
            dims,
            lats,
            marker="D",
            label=m_name,
            color=PALETTE.get(m_name, "#333333"),
            linewidth=2.0,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_ylabel(r"Latency Per Sample ($\mu s$)")
    ax.set_xlabel("Feature Dimension $D$")
    ax.set_title("Plot DK: Per-Sample Inference + Adaptation Latency vs Dimension $D$")
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dl_memory_vs_dim(scaling_results: dict[str, Any], output_path: Path) -> None:
    """Plot DL: Persistent state memory versus dimensionality D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    dims = scaling_results["dimensions"]
    models = scaling_results["models"]

    for m_name, entries in models.items():
        mems = [max(e["state_memory_bytes"], 1) for e in entries]
        ax.plot(
            dims,
            mems,
            marker="s",
            label=m_name,
            color=PALETTE.get(m_name, "#333333"),
            linewidth=2.0,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_ylabel("Persistent State Memory (Bytes, log scale)")
    ax.set_xlabel("Feature Dimension $D$")
    ax.set_title("Plot DL: Persistent State Memory Scaling vs Feature Dimension $D$")
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dm_state_norm_trajectory(
    retention_results: dict[str, Any], output_path: Path
) -> None:
    """Plot DM: Adaptive state Frobenius norm ||M_t||_F trajectory."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    norms_cont = retention_results["continuous"]["state_norms"]
    norms_reset = retention_results["reset_at_bounds"]["state_norms"]
    bounds = retention_results.get("regime_bounds", [120, 240, 360])

    t = np.arange(len(norms_cont))
    ax.plot(
        t,
        norms_cont,
        label="SafeAdaptiveDelta (Continuous)",
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.plot(
        t,
        norms_reset,
        label="SafeAdaptiveDelta (Reset-at-Shift)",
        color="#17becf",
        linestyle="--",
        linewidth=1.8,
    )

    for b in bounds:
        ax.axvline(b, color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel("Online Stream Step $t$")
    ax.set_ylabel(r"State Frobenius Norm $\|M_t\|_F$")
    ax.set_title(
        "Plot DM: Adaptive-State Norm Trajectory Across Non-Stationary Regimes ($A \\to B \\to C \\to A$)"
    )
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dn_adaptation_energy(shift_results: dict[str, Any], output_path: Path) -> None:
    """Plot DN: Cumulative adaptation energy across shifts."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    severities = ["mild", "moderate", "severe"]
    energies_cov = [
        shift_results["covariate"][s]["SafeAdaptiveDelta"]["adapt_energy"]
        for s in severities
    ]
    energies_db = [
        shift_results["decision_boundary"][s]["SafeAdaptiveDelta"]["adapt_energy"]
        for s in severities
    ]

    x = np.arange(len(severities))
    width = 0.35

    ax.bar(
        x - width / 2,
        energies_cov,
        width,
        label="Covariate Shift ($A \\to B \\to A$)",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        energies_db,
        width,
        label="Decision-Boundary Shift ($A \\to C \\to A$)",
        color="#d62728",
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_ylabel(
        r"Cumulative Adaptation Energy $E_{\mathrm{adapt}} = \sum_t \|\Delta M_t\|_F^2$"
    )
    ax.set_xlabel("Shift Severity")
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in severities])
    ax.set_title("Plot DN: Adaptation Energy Across Shift Type and Severity")
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_do_label_shuffle_control(
    config_data: dict[str, Any], output_path: Path
) -> None:
    """Plot DO: Label-shuffle negative control (verifying chance level)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    ctrls = config_data["controls"]
    models = list(ctrls.keys())
    normal_acc = [ctrls[m]["normal_accuracy"] for m in models]
    shuff_acc = [ctrls[m]["label_shuffled_accuracy"] for m in models]
    chance = ctrls[models[0]]["chance_level"]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        normal_acc,
        width,
        label="Normal Stream (True Signal)",
        color="#2ca02c",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        shuff_acc,
        width,
        label="Label Shuffled (Negative Control)",
        color="#d62728",
        alpha=0.85,
        edgecolor="black",
    )
    ax.axhline(
        chance,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Theoretical Chance ($1/K = {chance:.3f}$)",
    )

    ax.set_ylabel("Stream Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "Plot DO: Negative Control — Label Shuffle Reduces All Models to Chance"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dp_feature_permutation_control(
    config_data: dict[str, Any], output_path: Path
) -> None:
    """Plot DP: Feature-permutation control."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    ctrls = config_data["controls"]
    models = list(ctrls.keys())
    normal_acc = [ctrls[m]["normal_accuracy"] for m in models]
    perm_acc = [ctrls[m]["feature_permuted_accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        normal_acc,
        width,
        label="Unpermuted Coordinates",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        perm_acc,
        width,
        label="Permuted Coordinates",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_ylabel("Stream Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Plot DP: Coordinate Permutation Control Across Evaluated Models")
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def generate_all_phase_17_plots(
    artifacts_dir: Path | str = "docs/benchmarks/artifacts/phase_17",
) -> None:
    """Generate all 14 publication plots DC through DP from serialized Phase 17 artifacts."""
    base_dir = Path(artifacts_dir)
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(base_dir / "phase_17_results.json") as f:
        results = json.load(f)
    with open(base_dir / "phase_17_config.json") as f:
        config_data = json.load(f)
    with open(base_dir / "phase_17_shift_results.json") as f:
        shift_results = json.load(f)
    with open(base_dir / "phase_17_retention.json") as f:
        retention_results = json.load(f)
    with open(base_dir / "phase_17_scaling.json") as f:
        scaling_results = json.load(f)

    print(
        f"[Observatory] Generating all 14 Phase 17 Publication Plots into: {plots_dir} ..."
    )

    plot_dc_stationary_perf(results, plots_dir / "plot_dc_stationary_perf.png")
    plot_dd_post_shift_acc_vs_time(
        retention_results, plots_dir / "plot_dd_post_shift_acc_vs_time.png"
    )
    plot_de_recovery_curves(shift_results, plots_dir / "plot_de_recovery_curves.png")
    plot_df_continuous_vs_reset(
        retention_results, plots_dir / "plot_df_continuous_vs_reset.png"
    )
    plot_dg_retention_modes(
        retention_results, plots_dir / "plot_dg_retention_modes.png"
    )
    plot_dh_covariate_shift(shift_results, plots_dir / "plot_dh_covariate_shift.png")
    plot_di_decision_boundary_shift(
        shift_results, plots_dir / "plot_di_decision_boundary_shift.png"
    )
    plot_dj_acc_vs_dim(scaling_results, plots_dir / "plot_dj_acc_vs_dim.png")
    plot_dk_runtime_vs_dim(scaling_results, plots_dir / "plot_dk_runtime_vs_dim.png")
    plot_dl_memory_vs_dim(scaling_results, plots_dir / "plot_dl_memory_vs_dim.png")
    plot_dm_state_norm_trajectory(
        retention_results, plots_dir / "plot_dm_state_norm_trajectory.png"
    )
    plot_dn_adaptation_energy(
        shift_results, plots_dir / "plot_dn_adaptation_energy.png"
    )
    plot_do_label_shuffle_control(
        config_data, plots_dir / "plot_do_label_shuffle_control.png"
    )
    plot_dp_feature_permutation_control(
        config_data, plots_dir / "plot_dp_feature_permutation_control.png"
    )

    print("[Observatory] Successfully generated all 14 Phase 17 publication plots!")


if __name__ == "__main__":
    generate_all_phase_17_plots()
