"""DeltaCore Phase 12 Observatory Publication Plots AX through BJ.

Implements all 13 diagnostic figures specified in Phase 12 Section 22:
    - Plot AX: Spatio-temporal prediction error through regime transitions (plot_spatiotemporal_error_over_time_ax)
    - Plot AY: Adaptation vs negative-transfer Pareto frontier (plot_adaptation_negative_transfer_pareto_ay)
    - Plot AZ: Retention coefficient trajectories (plot_retention_trajectories_az)
    - Plot BA: Fixed-high vs fixed-low vs adaptive retention (plot_retention_ablation_ba)
    - Plot BB: A1 -> B -> A1 forgetting and recovery (plot_a1_b_a1_recovery_bb)
    - Plot BC: Adaptive retention vs state-conditioned retention (plot_error_vs_state_adaptive_bc)
    - Plot BD: Spatial order vs shuffled spatial order (plot_spatial_vs_shuffled_bd)
    - Plot BE: Performance versus dimensionality D in {64, 128, 256} (plot_perf_vs_dim_be)
    - Plot BF: Runtime/token versus dimensionality (plot_runtime_vs_dim_bf)
    - Plot BG: State memory versus dimensionality (plot_memory_vs_dim_bg)
    - Plot BH: Adaptation energy versus performance (plot_adaptation_energy_vs_perf_bh)
    - Plot BI: Safe versus unsafe variants (plot_safe_vs_unsafe_bi)
    - Plot BJ: Continuous state versus reset state (plot_continuous_vs_reset_bj)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

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


# Standard palette for Phase 12 models
PALETTE: dict[str, str] = {
    "Naive_persistence": "#7f7f7f",
    "FrozenLinear": "#6c757d",
    "FrozenMLP": "#adb5bd",
    "SpatialConvControl": "#343a40",
    "SpatialDownsampleControl": "#495057",
    "GRU": "#17a2b8",
    "LSTM": "#20c997",
    "OnlineRidge": "#fd7e14",
    "NonlinearOnlineRidge": "#e83e8c",
    "FixedDelta": "#007bff",
    "AdaptiveDelta": "#6f42c1",
    "SafeAdaptiveDelta": "#28a745",
    "SelfReferential": "#ffc107",
    "SafeSelfReferential": "#004085",
    "Selective_fixed_high": "#6c757d",
    "Selective_fixed_low": "#9b59b6",
    "Selective_adaptive": "#d63384",
    "Selective_state_adaptive": "#e74c3c",
    "Selective_shuffled_control": "#95a5a6",
    "Selective_oracle": "#198754",
}


def plot_spatiotemporal_error_over_time_ax(
    trajectories: Mapping[str, Sequence[float]],
    change_points: Sequence[int],
    output_path: Path,
) -> None:
    """Plot AX: Spatio-temporal prediction error through regime transitions (A -> B -> C -> A)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for name, errs in trajectories.items():
        color = PALETTE.get(name, "#333333")
        is_highlight = name in (
            "SafeAdaptiveDelta",
            "Selective_adaptive",
            "Selective_state_adaptive",
            "NonlinearOnlineRidge",
        )
        ax.plot(
            errs,
            label=name,
            color=color,
            alpha=0.90 if is_highlight else 0.45,
            lw=1.8 if is_highlight else 1.1,
        )

    for i, cp in enumerate(change_points):
        ax.axvline(
            cp,
            color="#d9534f",
            linestyle="--",
            alpha=0.75,
            lw=1.2,
            label="Regime Shift" if i == 0 else "",
        )

    ax.set_title(r"Plot AX: Spatio-Temporal Prediction Error ($A \to B \to C \to A$)")
    ax.set_xlabel(r"Streaming Timestep $t$")
    ax.set_ylabel(r"Relative Error $E_{\mathrm{rel}}(t)$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_adaptation_negative_transfer_pareto_ay(
    models: Sequence[str],
    post_shift_errors: Sequence[float],
    negative_transfers: Sequence[float],
    output_path: Path,
) -> None:
    """Plot AY: Adaptation vs Negative-Transfer Pareto Frontier."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    valid_pts = []
    for m, ps, nt in zip(models, post_shift_errors, negative_transfers, strict=False):
        if np.isnan(ps) or np.isnan(nt):
            continue
        color = PALETTE.get(m, "#333333")
        ax.scatter(ps, nt, s=110, color=color, zorder=4, edgecolor="black")
        ax.annotate(
            m,
            (ps, nt),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
        valid_pts.append((ps, nt))

    if valid_pts:
        # Sort by post_shift_error and identify non-dominated points
        pts = sorted(valid_pts, key=lambda x: x[0])
        pareto = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < pareto[-1][1]:
                pareto.append(pt)
        px, py = zip(*pareto, strict=False)
        ax.step(
            px,
            py,
            where="post",
            color="#e74c3c",
            linestyle="--",
            alpha=0.7,
            label="Empirical Pareto Frontier",
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("Plot AY: Adaptation vs. Negative-Transfer Pareto Frontier")
    ax.set_xlabel("Post-Shift Error (Lower is Better)")
    ax.set_ylabel(
        r"Negative Transfer: $E_{\mathrm{continuous}}(B) - E_{\mathrm{reset}}(B)$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_retention_trajectories_az(
    retention_series: Mapping[str, Sequence[float]],
    change_points: Sequence[int],
    output_path: Path,
) -> None:
    """Plot AZ: Retention coefficient alpha_t trajectories."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    for name, alphas in retention_series.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(alphas, label=name, color=color, alpha=0.85, lw=1.6)

    for i, cp in enumerate(change_points):
        ax.axvline(
            cp,
            color="#d9534f",
            linestyle="--",
            alpha=0.75,
            lw=1.2,
            label="Regime Shift" if i == 0 else "",
        )

    ax.set_title(r"Plot AZ: Retention Coefficient Dynamics $\alpha_t$ Across Regimes")
    ax.set_xlabel(r"Streaming Timestep $t$")
    ax.set_ylabel(r"Retention Factor $\alpha_t \in [\alpha_{\min}, 1.0]$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_retention_ablation_ba(
    retention_modes: Sequence[str],
    post_shift_errors: Sequence[float],
    negative_transfers: Sequence[float],
    output_path: Path,
) -> None:
    """Plot BA: Fixed-high vs fixed-low vs adaptive retention."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    x = np.arange(len(retention_modes))
    width = 0.55
    colors = [PALETTE.get(m, "#4a90e2") for m in retention_modes]

    ax1.bar(
        x, post_shift_errors, width=width, color=colors, edgecolor="black", alpha=0.85
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(retention_modes, rotation=25, ha="right", fontsize=8.5)
    ax1.set_title("Immediate Post-Shift Error")
    ax1.set_ylabel("Mean Relative Error")
    ax1.grid(True, alpha=0.3)

    ax2.bar(
        x, negative_transfers, width=width, color=colors, edgecolor="black", alpha=0.85
    )
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(retention_modes, rotation=25, ha="right", fontsize=8.5)
    ax2.set_title("Stale-Memory Negative Transfer")
    ax2.set_ylabel(r"$E_{\mathrm{cont}}(B) - E_{\mathrm{reset}}(B)$")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Plot BA: Retention Mode Ablation Analysis", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_a1_b_a1_recovery_bb(
    models: Sequence[str],
    adapt_b_errors: Sequence[float],
    forgetting_metrics: Sequence[float],
    output_path: Path,
) -> None:
    """Plot BB: A1 -> B -> A1 forgetting and recovery."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for m, b_err, fg in zip(models, adapt_b_errors, forgetting_metrics, strict=False):
        if np.isnan(b_err) or np.isnan(fg):
            continue
        color = PALETTE.get(m, "#333333")
        ax.scatter(b_err, fg, s=110, color=color, zorder=4, edgecolor="black")
        ax.annotate(
            m,
            (b_err, fg),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title(r"Plot BB: $A_1 \to B \to A_1$ Adaptation vs. Forgetting")
    ax.set_xlabel("Phase B Adaptation Error")
    ax.set_ylabel(
        r"Forgetting on Re-entry: $E_{\mathrm{return}, A} - E_{\mathrm{pre}, A}$"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_error_vs_state_adaptive_bc(
    error_trajectories: Mapping[str, Sequence[float]],
    change_points: Sequence[int],
    output_path: Path,
) -> None:
    """Plot BC: Error-only adaptive retention vs state-conditioned retention."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for name, errs in error_trajectories.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(errs, label=name, color=color, alpha=0.85, lw=1.6)

    for i, cp in enumerate(change_points):
        ax.axvline(
            cp,
            color="#d9534f",
            linestyle="--",
            alpha=0.75,
            lw=1.2,
            label="Regime Shift" if i == 0 else "",
        )

    ax.set_title("Plot BC: Error-Only vs. State-Conditioned Retention Controller")
    ax.set_xlabel(r"Streaming Timestep $t$")
    ax.set_ylabel(r"Relative Error $E_{\mathrm{rel}}(t)$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spatial_vs_shuffled_bd(
    models: Sequence[str],
    original_errors: Sequence[float],
    shuffled_errors: Sequence[float],
    output_path: Path,
) -> None:
    """Plot BD: True spatial ordering vs shuffled spatial permutation P_{spatial}."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        original_errors,
        width=width,
        label="True 2D Spatial Order",
        color="#2980b9",
        edgecolor="black",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        shuffled_errors,
        width=width,
        label=r"Shuffled Permutation $\mathcal{P}_{\mathrm{spatial}}$",
        color="#e67e22",
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax.set_title(
        r"Plot BD: Spatial Structure Ablation (Order vs. $\mathcal{P}_{\mathrm{spatial}}$)"
    )
    ax.set_ylabel("Mean Streaming Error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perf_vs_dim_be(
    dimensions: Sequence[int],
    model_errors: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Plot BE: Performance vs dimensionality D in {64, 128, 256}."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for name, errs in model_errors.items():
        color = PALETTE.get(name, "#333333")
        is_highlight = name in (
            "SafeAdaptiveDelta",
            "Selective_adaptive",
            "Selective_state_adaptive",
            "OnlineRidge",
        )
        ax.plot(
            dimensions,
            errs,
            marker="o",
            label=name,
            color=color,
            lw=2.0 if is_highlight else 1.2,
            alpha=0.90 if is_highlight else 0.50,
        )

    ax.set_title(r"Plot BE: Spatio-Temporal Prediction Error vs. Dimension $D$")
    ax.set_xlabel(r"Dimension $D = H \cdot W \cdot C$")
    ax.set_ylabel("Mean Relative Error")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_runtime_vs_dim_bf(
    dimensions: Sequence[int],
    model_runtimes: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Plot BF: Runtime/token (microseconds) vs dimensionality."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for name, runtimes in model_runtimes.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(
            dimensions,
            runtimes,
            marker="s",
            label=name,
            color=color,
            lw=1.8,
            alpha=0.85,
        )

    ax.set_title(r"Plot BF: Step Latency vs. Dimension $D$")
    ax.set_xlabel(r"Dimension $D = H \cdot W \cdot C$")
    ax.set_ylabel(r"Latency ($\mu\mathrm{s}$ / token)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dimensions)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_vs_dim_bg(
    dimensions: Sequence[int],
    model_memories: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Plot BG: Persistent state memory (bytes) vs dimensionality."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for name, mems in model_memories.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(
            dimensions,
            mems,
            marker="^",
            label=name,
            color=color,
            lw=1.8,
            alpha=0.85,
        )

    ax.set_title(r"Plot BG: Persistent State Memory vs. Dimension $D$")
    ax.set_xlabel(r"Dimension $D = H \cdot W \cdot C$")
    ax.set_ylabel("State Memory (Bytes)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dimensions)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_adaptation_energy_vs_perf_bh(
    models: Sequence[str],
    energies: Sequence[float],
    errors: Sequence[float],
    output_path: Path,
) -> None:
    """Plot BH: Adaptation energy vs prediction error."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for m, eng, err in zip(models, energies, errors, strict=False):
        if np.isnan(eng) or np.isnan(err):
            continue
        color = PALETTE.get(m, "#333333")
        ax.scatter(eng, err, s=110, color=color, zorder=4, edgecolor="black")
        ax.annotate(
            m,
            (eng, err),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_title(
        r"Plot BH: Adaptation Energy ($\sum ||\Delta S_t||_F^2$) vs. Prediction Error"
    )
    ax.set_xlabel(r"Cumulative Adaptation Energy $E_{\mathrm{adapt}}$")
    ax.set_ylabel("Mean Prediction Error")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_safe_vs_unsafe_bi(
    models: Sequence[str],
    max_norms: Sequence[float],
    min_margins: Sequence[float],
    output_path: Path,
) -> None:
    """Plot BI: Stability diagnostics across safe vs standard variants."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    x = np.arange(len(models))
    width = 0.55
    colors = [PALETTE.get(m, "#4a90e2") for m in models]

    ax1.bar(x, max_norms, width=width, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax1.set_title("Maximum State Norm")
    ax1.set_ylabel(r"$\max_t ||S_t||_F$")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.bar(x, min_margins, width=width, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax2.set_title("Minimum Stability Margin")
    ax2.set_ylabel(r"$\min_t \mu_t$ (Margin > 0 implies contractivity)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Plot BI: Safe vs. Unsafe Stability Diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_continuous_vs_reset_bj(
    models: Sequence[str],
    continuous_errors: Sequence[float],
    reset_errors: Sequence[float],
    output_path: Path,
) -> None:
    """Plot BJ: Continuous state vs state reset comparison."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        continuous_errors,
        width=width,
        label="Continuous Adaptive State",
        color="#27ae60",
        edgecolor="black",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        reset_errors,
        width=width,
        label="State Reset at Shift",
        color="#c0392b",
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax.set_title("Plot BJ: Causal Control B (Continuous State vs. State Reset)")
    ax.set_ylabel("Post-Shift Error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
