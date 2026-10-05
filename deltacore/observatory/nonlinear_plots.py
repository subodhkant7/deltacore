"""DeltaCore Phase 11 Observatory Publication Plots AN through AW.

Implements all 10 diagnostic figures specified in Phase 11 Section 17:
    - Plot AN: Nonlinear regime error over time (plot_nonlinear_error_over_time_an)
    - Plot AO: A -> B -> A adaptation, forgetting, and recovery (plot_adaptation_forgetting_ao)
    - Plot AP: Retention coefficient over time (plot_retention_trajectory_ap)
    - Plot AQ: Adaptive retention vs fixed retention & oracle (plot_retention_ablation_aq)
    - Plot AR: DeltaCore vs Linear RLS vs Nonlinear RLS (plot_model_comparison_ar)
    - Plot AS: Performance vs dimensionality D in {8, 16, 32, 64, 128} (plot_perf_vs_dim_as)
    - Plot AT: Runtime/token vs dimensionality (plot_runtime_vs_dim_at)
    - Plot AU: State memory vs dimensionality (plot_memory_vs_dim_au)
    - Plot AV: Adaptation energy vs error reduction (plot_adaptation_efficiency_av)
    - Plot AW: Safe vs unsafe variants (plot_safe_vs_unsafe_aw)
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


# Distinct color palette
PALETTE = {
    "Naive": "#7f7f7f",
    "FrozenLinear": "#6c757d",
    "FrozenMLP": "#adb5bd",
    "GRU": "#17a2b8",
    "LSTM": "#20c997",
    "OnlineRidge": "#fd7e14",
    "NonlinearOnlineRidge": "#e83e8c",
    "FixedDelta": "#007bff",
    "AdaptiveDelta": "#6f42c1",
    "SafeAdaptiveDelta": "#28a745",
    "SelfReferential": "#ffc107",
    "SafeSelfReferential": "#004085",
    "Selective_adaptive": "#d63384",
    "Selective_fixed_high": "#495057",
    "Selective_fixed_low": "#6f42c1",
    "Selective_oracle": "#198754",
}


def plot_nonlinear_error_over_time_an(
    trajectories: Mapping[str, Sequence[float]],
    change_points: Sequence[int],
    output_path: Path,
) -> None:
    """Plot AN: Relative error over time through nonlinear regime shifts."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for name, errs in trajectories.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(
            errs,
            label=name,
            color=color,
            alpha=0.85,
            lw=1.8 if "Safe" in name or "Nonlinear" in name else 1.2,
        )

    for cp in change_points:
        ax.axvline(
            cp,
            color="#d9534f",
            linestyle="--",
            alpha=0.7,
            lw=1.2,
            label="Regime Shift" if cp == change_points[0] else "",
        )

    ax.set_title(r"Plot AN: Nonlinear Regime Error Over Time ($A \to B \to C \to A$)")
    ax.set_xlabel(r"Streaming Timestep $t$")
    ax.set_ylabel(r"Relative Error $E_{\mathrm{rel}}(t)$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_adaptation_forgetting_ao(
    models: Sequence[str],
    adapt_errors: Sequence[float],
    forget_metrics: Sequence[float],
    output_path: Path,
) -> None:
    """Plot AO: Adaptation vs Forgetting in Stale-Penalty Stream (A -> B -> A)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for m, ad, fg in zip(models, adapt_errors, forget_metrics, strict=False):
        if np.isnan(ad) or np.isnan(fg):
            continue
        color = PALETTE.get(m, "#333333")
        ax.scatter(ad, fg, s=100, color=color, zorder=4, edgecolor="black")
        ax.annotate(
            m,
            (ad, fg),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title(
        r"Plot AO: Adaptation vs. Forgetting in Stale-Penalty Stream ($A \to B \to A$)"
    )
    ax.set_xlabel(r"Phase B Adaptation Error (Lower is Better)")
    ax.set_ylabel(r"Forgetting on Re-entry to A: $E_{return} - E_{pre}$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_retention_trajectory_ap(
    retention_trajectories: Mapping[str, Sequence[float]],
    change_points: Sequence[int],
    output_path: Path,
) -> None:
    """Plot AP: Retention coefficient alpha_t over time."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.0))

    for name, alphas in retention_trajectories.items():
        color = PALETTE.get(name, "#20c997")
        ax.plot(alphas, label=name, color=color, lw=1.6)

    for cp in change_points:
        ax.axvline(
            cp,
            color="#d9534f",
            linestyle="--",
            alpha=0.7,
            lw=1.2,
            label="Regime Shift" if cp == change_points[0] else "",
        )

    ax.set_title(r"Plot AP: Retention Coefficient $\alpha_t$ Trajectory")
    ax.set_xlabel(r"Streaming Timestep $t$")
    ax.set_ylabel(r"Retention Factor $\alpha_t \in [0, 1]$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_retention_ablation_aq(
    modes: Sequence[str],
    task_a_errors: Sequence[float],
    negative_transfers: Sequence[float],
    output_path: Path,
) -> None:
    """Plot AQ: Retention modes (Fixed High, Low, Adaptive, Oracle) comparison."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    x = np.arange(len(modes))
    width = 0.35

    ax.bar(
        x - width / 2,
        task_a_errors,
        width,
        label=r"Task A Error",
        color="#0d6efd",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        negative_transfers,
        width,
        label=r"Task B Negative Transfer",
        color="#dc3545",
        alpha=0.85,
    )

    ax.set_title(r"Plot AQ: Selective Retention vs. Fixed Retention & Oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9.5)
    ax.set_ylabel(r"Relative Metric Value")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_comparison_ar(
    models: Sequence[str],
    task_a_errors: Sequence[float],
    output_path: Path,
) -> None:
    """Plot AR: DeltaCore vs Linear RLS vs Nonlinear RLS on Task A."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    clean_m, clean_e = [], []
    for m, e in zip(models, task_a_errors, strict=False):
        if not np.isnan(e):
            clean_m.append(m)
            clean_e.append(e)

    colors = [PALETTE.get(m, "#495057") for m in clean_m]
    bars = ax.bar(clean_m, clean_e, color=colors, alpha=0.85)

    for bar, val in zip(bars, clean_e, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(r"Plot AR: Model Error Comparison on Nonlinear Task A ($D=8$)")
    ax.set_ylabel(r"Mean Relative Error $E_{\mathrm{rel}}$")
    ax.set_xticks(range(len(clean_m)))
    ax.set_xticklabels(clean_m, rotation=30, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perf_vs_dim_as(
    dims: Sequence[int],
    dim_results: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Plot AS: Performance vs Dimensionality D in {8, 16, 32, 64, 128}."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, errs in dim_results.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(dims, errs, marker="o", label=name, color=color, lw=1.6)

    ax.set_title(r"Plot AS: Adaptation Performance vs. Dimensionality $D$")
    ax.set_xlabel(r"Stream Dimension $D$")
    ax.set_ylabel(r"Mean Relative Error $E_{\mathrm{rel}}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_runtime_vs_dim_at(
    dims: Sequence[int],
    runtime_results: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Plot AT: Runtime per token vs Dimensionality D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, rtimes in runtime_results.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(dims, rtimes, marker="s", label=name, color=color, lw=1.6)

    ax.set_title(
        r"Plot AT: Streaming Latency per Token ($\mu\mathrm{s}$) vs. Dimension $D$"
    )
    ax.set_xlabel(r"Stream Dimension $D$")
    ax.set_ylabel(r"Latency per Token ($\mu\mathrm{s}$)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_vs_dim_au(
    dims: Sequence[int],
    memory_results: Mapping[str, Sequence[int]],
    output_path: Path,
) -> None:
    """Plot AU: State memory footprint (bytes) vs Dimensionality D."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, mem_bytes in memory_results.items():
        color = PALETTE.get(name, "#333333")
        ax.plot(dims, mem_bytes, marker="^", label=name, color=color, lw=1.6)

    ax.set_title(r"Plot AU: Internal State Memory Footprint (Bytes) vs. Dimension $D$")
    ax.set_xlabel(r"Stream Dimension $D$")
    ax.set_ylabel(r"State Memory (Bytes)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_adaptation_efficiency_av(
    models: Sequence[str],
    efficiency_values: Sequence[float],
    output_path: Path,
) -> None:
    """Plot AV: Adaptation Efficiency = error_reduction / (adaptation_energy + eps)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    clean_m, clean_eff = [], []
    for m, eff in zip(models, efficiency_values, strict=False):
        if not np.isnan(eff) and np.isfinite(eff):
            clean_m.append(m)
            clean_eff.append(eff)

    colors = [PALETTE.get(m, "#0d6efd") for m in clean_m]
    bars = ax.bar(clean_m, clean_eff, color=colors, alpha=0.85)

    for bar, val in zip(bars, clean_eff, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(
        r"Plot AV: Adaptation Efficiency Metric ($\Delta E / (E_{\mathrm{adapt}} + \epsilon)$)"
    )
    ax.set_ylabel(r"Efficiency Ratio")
    ax.set_xticks(range(len(clean_m)))
    ax.set_xticklabels(clean_m, rotation=30, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_safe_vs_unsafe_aw(
    model_pairs: Sequence[tuple[str, str]],
    errors_unsafe: Sequence[float],
    errors_safe: Sequence[float],
    output_path: Path,
) -> None:
    """Plot AW: Safe vs Unsafe variants under nonlinear stress."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    pair_names = [f"{u}\nvs\n{s}" for u, s in model_pairs]
    x = np.arange(len(model_pairs))
    width = 0.35

    # Replace nan with placeholder height for visualization
    u_vals = [e if not np.isnan(e) else 10.0 for e in errors_unsafe]
    s_vals = [e if not np.isnan(e) else 10.0 for e in errors_safe]

    ax.bar(
        x - width / 2,
        u_vals,
        width,
        label=r"Unconstrained (Unsafe)",
        color="#dc3545",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        s_vals,
        width,
        label=r"Contractive Bounded (Safe)",
        color="#198754",
        alpha=0.85,
    )

    for i, e in enumerate(errors_unsafe):
        if np.isnan(e):
            ax.text(
                i - width / 2,
                5.0,
                "DIVERGED\n(NaN)",
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=8,
            )

    ax.set_title(r"Plot AW: Safe vs. Unsafe Variants Under Nonlinear Stress")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=9)
    ax.set_ylabel(r"Relative Error (Capped at 10.0)")
    ax.set_ylim(0, 11.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
