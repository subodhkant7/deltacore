"""Observatory Plotting Extension for Phase 10: Streaming Adaptive-State Benchmark.

Generates publication-quality figures:
    - Plot AD: Error over time through regime transitions (with change point indicators)
    - Plot AE: First-passage and sustained-recovery distributions
    - Plot AF: Adaptation vs. Forgetting in A -> B -> A experiments
    - Plot AG: Performance vs. temporal delay (d in {16, 64, 256})
    - Plot AH: Adaptive-state norm trajectories
    - Plot AI: Adaptive-state update energy comparison
    - Plot AJ: Task performance vs. parameter count (capacity Pareto frontier)
    - Plot AK: Adaptive state ON vs. OFF ablation
    - Plot AL: Continuous state vs. Reset state at shift
    - Plot AM: Runtime per timestep (microseconds/token)
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
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "lines.linewidth": 1.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_error_over_time_ad(
    trajectories: dict[str, list[float]],
    change_points: list[int],
    output_path: Path | str,
) -> Path:
    """Plot AD: Error over time through regime transitions."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    colors = [
        "#2b5c8f",
        "#e6550d",
        "#31a354",
        "#756bb1",
        "#636363",
        "#bcbddc",
        "#fdae6b",
    ]
    for i, (name, errs) in enumerate(trajectories.items()):
        color = colors[i % len(colors)]
        ax.plot(errs, label=name, color=color, alpha=0.85)

    for cp in change_points:
        ax.axvline(
            cp,
            color="#de2d26",
            linestyle="--",
            alpha=0.7,
            label="Regime Switch" if cp == change_points[0] else "",
        )

    ax.set_title("Plot AD: Step Relative Error Over Time Across Regime Transitions")
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Step Error $E_t$")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_recovery_distributions_ae(
    models: Sequence[str],
    first_passage: Sequence[float],
    sustained: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Plot AE: First-passage and sustained-recovery distributions."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        first_passage,
        width,
        label="First-Passage Recovery ($t_{first}$)",
        color="#3182bd",
    )
    ax.bar(
        x + width / 2,
        sustained,
        width,
        label="Sustained Recovery ($t_{sust}$, K=10)",
        color="#e6550d",
    )

    ax.set_title("Plot AE: First-Passage and Sustained Recovery Steps by Model")
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Recovery Steps (lower is faster)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_adaptation_vs_forgetting_af(
    models: Sequence[str],
    adaptation_errors: Sequence[float],
    forgetting_metrics: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Plot AF: Adaptation vs. Forgetting in A -> B -> A experiments."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for i, model in enumerate(models):
        ax.scatter(
            adaptation_errors[i], forgetting_metrics[i], s=120, label=model, alpha=0.9
        )
        ax.annotate(
            model,
            (adaptation_errors[i], forgetting_metrics[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_title(
        "Plot AF: Online Adaptation Error vs. Catastrophic Forgetting (A -> B -> A)"
    )
    ax.set_xlabel("Adaptation Error on Regime B (lower is better)")
    ax.set_ylabel("Forgetting Metric upon Return to A (lower is better)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_performance_vs_delay_ag(
    delays: Sequence[int],
    delay_results: Mapping[str, Sequence[float]],
    output_path: Path | str,
) -> Path:
    """Plot AG: Performance vs. temporal delay in Task B."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for name, errs in delay_results.items():
        ax.plot(delays, errs, marker="o", label=name, linewidth=2)

    ax.set_title("Plot AG: Retrieval Error vs. Temporal Delay $d$ (Task B)")
    ax.set_xlabel("Temporal Delay $d$ (timesteps)")
    ax.set_ylabel("Retrieval Relative Error")
    ax.set_xscale("log", base=2)
    ax.set_xticks(delays)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_state_norm_trajectories_ah(
    trajectories: dict[str, list[float]],
    output_path: Path | str,
) -> Path:
    """Plot AH: Adaptive-state norm trajectories."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    for name, norms in trajectories.items():
        ax.plot(norms, label=name, alpha=0.85)

    ax.set_title(r"Plot AH: Adaptive State Norm $\|S_t\|_F$ Trajectories")
    ax.set_xlabel(r"Timestep $t$")
    ax.set_ylabel(r"State Frobenius Norm $\|S_t\|_F$")
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_adaptation_energy_ai(
    models: Sequence[str],
    energies: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Plot AI: Adaptive-state update energy."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    colors = [
        "#2b5c8f",
        "#41b6c4",
        "#238443",
        "#78c679",
        "#feb24c",
        "#e31a1c",
        "#800026",
    ]
    bars = ax.bar(
        models, energies, color=[colors[i % len(colors)] for i in range(len(models))]
    )

    ax.set_title(
        r"Plot AI: Total Adaptation Energy $E_{adapt} = \sum_t \|\Delta S_t\|_F^2$"
    )
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Cumulative Energy (Frobenius)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + max(energies) * 0.01,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_performance_vs_params_aj(
    params: Sequence[int],
    errors: Sequence[float],
    labels: Sequence[str],
    output_path: Path | str,
) -> Path:
    """Plot AJ: Task performance vs. parameter count (Pareto frontier)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.scatter(params, errors, s=100, color="#1f78b4", alpha=0.85)
    for p, e, lbl in zip(params, errors, labels, strict=False):
        ax.annotate(lbl, (p, e), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_title("Plot AJ: Online Adaptation Error vs. Trainable Parameter Count")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Mean Relative Error (lower is better)")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_adaptive_on_vs_off_ak(
    models: Sequence[str],
    on_errors: Sequence[float],
    off_errors: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Plot AK: Adaptive state ON vs. OFF ablation."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        on_errors,
        width,
        label="Adaptation ON (Dynamic State)",
        color="#2ca02c",
    )
    ax.bar(
        x + width / 2,
        off_errors,
        width,
        label="Adaptation OFF (Frozen State)",
        color="#d62728",
    )

    ax.set_title("Plot AK: Primary Causal Control — Adaptive State ON vs. OFF")
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Mean Relative Error")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_continuous_vs_reset_al(
    models: Sequence[str],
    continuous_errors: Sequence[float],
    reset_errors: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Plot AL: Continuous state vs. Reset state at shift."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        continuous_errors,
        width,
        label="Continuous Retained State",
        color="#1f77b4",
    )
    ax.bar(
        x + width / 2,
        reset_errors,
        width,
        label="State Reset at Shift",
        color="#ff7f0e",
    )

    ax.set_title("Plot AL: Retained Memory State — Continuous vs. Shift Reset")
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Mean Relative Error")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_runtime_per_timestep_am(
    models: Sequence[str],
    us_per_token: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Plot AM: Runtime per timestep (microseconds/token)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    bars = ax.bar(models, us_per_token, color="#41b6c4")
    ax.set_title(r"Plot AM: Streaming Inference Latency Per Timestep ($\mu$s/token)")
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel(r"Latency ($\mu$s / token)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + max(us_per_token) * 0.01,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
