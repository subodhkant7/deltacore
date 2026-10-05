# ==============================================================================
# DeltaCore: deltacore/benchmarks/reporting/visualization.py
# Optional plotting utilities for benchmark trajectories and distributions.
# Matplotlib is lazily imported and NOT a hard runtime dependency.
# ==============================================================================

from pathlib import Path
from typing import Any


def _ensure_matplotlib() -> Any:
    """Lazily import matplotlib.pyplot or raise an informative ImportError."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for DeltaCore visualization utilities. "
            "Install it via `pip install matplotlib`."
        ) from e


def plot_error_vs_time(
    trajectories: dict[str, list[float]],
    shift_position: int | None = None,
    save_path: str | Path | None = None,
    title: str = "Prediction Error vs. Time",
) -> None:
    """Plot Euclidean prediction error trajectories over time for multiple models."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model_name, errors in trajectories.items():
        ax.plot(errors, label=model_name, linewidth=1.5)

    if shift_position is not None:
        ax.axvline(
            shift_position,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Shift (t={shift_position})",
        )

    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Error Norm ||e_t||")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_state_norm_vs_time(
    state_norms: dict[str, list[float]],
    save_path: str | Path | None = None,
    title: str = "Memory Frobenius Norm vs. Time",
) -> None:
    """Plot associative memory Frobenius norm trajectories ||M_t||_F."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model_name, norms in state_norms.items():
        ax.plot(norms, label=model_name, linewidth=1.5)

    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Memory Norm ||M_t||_F")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_key_correlation_vs_error(
    correlations: list[float],
    errors_by_model: dict[str, list[float]],
    save_path: str | Path | None = None,
    title: str = "Retrieval Error vs. Key Correlation (Cosine Similarity)",
) -> None:
    """Plot cross-talk retrieval error as a function of key cosine similarity."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model_name, errs in errors_by_model.items():
        ax.plot(correlations, errs, marker="o", label=model_name, linewidth=1.5)

    ax.set_xlabel("Key Cosine Similarity")
    ax.set_ylabel("Mean Retrieval Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_stability_margin_distribution(
    margins_by_model: dict[str, list[float]],
    save_path: str | Path | None = None,
    title: str = "Stability Margin Distribution S_t = 2 - eta ||k||^2",
) -> None:
    """Plot histogram or KDE of stability margins across models."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model_name, margins in margins_by_model.items():
        ax.hist(margins, bins=20, alpha=0.5, label=model_name)

    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Non-expansion Boundary (S_t=0)",
    )
    ax.set_xlabel("Stability Margin S_t")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
