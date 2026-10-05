# ==============================================================================
# DeltaCore: deltacore/observatory/plots.py
# Static publication-quality plots for neural-state trajectories and benchmarks.
# ==============================================================================

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from deltacore.observatory.events import extract_events
from deltacore.observatory.schema import StateTrajectory


def _get_plt():
    """Lazily import matplotlib with Agg headless backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _apply_annotations(
    ax: Any,
    trajectory: StateTrajectory,
    annotate_shift: bool = True,
    annotate_recovery: bool = True,
    annotate_nonfinite: bool = True,
) -> None:
    """Optionally overlay discrete events on temporal axis without modifying data."""
    events = extract_events(trajectory)
    _, labels = ax.get_legend_handles_labels()

    for ev in events:
        if ev.event_type == "shift_boundary" and annotate_shift:
            ax.axvline(
                x=ev.step,
                color="black",
                linestyle="--",
                alpha=0.7,
                label="SHIFT" if "SHIFT" not in labels else None,
            )
            labels.append("SHIFT")
            ax.text(
                ev.step,
                ax.get_ylim()[1] * 0.95,
                " SHIFT",
                fontsize=8,
                verticalalignment="top",
                fontweight="bold",
            )
        elif ev.event_type == "first_threshold_crossing" and annotate_recovery:
            ax.axvline(
                x=ev.step,
                color="green",
                linestyle=":",
                alpha=0.7,
                label="FIRST RECOVERY" if "FIRST RECOVERY" not in labels else None,
            )
            labels.append("FIRST RECOVERY")
            ax.text(
                ev.step,
                ax.get_ylim()[1] * 0.85,
                " FIRST REC",
                fontsize=8,
                verticalalignment="top",
                color="green",
            )
        elif ev.event_type == "sustained_recovery" and annotate_recovery:
            ax.axvline(
                x=ev.step,
                color="teal",
                linestyle="-.",
                alpha=0.7,
                label="SUSTAINED RECOVERY"
                if "SUSTAINED RECOVERY" not in labels
                else None,
            )
            labels.append("SUSTAINED RECOVERY")
            ax.text(
                ev.step,
                ax.get_ylim()[1] * 0.75,
                " SUSTAINED",
                fontsize=8,
                verticalalignment="top",
                color="teal",
            )
        elif ev.event_type == "first_nonfinite_state" and annotate_nonfinite:
            ax.axvline(
                x=ev.step,
                color="red",
                linestyle="-",
                linewidth=1.5,
                label="FIRST NONFINITE" if "FIRST NONFINITE" not in labels else None,
            )
            labels.append("FIRST NONFINITE")
            ax.text(
                ev.step,
                ax.get_ylim()[1] * 0.90,
                f" DIVERGED (step {ev.step})",
                fontsize=8,
                verticalalignment="top",
                color="red",
                fontweight="bold",
            )
            # Shade post-failure region as unavailable
            ax.axvspan(ev.step, len(trajectory.steps) - 1, color="red", alpha=0.1)


# ------------------------------------------------------------------------------
# Core Plots A through I
# ------------------------------------------------------------------------------


def plot_error_vs_time(
    trajectory: StateTrajectory,
    output_path: Path | str | None = None,
    annotate: bool = True,
) -> Any:
    """Plot A: Prediction error norm ||e_t|| vs sequence step t."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    steps = [s.step for s in trajectory.steps]
    errs = [
        s.error_norm if s.error_norm is not None else float("nan")
        for s in trajectory.steps
    ]

    ax.plot(steps, errs, marker="o", markersize=3, label=f"{trajectory.model} Error")
    ax.set_xlabel("Sequence Step (t)")
    ax.set_ylabel("Error Norm $||e_t||_2$")
    ax.set_title(f"Plot A: Error vs Time ({trajectory.model}, Task: {trajectory.task})")
    ax.grid(True, linestyle=":", alpha=0.6)

    if annotate:
        _apply_annotations(ax, trajectory)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_step_size_vs_time(
    trajectory: StateTrajectory,
    output_path: Path | str | None = None,
    annotate: bool = True,
) -> Any:
    """Plot B: Adaptive learning rate eta_t vs sequence step t."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    steps = [s.step for s in trajectory.steps]
    etas = [
        s.step_size if s.step_size is not None else float("nan")
        for s in trajectory.steps
    ]

    if all(math.isnan(x) for x in etas):
        ax.text(
            0.5,
            0.5,
            "Step size is not applicable (model is non-adaptive/frozen)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    else:
        ax.plot(
            steps, etas, marker="s", markersize=3, color="orange", label=r"$\eta_t$"
        )

    ax.set_xlabel("Sequence Step (t)")
    ax.set_ylabel(r"Step Size $\eta_t$")
    ax.set_title(f"Plot B: Step Size vs Time ({trajectory.model})")
    ax.grid(True, linestyle=":", alpha=0.6)

    if annotate:
        _apply_annotations(ax, trajectory)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_memory_norm_vs_time(
    trajectory: StateTrajectory,
    output_path: Path | str | None = None,
    annotate: bool = True,
) -> Any:
    """Plot C: Memory Frobenius norm ||M_t||_F vs sequence step t."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    steps = [s.step for s in trajectory.steps]
    mem_norms = [
        s.memory_norm if s.memory_norm is not None else float("nan")
        for s in trajectory.steps
    ]
    dyn_norms = [
        s.dynamics_memory_norm if s.dynamics_memory_norm is not None else float("nan")
        for s in trajectory.steps
    ]

    ax.plot(
        steps,
        mem_norms,
        marker="^",
        markersize=3,
        color="blue",
        label=r"Content $||M_t||_F$",
    )
    if not all(math.isnan(x) for x in dyn_norms):
        ax.plot(
            steps,
            dyn_norms,
            marker="v",
            markersize=3,
            color="purple",
            linestyle="--",
            label=r"Dynamics $||C_t||_F$",
        )

    ax.set_xlabel("Sequence Step (t)")
    ax.set_ylabel("Frobenius Norm")
    ax.set_title(f"Plot C: Memory Norm vs Time ({trajectory.model})")
    ax.grid(True, linestyle=":", alpha=0.6)

    if annotate:
        _apply_annotations(ax, trajectory)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_update_norm_vs_time(
    trajectory: StateTrajectory,
    output_path: Path | str | None = None,
    annotate: bool = True,
) -> Any:
    """Plot D: Update Frobenius norm ||Delta M_t||_F vs sequence step t."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    steps = [s.step for s in trajectory.steps]
    upds = [
        s.update_norm if s.update_norm is not None else float("nan")
        for s in trajectory.steps
    ]
    dyn_upds = [
        s.dynamics_update_norm if s.dynamics_update_norm is not None else float("nan")
        for s in trajectory.steps
    ]

    ax.plot(
        steps,
        upds,
        marker="d",
        markersize=3,
        color="green",
        label=r"Content $||\Delta M_t||_F$",
    )
    if not all(math.isnan(x) for x in dyn_upds):
        ax.plot(
            steps,
            dyn_upds,
            marker="x",
            markersize=3,
            color="crimson",
            linestyle="--",
            label=r"Dynamics $||\Delta C_t||_F$",
        )

    ax.set_xlabel("Sequence Step (t)")
    ax.set_ylabel("Update Norm")
    ax.set_title(f"Plot D: Update Norm vs Time ({trajectory.model})")
    ax.grid(True, linestyle=":", alpha=0.6)

    if annotate:
        _apply_annotations(ax, trajectory)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_stability_margin_vs_time(
    trajectory: StateTrajectory,
    output_path: Path | str | None = None,
    annotate: bool = True,
) -> Any:
    """Plot E: Stability margin S_t = 2 - gamma_t vs sequence step t."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    steps = [s.step for s in trajectory.steps]
    margins = [
        s.stability_margin if s.stability_margin is not None else float("nan")
        for s in trajectory.steps
    ]

    if all(math.isnan(x) for x in margins):
        ax.text(
            0.5,
            0.5,
            "Stability margin is not applicable (model is non-adaptive/frozen)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    else:
        ax.plot(
            steps,
            margins,
            marker="x",
            markersize=3,
            color="teal",
            label=r"$S_t = 2 - \eta_t ||k_t||^2$",
        )
        ax.axhline(
            0.0,
            color="red",
            linestyle=":",
            linewidth=1.2,
            label="Contractive Threshold ($S_t = 0$)",
        )

    ax.set_xlabel("Sequence Step (t)")
    ax.set_ylabel("Stability Margin $S_t$")
    ax.set_title(f"Plot E: Stability Margin vs Time ({trajectory.model})")
    ax.grid(True, linestyle=":", alpha=0.6)

    if annotate:
        _apply_annotations(ax, trajectory)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_comparative_recovery_curves(
    trajectories: list[StateTrajectory],
    output_path: Path | str | None = None,
) -> Any:
    """Plot F: Comparative error recovery curves across multiple models."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(9, 5))

    for traj in trajectories:
        steps = [s.step for s in traj.steps]
        errs = [
            s.error_norm if s.error_norm is not None else float("nan")
            for s in traj.steps
        ]
        ax.plot(steps, errs, label=traj.model, linewidth=1.5)

        # Indicate failure step if non-finite
        if not traj.all_states_finite and traj.first_nonfinite_step is not None:
            ax.scatter(
                [traj.first_nonfinite_step],
                [
                    errs[traj.first_nonfinite_step]
                    if traj.first_nonfinite_step < len(errs)
                    and not math.isnan(errs[traj.first_nonfinite_step])
                    else 0.0
                ],
                marker="x",
                s=50,
                color="red",
            )

    ax.set_xlabel("Sequence Step (t)")
    ax.set_ylabel("Error Norm $||e_t||_2$")
    ax.set_title("Plot F: Comparative Recovery Curves Across Systems")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_key_correlation_vs_error(
    rhos: list[float],
    errors: list[float],
    output_path: Path | str | None = None,
) -> Any:
    """Plot G: Key correlation rho vs mean retrieval error bar/line chart."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        rhos,
        errors,
        marker="o",
        color="darkblue",
        linewidth=2.0,
        label="Observed Retrieval Error",
    )
    ax.set_xlabel(r"Key Cosine Similarity $\rho = \langle k_1, k_2 \rangle$")
    ax.set_ylabel("Mean Retrieval Error $\\bar{E}$")
    ax.set_title(r"Plot G: Key Correlation $\rho$ vs. Cross-Talk Retrieval Error")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_recovery_latency_vs_energy(
    models: list[str],
    latencies: list[int | None],
    energies: list[float],
    output_path: Path | str | None = None,
) -> Any:
    """Plot H: First-passage recovery latency T_FP vs cumulative update energy U_rec."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for m, lat, en in zip(models, latencies, energies, strict=False):
        if lat is not None:
            ax.scatter([lat], [en], s=80, label=m)
            ax.annotate(f" {m}", (lat, en), fontsize=8)
        else:
            # Indicate unrecovered run
            ax.axhline(en, linestyle=":", alpha=0.4)

    ax.set_xlabel("Recovery Latency $T_{\\text{FP}}$ (steps to 50% error reduction)")
    ax.set_ylabel(
        "Cumulative Update Energy $U_{\\text{rec}} = \\sum ||\\Delta M_t||_F$"
    )
    ax.set_title("Plot H: Recovery Latency vs. Update Energy Expenditure")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_failure_step_distribution(
    model_failure_steps: dict[str, list[int | None]],
    total_steps: int = 50,
    output_path: Path | str | None = None,
) -> Any:
    """Plot I: Histogram or bar distribution of failure steps across configurations."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    data_to_plot = []
    labels = []
    for model_name, steps in model_failure_steps.items():
        # Filter None (finite survival)
        fails = [s for s in steps if s is not None]
        data_to_plot.append(fails)
        labels.append(f"{model_name} (n_fail={len(fails)})")

    if any(len(d) > 0 for d in data_to_plot):
        ax.hist(
            data_to_plot, bins=range(0, total_steps + 5, 5), label=labels, alpha=0.7
        )
    else:
        ax.text(
            0.5,
            0.5,
            "100% Survival: Zero Numerical Failures Observed",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="green",
            fontweight="bold",
        )

    ax.set_xlabel("Failure Step Index $t^*$")
    ax.set_ylabel("Frequency of Divergence")
    ax.set_title("Plot I: Numerical Failure Step Distribution")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def generate_all_plots(
    trajectory: StateTrajectory,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Generate and save standard temporal plots (Plots A through E) for a single trajectory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["plot_a"] = plot_error_vs_time(
        trajectory, out_dir / "plot_a_error_vs_time.png"
    )
    paths["plot_b"] = plot_step_size_vs_time(
        trajectory, out_dir / "plot_b_step_size_vs_time.png"
    )
    paths["plot_c"] = plot_memory_norm_vs_time(
        trajectory, out_dir / "plot_c_memory_norm_vs_time.png"
    )
    paths["plot_d"] = plot_update_norm_vs_time(
        trajectory, out_dir / "plot_d_update_norm_vs_time.png"
    )
    paths["plot_e"] = plot_stability_margin_vs_time(
        trajectory, out_dir / "plot_e_stability_margin_vs_time.png"
    )
    return paths


def _to_2d_numpy(tensor_or_array: Any) -> Any:
    """Normalize input tensor or ndarray to a 2D spatial numpy array."""
    import numpy as np
    import torch

    if hasattr(tensor_or_array, "detach"):
        t = tensor_or_array.detach().cpu()
        if t.ndim == 4:
            # [B, C, H, W] -> take channel norm of first batch element
            arr = torch.linalg.norm(t[0], dim=0).numpy()
        elif t.ndim == 3:
            # [C, H, W] -> channel norm
            arr = torch.linalg.norm(t, dim=0).numpy()
        elif t.ndim == 2:
            arr = t.numpy()
        else:
            arr = t.flatten().numpy()
        return arr
    elif isinstance(tensor_or_array, np.ndarray):
        arr = tensor_or_array
        if arr.ndim == 4:
            return np.linalg.norm(arr[0], axis=0)
        elif arr.ndim == 3:
            return np.linalg.norm(arr, axis=0)
        return arr
    return np.array(tensor_or_array)


def plot_directional_output_magnitude(
    directional_outputs: dict[str, Any],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot J: 2x2 spatial heatmaps of directional output magnitude $\|Y_r(h, w)\|$."""
    plt = _get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    routes = [
        ("RIGHT", r"RIGHT ($\rightarrow$)"),
        ("LEFT", r"LEFT ($\leftarrow$)"),
        ("DOWN", r"DOWN ($\downarrow$)"),
        ("UP", r"UP ($\uparrow$)"),
    ]

    for idx, (key, label) in enumerate(routes):
        ax = axes[idx // 2, idx % 2]
        if key in directional_outputs:
            grid = _to_2d_numpy(directional_outputs[key])
            im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Width $w$")
        ax.set_ylabel("Height $h$")

    fig.suptitle(
        r"Plot J: Directional Output Magnitude $\|Y_r(h, w)\|$",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_directional_learning_rate_map(
    directional_lr_maps: dict[str, Any],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot K: 2x2 spatial heatmaps of directional learning-rate distribution $\eta_r(h, w)$."""
    plt = _get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    routes = [
        ("RIGHT", r"RIGHT ($\rightarrow$)"),
        ("LEFT", r"LEFT ($\leftarrow$)"),
        ("DOWN", r"DOWN ($\downarrow$)"),
        ("UP", r"UP ($\uparrow$)"),
    ]

    for idx, (key, label) in enumerate(routes):
        ax = axes[idx // 2, idx % 2]
        if key in directional_lr_maps:
            grid = _to_2d_numpy(directional_lr_maps[key])
            im = ax.imshow(grid, cmap="magma", interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Width $w$")
        ax.set_ylabel("Height $h$")

    fig.suptitle(
        r"Plot K: Directional Learning-Rate Map $\eta_r(h, w)$",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_directional_memory_activity_map(
    directional_activity_maps: dict[str, Any],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot L: 2x2 spatial heatmaps of directional memory-update activity $\|\Delta W_r(h, w)\|$."""
    plt = _get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    routes = [
        ("RIGHT", r"RIGHT ($\rightarrow$)"),
        ("LEFT", r"LEFT ($\leftarrow$)"),
        ("DOWN", r"DOWN ($\downarrow$)"),
        ("UP", r"UP ($\uparrow$)"),
    ]

    for idx, (key, label) in enumerate(routes):
        ax = axes[idx // 2, idx % 2]
        if key in directional_activity_maps:
            grid = _to_2d_numpy(directional_activity_maps[key])
            im = ax.imshow(grid, cmap="inferno", interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Width $w$")
        ax.set_ylabel("Height $h$")

    fig.suptitle(
        r"Plot L: Directional Memory-Activity Map $\|\Delta W_r(h, w)\|$",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_fused_spatial_output(
    fused_output: Any,
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot M: Heatmap of fused multi-directional spatial feature map $\|Y(h, w)\|$."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = _to_2d_numpy(fused_output)
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        r"Plot M: Fused Spatial Output $\|Y(h, w)\|$", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Width $w$")
    ax.set_ylabel("Height $h$")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_chunk_size_difference(
    differences: dict[Any, float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot N: Trajectory divergence and error vs chunk size $C$."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [f"C={k}" if str(k).isdigit() else str(k) for k in differences.keys()]
    values = list(differences.values())

    ax.bar(labels, values, color="steelblue", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Chunk Size $C$")
    ax.set_ylabel(r"Divergence from Token-Indexed Reference $\|Y_C - Y_{C=1}\|$")
    ax.set_title(
        "Plot N: Boundary-Refresh Chunk Size Sensitivity",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_train_val_loss(
    train_losses: list[float],
    val_losses: list[float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot O: Training vs validation loss trajectory across optimization steps."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = list(range(1, len(train_losses) + 1))

    ax.plot(steps, train_losses, label="Train Loss", color="royalblue", linewidth=1.8)
    if val_losses:
        val_steps = [
            int(i * (len(train_losses) / len(val_losses)))
            for i in range(1, len(val_losses) + 1)
        ]
        ax.plot(
            val_steps,
            val_losses,
            label="Validation Loss",
            color="crimson",
            linewidth=1.8,
            linestyle="--",
        )

    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("Loss")
    ax.set_title(
        "Plot O: Training vs Validation Loss Curve", fontsize=11, fontweight="bold"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_accuracy_vs_directions(
    direction_results: dict[str, float],
    metric_label: str = "Accuracy (%)",
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot P: Performance metric vs direction count / configuration."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(direction_results.keys())
    values = list(direction_results.values())

    bars = ax.bar(labels, values, color="teal", alpha=0.85, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Directional Configuration")
    ax.set_ylabel(metric_label)
    ax.set_title(
        "Plot P: Task Performance vs Directional Configuration",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_validation_error_vs_chunk_size(
    chunk_errors: dict[Any, float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot Q: Validation error vs chunk size C."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [f"C={k}" if str(k).isdigit() else str(k) for k in chunk_errors.keys()]
    values = list(chunk_errors.values())

    ax.plot(labels, values, marker="o", color="darkorange", linewidth=2.0, markersize=6)
    ax.set_xlabel("Chunk Size $C$")
    ax.set_ylabel("Validation Error")
    ax.set_title(
        "Plot Q: Validation Error vs Boundary Chunk Size",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_runtime_vs_chunk_size(
    chunk_runtimes: dict[Any, float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot R: Execution runtime (ms) vs chunk size C."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [f"C={k}" if str(k).isdigit() else str(k) for k in chunk_runtimes.keys()]
    values = list(chunk_runtimes.values())

    ax.bar(labels, values, color="mediumpurple", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Chunk Size $C$")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(
        "Plot R: CPU Execution Runtime vs Chunk Size", fontsize=11, fontweight="bold"
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_metric_vs_directional_route(
    route_metrics: dict[str, float],
    metric_name: str = "Validation Error",
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot S: Metric comparison across individual and paired routes."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(route_metrics.keys())
    values = list(route_metrics.values())

    ax.bar(labels, values, color="cornflowerblue", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Route Specification")
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Plot S: {metric_name} Across Directional Routes",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_absolute_vs_relative_error(
    models: list[str],
    abs_errors: list[float],
    rel_errors: list[float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot T: Dual-axis comparison of absolute vs relative error."""
    import numpy as np

    plt = _get_plt()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(
        x - width / 2,
        abs_errors,
        width,
        label=r"Absolute Error $\|Y - \hat{Y}\|_F$",
        color="steelblue",
        alpha=0.85,
    )
    ax1.set_xlabel("Model / Baseline")
    ax1.set_ylabel(r"Absolute Frobenius Error $\|Y - \hat{Y}\|_F$", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2,
        rel_errors,
        width,
        label=r"Relative Error $E_{\text{rel}}$",
        color="indianred",
        alpha=0.85,
    )
    ax2.set_ylabel(r"Relative Error $E_{\text{rel}}$", color="indianred")
    ax2.tick_params(axis="y", labelcolor="indianred")

    ax1.set_title(
        "Plot T: Absolute vs Relative Reconstruction Error Across Models",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_task_a_rel_error_by_model(
    models: list[str],
    rel_errors: list[float],
    errors_std: list[float] | None = None,
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot U: Task A validation relative error by model."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(models))
    yerr = errors_std if errors_std is not None else None
    ax.bar(
        x,
        rel_errors,
        yerr=yerr,
        capsize=4,
        color="royalblue",
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(r"Relative Error $E_{\text{rel}}$")
    ax.set_title(
        "Plot U: Task A Validation Relative Error Across Models",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_task_a_train_val_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot V: Task A train/validation curves for principal DeltaCore variants."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["royalblue", "darkorange", "forestgreen", "crimson", "purple"]
    for idx, (name, (tr, val)) in enumerate(curves.items()):
        c = colors[idx % len(colors)]
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr, linestyle="--", alpha=0.6, color=c, label=f"{name} (Train)")
        ax.plot(
            epochs, val, linestyle="-", linewidth=2.0, color=c, label=f"{name} (Val)"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error Loss")
    ax.set_title(
        "Plot V: Task A Optimization Trajectories for Principal Variants",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_coordinate_effect(
    conditions: list[str],
    rel_errors: list[float],
    errors_std: list[float] | None = None,
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot W: Task A spatial-address ablation (coordinate effects)."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = range(len(conditions))
    yerr = errors_std if errors_std is not None else None
    colors = ["salmon", "cornflowerblue", "sandybrown"]
    ax.bar(
        x,
        rel_errors,
        yerr=yerr,
        capsize=4,
        color=colors[: len(conditions)],
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel(r"Validation Relative Error $E_{\text{rel}}$")
    ax.set_title(
        "Plot W: Task A Spatial-Address Ablation (Coordinate Effects)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_task_b_acc_vs_params(
    models: list[str],
    accuracies: list[float],
    param_counts: list[int],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot X: Task B accuracy by model and parameter count."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(models))
    bars = ax.bar(x, accuracies, color="teal", alpha=0.85, edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(
        "Plot X: Task B Classification Accuracy and Parameter Count",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Annotate parameter count on top of bars
    for bar, params in zip(bars, param_counts, strict=False):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1.0,
            f"{params}p",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_translation_stress(
    models: list[str],
    orig_acc: list[float],
    trans_acc: list[float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot Y: Task B accuracy before and after translation stress."""
    import numpy as np

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        orig_acc,
        width,
        label="Original Patterns",
        color="steelblue",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        trans_acc,
        width,
        label="Translated Patterns (Stress)",
        color="coral",
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(
        "Plot Y: Task B Accuracy Before vs After Translation Stress",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_pixel_shuffle_control(
    models: list[str],
    orig_acc: list[float],
    shuff_acc: list[float],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot Z: Task B original vs pixel-shuffled accuracy."""
    import numpy as np

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        orig_acc,
        width,
        label="Original 2D Layout",
        color="forestgreen",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        shuff_acc,
        width,
        label="Pixel-Shuffled Control",
        color="indianred",
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(
        "Plot Z: Task B Original vs Pixel-Shuffled Spatial Sensitivity",
        fontsize=11,
        fontweight="bold",
    )
    ax.axhline(
        y=16.67, color="gray", linestyle="--", alpha=0.7, label="Chance (16.67%)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_directional_accuracy_points(
    routes: list[str],
    seed_points: dict[str, list[float]],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot AA: Directional accuracy comparison with per-seed points."""
    import numpy as np

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    x_indices = np.arange(len(routes))
    means = [np.mean(seed_points[r]) for r in routes]
    stds = [np.std(seed_points[r]) for r in routes]

    # Bar with error bars
    ax.bar(
        x_indices,
        means,
        yerr=stds,
        capsize=4,
        color="skyblue",
        alpha=0.6,
        edgecolor="steelblue",
    )

    # Scatter individual seeds
    for idx, r in enumerate(routes):
        points = seed_points[r]
        jitter = np.linspace(-0.15, 0.15, len(points))
        ax.scatter(
            [idx + j for j in jitter],
            points,
            color="navy",
            s=28,
            zorder=5,
            alpha=0.85,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(routes, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Plot AA: Directional Accuracy with Individual Seed Observations",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_state_and_gradient_norms(
    grad_norms: list[float],
    state_norms: dict[str, list[float]],
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot AB: Trainable-state norms and gradient norms over training."""
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    epochs = range(1, len(grad_norms) + 1)
    ax1.plot(
        epochs, grad_norms, color="crimson", marker="o", markersize=4, linewidth=1.5
    )
    ax1.set_ylabel(r"Gradient Norm $\|g\|_2$")
    ax1.set_title(
        "Plot AB: Training Diagnostics (Gradient and State Norm Dynamics)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)

    for state_name, norms in state_norms.items():
        ax2.plot(epochs, norms, label=state_name, linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Frobenius Norm $\|M\|_F$")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig


def plot_performance_vs_parameters(
    param_counts: list[int],
    metrics: list[float],
    labels: list[str],
    metric_label: str = "Validation Accuracy (%)",
    output_path: str | Path | None = None,
) -> Any:
    r"""Plot AC: Task performance vs parameter count."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        param_counts, metrics, color="darkviolet", s=80, zorder=5, edgecolor="black"
    )

    for p, m, lbl in zip(param_counts, metrics, labels, strict=False):
        ax.annotate(
            lbl,
            (p, m),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Trainable Parameter Count")
    ax.set_ylabel(metric_label)
    ax.set_title(
        "Plot AC: Model Capacity vs Task Performance", fontsize=11, fontweight="bold"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    return fig
