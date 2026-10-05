"""DeltaCore Phase 16 Observatory Publication Plots CQ through DB.

Implements all 12 diagnostic figures specified in Phase 16 Section 20:
    - Plot CQ: Error versus shift severity (plot_error_vs_severity_cq)
    - Plot CR: Eta adaptation trajectories (plot_eta_trajectories_cr)
    - Plot CS: Safety margin versus severity (plot_safety_margin_vs_severity_cs)
    - Plot CT: Continuous versus reset across shifts (plot_continuous_vs_reset_ct)
    - Plot CU: A->B->A->C retention stress trajectories (plot_retention_trajectories_cu)
    - Plot CV: D=256 empirical failure boundary map (plot_failure_boundary_map_cv)
    - Plot CW: Observation-noise robustness (plot_noise_robustness_cw)
    - Plot CX: Cross-domain robustness matrix (plot_cross_domain_matrix_cx)
    - Plot CY: Per-seed robustness differences (plot_per_seed_differences_cy)
    - Plot CZ: State norm under severe shift (plot_state_norm_severe_cz)
    - Plot DA: Adaptation energy versus severity (plot_adaptation_energy_vs_severity_da)
    - Plot DB: Resource metrics versus severity (plot_resource_metrics_vs_severity_db)
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
            "lines.linewidth": 1.5,
            "grid.alpha": 0.35,
        }
    )


PALETTE: dict[str, str] = {
    "Persistence": "#7f7f7f",
    "FrozenLinear": "#1f77b4",
    "OnlineRidge": "#ff7f0e",
    "FixedDelta": "#8c564b",
    "SafeAdaptiveDelta": "#2ca02c",
    "SpatialConv": "#e377c2",
}


def plot_error_vs_severity_cq(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CQ: Error versus shift severity across 5 shift types and key models."""
    _apply_style()
    agg = shift_matrix_data["aggregated_shift_matrix"]
    shift_types = ["mean", "variance", "temporal_speed", "noise", "combined"]
    severities = ["mild", "moderate", "severe"]
    models = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "FixedDelta",
        "SafeAdaptiveDelta",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    domains = [
        ("domain_a", "Domain A: NOAA OISST SST", axes[0]),
        ("domain_b", "Domain B: ECMWF ERA5 $T_{2m}$", axes[1]),
    ]

    for dom_key, dom_title, ax in domains:
        for m in models:
            err_by_sev = []
            for sev in severities:
                errs = []
                for st in shift_types:
                    m_data = agg.get(st, {}).get(sev, {}).get(dom_key, {}).get(m, {})
                    val = m_data.get("rel_error_mean")
                    if (
                        val is not None
                        and isinstance(val, (int, float))
                        and np.isfinite(val)
                        and val < 1000.0
                    ):
                        errs.append(float(val))
                err_by_sev.append(float(np.mean(errs)) if errs else float("nan"))

            ax.plot(
                severities,
                err_by_sev,
                "o-",
                label=m,
                color=PALETTE.get(m, "#333333"),
                linewidth=2.0 if m == "SafeAdaptiveDelta" else 1.4,
                markersize=6,
            )

        ax.set_title(dom_title, fontweight="semibold")
        ax.set_xlabel("Predetermined Shift Severity")
        ax.set_ylabel(r"Mean Relative Frobenius Error ($E_{\mathrm{rel}}$)")
        ax.legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Plot CQ: DeltaCore and Baselines Error versus Unseen Shift Severity",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_eta_trajectories_cr(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CR: Eta adaptation trajectories (eta_t over time) under shift."""
    _apply_style()
    psm = shift_matrix_data["per_seed_shift_matrix"]
    severities = ["mild", "moderate", "severe"]
    colors = {"mild": "#2ca02c", "moderate": "#ff7f0e", "severe": "#d62728"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    # Panel 1: Domain A Variance shift
    for sev in severities:
        run_data = (
            psm.get("variance", {})
            .get(sev, {})
            .get("domain_a", {})
            .get("SafeAdaptiveDelta", {})
            .get("seed_42", {})
        )
        eta_hist = run_data.get("eta_history", [])
        if eta_hist:
            ax1.plot(
                eta_hist,
                label=f"Severity: {sev}",
                color=colors[sev],
                linewidth=1.8,
            )

    ax1.axvline(
        x=80, color="gray", linestyle="--", alpha=0.7, label="Shift Onset ($t=80$)"
    )
    ax1.set_title(r"Domain A (OISST): $\eta_t$ Trajectory (Variance Shift)")
    ax1.set_xlabel("Streaming Step ($t$)")
    ax1.set_ylabel(r"Adaptive Learning Rate ($\eta_t$)")
    ax1.legend(loc="upper right", frameon=True)

    # Panel 2: Domain B Variance shift
    for sev in severities:
        run_data = (
            psm.get("variance", {})
            .get(sev, {})
            .get("domain_b", {})
            .get("SafeAdaptiveDelta", {})
            .get("seed_42", {})
        )
        eta_hist = run_data.get("eta_history", [])
        if eta_hist:
            ax2.plot(
                eta_hist,
                label=f"Severity: {sev}",
                color=colors[sev],
                linewidth=1.8,
            )

    ax2.axvline(
        x=80, color="gray", linestyle="--", alpha=0.7, label="Shift Onset ($t=80$)"
    )
    ax2.set_title(r"Domain B (ERA5): $\eta_t$ Trajectory (Variance Shift)")
    ax2.set_xlabel("Streaming Step ($t$)")
    ax2.set_ylabel(r"Adaptive Learning Rate ($\eta_t$)")
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle(
        r"Plot CR: SafeAdaptiveDelta Adaptive Step-Size ($\eta_t$) Response to Shift Onset",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_safety_margin_vs_severity_cs(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CS: Safety margin (2 - eta_t ||x_t||^2) versus shift severity."""
    _apply_style()
    agg = shift_matrix_data["aggregated_shift_matrix"]
    severities = ["mild", "moderate", "severe"]
    shift_types = ["mean", "variance", "temporal_speed", "noise", "combined"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Domain A
    for m in ["SafeAdaptiveDelta", "FixedDelta"]:
        margins = []
        for sev in severities:
            vals = []
            for st in shift_types:
                val = (
                    agg.get(st, {})
                    .get(sev, {})
                    .get("domain_a", {})
                    .get(m, {})
                    .get("min_safety_margin_mean")
                )
                if (
                    val is not None
                    and isinstance(val, (int, float))
                    and np.isfinite(val)
                ):
                    vals.append(float(val))
            margins.append(float(np.mean(vals)) if vals else float("nan"))

        ax1.plot(
            severities,
            margins,
            "o-",
            label=m,
            color=PALETTE[m],
            linewidth=2.0 if m == "SafeAdaptiveDelta" else 1.5,
            markersize=7,
        )

    ax1.axhline(
        y=0.5,
        color="red",
        linestyle=":",
        label=r"Theoretical Bound ($\rho=1.5 \Rightarrow 0.5$)",
    )
    ax1.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        alpha=0.7,
        label="Failure Boundary ($0.0$)",
    )
    ax1.set_title("Domain A (OISST): Minimum Safety Margin")
    ax1.set_xlabel("Shift Severity")
    ax1.set_ylabel(r"Minimum Safety Margin ($2.0 - \eta_t \|x_t\|_2^2$)")
    ax1.legend(loc="upper right", frameon=True)

    # Domain B
    for m in ["SafeAdaptiveDelta", "FixedDelta"]:
        margins = []
        for sev in severities:
            vals = []
            for st in shift_types:
                val = (
                    agg.get(st, {})
                    .get(sev, {})
                    .get("domain_b", {})
                    .get(m, {})
                    .get("min_safety_margin_mean")
                )
                if (
                    val is not None
                    and isinstance(val, (int, float))
                    and np.isfinite(val)
                ):
                    vals.append(float(val))
            margins.append(float(np.mean(vals)) if vals else float("nan"))

        ax2.plot(
            severities,
            margins,
            "o-",
            label=m,
            color=PALETTE[m],
            linewidth=2.0 if m == "SafeAdaptiveDelta" else 1.5,
            markersize=7,
        )

    ax2.axhline(
        y=0.5,
        color="red",
        linestyle=":",
        label=r"Theoretical Bound ($\rho=1.5 \Rightarrow 0.5$)",
    )
    ax2.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        alpha=0.7,
        label="Failure Boundary ($0.0$)",
    )
    ax2.set_title("Domain B (ERA5): Minimum Safety Margin")
    ax2.set_xlabel("Shift Severity")
    ax2.set_ylabel(r"Minimum Safety Margin ($2.0 - \eta_t \|x_t\|_2^2$)")
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Plot CS: Empirical Safety Margin Preservation across Predetermined Severities",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_continuous_vs_reset_ct(
    retention_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CT: Continuous versus reset ablation delta across shifts."""
    _apply_style()
    ra = retention_data["reset_ablation"]
    shift_types = ["mean", "variance", "temporal_speed", "noise", "combined"]
    severities = ["mild", "moderate", "severe"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    x = np.arange(len(shift_types))
    width = 0.25

    # Panel 1: Domain A
    for i, sev in enumerate(severities):
        deltas = [
            ra.get("domain_a", {}).get(st, {}).get(sev, {}).get("delta_e_rel", 0.0)
            for st in shift_types
        ]
        color = ["#74c476", "#31a354", "#006d2c"][i]
        ax1.bar(
            x + (i - 1) * width, deltas, width, label=f"Severity: {sev}", color=color
        )

    ax1.axhline(y=0.0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [st.replace("_", " ").title() for st in shift_types], rotation=20
    )
    ax1.set_title(
        r"Domain A (OISST): $\Delta_{\mathrm{reset}} = E_{\mathrm{cont}} - E_{\mathrm{reset}}$"
    )
    ax1.set_ylabel(r"Relative Error Difference ($\Delta_{\mathrm{reset}}$)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.text(
        0.03,
        0.92,
        "Negative = Continuous State Benefit\nPositive = Reset Advantage",
        transform=ax1.transAxes,
        fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Panel 2: Domain B
    for i, sev in enumerate(severities):
        deltas = [
            ra.get("domain_b", {}).get(st, {}).get(sev, {}).get("delta_e_rel", 0.0)
            for st in shift_types
        ]
        color = ["#fd8d3c", "#e6550d", "#a63603"][i]
        ax2.bar(
            x + (i - 1) * width, deltas, width, label=f"Severity: {sev}", color=color
        )

    ax2.axhline(y=0.0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [st.replace("_", " ").title() for st in shift_types], rotation=20
    )
    ax2.set_title(
        r"Domain B (ERA5): $\Delta_{\mathrm{reset}} = E_{\mathrm{cont}} - E_{\mathrm{reset}}$"
    )
    ax2.legend(loc="lower right", frameon=True)
    ax2.text(
        0.03,
        0.92,
        "Negative = Continuous State Benefit\nPositive = Reset Advantage",
        transform=ax2.transAxes,
        fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        r"Plot CT: State Persistence vs Shock-Mitigating State Reset ($\Delta_{\mathrm{reset}}$)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_retention_trajectories_cu(
    retention_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CU: A->B->A->C retention stress trajectories across 4 histories."""
    _apply_style()
    histories = retention_data["retention_stress_histories"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    coords = [
        ("A_to_B", r"$A \rightarrow B$", axes[0, 0]),
        ("A_to_B_to_A", r"$A \rightarrow B \rightarrow A$", axes[0, 1]),
        ("A_to_B_to_C", r"$A \rightarrow B \rightarrow C$", axes[1, 0]),
        (
            "A_to_severeB_to_A",
            r"$A \rightarrow \mathrm{severe\text{-}}B \rightarrow A$",
            axes[1, 1],
        ),
    ]

    for h_key, h_title, ax in coords:
        h_data = histories.get(h_key, {})
        cont_err = h_data.get("continuous_step_errors", [])
        reset_err = h_data.get("reset_step_errors", [])
        bounds = h_data.get("bounds", [])

        steps = np.arange(len(cont_err))
        ax.plot(
            steps, cont_err, label="Continuous State", color="#2ca02c", linewidth=1.7
        )
        ax.plot(
            steps,
            reset_err,
            label="Reset State at Boundaries",
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
        )

        for b in bounds:
            ax.axvline(
                x=b,
                color="black",
                linestyle=":",
                alpha=0.6,
                label="Regime Boundary" if b == bounds[0] else None,
            )

        ax.set_title(f"History: {h_title}", fontweight="semibold")
        ax.set_xlabel("Streaming Step ($t$)")
        ax.set_ylabel("Step Frobenius Error")
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle(
        r"Plot CU: Retention Stress Trajectories across Regime Switches ($A, B, C, \mathrm{severe\text{-}}B$)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_failure_boundary_map_cv(
    safety_boundary_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CV: D=256 empirical failure boundary map of FixedDelta vs SafeAdaptiveDelta."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Domain A
    da_data = safety_boundary_data["domain_a_oisst_256"]
    step_sizes = da_data["step_sizes"]
    fd_a = da_data["FixedDelta"]
    sad_a = da_data["SafeAdaptiveDelta"]

    fd_norms = [
        float(x["max_state_norm"]) if not x["diverged"] else float("nan") for x in fd_a
    ]
    sad_norms = [float(x["max_state_norm"]) for x in sad_a]

    ax1.plot(
        step_sizes,
        fd_norms,
        "s--",
        label=r"FixedDelta (Diverges $\eta \geq 0.012$)",
        color="#8c564b",
        linewidth=1.8,
        markersize=6,
    )
    ax1.plot(
        step_sizes,
        sad_norms,
        "o-",
        label="SafeAdaptiveDelta (Finite)",
        color="#2ca02c",
        linewidth=2.0,
        markersize=7,
    )

    # Mark divergence steps
    for item in fd_a:
        if item["diverged"]:
            ax1.scatter(
                [item["step_size"]], [3.0], marker="x", color="red", s=80, zorder=5
            )
            ax1.annotate(
                f"NaN\n(t={item['time_to_nonfinite']})",
                xy=(item["step_size"], 3.0),
                xytext=(item["step_size"] - 0.001, 3.2),
                fontsize=7.5,
                color="red",
                fontweight="bold",
            )

    ax1.set_title("Domain A (OISST SST, $D=256$): State Norm Boundary")
    ax1.set_xlabel(r"Baseline Step Size ($\eta$)")
    ax1.set_ylabel(r"Maximum State Norm ($\|M_t\|_F$)")
    ax1.set_ylim(0, 4.0)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Domain B
    db_data = safety_boundary_data["domain_b_era5_256"]
    fd_b = db_data["FixedDelta"]
    sad_b = db_data["SafeAdaptiveDelta"]

    fd_norms_b = [
        float(x["max_state_norm"]) if not x["diverged"] else float("nan") for x in fd_b
    ]
    sad_norms_b = [float(x["max_state_norm"]) for x in sad_b]

    ax2.plot(
        step_sizes,
        fd_norms_b,
        "s--",
        label=r"FixedDelta (Diverges $\eta \geq 0.005$)",
        color="#8c564b",
        linewidth=1.8,
        markersize=6,
    )
    ax2.plot(
        step_sizes,
        sad_norms_b,
        "o-",
        label="SafeAdaptiveDelta (Finite)",
        color="#2ca02c",
        linewidth=2.0,
        markersize=7,
    )

    for item in fd_b:
        if item["diverged"]:
            ax2.scatter(
                [item["step_size"]], [3.5], marker="x", color="red", s=80, zorder=5
            )
            ax2.annotate(
                f"NaN\n(t={item['time_to_nonfinite']})",
                xy=(item["step_size"], 3.5),
                xytext=(item["step_size"] - 0.001, 3.7),
                fontsize=7.5,
                color="red",
                fontweight="bold",
            )

    ax2.set_title(r"Domain B (ERA5 $T_{2m}$, $D=256$): State Norm Boundary")
    ax2.set_xlabel(r"Baseline Step Size ($\eta$)")
    ax2.set_ylabel(r"Maximum State Norm ($\|M_t\|_F$)")
    ax2.set_ylim(0, 4.5)
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Plot CV: D=256 Empirical Safety Boundary Mapping (FixedDelta Failure vs SafeAdaptiveDelta)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_noise_robustness_cw(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CW: Observation-noise robustness across 0%, 1%, 5%, 10% perturbations."""
    _apply_style()
    pr = shift_matrix_data["perturbation_robustness"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Domain A
    da_sad = pr["domain_a"]["SafeAdaptiveDelta"]
    da_fd = pr["domain_a"]["FixedDelta"]
    noise_pct = [x["noise_fraction"] * 100 for x in da_sad]
    sad_err_a = [x["rel_error"] for x in da_sad]
    fd_err_a = [x["rel_error"] for x in da_fd]

    ax1.plot(
        noise_pct,
        sad_err_a,
        "o-",
        label="SafeAdaptiveDelta",
        color="#2ca02c",
        linewidth=2.0,
        markersize=7,
    )
    ax1.plot(
        noise_pct,
        fd_err_a,
        "s--",
        label="FixedDelta",
        color="#8c564b",
        linewidth=1.8,
        markersize=6,
    )
    ax1.set_title("Domain A (OISST): Perturbation Robustness")
    ax1.set_xlabel("Observation Noise Perturbation (%)")
    ax1.set_ylabel(r"Relative Error ($E_{\mathrm{rel}}$)")
    ax1.legend(loc="upper left", frameon=True)

    # Domain B
    db_sad = pr["domain_b"]["SafeAdaptiveDelta"]
    sad_err_b = [x["rel_error"] for x in db_sad]

    ax2.plot(
        noise_pct,
        sad_err_b,
        "o-",
        label="SafeAdaptiveDelta",
        color="#2ca02c",
        linewidth=2.0,
        markersize=7,
    )
    ax2.set_title("Domain B (ERA5): Perturbation Robustness (SafeAdaptiveDelta)")
    ax2.set_xlabel("Observation Noise Perturbation (%)")
    ax2.set_ylabel(r"Relative Error ($E_{\mathrm{rel}}$)")
    ax2.annotate(
        "FixedDelta unstable on\nunscaled Domain B ($E > 10^4$)",
        xy=(5, 0.16),
        xytext=(2, 0.18),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d32f2f"),
    )
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle(
        r"Plot CW: Observation Perturbation Robustness ($0\%, 1\%, 5\%, 10\%$ Noise)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cross_domain_matrix_cx(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CX: Cross-domain robustness matrix comparing 5 shifts across both domains."""
    _apply_style()
    agg = shift_matrix_data["aggregated_shift_matrix"]
    shift_types = ["mean", "variance", "temporal_speed", "noise", "combined"]
    shift_labels = [
        "Mean Shift",
        "Variance Shift",
        "Temporal Speed",
        "Noise Shift",
        "Combined Shift",
    ]

    mat_err = np.zeros((len(shift_types), 2))
    mat_rec = np.zeros((len(shift_types), 2))

    for i, st in enumerate(shift_types):
        da_data = (
            agg.get(st, {})
            .get("moderate", {})
            .get("domain_a", {})
            .get("SafeAdaptiveDelta", {})
        )
        err_a = da_data.get("rel_error_mean")
        mat_err[i, 0] = (
            float(err_a) if err_a is not None and np.isfinite(err_a) else float("nan")
        )
        mat_rec[i, 0] = float(da_data.get("sustained_recovery_mean", 0.0))

        db_data = (
            agg.get(st, {})
            .get("moderate", {})
            .get("domain_b", {})
            .get("SafeAdaptiveDelta", {})
        )
        err_b = db_data.get("rel_error_mean")
        mat_err[i, 1] = (
            float(err_b) if err_b is not None and np.isfinite(err_b) else float("nan")
        )
        mat_rec[i, 1] = float(db_data.get("sustained_recovery_mean", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    im1 = ax1.imshow(mat_err, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Domain A (OISST)", "Domain B (ERA5)"], fontweight="semibold")
    ax1.set_yticks(np.arange(len(shift_types)))
    ax1.set_yticklabels(shift_labels, fontweight="medium")
    ax1.set_title(r"Relative Frobenius Error ($E_{\mathrm{rel}}$)")
    for i in range(len(shift_types)):
        for j in range(2):
            ax1.text(
                j,
                i,
                f"{mat_err[i, j]:.4f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(mat_rec * 100, cmap="Greens", aspect="auto", vmin=0, vmax=100)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Domain A (OISST)", "Domain B (ERA5)"], fontweight="semibold")
    ax2.set_yticks(np.arange(len(shift_types)))
    ax2.set_yticklabels(shift_labels, fontweight="medium")
    ax2.set_title("Sustained Recovery Rate (%)")
    for i in range(len(shift_types)):
        for j in range(2):
            ax2.text(
                j,
                i,
                f"{mat_rec[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        r"Plot CX: Cross-Domain Robustness Matrix (5 Shift Types $\times$ 2 Physical Domains)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_seed_differences_cy(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CY: Per-seed robustness differences across 5 seeds (42, 43, 44, 45, 46)."""
    _apply_style()
    psm = shift_matrix_data["per_seed_shift_matrix"]
    seeds = ["seed_42", "seed_43", "seed_44", "seed_45", "seed_46"]
    seed_labels = ["42", "43", "44", "45", "46"]
    models = ["FrozenLinear", "OnlineRidge", "SafeAdaptiveDelta"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Domain A - Moderate Combined Shift
    for m in models:
        vals = [
            psm.get("combined", {})
            .get("moderate", {})
            .get("domain_a", {})
            .get(m, {})
            .get(s, {})
            .get("rel_error", float("nan"))
            for s in seeds
        ]
        ax1.plot(
            seed_labels,
            vals,
            "o--",
            label=m,
            color=PALETTE[m],
            linewidth=1.8,
            markersize=6,
        )

    ax1.set_title("Domain A (OISST): Per-Seed Error (Combined Shift)")
    ax1.set_xlabel("Evaluation Seed")
    ax1.set_ylabel(r"Relative Error ($E_{\mathrm{rel}}$)")
    ax1.legend(loc="lower right", frameon=True)

    # Domain B - Moderate Combined Shift
    for m in models:
        vals = [
            psm.get("combined", {})
            .get("moderate", {})
            .get("domain_b", {})
            .get(m, {})
            .get(s, {})
            .get("rel_error", float("nan"))
            for s in seeds
        ]
        ax2.plot(
            seed_labels,
            vals,
            "o--",
            label=m,
            color=PALETTE[m],
            linewidth=1.8,
            markersize=6,
        )

    ax2.set_title("Domain B (ERA5): Per-Seed Error (Combined Shift)")
    ax2.set_xlabel("Evaluation Seed")
    ax2.set_ylabel(r"Relative Error ($E_{\mathrm{rel}}$)")
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Plot CY: Per-Seed Stability and Reproducibility Across 5 Deterministic Seeds",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_state_norm_severe_cz(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot CZ: State norm under severe shift over streaming steps."""
    _apply_style()
    psm = shift_matrix_data["per_seed_shift_matrix"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Domain A severe variance
    da_sad = (
        psm.get("variance", {})
        .get("severe", {})
        .get("domain_a", {})
        .get("SafeAdaptiveDelta", {})
        .get("seed_42", {})
        .get("state_norm_history", [])
    )
    da_fd = (
        psm.get("variance", {})
        .get("severe", {})
        .get("domain_a", {})
        .get("FixedDelta", {})
        .get("seed_42", {})
        .get("state_norm_history", [])
    )

    if da_sad:
        ax1.plot(da_sad, label="SafeAdaptiveDelta", color="#2ca02c", linewidth=2.0)
    if da_fd:
        ax1.plot(
            da_fd, label="FixedDelta", color="#8c564b", linewidth=1.6, linestyle="--"
        )

    ax1.axvline(x=80, color="gray", linestyle=":", label="Shift Onset ($t=80$)")
    ax1.set_title("Domain A (OISST): State Norm under Severe Variance Shift")
    ax1.set_xlabel("Streaming Step ($t$)")
    ax1.set_ylabel(r"State Frobenius Norm ($\|M_t\|_F$)")
    ax1.legend(loc="upper left", frameon=True)

    # Domain B severe variance
    db_sad = (
        psm.get("variance", {})
        .get("severe", {})
        .get("domain_b", {})
        .get("SafeAdaptiveDelta", {})
        .get("seed_42", {})
        .get("state_norm_history", [])
    )
    if db_sad:
        ax2.plot(db_sad, label="SafeAdaptiveDelta", color="#2ca02c", linewidth=2.0)

    ax2.axvline(x=80, color="gray", linestyle=":", label="Shift Onset ($t=80$)")
    ax2.set_title("Domain B (ERA5): State Norm under Severe Variance Shift")
    ax2.set_xlabel("Streaming Step ($t$)")
    ax2.set_ylabel(r"State Frobenius Norm ($\|M_t\|_F$)")
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle(
        r"Plot CZ: Memory Operator State Boundedness ($\|M_t\|_F$) under Severe Shift",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_adaptation_energy_vs_severity_da(
    shift_matrix_data: Mapping[str, Any], output_path: Path
) -> None:
    """Plot DA: Adaptation energy (sum ||Delta M_t||^2) versus shift severity."""
    _apply_style()
    agg = shift_matrix_data["aggregated_shift_matrix"]
    severities = ["mild", "moderate", "severe"]
    shift_types = ["mean", "variance", "temporal_speed", "noise", "combined"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Domain A
    for st, c in zip(shift_types, colors, strict=False):
        energies = []
        for sev in severities:
            val = (
                agg.get(st, {})
                .get(sev, {})
                .get("domain_a", {})
                .get("SafeAdaptiveDelta", {})
                .get("adaptation_energy_mean", 0.0)
            )
            energies.append(float(val) if val is not None and np.isfinite(val) else 0.0)
        ax1.plot(
            severities,
            energies,
            "o-",
            label=st.replace("_", " ").title(),
            color=c,
            linewidth=1.8,
            markersize=6,
        )

    ax1.set_title("Domain A (OISST): Adaptation Energy vs Severity")
    ax1.set_xlabel("Shift Severity")
    ax1.set_ylabel(r"Cumulative Adaptation Energy ($\sum \|\Delta M_t\|_F^2$)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Domain B
    for st, c in zip(shift_types, colors, strict=False):
        energies = []
        for sev in severities:
            val = (
                agg.get(st, {})
                .get(sev, {})
                .get("domain_b", {})
                .get("SafeAdaptiveDelta", {})
                .get("adaptation_energy_mean", 0.0)
            )
            energies.append(float(val) if val is not None and np.isfinite(val) else 0.0)
        ax2.plot(
            severities,
            energies,
            "o-",
            label=st.replace("_", " ").title(),
            color=c,
            linewidth=1.8,
            markersize=6,
        )

    ax2.set_title("Domain B (ERA5): Adaptation Energy vs Severity")
    ax2.set_xlabel("Shift Severity")
    ax2.set_ylabel(r"Cumulative Adaptation Energy ($\sum \|\Delta M_t\|_F^2$)")
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.suptitle(
        "Plot DA: Adaptation Energy Response across Predetermined Shift Types and Severities",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_resource_metrics_vs_severity_db(
    results_agg: Mapping[str, Any], output_path: Path
) -> None:
    """Plot DB: Resource metrics (step latency and persistent memory footprint)."""
    _apply_style()
    models = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "FixedDelta",
        "SafeAdaptiveDelta",
        "SpatialConv",
    ]

    latencies_a = [
        float(results_agg.get("domain_a", {}).get(m, {}).get("runtime_us_mean", 0.0))
        for m in models
    ]
    mem_a = [
        float(
            results_agg.get("domain_a", {}).get(m, {}).get("persistent_state_bytes", 0)
        )
        / 1024.0
        for m in models
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(models))
    colors = [PALETTE.get(m, "#555555") for m in models]

    # Panel 1: Latency
    bars1 = ax1.bar(
        x, latencies_a, color=colors, alpha=0.88, edgecolor="black", linewidth=0.5
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontweight="medium")
    ax1.set_ylabel(r"Per-Step Streaming Latency ($\mu$s)")
    ax1.set_title(r"Streaming Latency Overhead ($D=64$)")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.1,
            f"{yval:.1f} $\\mu$s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel 2: Memory
    bars2 = ax2.bar(
        x, mem_a, color=colors, alpha=0.88, edgecolor="black", linewidth=0.5
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontweight="medium")
    ax2.set_ylabel("Persistent State Footprint (KiB)")
    ax2.set_title(r"Persistent Online State Size ($D=64$)")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            f"{yval:.1f} KiB",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        "Plot DB: Resource Efficiency Under Online Streaming (Latency and Memory Footprint)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_phase_16_plots(
    artifacts_dir: Path | str = "docs/benchmarks/artifacts/phase_16",
) -> dict[str, Path]:
    """Generate all 12 Phase 16 Observatory plots from serialized JSON artifacts.

    Returns:
        Mapping from plot code (e.g. 'CQ', 'CR') to output image Path.
    """
    art_path = Path(artifacts_dir)
    plots_dir = art_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(art_path / "phase_16_shift_matrix.json") as f:
        shift_matrix_data = json.load(f)

    with open(art_path / "phase_16_retention.json") as f:
        retention_data = json.load(f)

    with open(art_path / "phase_16_safety_boundary.json") as f:
        safety_boundary_data = json.load(f)

    with open(art_path / "phase_16_results.json") as f:
        results_agg = json.load(f)

    generated: dict[str, Path] = {}

    p_cq = plots_dir / "plot_cq_error_vs_severity.png"
    plot_error_vs_severity_cq(shift_matrix_data, p_cq)
    generated["CQ"] = p_cq

    p_cr = plots_dir / "plot_cr_eta_trajectories.png"
    plot_eta_trajectories_cr(shift_matrix_data, p_cr)
    generated["CR"] = p_cr

    p_cs = plots_dir / "plot_cs_safety_margin_vs_severity.png"
    plot_safety_margin_vs_severity_cs(shift_matrix_data, p_cs)
    generated["CS"] = p_cs

    p_ct = plots_dir / "plot_ct_continuous_vs_reset.png"
    plot_continuous_vs_reset_ct(retention_data, p_ct)
    generated["CT"] = p_ct

    p_cu = plots_dir / "plot_cu_retention_trajectories.png"
    plot_retention_trajectories_cu(retention_data, p_cu)
    generated["CU"] = p_cu

    p_cv = plots_dir / "plot_cv_failure_boundary_map.png"
    plot_failure_boundary_map_cv(safety_boundary_data, p_cv)
    generated["CV"] = p_cv

    p_cw = plots_dir / "plot_cw_noise_robustness.png"
    plot_noise_robustness_cw(shift_matrix_data, p_cw)
    generated["CW"] = p_cw

    p_cx = plots_dir / "plot_cx_cross_domain_matrix.png"
    plot_cross_domain_matrix_cx(shift_matrix_data, p_cx)
    generated["CX"] = p_cx

    p_cy = plots_dir / "plot_cy_per_seed_differences.png"
    plot_per_seed_differences_cy(shift_matrix_data, p_cy)
    generated["CY"] = p_cy

    p_cz = plots_dir / "plot_cz_state_norm_severe.png"
    plot_state_norm_severe_cz(shift_matrix_data, p_cz)
    generated["CZ"] = p_cz

    p_da = plots_dir / "plot_da_adaptation_energy_vs_severity.png"
    plot_adaptation_energy_vs_severity_da(shift_matrix_data, p_da)
    generated["DA"] = p_da

    p_db = plots_dir / "plot_db_resource_metrics_vs_severity.png"
    plot_resource_metrics_vs_severity_db(results_agg, p_db)
    generated["DB"] = p_db

    print(f"[Observatory] All 12 Phase 16 plots successfully rendered to {plots_dir}.")
    return generated
