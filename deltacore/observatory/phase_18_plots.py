"""Phase 18 Observatory Publication Plots.

Generates all 18 required publication figures for Phase 18:
    1. plot_18_task_transfer_overview.png
    2. plot_18_boundary_rotation.png
    3. plot_18_boundary_translation.png
    4. plot_18_nonlinear_boundary.png
    5. plot_18_abrupt_vs_gradual.png
    6. plot_18_persistent_vs_reset.png
    7. plot_18_state_on_off.png
    8. plot_18_recovery_curves.png
    9. plot_18_cumulative_excess_loss.png
    10. plot_18_state_norm.png
    11. plot_18_adaptation_energy.png
    12. plot_18_safety_margin.png
    13. plot_18_scaling_accuracy.png
    14. plot_18_scaling_latency.png
    15. plot_18_scaling_memory.png
    16. plot_18_label_shuffle.png
    17. plot_18_feature_permutation.png
    18. plot_18_failure_cases.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(
        "/Users/urjasoft/.gemini/antigravity-ide/brain/6c485406-eeab-4ae3-9fa7-892b0159db3c/scratch/.mpl"
    )
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

PALETTE: dict[str, str] = {
    "FrozenLinear": "#1f77b4",
    "AdaptiveStateOFF": "#7f7f7f",
    "OnlineLogisticRegression": "#ff7f0e",
    "OnlineRidge": "#d62728",
    "OnlineMulticlassLinear": "#9467bd",
    "FixedDelta": "#bcbd22",
    "SafeAdaptiveDelta": "#2ca02c",
}


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


def generate_all_phase_18_plots(
    artifacts_dir: str | Path = "docs/benchmarks/artifacts/phase_18",
    plots_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all 18 publication-quality Observatory plots from serialized JSON artifacts.

    Args:
        artifacts_dir: Path to directory containing Phase 18 JSON artifacts.
        plots_dir: Destination directory for PNG plots.

    Returns:
        List of paths to generated PNG plot files.
    """
    _apply_style()
    art_path = Path(artifacts_dir)
    save_path = Path(plots_dir) if plots_dir else (art_path / "plots")
    save_path.mkdir(parents=True, exist_ok=True)

    with open(art_path / "phase_18_results.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(art_path / "phase_18_retention.json", encoding="utf-8") as f:
        retention = json.load(f)
    with open(art_path / "phase_18_scaling.json", encoding="utf-8") as f:
        scaling = json.load(f)
    with open(art_path / "phase_18_controls.json", encoding="utf-8") as f:
        controls = json.load(f)

    generated: list[Path] = []

    # --------------------------------------------------------------------------
    # 1. plot_18_task_transfer_overview.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    tasks = [
        "family_a_rotation",
        "family_b_translation",
        "family_c_nonlinear",
        "shift_covariate",
        "shift_prior",
    ]
    task_labels = [
        "Family A\n(Rotation)",
        "Family B\n(Translation)",
        "Family C\n(Nonlinear)",
        "Covariate\nShift",
        "Class Prior\nImbalance",
    ]
    models = [
        "FrozenLinear",
        "OnlineRidge",
        "FixedDelta",
        "SafeAdaptiveDelta",
        "AdaptiveStateOFF",
    ]
    x = np.arange(len(tasks))
    width = 0.16

    for idx, m_name in enumerate(models):
        accs = [results[t][m_name]["accuracy_mean"] * 100 for t in tasks]
        errs = [results[t][m_name]["accuracy_std"] * 100 for t in tasks]
        ax.bar(
            x + (idx - 2) * width,
            accs,
            width,
            yerr=errs,
            capsize=3,
            label=m_name,
            color=PALETTE.get(m_name, "#333333"),
            alpha=0.9,
        )

    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title(
        "Phase 18 Task Transfer Matrix Across Diverse Non-Stationary Environments"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylim(0, 100)
    ax.axhline(
        100.0 / 6,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Chance Level (16.7%)",
    )
    ax.legend(loc="lower right", framealpha=0.9)
    p1 = save_path / "plot_18_task_transfer_overview.png"
    fig.savefig(p1)
    plt.close(fig)
    generated.append(p1)

    # --------------------------------------------------------------------------
    # 2. plot_18_boundary_rotation.png (Family A)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m_keys = ["FrozenLinear", "AdaptiveStateOFF", "OnlineRidge", "SafeAdaptiveDelta"]
    f_a = results["family_a_rotation"]
    x_pos = np.arange(len(m_keys))
    means = [f_a[m]["accuracy_mean"] * 100 for m in m_keys]
    stds = [f_a[m]["accuracy_std"] * 100 for m in m_keys]
    ret_means = [f_a[m]["return_regime_accuracy_mean"] * 100 for m in m_keys]

    ax.bar(
        x_pos - 0.18,
        means,
        width=0.35,
        yerr=stds,
        capsize=4,
        label="Overall Stream Accuracy",
        color=[PALETTE[m] for m in m_keys],
        alpha=0.85,
    )
    ax.bar(
        x_pos + 0.18,
        ret_means,
        width=0.35,
        label="Return-Regime Accuracy",
        color=[PALETTE[m] for m in m_keys],
        hatch="//",
        alpha=0.6,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_keys, rotation=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Family A: Linear Decision Boundary Rotation (A -> C -> A)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    p2 = save_path / "plot_18_boundary_rotation.png"
    fig.savefig(p2)
    plt.close(fig)
    generated.append(p2)

    # --------------------------------------------------------------------------
    # 3. plot_18_boundary_translation.png (Family B)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    f_b = results["family_b_translation"]
    means_b = [f_b[m]["accuracy_mean"] * 100 for m in m_keys]
    stds_b = [f_b[m]["accuracy_std"] * 100 for m in m_keys]
    post_b = [f_b[m]["post_shift_accuracy_mean"] * 100 for m in m_keys]

    ax.bar(
        x_pos - 0.18,
        means_b,
        width=0.35,
        yerr=stds_b,
        capsize=4,
        label="Overall Stream Accuracy",
        color=[PALETTE[m] for m in m_keys],
        alpha=0.85,
    )
    ax.bar(
        x_pos + 0.18,
        post_b,
        width=0.35,
        label="Post-Shift Immediate Accuracy",
        color=[PALETTE[m] for m in m_keys],
        hatch="..",
        alpha=0.6,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_keys, rotation=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Family B: Decision Boundary Intercept Translation (A -> B_trans -> A)"
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    p3 = save_path / "plot_18_boundary_translation.png"
    fig.savefig(p3)
    plt.close(fig)
    generated.append(p3)

    # --------------------------------------------------------------------------
    # 4. plot_18_nonlinear_boundary.png (Family C)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    f_c = results["family_c_nonlinear"]
    means_c = [f_c[m]["accuracy_mean"] * 100 for m in m_keys]
    stds_c = [f_c[m]["accuracy_std"] * 100 for m in m_keys]

    ax.bar(
        x_pos,
        means_c,
        width=0.5,
        yerr=stds_c,
        capsize=4,
        color=[PALETTE[m] for m in m_keys],
        alpha=0.85,
    )
    ax.axhline(
        65.0,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="Linear Representation Ceiling (~65%)",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_keys, rotation=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Family C: Nonlinear Decision Boundary Deformation (Representation Ceiling)"
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    p4 = save_path / "plot_18_nonlinear_boundary.png"
    fig.savefig(p4)
    plt.close(fig)
    generated.append(p4)

    # --------------------------------------------------------------------------
    # 5. plot_18_abrupt_vs_gradual.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comp_models = [
        "FrozenLinear",
        "OnlineRidge",
        "SafeAdaptiveDelta",
        "AdaptiveStateOFF",
    ]
    abrupt_accs = [
        results["family_a_rotation"][m]["accuracy_mean"] * 100 for m in comp_models
    ]
    gradual_accs = [
        results["shift_gradual"][m]["accuracy_mean"] * 100 for m in comp_models
    ]
    x_comp = np.arange(len(comp_models))
    w_comp = 0.35

    ax.bar(
        x_comp - w_comp / 2,
        abrupt_accs,
        w_comp,
        label="Abrupt Boundary Shift",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x_comp + w_comp / 2,
        gradual_accs,
        w_comp,
        label="Gradual Continuous Drift",
        color="#2ca02c",
        alpha=0.85,
    )
    ax.set_xticks(x_comp)
    ax.set_xticklabels(comp_models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Adaptation Performance: Abrupt Transition vs Gradual Drift")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    p5 = save_path / "plot_18_abrupt_vs_gradual.png"
    fig.savefig(p5)
    plt.close(fig)
    generated.append(p5)

    # --------------------------------------------------------------------------
    # 6. plot_18_persistent_vs_reset.png (Causal Ablation)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    abl_tasks = ["family_a_rotation", "family_b_translation", "shift_strong_mismatch"]
    abl_labels = [
        "Family A (Rotation)\nA -> C -> A",
        "Family B (Translation)\nA -> B -> A",
        "Strong Mismatch\nA -> B (Adversarial)",
    ]
    c_vals = [retention[t]["continuous_acc_mean"] * 100 for t in abl_tasks]
    r_vals = [retention[t]["reset_acc_mean"] * 100 for t in abl_tasks]
    x_abl = np.arange(len(abl_tasks))
    w_abl = 0.35

    ax.bar(
        x_abl - w_abl / 2,
        c_vals,
        w_abl,
        label="Continuous Persistent State",
        color="#2ca02c",
        alpha=0.85,
    )
    ax.bar(
        x_abl + w_abl / 2,
        r_vals,
        w_abl,
        label="Oracle Reset at Boundary",
        color="#d62728",
        alpha=0.85,
    )
    ax.set_xticks(x_abl)
    ax.set_xticklabels(abl_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Causal Ablation: Continuous Persistent State vs Boundary Reset")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    p6 = save_path / "plot_18_persistent_vs_reset.png"
    fig.savefig(p6)
    plt.close(fig)
    generated.append(p6)

    # --------------------------------------------------------------------------
    # 7. plot_18_state_on_off.png (State ON vs State OFF)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    eval_tasks = [
        "family_a_rotation",
        "family_b_translation",
        "family_c_nonlinear",
        "shift_covariate",
        "shift_gradual",
        "shift_prior",
    ]
    eval_labels = [
        "Fam A (Rot)",
        "Fam B (Trans)",
        "Fam C (Nonlin)",
        "Covariate",
        "Gradual",
        "Prior",
    ]
    on_vals = [
        results[t]["SafeAdaptiveDelta"]["accuracy_mean"] * 100 for t in eval_tasks
    ]
    off_vals = [
        results[t]["AdaptiveStateOFF"]["accuracy_mean"] * 100 for t in eval_tasks
    ]
    x_ev = np.arange(len(eval_tasks))
    w_ev = 0.35

    ax.bar(
        x_ev - w_ev / 2,
        on_vals,
        w_ev,
        label="SafeAdaptiveDelta (State ON, M_t != 0)",
        color="#2ca02c",
        alpha=0.85,
    )
    ax.bar(
        x_ev + w_ev / 2,
        off_vals,
        w_ev,
        label="AdaptiveStateOFF (State OFF, M_t == 0)",
        color="#7f7f7f",
        alpha=0.85,
    )
    ax.set_xticks(x_ev)
    ax.set_xticklabels(eval_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Primary Causal Ablation: Associative State ON vs Clamped OFF")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    p7 = save_path / "plot_18_state_on_off.png"
    fig.savefig(p7)
    plt.close(fig)
    generated.append(p7)

    # --------------------------------------------------------------------------
    # 8. plot_18_recovery_curves.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rec_models = [
        "SafeAdaptiveDelta",
        "FixedDelta",
        "OnlineRidge",
        "OnlineLogisticRegression",
    ]
    rec_tasks = ["family_a_rotation", "family_b_translation", "shift_gradual"]
    x_rec = np.arange(len(rec_tasks))
    w_rec = 0.20

    for idx, m in enumerate(rec_models):
        passages = [results[t][m]["first_passage_recovery_mean"] for t in rec_tasks]
        ax.bar(
            x_rec + (idx - 1.5) * w_rec,
            passages,
            w_rec,
            label=m,
            color=PALETTE[m],
            alpha=0.85,
        )

    ax.set_xticks(x_rec)
    ax.set_xticklabels(
        ["Family A (Rotation)", "Family B (Translation)", "Gradual Drift"]
    )
    ax.set_ylabel("First-Passage Recovery Steps (Lower is Faster)")
    ax.set_title("Shift Recovery Dynamics: First-Passage Step Latency")
    ax.legend(loc="upper right")
    p8 = save_path / "plot_18_recovery_curves.png"
    fig.savefig(p8)
    plt.close(fig)
    generated.append(p8)

    # --------------------------------------------------------------------------
    # 9. plot_18_cumulative_excess_loss.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, m in enumerate(rec_models):
        excess = [results[t][m]["cumulative_excess_loss_mean"] for t in rec_tasks]
        ax.bar(
            x_rec + (idx - 1.5) * w_rec,
            excess,
            w_rec,
            label=m,
            color=PALETTE[m],
            alpha=0.85,
        )

    ax.set_xticks(x_rec)
    ax.set_xticklabels(
        ["Family A (Rotation)", "Family B (Translation)", "Gradual Drift"]
    )
    ax.set_ylabel("Cumulative Excess Loss (Nats, Lower is Better)")
    ax.set_title("Cumulative Excess Loss Incurred During Non-Stationary Regimes")
    ax.legend(loc="upper left")
    p9 = save_path / "plot_18_cumulative_excess_loss.png"
    fig.savefig(p9)
    plt.close(fig)
    generated.append(p9)

    # --------------------------------------------------------------------------
    # 10. plot_18_state_norm.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    state_models = ["SafeAdaptiveDelta", "FixedDelta"]
    t_list = [
        "family_a_rotation",
        "family_b_translation",
        "family_c_nonlinear",
        "shift_covariate",
    ]
    x_sn = np.arange(len(t_list))
    w_sn = 0.35

    for idx, m in enumerate(state_models):
        norms = [results[t][m]["max_state_norm_mean"] for t in t_list]
        stds = [results[t][m]["max_state_norm_std"] for t in t_list]
        ax.bar(
            x_sn + (idx - 0.5) * w_sn,
            norms,
            w_sn,
            yerr=stds,
            capsize=4,
            label=m,
            color=PALETTE[m],
            alpha=0.85,
        )

    ax.set_xticks(x_sn)
    ax.set_xticklabels(
        ["Family A (Rot)", "Family B (Trans)", "Family C (Nonlin)", "Covariate"]
    )
    ax.set_ylabel(r"Maximum State Norm $\|M_t\|_F$")
    ax.set_title("Associative State Norm Growth Envelope Across Task Families")
    ax.legend(loc="upper right")
    p10 = save_path / "plot_18_state_norm.png"
    fig.savefig(p10)
    plt.close(fig)
    generated.append(p10)

    # --------------------------------------------------------------------------
    # 11. plot_18_adaptation_energy.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    energies_safe = [
        results[t]["SafeAdaptiveDelta"]["adaptation_energy_mean"] for t in t_list
    ]
    energies_fixed = [
        results[t]["FixedDelta"]["adaptation_energy_mean"] for t in t_list
    ]

    ax.bar(
        x_sn - w_sn / 2,
        energies_safe,
        w_sn,
        label="SafeAdaptiveDelta",
        color="#2ca02c",
        alpha=0.85,
    )
    ax.bar(
        x_sn + w_sn / 2,
        energies_fixed,
        w_sn,
        label="FixedDelta",
        color="#bcbd22",
        alpha=0.85,
    )
    ax.set_xticks(x_sn)
    ax.set_xticklabels(
        ["Family A (Rot)", "Family B (Trans)", "Family C (Nonlin)", "Covariate"]
    )
    ax.set_ylabel(r"Total Adaptation Energy $\sum \|\Delta M_t\|_F^2$")
    ax.set_title(
        "Total Adaptation Energy Dissipated Across Non-Stationary Environments"
    )
    ax.legend(loc="upper right")
    p11 = save_path / "plot_18_adaptation_energy.png"
    fig.savefig(p11)
    plt.close(fig)
    generated.append(p11)

    # --------------------------------------------------------------------------
    # 12. plot_18_safety_margin.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    safeties = [
        results[t]["SafeAdaptiveDelta"]["min_safety_margin_mean"] for t in t_list
    ]
    ax.bar(
        x_sn,
        safeties,
        width=0.45,
        color="#2ca02c",
        alpha=0.85,
        label="SafeAdaptiveDelta Margin",
    )
    ax.axhline(
        0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="Contraction Bound Violation (Margin = 0)",
    )
    ax.set_xticks(x_sn)
    ax.set_xticklabels(
        ["Family A (Rot)", "Family B (Trans)", "Family C (Nonlin)", "Covariate"]
    )
    ax.set_ylabel(r"Minimum Safety Margin $1 - \frac{\eta_t \|x_t\|^2}{\rho}$")
    ax.set_title("Strict Contraction Safety Margin Across Task Families")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right")
    p12 = save_path / "plot_18_safety_margin.png"
    fig.savefig(p12)
    plt.close(fig)
    generated.append(p12)

    # --------------------------------------------------------------------------
    # 13. plot_18_scaling_accuracy.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    dims = [s["dim"] for s in scaling]
    acc_means = [s["accuracy_mean"] * 100 for s in scaling]
    acc_stds = [s["accuracy_std"] * 100 for s in scaling]

    ax.errorbar(
        dims,
        acc_means,
        yerr=acc_stds,
        fmt="o-",
        color="#2ca02c",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="SafeAdaptiveDelta",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Feature Dimension D")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title(
        "Dimensional Scaling: Classification Accuracy vs Dimension D in {32, 64, 128, 256}"
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    p13 = save_path / "plot_18_scaling_accuracy.png"
    fig.savefig(p13)
    plt.close(fig)
    generated.append(p13)

    # --------------------------------------------------------------------------
    # 14. plot_18_scaling_latency.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    lats = [s["latency_us_mean"] for s in scaling]
    lat_stds = [s["latency_us_std"] for s in scaling]

    ax.errorbar(
        dims,
        lats,
        yerr=lat_stds,
        fmt="s-",
        color="#1f77b4",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="Latency per Sample (µs)",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Feature Dimension D")
    ax.set_ylabel("Streaming Latency per Sample (µs, Log Scale)")
    ax.set_title("Runtime Scaling: Online Step Latency vs Feature Dimension D")
    ax.legend(loc="upper left")
    p14 = save_path / "plot_18_scaling_latency.png"
    fig.savefig(p14)
    plt.close(fig)
    generated.append(p14)

    # --------------------------------------------------------------------------
    # 15. plot_18_scaling_memory.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bytes_vals = [s["persistent_state_bytes"] for s in scaling]
    kb_vals = [b / 1024.0 for b in bytes_vals]

    ax.plot(
        dims,
        kb_vals,
        "D-",
        color="#d62728",
        linewidth=2,
        markersize=7,
        label=r"Persistent State: $4D^2$ bytes (FP32)",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Feature Dimension D")
    ax.set_ylabel("Persistent State Memory (KB, Log Scale)")
    ax.set_title("Memory Scaling: Exact Persistent State Footprint 4D^2 Bytes")
    ax.legend(loc="upper left")
    p15 = save_path / "plot_18_scaling_memory.png"
    fig.savefig(p15)
    plt.close(fig)
    generated.append(p15)

    # --------------------------------------------------------------------------
    # 16. plot_18_label_shuffle.png (Negative Control)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    shuf = controls["label_shuffle"]
    true_acc = results["family_a_rotation"]["SafeAdaptiveDelta"]["accuracy_mean"] * 100
    shuf_acc = shuf["mean_accuracy"] * 100
    chance_acc = shuf["chance_theoretical"] * 100

    ax.bar(
        ["True Target Stream", "Label-Shuffled Stream (Control)"],
        [true_acc, shuf_acc],
        width=0.45,
        color=["#2ca02c", "#d62728"],
        alpha=0.85,
    )
    ax.axhline(
        chance_acc,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Theoretical Chance (1/K = {chance_acc:.1f}%)",
    )
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Negative Control: Label Shuffle Destroys Online Adaptation")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    p16 = save_path / "plot_18_label_shuffle.png"
    fig.savefig(p16)
    plt.close(fig)
    generated.append(p16)

    # --------------------------------------------------------------------------
    # 17. plot_18_feature_permutation.png (Permutation Control)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    perm = controls["feature_permutation_invariance"]
    diffs = [d * 100 for d in perm["per_seed_absolute_differences"]]
    seeds_list = [f"Seed {42 + i}" for i in range(len(diffs))]

    ax.bar(seeds_list, diffs, width=0.45, color="#1f77b4", alpha=0.85)
    ax.set_ylabel("Absolute Accuracy Difference (%)")
    ax.set_title(
        "Coordinate Permutation Control: Performance Invariance (|Delta| < 1e-4)"
    )
    ax.set_ylim(0, 0.05)
    p17 = save_path / "plot_18_feature_permutation.png"
    fig.savefig(p17)
    plt.close(fig)
    generated.append(p17)

    # --------------------------------------------------------------------------
    # 18. plot_18_failure_cases.png (Scientific Falsification Summary)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fail_labels = [
        "Adversarial Mismatch\n(Stale State Penalty)",
        "Nonlinear Deformation\n(Representation Ceiling)",
        "Class Prior Shift\n(vs Online Ridge)",
    ]
    # Penalties or underperformances
    pen_mismatch = (
        abs(retention["shift_strong_mismatch"]["delta_persistent_vs_reset"]) * 100
    )
    pen_nonlin = (
        1.0 - results["family_c_nonlinear"]["SafeAdaptiveDelta"]["accuracy_mean"]
    ) * 100
    pen_prior = (
        max(
            0.0,
            (
                results["shift_prior"]["OnlineRidge"]["accuracy_mean"]
                - results["shift_prior"]["SafeAdaptiveDelta"]["accuracy_mean"]
            ),
        )
        * 100
    )
    penalties = [pen_mismatch, pen_nonlin, pen_prior]

    ax.bar(
        fail_labels,
        penalties,
        width=0.45,
        color=["#d62728", "#ff7f0e", "#9467bd"],
        alpha=0.85,
    )
    ax.set_ylabel("Performance Penalty / Gap (%)")
    ax.set_title(
        "Scientific Falsification: Identified Failure Modes & Architectural Ceilings"
    )
    ax.set_ylim(0, 60)
    for idx, v in enumerate(penalties):
        ax.text(idx, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    p18 = save_path / "plot_18_failure_cases.png"
    fig.savefig(p18)
    plt.close(fig)
    generated.append(p18)

    return generated
