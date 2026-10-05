# ==============================================================================
# DeltaCore: deltacore/observatory/report.py
# Machine-readable and human-readable Markdown reproducibility reports.
# ==============================================================================

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from deltacore.observatory.analysis import (
    analyze_shift_response,
    analyze_stability_events,
)
from deltacore.observatory.events import extract_events
from deltacore.observatory.fingerprint import generate_fingerprint
from deltacore.observatory.replay import ReplayResult
from deltacore.observatory.schema import StateTrajectory


def generate_observatory_report(
    trajectory: StateTrajectory,
    replay_result: ReplayResult | None = None,
    plot_paths: Mapping[str, Path | str] | None = None,
    output_path: Path | str | None = None,
) -> str:
    """Generate a comprehensive Markdown reproducibility report for a neural-state trajectory.

    Guarantees:
        1. Fully reproducible provenance (seeds, system, versions).
        2. Discrete temporal events with zero causal assertions.
        3. Strict missing-data and failure visibility.
        4. Explicit statement of limitations and epistemic boundaries.
        5. Zero automated qualitative winner labels.
    """
    shift_res = analyze_shift_response(trajectory)
    stab_res = analyze_stability_events(trajectory)
    fp = generate_fingerprint(trajectory)
    events = extract_events(trajectory)

    lines: list[str] = []
    lines.append(
        f"# DeltaCore State Observatory Report: {trajectory.model} on {trajectory.task}"
    )
    lines.append("")
    lines.append("## 1. Experiment Metadata & Provenance")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Model / Baseline** | `{trajectory.model}` |")
    lines.append(f"| **Benchmark Task** | `{trajectory.task}` |")
    lines.append(f"| **Random Seed** | `{trajectory.seed}` |")
    lines.append(f"| **Sequence Length** | {len(trajectory.steps)} |")
    lines.append(
        f"| **All States Finite** | `{'YES' if stab_res.all_states_finite else 'NO (DIVERGED)'}` |"
    )
    lines.append(f"| **Python Version** | `{sys.version.split()[0]}` |")
    lines.append(f"| **PyTorch Version** | `{torch.__version__}` |")
    lines.append(f"| **Platform / OS** | `{platform.platform()}` |")
    lines.append(
        f"| **Hardware Device** | `{trajectory.metadata.get('device', 'cpu')}` |"
    )
    lines.append(
        f"| **Floating-Point Precision** | `{trajectory.metadata.get('dtype', 'float32')}` |"
    )
    lines.append("")

    lines.append("## 2. Trajectory Summary & Fingerprint")
    lines.append("")
    lines.append("| Metric Axis | Value | Formal Definition / Units |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(
        f"| Mean Error $\\bar{{E}}$ | {f'{fp.mean_error:.4f}' if fp.mean_error is not None else 'N/A'} | Arithmetic mean of $||e_t||_2$ |"
    )
    lines.append(
        f"| Final Error $E_T$ | {f'{fp.final_error:.4f}' if fp.final_error is not None else 'N/A'} | Sequence terminal $||e_T||_2$ |"
    )
    lines.append(
        f"| Recovery Latency $T_{{\\text{{FP}}}}$ | {fp.recovery_latency if fp.recovery_latency is not None else 'N/A'} | Steps to 50% shock error reduction |"
    )
    lines.append(
        f"| Sustained Recovery $T_{{\\text{{sust}}}}$ | {fp.sustained_recovery if fp.sustained_recovery is not None else 'N/A'} | Consecutive window error maintenance |"
    )
    lines.append(
        f"| Mean Step Size $\\bar{{\\eta}}$ | {f'{fp.mean_eta:.4f}' if fp.mean_eta is not None else 'N/A (Frozen)'} | Mean adaptive rate $\\eta_t$ |"
    )
    lines.append(
        f"| Step Size Variance | {f'{fp.eta_variance:.6f}' if fp.eta_variance is not None else 'N/A (Frozen)'} | Sample variance $\\text{{Var}}(\\eta_t)$ |"
    )
    lines.append(
        f"| Total Update Energy $U_M$ | {f'{fp.update_energy:.4f}' if fp.update_energy is not None else 'N/A'} | Cumulative $\\sum ||\\Delta M_t||_F$ |"
    )
    lines.append(
        f"| Peak Update $\\max ||\\Delta M_t||_F$ | {f'{fp.max_update:.4f}' if fp.max_update is not None else 'N/A'} | Peak single-step state displacement |"
    )
    lines.append(
        f"| Max State Norm $\\max ||M_t||_F$ | {f'{fp.max_state_norm:.4f}' if fp.max_state_norm is not None else 'N/A'} | Peak state Frobenius norm |"
    )
    lines.append(
        f"| Min Stability Margin $\\min S_t$ | {f'{fp.min_stability_margin:.4f}' if fp.min_stability_margin is not None else 'N/A (Frozen)'} | Minimum $2 - \\eta_t ||k_t||^2$ |"
    )
    lines.append(
        f"| Divergence Step $t^*$ | {fp.failure_step if fp.failure_step is not None else 'None (Finite)'} | First non-finite step index |"
    )
    lines.append("")

    lines.append("## 3. Detected Trajectory Events")
    lines.append("")
    if events:
        lines.append("| Step | Event Type | Value | Context |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for ev in events:
            val_str = (
                f"{ev.value:.4f}"
                if isinstance(ev.value, (int, float))
                else str(ev.value)
            )
            ctx_str = (
                ", ".join(f"{k}={v}" for k, v in ev.context.items())
                if ev.context
                else "—"
            )
            lines.append(f"| {ev.step} | `{ev.event_type}` | {val_str} | {ctx_str} |")
    else:
        lines.append("*No discrete threshold or failure events detected.*")
    lines.append("")

    lines.append("## 4. Adaptation Response & Distribution Shift")
    lines.append("")
    lines.append("| Metric | Value | Interpretation Rule |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(
        f"| Pre-Shift Error | {f'{shift_res.pre_shift_error:.4f}' if shift_res.pre_shift_error is not None else 'N/A'} | Baseline accuracy prior to perturbation |"
    )
    lines.append(
        f"| Shock Error | {f'{shift_res.shock_error:.4f}' if shift_res.shock_error is not None else 'N/A'} | Immediate post-perturbation error |"
    )
    lines.append(
        f"| Recovery Slope | {f'{shift_res.recovery_slope:.4f}' if shift_res.recovery_slope is not None else 'N/A'} | Empirical error reduction rate |"
    )
    lines.append(
        f"| Post-Shift Energy | {f'{shift_res.update_energy_after_shift:.4f}' if shift_res.update_energy_after_shift is not None else 'N/A'} | Update budget expended post-shock |"
    )
    lines.append(
        f"| State Growth Ratio | {f'{shift_res.state_growth_ratio:.4f}' if shift_res.state_growth_ratio is not None else 'N/A'} | Relative state expansion $G_M$ |"
    )
    lines.append("")

    lines.append("## 5. Stability & Contraction Invariants")
    lines.append("")
    lines.append("| Invariant Property | Assessment | Contractive Bound |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(
        f"| Content Margin $\\min S_t$ | {f'{stab_res.minimum_content_margin:.4f}' if stab_res.minimum_content_margin is not None else 'N/A'} | Contractive if $S_t \\ge 0$ |"
    )
    lines.append(
        f"| Dynamics Margin $\\min S_t^{{\\text{{dyn}}}}$ | {f'{stab_res.minimum_dynamics_margin:.4f}' if stab_res.minimum_dynamics_margin is not None else 'N/A'} | Contractive if $S_t^{{\\text{{dyn}}}} \\ge 0$ |"
    )
    lines.append(
        f"| Max Normalized Step $\\max \\gamma_t$ | {f'{stab_res.maximum_normalized_step:.4f}' if stab_res.maximum_normalized_step is not None else 'N/A'} | Non-expansive if $\\le 2.0$ |"
    )
    lines.append(
        f"| Active Clip Count | {stab_res.clip_count} | Steps where safety controller intervened |"
    )
    lines.append(
        f"| Terminal State Finite | `{stab_res.terminal_state_finite}` | Strict finiteness check at $t=T-1$ |"
    )
    lines.append("")

    lines.append("## 6. Replay & Provenance Verification")
    lines.append("")
    if replay_result is not None:
        lines.append(f"- **Replay Status**: `{replay_result.status.upper()}`")
        lines.append(
            f"- **Max Absolute Metric Difference**: `{replay_result.max_absolute_diff:.2e}`"
        )
        lines.append(
            f"- **Max Relative Metric Difference**: `{replay_result.max_relative_diff:.2e}`"
        )
        if replay_result.error_message:
            lines.append(
                f"- **Replay Diagnostic Error**: {replay_result.error_message}"
            )
    else:
        lines.append("*Replay was not executed for this report.*")
    lines.append("")

    if plot_paths:
        lines.append("## 7. Generated Visual Artifacts")
        lines.append("")
        for name, p in plot_paths.items():
            lines.append(
                f"- **{name.replace('_', ' ').title()}**: [{Path(p).name}]({p})"
            )
        lines.append("")

    lines.append("## 8. Epistemic Guardrails & Scientific Limitations")
    lines.append("")
    lines.append(
        "1. **Absence of Proof from Finite Sequences**: Numerical survival over the tested sequence length does not constitute mathematical proof of global Lyapunov stability. Bounds verified here are local sufficient conditions."
    )
    lines.append(
        "2. **Determinism Boundaries**: Exact numerical bit-for-bit equivalence is only guaranteed within identical hardware architecture, PyTorch build, and seed. Cross-platform floating point divergence is expected."
    )
    lines.append(
        "3. **Non-Causal Event Attribution**: Temporal event ordering (e.g. update peak preceded recovery) indicates correlation and sequence order, not proven causal necessity."
    )
    lines.append(
        "4. **No Automated Winner Designations**: Performance involves multi-objective trade-offs. Fast adaptation generally expends higher state update energy and operates with lower stability margins."
    )
    lines.append("")

    content = "\n".join(lines)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return content
