# ==============================================================================
# DeltaCore: deltacore/observatory/fingerprint.py
# Compact state-transition fingerprint and multi-axial comparison.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from deltacore.observatory.analysis import (
    analyze_shift_response,
    analyze_stability_events,
)
from deltacore.observatory.schema import StateTrajectory


@dataclass(frozen=True)
class StateTransitionFingerprint:
    """Compact numerical descriptor of a neural-state trajectory.

    Captures adaptation speed, energy expenditure, stability margins, and survival.
    Preserves meaningful absolute units. Missing data semantics strictly apply.

    Attributes:
        model: Identifier of the evaluated model.
        task: Benchmark task identifier.
        seed: Random seed or None.
        mean_error: Arithmetic mean of prediction error ||e_t|| over finite steps.
        final_error: Retrieval error at the final sequence step.
        recovery_latency: Steps from shift to first passage recovery (or None).
        sustained_recovery: Steps from shift to sustained recovery (or None).
        mean_eta: Mean adaptive step size eta_t (None if frozen/non-adaptive).
        eta_variance: Variance of step size eta_t (None if frozen/non-adaptive).
        update_energy: Cumulative Frobenius norm sum of memory updates ||Delta M_t||_F.
        max_update: Maximum single-step update norm ||Delta M_t||_F.
        max_state_norm: Maximum state Frobenius norm ||M_t||_F.
        min_stability_margin: Minimum contractive margin S_t = 2 - eta_t ||k_t||^2.
        all_states_finite: True iff all sequence states remained strictly finite.
        failure_step: Step index of numerical divergence, or None if finite.
    """

    model: str
    task: str
    seed: int | None
    mean_error: float | None
    final_error: float | None
    recovery_latency: int | None
    sustained_recovery: int | None
    mean_eta: float | None
    eta_variance: float | None
    update_energy: float | None
    max_update: float | None
    max_state_norm: float | None
    min_stability_margin: float | None
    all_states_finite: bool
    failure_step: int | None
    five_memory_state_activity: float | None = None
    learning_rate_variance: float | None = None
    retention_variance: float | None = None
    key_memory_activity: float | None = None
    value_memory_activity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_fingerprint(trajectory: StateTrajectory) -> StateTransitionFingerprint:
    """Generate the standardized numerical fingerprint for a given StateTrajectory."""
    steps = trajectory.steps
    first_fail = trajectory.first_nonfinite_step

    # Finite steps for statistics
    finite_steps = [
        s
        for s in steps
        if s.observed
        and (s.finite_state is True)
        and (first_fail is None or s.step < first_fail)
    ]

    # Error statistics
    valid_errs = [
        s.error_norm
        for s in finite_steps
        if s.error_norm is not None and math.isfinite(s.error_norm)
    ]
    mean_err = sum(valid_errs) / len(valid_errs) if valid_errs else None
    final_err = (
        finite_steps[-1].error_norm
        if finite_steps and finite_steps[-1].error_norm is not None
        else None
    )

    # Shift response metrics
    shift_res = analyze_shift_response(trajectory)
    stab_res = analyze_stability_events(trajectory)

    # Step-size statistics
    etas = [
        s.step_size
        for s in finite_steps
        if s.step_size is not None and math.isfinite(s.step_size)
    ]
    if etas:
        mean_eta = float(sum(etas) / len(etas))
        if len(etas) > 1:
            eta_var = float(sum((x - mean_eta) ** 2 for x in etas) / (len(etas) - 1))
        else:
            eta_var = 0.0
        eta_variance = eta_var
    else:
        mean_eta = None
        eta_variance = None

    # Energy and updates
    upds = [
        s.update_norm
        for s in finite_steps
        if s.update_norm is not None and math.isfinite(s.update_norm)
    ]
    update_energy = float(sum(upds)) if upds else 0.0
    max_upd = float(max(upds)) if upds else 0.0

    # Phase 8: Five-memory metrics (None if model lacks these memories)
    k_norms = [
        s.key_memory_norm
        for s in finite_steps
        if s.key_memory_norm is not None and math.isfinite(s.key_memory_norm)
    ]
    v_norms = [
        s.value_memory_norm
        for s in finite_steps
        if s.value_memory_norm is not None and math.isfinite(s.value_memory_norm)
    ]
    safe_lrs = [
        s.safe_learning_rate
        for s in finite_steps
        if s.safe_learning_rate is not None and math.isfinite(s.safe_learning_rate)
    ]
    safe_rets = [
        s.safe_retention
        for s in finite_steps
        if s.safe_retention is not None and math.isfinite(s.safe_retention)
    ]

    key_activity: float | None = None
    if len(k_norms) > 1:
        key_activity = float(
            sum(abs(k_norms[i] - k_norms[i - 1]) for i in range(1, len(k_norms)))
        )

    val_activity: float | None = None
    if len(v_norms) > 1:
        val_activity = float(
            sum(abs(v_norms[i] - v_norms[i - 1]) for i in range(1, len(v_norms)))
        )

    lr_var: float | None = None
    if len(safe_lrs) > 1:
        mean_l = sum(safe_lrs) / len(safe_lrs)
        lr_var = float(sum((x - mean_l) ** 2 for x in safe_lrs) / (len(safe_lrs) - 1))
    elif safe_lrs:
        lr_var = 0.0

    ret_var: float | None = None
    if len(safe_rets) > 1:
        mean_r = sum(safe_rets) / len(safe_rets)
        ret_var = float(
            sum((x - mean_r) ** 2 for x in safe_rets) / (len(safe_rets) - 1)
        )
    elif safe_rets:
        ret_var = 0.0

    five_mem_activity: float | None = None
    active_components = [
        a for a in (key_activity, val_activity, lr_var, ret_var) if a is not None
    ]
    if active_components:
        five_mem_activity = float(sum(active_components))

    return StateTransitionFingerprint(
        model=trajectory.model,
        task=trajectory.task,
        seed=trajectory.seed,
        mean_error=mean_err,
        final_error=final_err,
        recovery_latency=shift_res.first_passage_recovery,
        sustained_recovery=shift_res.sustained_recovery,
        mean_eta=mean_eta,
        eta_variance=eta_variance if etas else None,
        update_energy=update_energy,
        max_update=max_upd,
        max_state_norm=stab_res.maximum_memory_norm,
        min_stability_margin=stab_res.minimum_content_margin,
        all_states_finite=stab_res.all_states_finite,
        failure_step=stab_res.first_nonfinite_step,
        five_memory_state_activity=five_mem_activity,
        learning_rate_variance=lr_var,
        retention_variance=ret_var,
        key_memory_activity=key_activity,
        value_memory_activity=val_activity,
    )


def compare_fingerprints(
    fingerprints: list[StateTransitionFingerprint],
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare multiple systems across primary analytical axes.

    Primary comparison axes:
        1. Recovery Speed (recovery_latency, sustained_recovery)
        2. Adaptation Cost (update_energy, max_update)
        3. Stability Margin (min_stability_margin, all_states_finite)

    No scalar winner score is computed unless custom_weights are explicitly provided.
    """
    table: list[dict[str, Any]] = []
    for fp in fingerprints:
        row = {
            "model": fp.model,
            "seed": fp.seed,
            "survival": "SURVIVED"
            if fp.all_states_finite
            else f"FAILED (step {fp.failure_step})",
            "recovery_latency": fp.recovery_latency
            if fp.recovery_latency is not None
            else "N/A",
            "sustained_recovery": fp.sustained_recovery
            if fp.sustained_recovery is not None
            else "N/A",
            "update_energy": round(fp.update_energy, 4)
            if fp.update_energy is not None
            else "N/A",
            "min_margin": round(fp.min_stability_margin, 4)
            if fp.min_stability_margin is not None
            else "N/A",
            "max_state_norm": round(fp.max_state_norm, 4)
            if fp.max_state_norm is not None
            else "N/A",
            "final_error": round(fp.final_error, 4)
            if fp.final_error is not None
            else "N/A",
        }

        if custom_weights is not None:
            # Only compute scalar score if weights are explicitly provided by user
            score = 0.0
            for k, w in custom_weights.items():
                val = getattr(fp, k, None)
                if val is not None and isinstance(val, (int, float)):
                    score += w * float(val)
            row["weighted_score"] = round(score, 4)

        table.append(row)

    return {
        "num_models": len(fingerprints),
        "comparison_table": table,
        "axes": ["recovery_speed", "update_energy", "stability_margin"],
        "has_custom_weights": custom_weights is not None,
    }
