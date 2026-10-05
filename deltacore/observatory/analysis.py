# ==============================================================================
# DeltaCore: deltacore/observatory/analysis.py
# Temporal, shift response, stability, and comparative analysis functions.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from deltacore.benchmarks.metrics.energy import compute_state_growth_ratio
from deltacore.benchmarks.metrics.recovery import (
    compute_first_passage_recovery,
    compute_sustained_recovery,
)
from deltacore.observatory.schema import StateTrajectory

# ------------------------------------------------------------------------------
# 1. Temporal Trajectory Extractors
# ------------------------------------------------------------------------------


def extract_error_trajectory(trajectory: StateTrajectory) -> list[float | None]:
    r"""Extract error trajectory $E_t = \|e_t\|_2$ preserving missing/failure semantics."""
    return trajectory.error_norms


def extract_state_trajectory(trajectory: StateTrajectory) -> list[float | None]:
    r"""Extract memory state norm trajectory $\|M_t\|_F$ preserving missing semantics."""
    return trajectory.memory_norms


def extract_dynamics_trajectory(trajectory: StateTrajectory) -> list[float | None]:
    r"""Extract dynamics memory norm trajectory $\|C_t\|_F$ (None if unsupported)."""
    return trajectory.dynamics_memory_norms


def extract_update_trajectory(trajectory: StateTrajectory) -> list[float | None]:
    r"""Extract update norm trajectory $\|\Delta M_t\|_F$."""
    return trajectory.update_norms


def extract_step_size_trajectory(trajectory: StateTrajectory) -> list[float | None]:
    r"""Extract step-size trajectory $\eta_t$ (None if non-adaptive)."""
    return trajectory.step_sizes


def extract_stability_trajectory(trajectory: StateTrajectory) -> list[float | None]:
    r"""Extract content stability margin trajectory $S_t = 2 - \eta_t \|k_t\|_2^2$."""
    return trajectory.stability_margins


def extract_dynamics_stability_trajectory(
    trajectory: StateTrajectory,
) -> list[float | None]:
    r"""Extract dynamics stability margin trajectory $S_t^{\text{dyn}} = 2 - \rho_t \|z_t\|_2^2$."""
    return trajectory.get_series("dynamics_stability_margin")


# ------------------------------------------------------------------------------
# 2. Adaptation Response Analysis
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftResponseResult:
    """Quantitative evaluation of online adaptation response to a distribution shift.

    Attributes:
        pre_shift_error: Mean error immediately preceding the shift point.
        shock_error: Immediate error at the first post-shift observation.
        first_passage_recovery: Steps from shift to first error <= tau * shock_error (or None).
        sustained_recovery: Steps from shift to window-sustained recovery (or None).
        recovery_slope: Empirical error reduction rate (E_shock - E_rec) / T_FP (or None).
        final_error: Retrieval error at the end of the sequence (or None if failed).
        peak_update_after_shift: Maximum update norm observed at or after the shift.
        state_growth_ratio: Ratio max_{t >= t_s} ||M_t||_F / ||M_{t_s}||_F.
        update_energy_after_shift: Cumulative sum of ||Delta M_t||_F at or after shift.
    """

    pre_shift_error: float | None
    shock_error: float | None
    first_passage_recovery: int | None
    sustained_recovery: int | None
    recovery_slope: float | None
    final_error: float | None
    peak_update_after_shift: float | None
    state_growth_ratio: float | None
    update_energy_after_shift: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_shift_response(
    trajectory: StateTrajectory,
    tau: float | None = None,
    window: int | None = None,
) -> ShiftResponseResult:
    """Compute benchmark-conforming adaptation response metrics for a shift trajectory.

    Uses formal definitions from deltacore.benchmarks.metrics. Do not invent alternative thresholds.
    """
    steps = trajectory.steps
    if not steps:
        return ShiftResponseResult(
            pre_shift_error=None,
            shock_error=None,
            first_passage_recovery=None,
            sustained_recovery=None,
            recovery_slope=None,
            final_error=None,
            peak_update_after_shift=None,
            state_growth_ratio=None,
            update_energy_after_shift=None,
        )

    meta = trajectory.metadata or {}
    t_shift = meta.get("shift_position")
    tau_val = tau if tau is not None else meta.get("tau", 0.5)
    win_val = window if window is not None else meta.get("window", 4)

    # If shift position is not defined, default to 0
    shift_pos = t_shift if t_shift is not None else 0
    shift_pos = max(0, min(shift_pos, len(steps) - 1))

    # Pre-shift error: average of up to 10 steps prior to shift
    pre_steps: list[float] = []
    for i in range(max(0, shift_pos - 10), shift_pos):
        val = steps[i].error_norm
        if steps[i].observed and val is not None and math.isfinite(val):
            pre_steps.append(val)
    pre_shift_err = sum(pre_steps) / len(pre_steps) if pre_steps else None

    # Shock error
    shock_step = steps[shift_pos]
    shock_err = (
        shock_step.error_norm
        if shock_step.observed
        and shock_step.error_norm is not None
        and math.isfinite(shock_step.error_norm)
        else None
    )

    # Convert errors to tensor format for canonical benchmark functions
    import torch

    valid_errs = [
        s.error_norm if (s.observed and s.error_norm is not None) else float("nan")
        for s in steps
    ]
    err_tensor = torch.tensor([[e] for e in valid_errs], dtype=torch.float32)

    fp_rec = compute_first_passage_recovery(
        err_tensor, shift_index=shift_pos, tau=tau_val
    )
    sust_rec = compute_sustained_recovery(
        err_tensor, shift_index=shift_pos, window=win_val, tau=tau_val
    )

    # Recovery slope: (shock_error - error_at_fp) / fp_rec
    rec_slope: float | None = None
    if fp_rec is not None and shock_err is not None and fp_rec > 0:
        fp_step_idx = shift_pos + fp_rec
        if fp_step_idx < len(steps):
            e_at_fp = (
                steps[fp_step_idx].error_norm if steps[fp_step_idx].observed else None
            )
            if e_at_fp is not None:
                rec_slope = float((shock_err - e_at_fp) / fp_rec)
    elif fp_rec == 0:
        rec_slope = 0.0

    # Final error
    fin_step = steps[-1]
    final_err = (
        fin_step.error_norm
        if fin_step.observed
        and fin_step.error_norm is not None
        and math.isfinite(fin_step.error_norm)
        else None
    )

    # Post-shift updates and energy
    post_upds = [
        s.update_norm
        for s in steps[shift_pos:]
        if s.observed and s.update_norm is not None and math.isfinite(s.update_norm)
    ]
    peak_upd = max(post_upds) if post_upds else None
    upd_energy = sum(post_upds) if post_upds else None

    # State growth ratio
    mem_norms_list = [
        s.memory_norm if s.memory_norm is not None else 0.0 for s in steps
    ]
    growth_ratio = compute_state_growth_ratio(mem_norms_list, shift_index=shift_pos)

    return ShiftResponseResult(
        pre_shift_error=pre_shift_err,
        shock_error=shock_err,
        first_passage_recovery=fp_rec,
        sustained_recovery=sust_rec,
        recovery_slope=rec_slope,
        final_error=final_err,
        peak_update_after_shift=peak_upd,
        state_growth_ratio=growth_ratio,
        update_energy_after_shift=upd_energy,
    )


# ------------------------------------------------------------------------------
# 3. Stability Event Analysis
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityAnalysisResult:
    """Structured assessment of numerical stability and contractive bounds.

    Guarantees failure visibility:
        If an intermediate state was non-finite, all_states_finite is False,
        first_nonfinite_step is populated, and terminal_state_finite cannot be True.
    """

    minimum_content_margin: float | None
    minimum_dynamics_margin: float | None
    maximum_normalized_step: float | None
    maximum_normalized_dynamics_rate: float | None
    clip_count: int
    all_states_finite: bool
    terminal_state_finite: bool
    first_nonfinite_step: int | None
    maximum_memory_norm: float | None
    maximum_dynamics_norm: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_stability_events(trajectory: StateTrajectory) -> StabilityAnalysisResult:
    """Analyze stability invariants, contraction margins, and numerical survival.

    Enforces that intermediate numerical failures remain visible and cannot
    be masked as healthy terminal runs.
    """
    steps = trajectory.steps

    first_nonfin = trajectory.first_nonfinite_step
    all_finite = trajectory.all_states_finite and (first_nonfin is None)

    # Check each step for any hidden non-finite states or unobserved steps
    for s in steps:
        if not s.observed:
            all_finite = False
            # Unobserved post-failure step must never be identified as the failure step
            continue
        if s.finite_state is False:
            all_finite = False
            if first_nonfin is None:
                first_nonfin = s.step
        for val in (s.error_norm, s.memory_norm, s.update_norm, s.step_size):
            if val is not None and not math.isfinite(val):
                all_finite = False
                if first_nonfin is None:
                    first_nonfin = s.step

    # Terminal state finiteness invariant
    terminal_finite = False
    if all_finite and steps:
        terminal_finite = (
            steps[-1].observed
            and (steps[-1].finite_state is True)
            and (trajectory.terminal_state_finite is not False)
        )

    # Compute valid bounds over finite steps prior to any failure
    finite_steps = [
        s
        for s in steps
        if s.observed
        and (s.finite_state is True)
        and (first_nonfin is None or s.step < first_nonfin)
    ]

    c_margins = [
        s.stability_margin
        for s in finite_steps
        if s.stability_margin is not None and math.isfinite(s.stability_margin)
    ]
    min_c_margin = min(c_margins) if c_margins else None

    d_margins = [
        s.dynamics_stability_margin
        for s in finite_steps
        if s.dynamics_stability_margin is not None
        and math.isfinite(s.dynamics_stability_margin)
    ]
    min_d_margin = min(d_margins) if d_margins else None

    norm_steps = [
        s.normalized_step
        for s in finite_steps
        if s.normalized_step is not None and math.isfinite(s.normalized_step)
    ]
    max_norm_step = max(norm_steps) if norm_steps else None

    # Dynamics rate bounds from extra metadata if present
    d_rates: list[float] = [
        float(s.extra["normalized_dynamics_rate"])
        for s in finite_steps
        if s.extra.get("normalized_dynamics_rate") is not None
        and math.isfinite(float(s.extra["normalized_dynamics_rate"]))
    ]
    max_d_rate = max(d_rates) if d_rates else None

    clips = sum(1 for s in steps if s.clip_event)
    # Check if metrics contains clip_count
    if (
        "clip_count" in trajectory.metrics
        and trajectory.metrics["clip_count"] is not None
    ):
        clips = max(clips, int(trajectory.metrics["clip_count"]))

    mem_norms = [
        s.memory_norm
        for s in finite_steps
        if s.memory_norm is not None and math.isfinite(s.memory_norm)
    ]
    max_m_norm = max(mem_norms) if mem_norms else None

    dyn_norms = [
        s.dynamics_memory_norm
        for s in finite_steps
        if s.dynamics_memory_norm is not None and math.isfinite(s.dynamics_memory_norm)
    ]
    max_c_norm = max(dyn_norms) if dyn_norms else None

    return StabilityAnalysisResult(
        minimum_content_margin=min_c_margin,
        minimum_dynamics_margin=min_d_margin,
        maximum_normalized_step=max_norm_step,
        maximum_normalized_dynamics_rate=max_d_rate,
        clip_count=clips,
        all_states_finite=all_finite,
        terminal_state_finite=terminal_finite,
        first_nonfinite_step=first_nonfin,
        maximum_memory_norm=max_m_norm,
        maximum_dynamics_norm=max_c_norm,
    )


# ------------------------------------------------------------------------------
# 4. Comparative Run Analysis
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunComparisonResult:
    """Pairwise comparative delta between an evaluation run and a reference baseline.

    Sign Conventions (Delta = Eval - Baseline):
        - error_delta: Final error difference E_T(eval) - E_T(base).
          Negative (< 0) indicates lower final error for eval.
        - recovery_delta: Recovery latency difference T_FP(eval) - T_FP(base).
          Negative (< 0) indicates faster recovery for eval.
          None if either model failed to achieve recovery.
        - energy_delta: Update energy difference U_M(eval) - U_M(base).
          Positive (> 0) indicates higher update expenditure for eval.
        - state_norm_delta: Maximum state norm difference max ||M||_F(eval) - max ||M||_F(base).
        - stability_margin_delta: Minimum margin difference min S_t(eval) - min S_t(base).
          Positive (> 0) indicates eval operated with a safer/more contractive margin.
        - failure_delta: Failure indicator difference (1 if eval failed else 0) - (1 if base failed else 0).
          -1: Eval succeeded while baseline failed.
          0: Both had identical numerical survival outcome.
          +1: Eval failed while baseline survived.

    NO WINNER IS AUTOMATICALLY CHOSEN. Trade-offs remain explicit.
    """

    eval_model: str
    baseline_model: str
    error_delta: float | None
    recovery_delta: int | None
    energy_delta: float | None
    state_norm_delta: float | None
    stability_margin_delta: float | None
    failure_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_runs(
    eval_trajectory: StateTrajectory,
    baseline_trajectory: StateTrajectory,
) -> RunComparisonResult:
    """Calculate mathematically explicit pairwise deltas between two trajectories.

    Sign convention: Delta = Eval - Baseline.
    Does not choose a winner or assign normative value judgments.
    """
    eval_shift = analyze_shift_response(eval_trajectory)
    base_shift = analyze_shift_response(baseline_trajectory)

    eval_stab = analyze_stability_events(eval_trajectory)
    base_stab = analyze_stability_events(baseline_trajectory)

    # 1. Error delta
    err_delta: float | None = None
    if eval_shift.final_error is not None and base_shift.final_error is not None:
        err_delta = eval_shift.final_error - base_shift.final_error

    # 2. Recovery delta
    rec_delta: int | None = None
    if (
        eval_shift.first_passage_recovery is not None
        and base_shift.first_passage_recovery is not None
    ):
        rec_delta = (
            eval_shift.first_passage_recovery - base_shift.first_passage_recovery
        )

    # 3. Energy delta
    energy_delta: float | None = None
    if (
        eval_shift.update_energy_after_shift is not None
        and base_shift.update_energy_after_shift is not None
    ):
        energy_delta = (
            eval_shift.update_energy_after_shift - base_shift.update_energy_after_shift
        )

    # 4. State norm delta
    state_norm_delta: float | None = None
    if (
        eval_stab.maximum_memory_norm is not None
        and base_stab.maximum_memory_norm is not None
    ):
        state_norm_delta = eval_stab.maximum_memory_norm - base_stab.maximum_memory_norm

    # 5. Stability margin delta
    stab_delta: float | None = None
    if (
        eval_stab.minimum_content_margin is not None
        and base_stab.minimum_content_margin is not None
    ):
        stab_delta = eval_stab.minimum_content_margin - base_stab.minimum_content_margin

    # 6. Failure delta
    eval_failed = 0 if eval_stab.all_states_finite else 1
    base_failed = 0 if base_stab.all_states_finite else 1
    fail_delta = eval_failed - base_failed

    return RunComparisonResult(
        eval_model=eval_trajectory.model,
        baseline_model=baseline_trajectory.model,
        error_delta=err_delta,
        recovery_delta=rec_delta,
        energy_delta=energy_delta,
        state_norm_delta=state_norm_delta,
        stability_margin_delta=stab_delta,
        failure_delta=fail_delta,
    )
