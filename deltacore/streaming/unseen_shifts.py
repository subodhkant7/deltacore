"""Phase 16: Unseen Shift Robustness & Adaptive Safety Module.

Implements predetermined unseen distribution shifts, multi-severity evaluations,
full adaptive safety telemetry (eta_t, eta_t * ||x_t||^2, safety margin, state norm,
retention, update norm), state reset interventions, retention stress histories
(A -> B, A -> B -> A, A -> B -> C, A -> severe-B -> A), failure boundary mapping
at D=256, and spatial permutation equivariance diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from deltacore.streaming.atmospheric_spatiotemporal import (
    compute_relative_frobenius_error,
)
from deltacore.streaming.models import (
    FixedDeltaPredictor,
    SafeAdaptiveDeltaPredictor,
    StreamingPredictor,
    compute_parameter_hash,
)


@dataclass(frozen=True)
class ShiftParameters:
    """Predefined scaling parameters for unseen shift generation."""

    shift_type: str
    severity: str
    mean_delta: float = 0.0
    var_scale: float = 1.0
    temporal_stride: int = 1
    noise_sigma: float = 0.0


SEVERITY_SCALES: dict[str, dict[str, ShiftParameters]] = {
    "mean": {
        "mild": ShiftParameters("mean", "mild", mean_delta=0.5),
        "moderate": ShiftParameters("mean", "moderate", mean_delta=1.0),
        "severe": ShiftParameters("mean", "severe", mean_delta=2.0),
    },
    "variance": {
        "mild": ShiftParameters("variance", "mild", var_scale=1.5),
        "moderate": ShiftParameters("variance", "moderate", var_scale=2.0),
        "severe": ShiftParameters("variance", "severe", var_scale=3.0),
    },
    "temporal_speed": {
        "mild": ShiftParameters("temporal_speed", "mild", temporal_stride=2),
        "moderate": ShiftParameters("temporal_speed", "moderate", temporal_stride=3),
        "severe": ShiftParameters("temporal_speed", "severe", temporal_stride=4),
    },
    "noise": {
        "mild": ShiftParameters("noise", "mild", noise_sigma=0.10),
        "moderate": ShiftParameters("noise", "moderate", noise_sigma=0.25),
        "severe": ShiftParameters("noise", "severe", noise_sigma=0.50),
    },
    "combined": {
        "mild": ShiftParameters(
            "combined", "mild", mean_delta=0.5, var_scale=1.5, noise_sigma=0.10
        ),
        "moderate": ShiftParameters(
            "combined", "moderate", mean_delta=1.0, var_scale=2.0, noise_sigma=0.25
        ),
        "severe": ShiftParameters(
            "combined", "severe", mean_delta=2.0, var_scale=3.0, noise_sigma=0.50
        ),
    },
}


def apply_unseen_shift(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    shift_type: str,
    severity: str,
    shift_start: int = 40,
    shift_end: int = 110,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply predetermined unseen distribution shift during designated interval."""
    if shift_type not in SEVERITY_SCALES or severity not in SEVERITY_SCALES[shift_type]:
        raise ValueError(f"Unknown shift configuration: {shift_type}/{severity}")

    cfg = SEVERITY_SCALES[shift_type][severity]
    x_shifted = inputs.clone()
    y_shifted = targets.clone()

    t_max = len(inputs)
    t_start = min(shift_start, t_max)
    t_end = min(shift_end, t_max)

    if t_start >= t_end:
        return x_shifted, y_shifted

    std_x = inputs.std(dim=0, keepdim=True).clamp_min(1e-4)
    mean_x = inputs.mean(dim=0, keepdim=True)

    gen = torch.Generator().manual_seed(seed)

    if shift_type == "mean":
        shift_vec = cfg.mean_delta * std_x
        x_shifted[t_start:t_end] = x_shifted[t_start:t_end] + shift_vec
        y_shifted[t_start:t_end] = y_shifted[t_start:t_end] + shift_vec

    elif shift_type == "variance":
        scaled_x = mean_x + cfg.var_scale * (x_shifted[t_start:t_end] - mean_x)
        scaled_y = mean_x + cfg.var_scale * (y_shifted[t_start:t_end] - mean_x)
        x_shifted[t_start:t_end] = scaled_x
        y_shifted[t_start:t_end] = scaled_y

    elif shift_type == "temporal_speed":
        # Stride sampling during shift interval: advances through underlying dynamics faster
        stride = cfg.temporal_stride
        sub_indices = torch.arange(
            t_start, min(t_start + (t_end - t_start) * stride, t_max), stride
        )
        L = len(sub_indices)
        if L > 0:
            x_shifted[t_start : t_start + L] = inputs[sub_indices]
            y_shifted[t_start : t_start + L] = targets[sub_indices]

    elif shift_type == "noise":
        noise_x = torch.randn(t_end - t_start, inputs.shape[-1], generator=gen) * (
            cfg.noise_sigma * std_x
        )
        noise_y = torch.randn(t_end - t_start, targets.shape[-1], generator=gen) * (
            cfg.noise_sigma * std_x
        )
        x_shifted[t_start:t_end] = x_shifted[t_start:t_end] + noise_x
        y_shifted[t_start:t_end] = y_shifted[t_start:t_end] + noise_y

    elif shift_type == "combined":
        # Simultaneous mean, variance, and noise shift
        var_x = mean_x + cfg.var_scale * (x_shifted[t_start:t_end] - mean_x)
        var_y = mean_x + cfg.var_scale * (y_shifted[t_start:t_end] - mean_x)
        mean_vec = cfg.mean_delta * std_x
        noise_x = torch.randn(t_end - t_start, inputs.shape[-1], generator=gen) * (
            cfg.noise_sigma * std_x
        )
        noise_y = torch.randn(t_end - t_start, targets.shape[-1], generator=gen) * (
            cfg.noise_sigma * std_x
        )
        x_shifted[t_start:t_end] = var_x + mean_vec + noise_x
        y_shifted[t_start:t_end] = var_y + mean_vec + noise_y

    return x_shifted, y_shifted


def evaluate_shift_streaming_run(
    model: StreamingPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    shift_start: int = 40,
    shift_end: int = 110,
    reset_at_step: int | None = None,
) -> dict[str, Any]:
    r"""Execute streaming run with comprehensive telemetry recording.

    Tracks:
        - Relative prediction error e_t
        - Effective step size eta_t
        - Contraction factor eta_t * ||x_t||^2
        - Local safety margin 2.0 - eta_t * ||x_t||^2
        - State Frobenius norm ||M_t||_F
        - Update norm ||\Delta M_t||_F
        - State retention factor alpha_t
        - Step latency in microseconds
    """
    model.reset_state()
    h_before = compute_parameter_hash(model)

    t_steps = len(inputs)
    preds: list[torch.Tensor] = []
    step_errors: list[float] = []
    step_latencies: list[float] = []

    # Telemetry histories
    eta_history: list[float] = []
    contraction_factor_history: list[float] = []
    safety_margin_history: list[float] = []
    state_norm_history: list[float] = []
    update_norm_history: list[float] = []
    retention_history: list[float] = []

    diverged = False
    time_to_nonfinite: int | None = None

    for t in range(t_steps):
        if reset_at_step is not None and t == reset_at_step:
            model.reset_state()

        xt = inputs[t]
        yt = targets[t]

        t0 = time.perf_counter()
        y_hat = model.predict_step(xt)
        t_eval = time.perf_counter() - t0
        step_latencies.append(t_eval * 1e6)

        if not torch.isfinite(y_hat).all():
            diverged = True
            time_to_nonfinite = t
            break

        preds.append(y_hat.detach().clone())
        err_rel = float(
            torch.linalg.norm(yt - y_hat).item()
            / max(torch.linalg.norm(yt).item(), 1e-8)
        )
        step_errors.append(err_rel)

        # State adaptation
        model.adapt_step(xt, yt)

        state_norm = model.get_state_norm()
        if not math.isfinite(state_norm):
            diverged = True
            time_to_nonfinite = t
            break

        state_norm_history.append(state_norm)
        update_norm_history.append(getattr(model, "last_update_norm", 0.0))

        # Track adaptive step-size telemetry if supported by model
        x_norm_sq = float((xt @ xt).item())
        eta_t = getattr(model, "last_step_size", getattr(model, "step_size", 0.015))
        c_factor = eta_t * x_norm_sq
        margin = getattr(model, "last_margin", 2.0 - c_factor)
        retention = getattr(model, "last_retention", 1.0)

        eta_history.append(eta_t)
        contraction_factor_history.append(c_factor)
        safety_margin_history.append(margin)
        retention_history.append(retention)

    h_after = compute_parameter_hash(model)
    assert h_before == h_after, (
        f"Parameter immutability violation in {model.name}! Offline parameters mutated."
    )

    if diverged:
        overall_rel_err = float("nan")
        overall_mae = float("nan")
        overall_rmse = float("nan")
    else:
        preds_tensor = torch.stack(preds)
        overall_rel_err = compute_relative_frobenius_error(targets, preds_tensor)
        overall_mae = float(torch.mean(torch.abs(targets - preds_tensor)).item())
        overall_rmse = float(
            torch.sqrt(torch.mean((targets - preds_tensor) ** 2)).item()
        )

    # Pre-shift error (t in [0, shift_start))
    pre_errors = step_errors[:shift_start]
    e_pre = float(np.mean(pre_errors)) if pre_errors else 0.0

    # Shift-period error (t in [shift_start, min(shift_end, len(step_errors))])
    shift_errors = step_errors[shift_start : min(shift_end, len(step_errors))]
    e_shift = float(np.mean(shift_errors)) if shift_errors else 0.0

    # Recovery metrics
    recovery_thresh = e_pre + 0.05
    fp_recovery = None
    if shift_start < len(step_errors):
        for offset, err in enumerate(step_errors[shift_start:]):
            if err <= recovery_thresh:
                fp_recovery = offset
                break
    if fp_recovery is None:
        fp_recovery = max(0, len(step_errors) - shift_start)

    # Sustained recovery (5 consecutive steps below threshold)
    sustained_recovery = None
    w_sz = 5
    if shift_start < len(step_errors):
        remaining = step_errors[shift_start:]
        for offset in range(max(0, len(remaining) - w_sz)):
            window = remaining[offset : offset + w_sz]
            if all(e <= recovery_thresh for e in window):
                sustained_recovery = offset
                break
    if sustained_recovery is None:
        sustained_recovery = max(0, len(step_errors) - shift_start)

    cum_excess = float(
        sum(
            max(0.0, err - e_pre)
            for err in step_errors[shift_start : min(shift_end, len(step_errors))]
        )
    )
    adapt_energy = float(sum(u**2 for u in update_norm_history))
    max_state_norm = max(state_norm_history) if state_norm_history else 0.0
    min_margin = min(safety_margin_history) if safety_margin_history else 0.0

    return {
        "model_name": model.name,
        "diverged": diverged,
        "time_to_nonfinite": time_to_nonfinite,
        "rel_error": overall_rel_err,
        "mae": overall_mae,
        "rmse": overall_rmse,
        "e_pre": e_pre,
        "e_shift": e_shift,
        "first_passage_recovery": fp_recovery,
        "sustained_recovery": sustained_recovery,
        "cumulative_excess_error": cum_excess,
        "adaptation_energy": adapt_energy,
        "max_state_norm": max_state_norm,
        "min_safety_margin": min_margin,
        "mean_latency_us": float(np.mean(step_latencies)) if step_latencies else 0.0,
        "persistent_state_bytes": model.get_state_memory_bytes(),
        "step_errors": step_errors,
        "eta_history": eta_history,
        "contraction_factor_history": contraction_factor_history,
        "safety_margin_history": safety_margin_history,
        "state_norm_history": state_norm_history,
        "update_norm_history": update_norm_history,
        "retention_history": retention_history,
    }


def evaluate_shift_reset_ablation(
    model: StreamingPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    shift_start: int = 40,
    shift_end: int = 110,
) -> dict[str, Any]:
    """Evaluate continuous state vs state reset at shift onset."""
    res_cont = evaluate_shift_streaming_run(
        model=model,
        inputs=inputs,
        targets=targets,
        shift_start=shift_start,
        shift_end=shift_end,
        reset_at_step=None,
    )
    res_reset = evaluate_shift_streaming_run(
        model=model,
        inputs=inputs,
        targets=targets,
        shift_start=shift_start,
        shift_end=shift_end,
        reset_at_step=shift_start,
    )

    delta_e_rel = res_cont["rel_error"] - res_reset["rel_error"]
    delta_e_shift = res_cont["e_shift"] - res_reset["e_shift"]
    delta_cum_excess = (
        res_cont["cumulative_excess_error"] - res_reset["cumulative_excess_error"]
    )

    return {
        "continuous": res_cont,
        "reset": res_reset,
        "delta_e_rel": delta_e_rel,
        "delta_e_shift": delta_e_shift,
        "delta_cum_excess": delta_cum_excess,
        "continuous_advantage": delta_cum_excess < 0,
    }


def generate_retention_stress_stream(
    domain_a_inputs: torch.Tensor,
    domain_a_targets: torch.Tensor,
    domain_b_inputs: torch.Tensor,
    domain_b_targets: torch.Tensor,
    history_type: str,
    seg_len: int = 40,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Construct controlled retention stress histories.

    Histories:
        - "A_to_B": [A (seg_len), B (seg_len)]
        - "A_to_B_to_A": [A (seg_len), B (seg_len), A (seg_len)]
        - "A_to_B_to_C": [A (seg_len), B (seg_len), perturbed_A (seg_len)]
        - "A_to_severeB_to_A": [A (seg_len), severe_variance_B (seg_len), A (seg_len)]
    """

    def _std(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, keepdim=True).clamp_min(1e-4)
        return (x - m) / s, (y - m) / s

    a_in, a_tgt = _std(domain_a_inputs[:seg_len], domain_a_targets[:seg_len])
    b_in, b_tgt = _std(domain_b_inputs[:seg_len], domain_b_targets[:seg_len])
    a_return_in, a_return_tgt = _std(
        domain_a_inputs[seg_len : 2 * seg_len], domain_a_targets[seg_len : 2 * seg_len]
    )

    if history_type == "A_to_B":
        inputs = torch.cat([a_in, b_in], dim=0)
        targets = torch.cat([a_tgt, b_tgt], dim=0)
        bounds = [seg_len]

    elif history_type == "A_to_B_to_A":
        inputs = torch.cat([a_in, b_in, a_return_in], dim=0)
        targets = torch.cat([a_tgt, b_tgt, a_return_tgt], dim=0)
        bounds = [seg_len, 2 * seg_len]

    elif history_type == "A_to_B_to_C":
        # C is perturbed Domain A with mean and variance shift
        gen = torch.Generator().manual_seed(seed)
        c_in = (
            a_return_in * 1.8
            + 1.0
            + torch.randn(a_return_in.shape, generator=gen) * 0.2
        )
        c_tgt = a_return_tgt * 1.8 + 1.0
        inputs = torch.cat([a_in, b_in, c_in], dim=0)
        targets = torch.cat([a_tgt, b_tgt, c_tgt], dim=0)
        bounds = [seg_len, 2 * seg_len]

    elif history_type == "A_to_severeB_to_A":
        severe_b_in = b_in * 3.0 + 2.0
        severe_b_tgt = b_tgt * 3.0 + 2.0
        inputs = torch.cat([a_in, severe_b_in, a_return_in], dim=0)
        targets = torch.cat([a_tgt, severe_b_tgt, a_return_tgt], dim=0)
        bounds = [seg_len, 2 * seg_len]

    else:
        raise ValueError(f"Unknown retention stress history: {history_type}")

    return inputs, targets, bounds


def map_failure_boundary_grid(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    step_sizes: list[float] | None = None,
    dim: int = 256,
) -> dict[str, Any]:
    """Map empirical failure boundary at D=256 for FixedDelta vs SafeAdaptiveDelta."""
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.008, 0.012, 0.016, 0.020, 0.025]

    fixed_results: list[dict[str, Any]] = []
    safe_results: list[dict[str, Any]] = []

    for eta in step_sizes:
        # 1. FixedDelta
        m_fixed = FixedDeltaPredictor(dim=dim, step_size=eta)
        res_f = evaluate_shift_streaming_run(m_fixed, inputs, targets)
        fixed_results.append(
            {
                "step_size": eta,
                "diverged": res_f["diverged"],
                "time_to_nonfinite": res_f["time_to_nonfinite"],
                "max_state_norm": res_f["max_state_norm"],
                "rel_error": res_f["rel_error"],
                "min_safety_margin": res_f["min_safety_margin"],
            }
        )

        # 2. SafeAdaptiveDelta (rho=1.50)
        m_safe = SafeAdaptiveDeltaPredictor(
            dim=dim, eta_max=eta, rho=1.50, alpha_min=0.95, gamma=0.05
        )
        res_s = evaluate_shift_streaming_run(m_safe, inputs, targets)
        safe_results.append(
            {
                "step_size": eta,
                "diverged": res_s["diverged"],
                "time_to_nonfinite": res_s["time_to_nonfinite"],
                "max_state_norm": res_s["max_state_norm"],
                "rel_error": res_s["rel_error"],
                "min_safety_margin": res_s["min_safety_margin"],
            }
        )

    return {
        "step_sizes": step_sizes,
        "dimension": dim,
        "FixedDelta": fixed_results,
        "SafeAdaptiveDelta": safe_results,
    }


def evaluate_perturbation_robustness(
    model: StreamingPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    noise_fractions: list[float] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate model robustness against pure observation noise perturbations."""
    if noise_fractions is None:
        noise_fractions = [0.0, 0.01, 0.05, 0.10]

    results = []
    std_x = inputs.std(dim=0, keepdim=True).clamp_min(1e-4)
    gen = torch.Generator().manual_seed(seed)

    for frac in noise_fractions:
        if frac == 0.0:
            p_inputs = inputs
            p_targets = targets
        else:
            noise = torch.randn(inputs.shape, generator=gen) * (frac * std_x)
            p_inputs = inputs + noise
            p_targets = targets + noise

        res = evaluate_shift_streaming_run(model, p_inputs, p_targets)
        results.append(
            {
                "noise_fraction": frac,
                "rel_error": res["rel_error"],
                "e_shift": res["e_shift"],
                "max_state_norm": res["max_state_norm"],
                "adaptation_energy": res["adaptation_energy"],
                "diverged": res["diverged"],
            }
        )

    return results


def check_permutation_equivariance(
    model: StreamingPredictor,
    inputs: torch.Tensor,
    seed: int = 42,
) -> float:
    r"""Verify exact output permutation equivariance: ||f(PX) - P f(X)||_F."""
    dim = inputs.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dim, generator=gen)
    p_mat = torch.eye(dim)[perm]  # (D, D)

    # 1. Unpermuted pass
    model.reset_state()
    preds_orig = []
    for t in range(min(30, len(inputs))):
        p = model.predict_step(inputs[t])
        preds_orig.append(p.detach().clone())
        model.adapt_step(inputs[t], inputs[t])  # autoadapt step

    # 2. Permuted input pass
    perm_inputs = inputs @ p_mat.T
    model.reset_state()
    preds_perm = []
    for t in range(min(30, len(inputs))):
        p = model.predict_step(perm_inputs[t])
        preds_perm.append(p.detach().clone())
        model.adapt_step(perm_inputs[t], perm_inputs[t])

    # Compute ||preds_perm - preds_orig * P^T||_F / ||preds_orig||_F
    t_orig = torch.stack(preds_orig) @ p_mat.T
    t_perm = torch.stack(preds_perm)

    equiv_err = float(
        torch.linalg.norm(t_perm - t_orig).item()
        / max(torch.linalg.norm(t_orig).item(), 1e-8)
    )
    return equiv_err
