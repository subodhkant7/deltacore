"""Phase 15: Adaptive Regime Transfer & Robustness Module.

Implements mathematical utilities, validation calibration sweeps, cross-domain
parameter and state transfer, regime-switch streams, and failure boundary diagnostics
comparing Domain A (NOAA OISST slow thermal) and Domain B (ERA5 fast synoptic advection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from deltacore.streaming.atmospheric_benchmark import evaluate_streaming_run
from deltacore.streaming.atmospheric_spatiotemporal import (
    AtmosphericConfig,
    AtmosphericData,
    compute_relative_frobenius_error,
    generate_atmospheric_dataset,
)
from deltacore.streaming.models import (
    FixedDeltaPredictor,
    SafeAdaptiveDeltaPredictor,
    StreamingPredictor,
)
from deltacore.streaming.real_spatiotemporal import (
    RealSpatioTemporalConfig,
    RealSpatioTemporalData,
    generate_real_spatiotemporal_dataset,
)


@dataclass(frozen=True)
class SafeAdaptiveDeltaConfig:
    """Hyperparameter specification for SafeAdaptiveDelta."""

    eta_max: float = 0.008
    rho: float = 1.90
    alpha_min: float = 0.85
    gamma: float = 0.05

    def instantiate(self, dim: int) -> SafeAdaptiveDeltaPredictor:
        return SafeAdaptiveDeltaPredictor(
            dim=dim,
            eta_max=self.eta_max,
            rho=self.rho,
            alpha_min=self.alpha_min,
            gamma=self.gamma,
        )


@dataclass
class CrossDomainDatasets:
    """Container holding preprocessed Domain A and Domain B datasets."""

    domain_a: RealSpatioTemporalData
    domain_b: AtmosphericData


def load_cross_domain_datasets(
    seed: int = 42, resolution: str = "small"
) -> CrossDomainDatasets:
    """Load both Domain A (OISST) and Domain B (ERA5) for a given seed and resolution."""
    if resolution == "small":
        h, w = 8, 8
    elif resolution == "medium":
        h, w = 16, 16
    else:
        raise ValueError(f"Unknown resolution {resolution}")

    cfg_a = RealSpatioTemporalConfig(height=h, width=w, seed=seed)
    data_a = generate_real_spatiotemporal_dataset(cfg_a)

    cfg_b = AtmosphericConfig(resolution=resolution, height=h, width=w, seed=seed)
    data_b = generate_atmospheric_dataset(cfg_b)

    return CrossDomainDatasets(domain_a=data_a, domain_b=data_b)


def calibrate_validation_grid(
    val_inputs_a: torch.Tensor,
    val_targets_a: torch.Tensor,
    val_inputs_b: torch.Tensor,
    val_targets_b: torch.Tensor,
    dim: int = 64,
    grid: list[SafeAdaptiveDeltaConfig] | None = None,
) -> dict[str, Any]:
    """Execute predetermined grid sweep on validation splits only.

    Grid:
        alpha_min in {0.70, 0.85, 0.95}
        rho in {1.5, 1.9}
        eta_max in {0.008, 0.015}
    """
    if grid is None:
        grid = []
        for eta0 in [0.008, 0.015]:
            for rho in [1.5, 1.9]:
                for a_min in [0.70, 0.85, 0.95]:
                    grid.append(
                        SafeAdaptiveDeltaConfig(
                            eta_max=eta0, rho=rho, alpha_min=a_min, gamma=0.05
                        )
                    )

    sweep_results: list[dict[str, Any]] = []
    best_a: tuple[float, SafeAdaptiveDeltaConfig] = (float("inf"), grid[0])
    best_b: tuple[float, SafeAdaptiveDeltaConfig] = (float("inf"), grid[0])
    best_pooled: tuple[float, SafeAdaptiveDeltaConfig] = (float("inf"), grid[0])

    for cfg in grid:
        # Domain A validation
        mA = cfg.instantiate(dim)
        resA = evaluate_streaming_run(mA, val_inputs_a, val_targets_a)
        errA = float(resA["rel_error"])

        # Domain B validation
        mB = cfg.instantiate(dim)
        resB = evaluate_streaming_run(mB, val_inputs_b, val_targets_b)
        errB = float(resB["rel_error"])

        pooled_err = 0.5 * (errA + errB)

        sweep_results.append(
            {
                "config": {
                    "eta_max": cfg.eta_max,
                    "rho": cfg.rho,
                    "alpha_min": cfg.alpha_min,
                    "gamma": cfg.gamma,
                },
                "val_error_domain_a": errA,
                "val_error_domain_b": errB,
                "val_error_pooled": pooled_err,
            }
        )

        if errA < best_a[0]:
            best_a = (errA, cfg)
        if errB < best_b[0]:
            best_b = (errB, cfg)
        if pooled_err < best_pooled[0]:
            best_pooled = (pooled_err, cfg)

    return {
        "sweep_results": sweep_results,
        "best_domain_a": {
            "config": {
                "eta_max": best_a[1].eta_max,
                "rho": best_a[1].rho,
                "alpha_min": best_a[1].alpha_min,
                "gamma": best_a[1].gamma,
            },
            "val_error": best_a[0],
        },
        "best_domain_b": {
            "config": {
                "eta_max": best_b[1].eta_max,
                "rho": best_b[1].rho,
                "alpha_min": best_b[1].alpha_min,
                "gamma": best_b[1].gamma,
            },
            "val_error": best_b[0],
        },
        "best_pooled": {
            "config": {
                "eta_max": best_pooled[1].eta_max,
                "rho": best_pooled[1].rho,
                "alpha_min": best_pooled[1].alpha_min,
                "gamma": best_pooled[1].gamma,
            },
            "val_error": best_pooled[0],
        },
    }


def generate_regime_switch_stream(
    test_inputs_a: torch.Tensor,
    test_targets_a: torch.Tensor,
    test_inputs_b: torch.Tensor,
    test_targets_b: torch.Tensor,
    segment_len: int = 70,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Construct controlled mixed spatio-temporal benchmark stream.

    Stream Structure:
        Phase 1 (t in [0, segment_len)): Domain A dynamics (OISST slow thermal)
        Phase 2 (t in [segment_len, 2*segment_len)): Domain B dynamics (ERA5 fast synoptic advection)
        Phase 3 (t in [2*segment_len, 3*segment_len)): Domain A dynamics (return to OISST slow thermal)

    Both segments standardized to 0-mean unit-variance coordinates to ensure
    physically compatible input-space representation.
    """
    # Slice first segment_len steps
    assert len(test_inputs_a) >= segment_len
    assert len(test_inputs_b) >= segment_len

    seg1_in = test_inputs_a[:segment_len]
    seg1_tgt = test_targets_a[:segment_len]

    seg2_in = test_inputs_b[:segment_len]
    seg2_tgt = test_targets_b[:segment_len]

    seg3_in = test_inputs_a[segment_len : 2 * segment_len]
    seg3_tgt = test_targets_a[segment_len : 2 * segment_len]

    # Standardize each segment internally to zero-mean unit-variance per coordinate
    def _std_pair(
        x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_x = x.mean(dim=0, keepdim=True)
        std_x = x.std(dim=0, keepdim=True).clamp_min(1e-4)
        return (x - mean_x) / std_x, (y - mean_x) / std_x

    s1_in, s1_tgt = _std_pair(seg1_in, seg1_tgt)
    s2_in, s2_tgt = _std_pair(seg2_in, seg2_tgt)
    s3_in, s3_tgt = _std_pair(seg3_in, seg3_tgt)

    mixed_inputs = torch.cat([s1_in, s2_in, s3_in], dim=0)
    mixed_targets = torch.cat([s1_tgt, s2_tgt, s3_tgt], dim=0)

    boundary_1 = segment_len
    boundary_2 = 2 * segment_len

    return mixed_inputs, mixed_targets, (boundary_1, boundary_2)


def evaluate_state_initialization(
    model: SafeAdaptiveDeltaPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate streaming performance under specific state initialization condition."""
    model.reset_state()
    dim = model.dim

    if initial_state is not None:
        model.set_state(initial_state)
    elif noise_std > 0.0:
        gen = torch.Generator().manual_seed(seed)
        rand_m = torch.randn(dim, dim, generator=gen, dtype=torch.float32) * noise_std
        model.set_state(rand_m)
    else:
        # Zero state
        model.set_state(torch.zeros(dim, dim, dtype=torch.float32))

    t_steps = len(inputs)
    preds: list[torch.Tensor] = []
    state_norms: list[float] = []

    for t in range(t_steps):
        xt = inputs[t]
        yt = targets[t]

        y_hat = model.predict_step(xt)
        preds.append(y_hat.detach().clone())

        model.adapt_step(xt, yt)
        state_norms.append(model.get_state_norm())

    preds_tensor = torch.stack(preds)
    overall_rel_err = compute_relative_frobenius_error(targets, preds_tensor)

    # Compute early error (first 10 steps) to capture initialization transient
    early_err = compute_relative_frobenius_error(targets[:10], preds_tensor[:10])
    max_state_norm = max(state_norms) if state_norms else 0.0

    return {
        "rel_error": overall_rel_err,
        "early_rel_error_first_10": early_err,
        "max_state_norm": max_state_norm,
        "final_state": model.M.data.clone(),
    }


def evaluate_failure_boundary(
    dim: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    is_safe: bool = True,
    step_size: float = 0.008,
    rho: float = 1.90,
) -> dict[str, Any]:
    """Measure exact numerical stability / failure boundary at scaled dimension D=256."""
    if is_safe:
        model: StreamingPredictor = SafeAdaptiveDeltaPredictor(
            dim=dim, eta_max=step_size, rho=rho
        )
    else:
        model = FixedDeltaPredictor(dim=dim, step_size=step_size)

    model.reset_state()
    t_steps = len(inputs)

    diverged = False
    time_to_nonfinite = -1
    max_state_norm = 0.0
    min_safety_margin = 2.0
    final_error_before_failure = 0.0
    preds: list[torch.Tensor] = []

    for t in range(t_steps):
        x = inputs[t]
        y = targets[t]

        pred = model.predict_step(x)
        if not torch.isfinite(pred).all():
            diverged = True
            time_to_nonfinite = t
            break

        preds.append(pred.detach().clone())
        err = float(
            torch.linalg.norm(y - pred).item() / max(torch.linalg.norm(y).item(), 1e-8)
        )
        final_error_before_failure = err

        model.adapt_step(x, y)

        state_norm = model.get_state_norm()
        if not math.isfinite(state_norm):
            diverged = True
            time_to_nonfinite = t
            break

        max_state_norm = max(max_state_norm, state_norm)

        # Margin: distance to unconstrained 2.0 instability threshold
        x_norm_sq = float((x @ x).item())
        eta_eff = getattr(model, "last_step_size", step_size)
        current_margin = 2.0 - (eta_eff * x_norm_sq)
        min_safety_margin = min(min_safety_margin, current_margin)

    rel_error = (
        float("nan")
        if diverged
        else compute_relative_frobenius_error(targets, torch.stack(preds))
    )

    return {
        "model_type": "SafeAdaptiveDelta" if is_safe else "FixedDelta",
        "dimension": dim,
        "diverged": diverged,
        "time_to_nonfinite": time_to_nonfinite if diverged else None,
        "max_state_norm": max_state_norm,
        "min_safety_margin": min_safety_margin,
        "final_error_before_failure": final_error_before_failure,
        "rel_error": rel_error,
    }
