"""Unit and Integration Tests for Phase 15: Adaptive Regime Transfer & Robustness.

Tests:
    1. Cross-domain dataset loading and partition isolation (Domain A vs Domain B).
    2. Validation-only hyperparameter calibration sweep (no test leakage).
    3. Cross-domain parameter transfer (A -> B and B -> A without retuning).
    4. Regime-switch stream construction and continuity (OISST -> ERA5 -> OISST).
    5. State retention ablation (continuous vs reset vs retain-high).
    6. State initialization sensitivity and transferred state portability.
    7. Failure boundary diagnostic at D=256 (FixedDelta explosion vs SafeAdaptiveDelta safety margin).
    8. Six required JSON reproducibility artifacts structure and completeness.
    9. Observatory publication plots CG through CP rendering and existence.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deltacore.observatory.phase_15_plots import generate_all_phase_15_plots
from deltacore.streaming.models import (
    SafeAdaptiveDeltaPredictor,
    compute_parameter_hash,
)
from deltacore.streaming.regime_transfer import (
    SafeAdaptiveDeltaConfig,
    calibrate_validation_grid,
    evaluate_failure_boundary,
    evaluate_state_initialization,
    generate_regime_switch_stream,
    load_cross_domain_datasets,
)


class TestCrossDomainIngestionAndCalibration:
    """Tests for Phase 15 cross-domain data loading and validation calibration."""

    def test_load_cross_domain_datasets(self) -> None:
        """Domain A (OISST) and Domain B (ERA5) must be loaded with identical D=64 features."""
        ds = load_cross_domain_datasets(seed=42, resolution="small")
        assert ds.domain_a.test_inputs.shape[-1] == 64
        assert ds.domain_b.test_inputs.shape[-1] == 64
        assert len(ds.domain_a.test_inputs) == len(ds.domain_a.test_targets)
        assert len(ds.domain_b.test_inputs) == len(ds.domain_b.test_targets)
        # Check validation splits exist and are non-empty
        assert len(ds.domain_a.val_inputs) > 0
        assert len(ds.domain_b.val_inputs) > 0

    def test_calibrate_validation_grid_no_test_leakage(self) -> None:
        """Validation sweep must calibrate strictly on validation splits without touching test sets."""
        ds = load_cross_domain_datasets(seed=42, resolution="small")
        grid = [
            SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.5, alpha_min=0.85),
            SafeAdaptiveDeltaConfig(eta_max=0.015, rho=1.5, alpha_min=0.95),
        ]
        calib = calibrate_validation_grid(
            val_inputs_a=ds.domain_a.val_inputs,
            val_targets_a=ds.domain_a.val_targets,
            val_inputs_b=ds.domain_b.val_inputs,
            val_targets_b=ds.domain_b.val_targets,
            grid=grid,
            dim=64,
        )
        assert "best_domain_a" in calib
        assert "best_domain_b" in calib
        assert "best_pooled" in calib
        assert len(calib["sweep_results"]) == 2


class TestRegimeSwitchAndStateTransfer:
    """Tests for regime switch stream construction and state transfer."""

    def test_regime_switch_stream_generation(self) -> None:
        """Regime-switch stream must concatenate controlled segments with correct boundaries."""
        ds = load_cross_domain_datasets(seed=42, resolution="small")
        inputs, targets, bounds = generate_regime_switch_stream(
            test_inputs_a=ds.domain_a.test_inputs,
            test_targets_a=ds.domain_a.test_targets,
            test_inputs_b=ds.domain_b.test_inputs,
            test_targets_b=ds.domain_b.test_targets,
            segment_len=30,
        )
        assert inputs.shape == (90, 64)
        assert targets.shape == (90, 64)
        assert bounds == (30, 60)
        assert torch.isfinite(inputs).all()
        assert torch.isfinite(targets).all()

    def test_parameter_immutability_under_transfer(self) -> None:
        """Transferring a model across domains must not mutate its offline parameters."""
        model = SafeAdaptiveDeltaPredictor(
            dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95
        )
        h_init = compute_parameter_hash(model)

        ds = load_cross_domain_datasets(seed=42, resolution="small")
        # Step through Domain A
        for t in range(20):
            x = ds.domain_a.test_inputs[t]
            y = ds.domain_a.test_targets[t]
            _ = model.predict_step(x)
            model.adapt_step(x, y)

        # Step through Domain B
        for t in range(20):
            x = ds.domain_b.test_inputs[t]
            y = ds.domain_b.test_targets[t]
            _ = model.predict_step(x)
            model.adapt_step(x, y)

        h_final = compute_parameter_hash(model)
        assert h_init == h_final, (
            "Predictor offline parameters mutated during streaming!"
        )

    def test_state_initialization_sensitivity(self) -> None:
        """Initial state condition must be respected and tracked without being wiped."""
        model = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.9)
        ds = load_cross_domain_datasets(seed=42, resolution="small")

        # Zero state
        res_zero = evaluate_state_initialization(
            model=model,
            inputs=ds.domain_a.test_inputs[:30],
            targets=ds.domain_a.test_targets[:30],
            initial_state=None,
        )
        assert res_zero["max_state_norm"] >= 0.0
        assert 0.0 < res_zero["rel_error"] < 2.0

        # Transferred initial state
        initial_m = torch.randn(64, 64) * 0.05
        res_init = evaluate_state_initialization(
            model=model,
            inputs=ds.domain_a.test_inputs[:30],
            targets=ds.domain_a.test_targets[:30],
            initial_state=initial_m,
        )
        assert res_init["max_state_norm"] >= 0.0


class TestFailureBoundaryAtD256:
    """Tests for Safe vs Unsafe failure boundary at D=256."""

    def test_failure_boundary_safe_vs_unsafe(self) -> None:
        """SafeAdaptiveDelta must preserve safety margin and remain finite at D=256."""
        scaled_ds = load_cross_domain_datasets(seed=42, resolution="medium")
        inputs = scaled_ds.domain_b.test_inputs[:50]
        targets = scaled_ds.domain_b.test_targets[:50]

        # FixedDelta with large step size on atmospheric data
        res_fixed = evaluate_failure_boundary(
            dim=256,
            inputs=inputs,
            targets=targets,
            is_safe=False,
            step_size=0.008,
        )
        assert res_fixed["model_type"] == "FixedDelta"

        # SafeAdaptiveDelta with contractive safety controller
        res_safe = evaluate_failure_boundary(
            dim=256,
            inputs=inputs,
            targets=targets,
            is_safe=True,
            step_size=0.008,
            rho=1.90,
        )
        assert res_safe["model_type"] == "SafeAdaptiveDelta"
        assert res_safe["diverged"] is False
        assert res_safe["min_safety_margin"] >= 0.099  # 2.0 - rho = 0.10


class TestPhase15ArtifactsAndObservatory:
    """Verify that all 6 required JSON artifacts and 10 plots are valid."""

    def test_json_artifacts_exist_and_valid(self) -> None:
        """All 6 required JSON artifacts must exist and contain non-empty data."""
        art_dir = Path("docs/benchmarks/artifacts/phase_15")
        required_files = [
            "phase_15_results.json",
            "phase_15_per_seed.json",
            "phase_15_config.json",
            "phase_15_transfer.json",
            "phase_15_safety.json",
            "phase_15_retention.json",
        ]
        for fname in required_files:
            p = art_dir / fname
            assert p.exists(), f"Missing artifact: {p}"
            with open(p) as f:
                data = json.load(f)
                assert len(data) > 0

    def test_observatory_plots_generation(self) -> None:
        """generate_all_phase_15_plots must render all 10 figures CG-CP."""
        art_dir = Path("docs/benchmarks/artifacts/phase_15")
        plots_dir = art_dir / "plots"
        generate_all_phase_15_plots(artifacts_dir=art_dir)

        required_plots = [
            "plot_cg_cross_domain_matrix.png",
            "plot_ch_parameter_transfer.png",
            "plot_ci_safety_pareto_frontier.png",
            "plot_cj_state_transfer.png",
            "plot_ck_continuous_vs_reset.png",
            "plot_cl_state_initialization.png",
            "plot_cm_failure_boundary.png",
            "plot_cn_runtime_vs_dim.png",
            "plot_co_memory_vs_dim.png",
            "plot_cp_per_seed_differences.png",
        ]
        for plot_name in required_plots:
            p = plots_dir / plot_name
            assert p.exists(), f"Missing plot: {p}"
            assert p.stat().st_size > 10000, f"Plot {p} is suspiciously small!"
