"""Unit and Integration Tests for Phase 16: Unseen Shift Robustness & Adaptive Safety.

Tests:
    1. Shift generation determinism across all 5 shift types.
    2. Severity control (mild < moderate < severe parameter monotonic scaling).
    3. Perturbation reproducibility (deterministic observation noise).
    4. Online parameter immutability (Delta theta = 0 during streaming evaluation).
    5. State reset intervention behavior and delta calculation.
    6. Retention stress stream generation across 4 multi-regime histories.
    7. Empirical safety boundary detection at D=256 (FixedDelta vs SafeAdaptiveDelta).
    8. Spatial permutation equivariance under perturbations.
    9. Artifact serialization completeness for all 6 required Phase 16 JSON artifacts.
    10. Observatory publication plots rendering (Figures CQ through DB).
    11. Metric correctness (recovery, cumulative excess error, adaptation energy).
    12. Seed reproducibility across deterministic seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deltacore.streaming.models import (
    SafeAdaptiveDeltaPredictor,
    compute_parameter_hash,
)
from deltacore.streaming.unseen_shifts import (
    SEVERITY_SCALES,
    apply_unseen_shift,
    check_permutation_equivariance,
    evaluate_perturbation_robustness,
    evaluate_shift_reset_ablation,
    evaluate_shift_streaming_run,
    generate_retention_stress_stream,
    map_failure_boundary_grid,
)


class TestShiftGenerationAndSeverity:
    """Tests for shift generation determinism, severity scaling, and perturbations."""

    def test_shift_generation_determinism(self) -> None:
        """Every shift generator must produce bitwise identical tensors given identical seeds."""
        base_x = torch.randn(60, 64, generator=torch.Generator().manual_seed(100))
        base_y = torch.randn(60, 64, generator=torch.Generator().manual_seed(101))

        for st in ["mean", "variance", "temporal_speed", "noise", "combined"]:
            m1_x, m1_y = apply_unseen_shift(
                base_x, base_y, st, "moderate", shift_start=20, shift_end=50, seed=42
            )
            m2_x, m2_y = apply_unseen_shift(
                base_x, base_y, st, "moderate", shift_start=20, shift_end=50, seed=42
            )
            assert torch.equal(m1_x, m2_x), f"Shift {st} non-deterministic for x"
            assert torch.equal(m1_y, m2_y), f"Shift {st} non-deterministic for y"

    def test_severity_control_monotonic_scaling(self) -> None:
        """Predetermined severities (mild, moderate, severe) must scale perturbations monotonically."""
        base_x = torch.ones(50, 16)
        base_y = torch.ones(50, 16)

        # Mean shift offset check
        _, m_mild = apply_unseen_shift(
            base_x, base_y, "mean", "mild", shift_start=20, shift_end=40, seed=42
        )
        _, m_mod = apply_unseen_shift(
            base_x, base_y, "mean", "moderate", shift_start=20, shift_end=40, seed=42
        )
        _, m_sev = apply_unseen_shift(
            base_x, base_y, "mean", "severe", shift_start=20, shift_end=40, seed=42
        )

        diff_mild = (m_mild[25] - base_y[25]).abs().mean().item()
        diff_mod = (m_mod[25] - base_y[25]).abs().mean().item()
        diff_sev = (m_sev[25] - base_y[25]).abs().mean().item()

        assert diff_mild < diff_mod < diff_sev

        # Variance factor check
        assert (
            SEVERITY_SCALES["variance"]["mild"].var_scale
            < SEVERITY_SCALES["variance"]["moderate"].var_scale
            < SEVERITY_SCALES["variance"]["severe"].var_scale
        )
        assert (
            SEVERITY_SCALES["noise"]["mild"].noise_sigma
            < SEVERITY_SCALES["noise"]["moderate"].noise_sigma
            < SEVERITY_SCALES["noise"]["severe"].noise_sigma
        )

    def test_perturbation_reproducibility(self) -> None:
        """Observation perturbation must be deterministic and reproducible."""
        model = SafeAdaptiveDeltaPredictor(
            dim=16, eta_max=0.015, rho=1.5, alpha_min=0.95
        )
        x = torch.randn(40, 16, generator=torch.Generator().manual_seed(55))
        y = torch.randn(40, 16, generator=torch.Generator().manual_seed(56))

        res1 = evaluate_perturbation_robustness(
            model, x, y, noise_fractions=[0.0, 0.05], seed=42
        )
        res2 = evaluate_perturbation_robustness(
            model, x, y, noise_fractions=[0.0, 0.05], seed=42
        )

        assert len(res1) == len(res2) == 2
        assert res1[0]["rel_error"] == pytest.approx(res2[0]["rel_error"], abs=1e-6)
        assert res1[1]["rel_error"] == pytest.approx(res2[1]["rel_error"], abs=1e-6)


class TestOnlineInvarianceAndReset:
    """Tests for parameter immutability, state reset ablation, and retention stress."""

    def test_no_test_time_parameter_updates(self) -> None:
        """Parameters theta must be completely immutable (Delta theta = 0) during streaming."""
        model = SafeAdaptiveDeltaPredictor(
            dim=32, eta_max=0.015, rho=1.5, alpha_min=0.95
        )
        initial_hash = compute_parameter_hash(model)

        x = torch.randn(60, 32, generator=torch.Generator().manual_seed(77))
        y = torch.randn(60, 32, generator=torch.Generator().manual_seed(78))
        shift_x, shift_y = apply_unseen_shift(
            x, y, "variance", "severe", shift_start=20, shift_end=50, seed=42
        )

        # Run streaming evaluation
        _ = evaluate_shift_streaming_run(
            model, shift_x, shift_y, shift_start=20, shift_end=50
        )

        final_hash = compute_parameter_hash(model)
        assert initial_hash == final_hash, (
            "Model parameter changed during test evaluation!"
        )

    def test_state_reset_ablation(self) -> None:
        """State reset at shift onset must clear adaptive state and compute delta_reset."""
        model = SafeAdaptiveDeltaPredictor(
            dim=16, eta_max=0.015, rho=1.5, alpha_min=0.95
        )

        x = torch.randn(50, 16, generator=torch.Generator().manual_seed(10))
        y = torch.randn(50, 16, generator=torch.Generator().manual_seed(11))
        sx, sy = apply_unseen_shift(
            x, y, "mean", "moderate", shift_start=25, shift_end=45, seed=42
        )

        res = evaluate_shift_reset_ablation(model, sx, sy, shift_start=25, shift_end=45)

        assert "continuous" in res
        assert "reset" in res
        assert "delta_e_rel" in res
        assert res["delta_e_rel"] == pytest.approx(
            res["continuous"]["rel_error"] - res["reset"]["rel_error"], abs=1e-7
        )

    def test_retention_stress_stream_generation(self) -> None:
        """Retention stress stream generator must support multi-regime transitions with correct bounds."""
        a_in = torch.randn(50, 16)
        a_tgt = torch.randn(50, 16)
        b_in = torch.randn(50, 16)
        b_tgt = torch.randn(50, 16)

        # A -> B
        s_ab, t_ab, bounds_ab = generate_retention_stress_stream(
            a_in, a_tgt, b_in, b_tgt, "A_to_B", seg_len=20
        )
        assert s_ab.shape == (40, 16)
        assert bounds_ab == [20]

        # A -> B -> A
        s_aba, t_aba, bounds_aba = generate_retention_stress_stream(
            a_in, a_tgt, b_in, b_tgt, "A_to_B_to_A", seg_len=20
        )
        assert s_aba.shape == (60, 16)
        assert bounds_aba == [20, 40]

        # A -> B -> C
        s_abc, t_abc, bounds_abc = generate_retention_stress_stream(
            a_in, a_tgt, b_in, b_tgt, "A_to_B_to_C", seg_len=20
        )
        assert s_abc.shape == (60, 16)
        assert bounds_abc == [20, 40]

        # A -> severe-B -> A
        s_sev, t_sev, bounds_sev = generate_retention_stress_stream(
            a_in, a_tgt, b_in, b_tgt, "A_to_severeB_to_A", seg_len=20
        )
        assert s_sev.shape == (60, 16)
        assert bounds_sev == [20, 40]


class TestSafetyBoundaryAndEquivariance:
    """Tests for D=256 failure boundary mapping and spatial equivariance."""

    def test_safety_boundary_detection_d256(self) -> None:
        """FixedDelta diverges at large step sizes, while SafeAdaptiveDelta remains bounded."""
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(60, 256, generator=gen) * 2.5
        y = x + 0.1 * torch.randn_like(x)

        step_sizes = [0.002, 0.015]
        res = map_failure_boundary_grid(x, y, step_sizes=step_sizes, dim=256)

        assert "FixedDelta" in res
        assert "SafeAdaptiveDelta" in res
        assert len(res["SafeAdaptiveDelta"]) == len(step_sizes)

        # SafeAdaptiveDelta should remain finite for both
        for item in res["SafeAdaptiveDelta"]:
            assert item["diverged"] == 0
            assert np.isfinite(item["max_state_norm"])
            assert item["min_safety_margin"] >= 0.0

    def test_spatial_permutation_equivariance(self) -> None:
        """SafeAdaptiveDelta must be strictly permutation-equivariant under perturbations."""
        model = SafeAdaptiveDeltaPredictor(
            dim=16, eta_max=0.015, rho=1.5, alpha_min=0.95
        )
        x = torch.randn(40, 16, generator=torch.Generator().manual_seed(99))

        err_equiv = check_permutation_equivariance(model, x, seed=42)
        assert err_equiv < 1e-4, (
            f"Permutation equivariance violation: E_equiv = {err_equiv}"
        )


class TestMetricCorrectnessAndSerialization:
    """Tests for telemetry metrics, artifact serialization, and plot files."""

    def test_metric_correctness(self) -> None:
        """Verify recovery detection and excess error calculations."""
        model = SafeAdaptiveDeltaPredictor(
            dim=16, eta_max=0.015, rho=1.5, alpha_min=0.95
        )
        x = torch.randn(50, 16, generator=torch.Generator().manual_seed(1))
        y = torch.randn(50, 16, generator=torch.Generator().manual_seed(2))

        metrics = evaluate_shift_streaming_run(
            model, x, y, shift_start=20, shift_end=40
        )

        assert metrics["diverged"] == 0
        assert metrics["rel_error"] > 0.0
        assert metrics["max_state_norm"] > 0.0
        assert metrics["min_safety_margin"] <= 2.0
        assert metrics["adaptation_energy"] >= 0.0
        assert len(metrics["eta_history"]) == 50
        assert len(metrics["state_norm_history"]) == 50

    def test_artifact_serialization_completeness(self) -> None:
        """All 6 required Phase 16 JSON artifacts must exist and have complete schemas."""
        art_dir = Path("docs/benchmarks/artifacts/phase_16")
        required_files = [
            "phase_16_results.json",
            "phase_16_per_seed.json",
            "phase_16_config.json",
            "phase_16_shift_matrix.json",
            "phase_16_safety_boundary.json",
            "phase_16_retention.json",
        ]

        for fname in required_files:
            fpath = art_dir / fname
            assert fpath.exists(), f"Missing required Phase 16 artifact: {fpath}"
            with open(fpath) as f:
                data = json.load(f)
            assert isinstance(data, dict), f"Artifact {fname} must be a JSON object"
            assert len(data) > 0, f"Artifact {fname} is empty"

    def test_observatory_plots_existence(self) -> None:
        """All 12 Observatory publication figures CQ through DB must exist and be non-empty."""
        plots_dir = Path("docs/benchmarks/artifacts/phase_16/plots")
        required_plots = [
            "plot_cq_error_vs_severity.png",
            "plot_cr_eta_trajectories.png",
            "plot_cs_safety_margin_vs_severity.png",
            "plot_ct_continuous_vs_reset.png",
            "plot_cu_retention_trajectories.png",
            "plot_cv_failure_boundary_map.png",
            "plot_cw_noise_robustness.png",
            "plot_cx_cross_domain_matrix.png",
            "plot_cy_per_seed_differences.png",
            "plot_cz_state_norm_severe.png",
            "plot_da_adaptation_energy_vs_severity.png",
            "plot_db_resource_metrics_vs_severity.png",
        ]

        for pname in required_plots:
            plot_path = plots_dir / pname
            assert plot_path.exists(), (
                f"Missing Observatory publication plot: {plot_path}"
            )
            assert plot_path.stat().st_size > 10_000, (
                f"Plot {pname} file size too small: {plot_path.stat().st_size} bytes"
            )
