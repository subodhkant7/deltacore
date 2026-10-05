"""Unit and Property Tests for DeltaCore Phase 11: Nonlinear Adaptive State & Selective Retention."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from deltacore.observatory.nonlinear_plots import (
    plot_adaptation_efficiency_av,
    plot_adaptation_forgetting_ao,
    plot_memory_vs_dim_au,
    plot_model_comparison_ar,
    plot_nonlinear_error_over_time_an,
    plot_perf_vs_dim_as,
    plot_retention_ablation_aq,
    plot_retention_trajectory_ap,
    plot_runtime_vs_dim_at,
    plot_safe_vs_unsafe_aw,
)
from deltacore.streaming.metrics import (
    compute_adaptation_efficiency,
    compute_negative_transfer,
)
from deltacore.streaming.models import (
    NaivePredictor,
    NonlinearOnlineRidgePredictor,
    SelectiveRetentionPredictor,
)
from deltacore.streaming.nonlinear_regimes import (
    generate_bounded_nonlinear_operator,
    generate_nonlinear_delayed_retrieval_stream,
    generate_nonlinear_regime_stream,
    generate_stale_penalty_stream,
    simulate_nonlinear_step,
)


class TestNonlinearRegimeGeneration:
    """Tests for deterministic generation and boundedness of nonlinear dynamical streams."""

    def test_bounded_operator_generation(self) -> None:
        gen = torch.Generator().manual_seed(42)
        A, B, l_quad = generate_bounded_nonlinear_operator(
            dim=8, spectral_radius=0.80, lambda_quad=0.08, generator=gen
        )
        assert A.shape == (8, 8)
        assert B.shape == (8, 8)
        assert l_quad == 0.08
        evs = torch.linalg.eigvals(A)
        assert float(torch.max(torch.abs(evs)).item()) <= 0.81

    def test_simulate_nonlinear_step(self) -> None:
        gen = torch.Generator().manual_seed(42)
        A, B, l_quad = generate_bounded_nonlinear_operator(dim=8, generator=gen)
        x = torch.randn(8, generator=gen)
        f_x = simulate_nonlinear_step(x, A, B, l_quad)
        assert f_x.shape == (8,)
        assert torch.isfinite(f_x).all()

    def test_nonlinear_regime_stream_determinism_and_bounds(self) -> None:
        s1 = generate_nonlinear_regime_stream(seq_len=128, dim=8, seed=123)
        s2 = generate_nonlinear_regime_stream(seq_len=128, dim=8, seed=123)
        assert torch.allclose(s1.inputs, s2.inputs)
        assert torch.allclose(s1.targets, s2.targets)
        assert s1.change_points == [32, 64, 96]
        assert float(torch.max(torch.abs(s1.inputs)).item()) < 10.0
        assert float(torch.max(torch.abs(s1.targets)).item()) < 10.0

    def test_stale_penalty_stream_phases(self) -> None:
        stream = generate_stale_penalty_stream(seq_len=120, dim=8, seed=42)
        assert stream.inputs.shape == (120, 8)
        assert stream.targets.shape == (120, 8)
        assert stream.change_points == [40, 80]
        assert stream.regime_ids[:40] == [0] * 40
        assert stream.regime_ids[40:80] == [1] * 40
        assert stream.regime_ids[80:] == [0] * 40

    def test_nonlinear_delayed_retrieval_stream(self) -> None:
        stream = generate_nonlinear_delayed_retrieval_stream(
            delay=32, seq_len=128, dim=8, seed=42
        )
        assert stream.inputs.shape == (128, 8)
        assert stream.targets.shape == (128, 8)
        q_time = int(stream.metadata["query_time"])
        assert q_time == 10 + 32 + 1
        assert float(torch.linalg.norm(stream.targets[q_time]).item()) > 0.0

    def test_delayed_retrieval_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="less than seq_len"):
            generate_nonlinear_delayed_retrieval_stream(delay=200, seq_len=128)


class TestPhase11PredictorsAndIsolation:
    """Tests for Naive, NonlinearOnlineRidge, and SelectiveRetention predictors."""

    def test_naive_persistence_predictor(self) -> None:
        model = NaivePredictor(dim=8, mode="persistence")
        x1 = torch.ones(8)
        model.adapt_step(x1, x1)
        pred = model.predict_step(torch.zeros(8))
        assert torch.allclose(pred, x1)
        model.reset_state()
        assert torch.allclose(model.predict_step(x1), torch.zeros(8))

    def test_nonlinear_online_ridge_update_and_isolation(self) -> None:
        model = NonlinearOnlineRidgePredictor(dim=8, rff_dim=16, lam=0.98)
        tot_p, tr_p = model.get_param_count()
        assert tr_p == 0

        # Verify zero parameter updates during adapt
        for p in model.parameters():
            p.requires_grad = False

        x = torch.randn(8)
        y = torch.randn(8)
        pred_before = model.predict_step(x)
        model.adapt_step(x, y)
        pred_after = model.predict_step(x)

        # Output adapts toward y
        assert not torch.allclose(pred_before, pred_after)
        assert model.get_state_norm() > 0.0

        model.reset_state()
        assert model.get_state_norm() == 0.0

    def test_selective_retention_modes(self) -> None:
        x = torch.randn(8)
        y = torch.randn(8)

        # Fixed high
        m_high = SelectiveRetentionPredictor(
            dim=8, retention_mode="fixed_high", alpha_high=0.99
        )
        m_high.adapt_step(x, y)
        assert m_high.last_retention == 0.99

        # Fixed low
        m_low = SelectiveRetentionPredictor(
            dim=8, retention_mode="fixed_low", alpha_low=0.70
        )
        m_low.adapt_step(x, y)
        assert m_low.last_retention == 0.70

        # Adaptive: when error is large, alpha drops
        m_adapt = SelectiveRetentionPredictor(
            dim=8, retention_mode="adaptive", gamma=1.5
        )
        m_adapt.adapt_step(x, y * 5.0)
        assert m_adapt.last_retention < 0.99
        assert m_adapt.last_retention >= m_adapt.alpha_min

        # Oracle: when t in oracle_change_points, alpha = 0.0
        m_oracle = SelectiveRetentionPredictor(
            dim=8, retention_mode="oracle", oracle_change_points=[0]
        )
        assert m_oracle.current_timestep == 0
        m_oracle.adapt_step(x, y)
        assert m_oracle.last_retention == 0.0

    def test_selective_retention_linear_vs_nonlinear_ablation_e(self) -> None:
        m_nl = SelectiveRetentionPredictor(
            dim=8, use_nonlinear_features=True, feat_dim=8
        )
        m_lin = SelectiveRetentionPredictor(
            dim=8, use_nonlinear_features=False, feat_dim=8
        )
        x = torch.ones(8) * 2.0
        phi_nl = m_nl._phi(x)
        phi_lin = m_lin._phi(x)
        assert torch.allclose(phi_lin, x)
        assert torch.allclose(phi_nl, torch.tanh(x))

    def test_zero_parameter_updates_in_selective_retention(self) -> None:
        model = SelectiveRetentionPredictor(dim=8, use_nonlinear_features=True)
        for p in model.parameters():
            p.requires_grad = False
        initial_params = [p.clone() for p in model.parameters()]

        x = torch.randn(8)
        y = torch.randn(8)
        model.adapt_step(x, y)

        for p_init, p_curr in zip(initial_params, model.parameters(), strict=True):
            assert torch.allclose(p_init, p_curr)
            assert p_curr.grad is None


class TestPhase11Metrics:
    """Tests for negative transfer and adaptation efficiency metrics."""

    def test_negative_transfer_calculation(self) -> None:
        # If continuous error is 0.80 and reset error is 0.50, negative transfer is +0.30
        nt = compute_negative_transfer(error_continuous_b=0.80, error_reset_b=0.50)
        assert pytest.approx(nt) == 0.30

    def test_adaptation_efficiency_calculation(self) -> None:
        eff = compute_adaptation_efficiency(
            error_reduction=0.40, adaptation_energy=10.0
        )
        assert pytest.approx(eff, rel=1e-3) == 0.04


class TestObservatoryPlotsANThroughAW:
    """Smoke test ensuring plots AN through AW generate cleanly without errors."""

    def test_plots_an_through_aw_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            trajs = {"ModelA": [0.5, 0.4, 0.3], "ModelB": [0.6, 0.5, 0.4]}
            plot_nonlinear_error_over_time_an(trajs, [1], out_dir / "plot_an.png")
            assert (out_dir / "plot_an.png").exists()

            plot_adaptation_forgetting_ao(
                ["M1", "M2"], [0.2, 0.3], [0.05, 0.10], out_dir / "plot_ao.png"
            )
            assert (out_dir / "plot_ao.png").exists()

            plot_retention_trajectory_ap(
                {"M1": [1.0, 0.8, 0.9]}, [1], out_dir / "plot_ap.png"
            )
            assert (out_dir / "plot_ap.png").exists()

            plot_retention_ablation_aq(
                ["High", "Low"], [0.5, 0.6], [0.1, -0.05], out_dir / "plot_aq.png"
            )
            assert (out_dir / "plot_aq.png").exists()

            plot_model_comparison_ar(["M1", "M2"], [0.4, 0.5], out_dir / "plot_ar.png")
            assert (out_dir / "plot_ar.png").exists()

            dims = [8, 16]
            plot_perf_vs_dim_as(dims, {"M1": [0.5, 0.6]}, out_dir / "plot_as.png")
            assert (out_dir / "plot_as.png").exists()

            plot_runtime_vs_dim_at(dims, {"M1": [30.0, 45.0]}, out_dir / "plot_at.png")
            assert (out_dir / "plot_at.png").exists()

            plot_memory_vs_dim_au(dims, {"M1": [256, 1024]}, out_dir / "plot_au.png")
            assert (out_dir / "plot_au.png").exists()

            plot_adaptation_efficiency_av(
                ["M1", "M2"], [0.05, 0.08], out_dir / "plot_av.png"
            )
            assert (out_dir / "plot_av.png").exists()

            plot_safe_vs_unsafe_aw(
                [("Unsafe", "Safe")], [np.nan], [0.5], out_dir / "plot_aw.png"
            )
            assert (out_dir / "plot_aw.png").exists()
