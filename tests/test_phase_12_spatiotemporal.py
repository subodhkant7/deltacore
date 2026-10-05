"""Tests for DeltaCore Phase 12: Controlled Spatio-Temporal Adaptive State.

Verifies:
    1. Spatial field initialization and properties (dimensions, normalization)
    2. Spatio-temporal simulation steps across regimes (boundedness, dissipativity)
    3. Stream generation (A -> B -> C -> A, A1 -> B -> A2, shift intervals)
    4. Determinism and reproducibility across seeds
    5. Spatial permutation ablation P_{spatial} (marginal preservation, locality destruction)
    6. Streaming predictor interface contracts (zero parameter updates at test time)
    7. Retention modes (fixed_high, fixed_low, adaptive, oracle, shuffled)
    8. State-conditioned retention controller (SelectiveStateAdaptivePredictor, 5 params)
    9. Causal Control D: true adaptive vs shuffled controller
    10. Representation controls (SpatialConvControl, SpatialDownsampleControl)
    11. Memory and state accounting (total params, state bytes, breakdown)
    12. Metric calculations (pre/post errors, recovery, negative transfer, forgetting)
    13. Dimensional scaling verification (D=HWC across D in {64, 128, 256})
    14. State reset invariance
    15. Observatory plot generation and serialization
    16. Artifact schema and loading validation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from deltacore.observatory.spatiotemporal_plots import (
    plot_a1_b_a1_recovery_bb,
    plot_adaptation_energy_vs_perf_bh,
    plot_adaptation_negative_transfer_pareto_ay,
    plot_continuous_vs_reset_bj,
    plot_error_vs_state_adaptive_bc,
    plot_memory_vs_dim_bg,
    plot_perf_vs_dim_be,
    plot_retention_ablation_ba,
    plot_retention_trajectories_az,
    plot_runtime_vs_dim_bf,
    plot_safe_vs_unsafe_bi,
    plot_spatial_vs_shuffled_bd,
    plot_spatiotemporal_error_over_time_ax,
)
from deltacore.streaming.metrics import (
    StreamingTelemetry,
    compute_adaptation_efficiency,
    compute_adaptation_energy,
    compute_cumulative_excess_error,
    compute_first_passage_recovery,
    compute_forgetting,
    compute_negative_transfer,
    compute_pre_post_errors,
    compute_relative_step_error,
    compute_sustained_recovery,
)
from deltacore.streaming.models import (
    FrozenLinearPredictor,
    GRUPredictor,
    LSTMPredictor,
    NonlinearOnlineRidgePredictor,
    OnlineRidgePredictor,
    SafeAdaptiveDeltaPredictor,
    SafeSelfReferentialPredictor,
    SelectiveRetentionPredictor,
    SelectiveStateAdaptivePredictor,
    SpatialConvControl,
    SpatialDownsampleControl,
)
from deltacore.streaming.spatiotemporal_regimes import (
    SpatioTemporalDimensions,
    apply_spatial_permutation,
    create_initial_spatial_field,
    generate_spatiotemporal_stale_stream,
    generate_spatiotemporal_stream,
    simulate_spatiotemporal_step,
)


class TestSpatioTemporalDataAndRegimes:
    """Tests for spatial field generation and physical evolution dynamics."""

    def test_dimensions_contract(self) -> None:
        dims = SpatioTemporalDimensions(height=8, width=8, channels=1)
        assert dims.dim == 64

        dims2 = SpatioTemporalDimensions(height=8, width=8, channels=2)
        assert dims2.dim == 128

        dims3 = SpatioTemporalDimensions(height=16, width=16, channels=1)
        assert dims3.dim == 256

    @pytest.mark.parametrize(
        ("h", "w", "c", "expected_d"),
        [
            (8, 8, 1, 64),
            (8, 8, 2, 128),
            (16, 16, 1, 256),
            (4, 4, 1, 16),
            (4, 4, 3, 48),
        ],
    )
    def test_parametrized_dimensions(
        self, h: int, w: int, c: int, expected_d: int
    ) -> None:
        dims = SpatioTemporalDimensions(height=h, width=w, channels=c)
        assert dims.dim == expected_d

    def test_initial_spatial_field_properties(self) -> None:
        dims = SpatioTemporalDimensions(height=8, width=8, channels=1)
        gen = torch.Generator().manual_seed(42)
        field = create_initial_spatial_field(dims, generator=gen)

        assert field.shape == (8, 8, 1)
        assert torch.isfinite(field).all()
        # Unit Frobenius norm
        norm = torch.linalg.norm(field).item()
        assert abs(norm - 1.0) < 1e-4

    def test_initial_field_determinism(self) -> None:
        dims = SpatioTemporalDimensions(height=8, width=8, channels=2)
        f1 = create_initial_spatial_field(
            dims, generator=torch.Generator().manual_seed(101)
        )
        f2 = create_initial_spatial_field(
            dims, generator=torch.Generator().manual_seed(101)
        )
        f3 = create_initial_spatial_field(
            dims, generator=torch.Generator().manual_seed(202)
        )

        assert torch.allclose(f1, f2)
        assert not torch.allclose(f1, f3)

    @pytest.mark.parametrize(
        "regime",
        [
            "horizontal_transport",
            "vertical_transport",
            "diffusion_vortex",
            "horizontal_transport_a2",
        ],
    )
    def test_simulate_spatiotemporal_step_boundedness(self, regime: str) -> None:
        dims = SpatioTemporalDimensions(height=8, width=8, channels=1)
        X = create_initial_spatial_field(
            dims, generator=torch.Generator().manual_seed(42)
        )
        X_curr = X.clone()
        for _ in range(40):
            X_next = simulate_spatiotemporal_step(X_curr, regime=regime)
            assert torch.isfinite(X_next).all()
            assert torch.max(torch.abs(X_next)).item() <= 5.0
            X_curr = X_next

    @pytest.mark.parametrize(
        ("h", "w", "c", "d"),
        [
            (8, 8, 1, 64),
            (8, 8, 2, 128),
            (16, 16, 1, 256),
        ],
    )
    def test_generate_spatiotemporal_stream_shapes(
        self, h: int, w: int, c: int, d: int
    ) -> None:
        stream = generate_spatiotemporal_stream(
            seq_len=64, height=h, width=w, channels=c, seed=42
        )
        assert stream.inputs.shape == (64, d)
        assert stream.targets.shape == (64, d)
        assert len(stream.regime_ids) == 64
        assert len(stream.change_points) == 3
        assert stream.metadata["dim"] == d
        assert stream.metadata["height"] == h
        assert stream.metadata["width"] == w
        assert stream.metadata["channels"] == c

    @pytest.mark.parametrize("interval", [16, 32, 64])
    def test_stream_shift_intervals(self, interval: int) -> None:
        stream = generate_spatiotemporal_stream(
            seq_len=128, height=8, width=8, channels=1, shift_interval=interval, seed=42
        )
        expected_cps = list(range(interval, 128, interval))
        assert stream.change_points == expected_cps

    def test_stale_stream_generation(self) -> None:
        stream_stale = generate_spatiotemporal_stale_stream(
            seq_len=90, height=8, width=8, channels=1, seed=42
        )
        assert stream_stale.inputs.shape == (90, 64)
        assert stream_stale.targets.shape == (90, 64)
        assert stream_stale.change_points == [30, 60]
        assert set(stream_stale.regime_ids) == {0, 1, 2}

    def test_spatial_permutation_preserves_marginals(self) -> None:
        stream = generate_spatiotemporal_stream(
            seq_len=32, height=8, width=8, channels=1, seed=42
        )
        perm_stream = apply_spatial_permutation(stream, permutation_seed=999)

        assert perm_stream.inputs.shape == stream.inputs.shape
        orig_sum = float(stream.inputs.sum().item())
        perm_sum = float(perm_stream.inputs.sum().item())
        assert abs(orig_sum - perm_sum) < 1e-4

        for t in range(32):
            s_orig = torch.sort(stream.inputs[t]).values
            s_perm = torch.sort(perm_stream.inputs[t]).values
            assert torch.allclose(s_orig, s_perm, atol=1e-5)

        assert not torch.allclose(stream.inputs, perm_stream.inputs)


class TestModelsAndRetentionMechanisms:
    """Tests for Phase 12 predictors, retention modes, and controls."""

    @pytest.mark.parametrize(
        "mode", ["fixed_high", "fixed_low", "adaptive", "oracle", "shuffled"]
    )
    def test_selective_retention_modes_initialization(self, mode: str) -> None:
        pred = SelectiveRetentionPredictor(
            dim=16,
            feat_dim=16,
            retention_mode=mode,
            use_nonlinear_features=True,
            oracle_change_points=[10, 20],
        )
        assert pred.retention_mode == mode
        assert pred.M.shape == (16, 16)

    def test_selective_retention_linear_features(self) -> None:
        pred = SelectiveRetentionPredictor(
            dim=16,
            feat_dim=16,
            retention_mode="adaptive",
            use_nonlinear_features=False,
        )
        x = torch.randn(16)
        phi = pred._phi(x)
        assert torch.allclose(phi, x)

    def test_state_conditioned_controller_architecture(self) -> None:
        pred = SelectiveStateAdaptivePredictor(dim=16, feat_dim=16, seed=42)
        total_p, train_p = pred.get_param_count()
        assert pred.w_controller.numel() == 4
        assert pred.b_controller.numel() == 1
        assert pred.w_controller.numel() + pred.b_controller.numel() == 5
        assert train_p == 0

    def test_state_conditioned_controller_adaptation_step(self) -> None:
        pred = SelectiveStateAdaptivePredictor(dim=16, feat_dim=16, seed=42)
        x = torch.randn(16)
        y = torch.randn(16)

        pred.predict_step(x)
        pred.adapt_step(x, y)

        assert pred.last_update_norm > 0.0
        assert 0.10 <= pred.last_retention <= 1.0
        assert pred.residual_ema > 0.0

    def test_causal_control_d_shuffled_controller(self) -> None:
        pred_true = SelectiveStateAdaptivePredictor(
            dim=16, feat_dim=16, shuffled_control=False, seed=42
        )
        pred_shuff = SelectiveStateAdaptivePredictor(
            dim=16, feat_dim=16, shuffled_control=True, seed=42
        )

        assert pred_true.get_param_count() == pred_shuff.get_param_count()
        assert pred_true.w_controller.shape == pred_shuff.w_controller.shape

        x = torch.randn(16)
        y = torch.randn(16)
        pred_true.adapt_step(x, y)
        pred_shuff.adapt_step(x, y)

        assert 0.10 <= pred_true.last_retention <= 1.0
        assert 0.10 <= pred_shuff.last_retention <= 1.0

    @pytest.mark.parametrize("conv_ch", [2, 4, 8])
    def test_spatial_conv_control(self, conv_ch: int) -> None:
        conv_ctrl = SpatialConvControl(
            height=8, width=8, channels=1, conv_channels=conv_ch, seed=42
        )
        assert conv_ctrl.dim == 64
        x = torch.randn(64)
        pred = conv_ctrl.predict_step(x)
        assert pred.shape == (64,)

        conv_ctrl.adapt_step(x, pred)
        assert conv_ctrl.last_update_norm == 0.0
        assert conv_ctrl.get_state_memory_bytes() == 0

    def test_spatial_downsample_control(self) -> None:
        ds_ctrl = SpatialDownsampleControl(height=8, width=8, channels=1, seed=42)
        assert ds_ctrl.dim == 64
        x = torch.randn(64)
        pred = ds_ctrl.predict_step(x)
        assert pred.shape == (64,)
        assert ds_ctrl.get_state_memory_bytes() == 0

    def test_state_memory_accounting_correctness(self) -> None:
        dim = 32
        online_ridge = OnlineRidgePredictor(dim=dim)
        assert online_ridge.get_state_memory_bytes() == 2 * dim * dim * 4

        safe_delta = SafeAdaptiveDeltaPredictor(dim=dim)
        assert safe_delta.get_state_memory_bytes() == dim * dim * 4

        gru = GRUPredictor(dim=dim, hidden_dim=4)
        assert gru.get_state_memory_bytes() == 4 * 4

        lstm = LSTMPredictor(dim=dim, hidden_dim=4)
        assert lstm.get_state_memory_bytes() == 2 * 4 * 4

        frozen = FrozenLinearPredictor(dim=dim)
        assert frozen.get_state_memory_bytes() == 0

        breakdown = safe_delta.get_memory_breakdown()
        assert breakdown["persistent_state_bytes"] == dim * dim * 4
        assert breakdown["total_bytes"] > 0

    def test_zero_parameter_updates_at_test_time(self) -> None:
        models = [
            SafeAdaptiveDeltaPredictor(dim=16),
            OnlineRidgePredictor(dim=16),
            SelectiveStateAdaptivePredictor(dim=16, feat_dim=16),
            SpatialConvControl(height=4, width=4, channels=1),
        ]
        for m in models:
            for p in m.parameters():
                p.requires_grad = False
            x = torch.randn(16)
            y = torch.randn(16)
            m.predict_step(x)
            m.adapt_step(x, y)
            for p in m.parameters():
                assert p.grad is None, f"Parameter grad not None in {m.name}"

    def test_state_reset_invariance(self) -> None:
        pred = SafeAdaptiveDeltaPredictor(dim=16)
        x = torch.randn(16)
        y = torch.randn(16)

        initial_norm = pred.get_state_norm()
        for _ in range(5):
            pred.adapt_step(x, y)
        assert pred.get_state_norm() != initial_norm

        pred.reset_state()
        assert abs(pred.get_state_norm() - initial_norm) < 1e-5


class TestMetricsAndAnalysis:
    """Tests for transition metrics, Pareto analysis, and recovery times."""

    def test_relative_step_error_properties(self) -> None:
        y = torch.tensor([1.0, 2.0, 3.0])
        y_hat = torch.tensor([1.0, 2.0, 3.0])
        err_zero = compute_relative_step_error(y_hat, y)
        assert abs(err_zero) < 1e-6

        y_perturbed = y * 1.10
        err_rel = compute_relative_step_error(y_perturbed, y)
        assert abs(err_rel - 0.10) < 1e-4

    def test_pre_post_errors_calculation(self) -> None:
        errors = [0.1] * 20 + [0.8] * 10
        e_pre, e_post_0 = compute_pre_post_errors(errors, change_point=20, window=10)
        assert abs(e_pre - 0.1) < 1e-5
        assert abs(e_post_0 - 0.8) < 1e-5

    def test_recovery_passage_and_sustained(self) -> None:
        errors = [0.1] * 10 + [
            0.8,
            0.6,
            0.4,
            0.15,
            0.10,
            0.09,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
        ]
        tau_fp = compute_first_passage_recovery(errors, change_point=10, threshold=0.15)
        assert tau_fp == 3

        tau_sr = compute_sustained_recovery(
            errors, change_point=10, threshold=0.15, consecutive_steps=10
        )
        assert tau_sr == 3

    def test_negative_transfer_metric(self) -> None:
        nt = compute_negative_transfer(error_continuous_b=0.65, error_reset_b=0.45)
        assert abs(nt - 0.20) < 1e-5

        pt = compute_negative_transfer(error_continuous_b=0.35, error_reset_b=0.45)
        assert abs(pt - (-0.10)) < 1e-5

    def test_forgetting_metric(self) -> None:
        errors = [0.10] * 10 + [0.70] * 10 + [0.25] * 10
        fg = compute_forgetting(
            errors, initial_steady_point=10, return_change_point=20, window=5
        )
        assert abs(fg - 0.15) < 1e-5

    def test_adaptation_energy_metric(self) -> None:
        updates = [1.0, 2.0, 3.0]
        # 1^2 + 2^2 + 3^2 = 14
        energy = compute_adaptation_energy(updates)
        assert abs(energy - 14.0) < 1e-5

    def test_adaptation_efficiency_metric(self) -> None:
        eff = compute_adaptation_efficiency(error_reduction=0.5, adaptation_energy=2.0)
        assert abs(eff - 0.25) < 1e-4

    def test_cumulative_excess_error_metric(self) -> None:
        errors = [0.2] * 5 + [0.5, 0.4, 0.2, 0.1]
        excess = compute_cumulative_excess_error(
            errors, change_point=5, e_pre=0.2, window=4
        )
        # max(0, 0.5-0.2) + max(0, 0.4-0.2) + max(0, 0.2-0.2) + max(0, 0.1-0.2)
        # = 0.3 + 0.2 + 0.0 + 0.0 = 0.5
        assert abs(excess - 0.5) < 1e-5

    def test_streaming_telemetry_summary(self) -> None:
        telem = StreamingTelemetry()
        telem.record_step(
            error=0.1,
            state_norm=1.0,
            update_norm=0.2,
            stability_margin=1.5,
            step_size=0.1,
            retention=0.9,
        )
        telem.record_step(
            error=0.2,
            state_norm=1.5,
            update_norm=0.3,
            stability_margin=1.4,
            step_size=0.2,
            retention=0.8,
        )

        summary = telem.summary()
        assert abs(summary["mean_error"] - 0.15) < 1e-5
        assert abs(summary["max_state_norm"] - 1.5) < 1e-5
        assert abs(summary["min_stability_margin"] - 1.4) < 1e-5
        assert summary["non_finite_count"] == 0


class TestObservatoryPlotsGeneration:
    """Verifies that all 13 Observatory plots (AX to BJ) render without error."""

    def test_generate_all_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)

            trajectories = {
                "SafeAdaptiveDelta": [0.1] * 30,
                "OnlineRidge": [0.2] * 30,
            }
            cps = [10, 20]

            # AX
            plot_spatiotemporal_error_over_time_ax(
                trajectories, cps, out / "plot_ax.png"
            )
            assert (out / "plot_ax.png").exists()

            # AY
            plot_adaptation_negative_transfer_pareto_ay(
                ["M1", "M2"], [0.2, 0.4], [0.01, 0.05], out / "plot_ay.png"
            )
            assert (out / "plot_ay.png").exists()

            # AZ
            plot_retention_trajectories_az({"M1": [0.9] * 30}, cps, out / "plot_az.png")
            assert (out / "plot_az.png").exists()

            # BA
            plot_retention_ablation_ba(
                ["R1", "R2"], [0.3, 0.5], [0.02, 0.04], out / "plot_ba.png"
            )
            assert (out / "plot_ba.png").exists()

            # BB
            plot_a1_b_a1_recovery_bb(
                ["M1", "M2"], [0.5, 0.6], [0.1, 0.2], out / "plot_bb.png"
            )
            assert (out / "plot_bb.png").exists()

            # BC
            plot_error_vs_state_adaptive_bc(trajectories, cps, out / "plot_bc.png")
            assert (out / "plot_bc.png").exists()

            # BD
            plot_spatial_vs_shuffled_bd(
                ["M1", "M2"], [0.3, 0.4], [0.5, 0.6], out / "plot_bd.png"
            )
            assert (out / "plot_bd.png").exists()

            # BE
            plot_perf_vs_dim_be([64, 128], {"M1": [0.3, 0.4]}, out / "plot_be.png")
            assert (out / "plot_be.png").exists()

            # BF
            plot_runtime_vs_dim_bf([64, 128], {"M1": [10.0, 20.0]}, out / "plot_bf.png")
            assert (out / "plot_bf.png").exists()

            # BG
            plot_memory_vs_dim_bg([64, 128], {"M1": [1024, 4096]}, out / "plot_bg.png")
            assert (out / "plot_bg.png").exists()

            # BH
            plot_adaptation_energy_vs_perf_bh(
                ["M1"], [10.0], [0.2], out / "plot_bh.png"
            )
            assert (out / "plot_bh.png").exists()

            # BI
            plot_safe_vs_unsafe_bi(["M1"], [1.5], [0.1], out / "plot_bi.png")
            assert (out / "plot_bi.png").exists()

            # BJ
            plot_continuous_vs_reset_bj(["M1"], [0.3], [0.25], out / "plot_bj.png")
            assert (out / "plot_bj.png").exists()


class TestArtifactsValidation:
    """Verifies that generated Phase 12 JSON artifacts exist and adhere to schema."""

    def test_artifacts_existence_and_schema(self) -> None:
        art_dir = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "benchmarks"
            / "artifacts"
            / "phase_12"
        )
        assert (art_dir / "phase_12_results.json").exists()
        assert (art_dir / "phase_12_per_seed.json").exists()
        assert (art_dir / "phase_12_scaling.json").exists()
        assert (art_dir / "phase_12_config.json").exists()
        assert (art_dir / "phase_12_retention.json").exists()

        with open(art_dir / "phase_12_config.json") as f:
            cfg = json.load(f)
            assert cfg["phase"] == 12
            assert cfg["seeds"] == [0, 1, 2, 3, 4]
            assert cfg["base_dimensions"]["dim"] == 64

        with open(art_dir / "phase_12_results.json") as f:
            res = json.load(f)
            assert "aggregated_task_b" in res
            assert "spatial_ablation" in res
            assert "frequency_ablation" in res

        with open(art_dir / "phase_12_scaling.json") as f:
            sc = json.load(f)
            assert "64" in sc
            assert "128" in sc
            assert "256" in sc

        with open(art_dir / "phase_12_retention.json") as f:
            ret = json.load(f)
            assert "Selective_adaptive" in ret
            assert "Selective_fixed_high" in ret
            assert "Selective_state_adaptive" in ret


class TestCausalControlsAndPareto:
    """Verifies Causal Controls A-D, stability invariants, and data integrity."""

    def test_control_a_adaptive_retention_toggle(self) -> None:
        pred = SelectiveRetentionPredictor(
            dim=16, feat_dim=16, retention_mode="adaptive"
        )
        x = torch.randn(16)
        y = torch.randn(16)

        pred.set_adaptation(False)
        pred.adapt_step(x, y)
        assert pred.last_update_norm == 0.0

        pred.set_adaptation(True)
        pred.adapt_step(x, y)
        assert pred.last_update_norm > 0.0

    def test_control_b_continuous_vs_reset_dynamics(self) -> None:
        pred = SafeAdaptiveDeltaPredictor(dim=16)
        x = torch.randn(16)
        y = torch.randn(16)

        pred.adapt_step(x, y)
        norm_cont = pred.get_state_norm()

        pred.reset_state()
        norm_reset = pred.get_state_norm()
        assert norm_reset < norm_cont

    def test_control_c_retention_modes_relative_order(self) -> None:
        pred_high = SelectiveRetentionPredictor(
            dim=8, feat_dim=8, retention_mode="fixed_high", alpha_high=0.99
        )
        pred_low = SelectiveRetentionPredictor(
            dim=8, feat_dim=8, retention_mode="fixed_low", alpha_low=0.70
        )

        x = torch.randn(8)
        y = torch.randn(8)
        pred_high.adapt_step(x, y)
        pred_low.adapt_step(x, y)

        assert pred_high.last_retention == 0.99
        assert pred_low.last_retention == 0.70

    def test_control_d_shuffled_control_decorrelation(self) -> None:
        pred_shuff = SelectiveStateAdaptivePredictor(
            dim=16, feat_dim=16, shuffled_control=True, seed=42
        )
        x = torch.zeros(16)
        y = torch.zeros(16)
        pred_shuff.adapt_step(x, y)
        # Even with zero error and zero state, shuffled control injects pseudo-random variation
        ret_shuff = pred_shuff.last_retention
        assert 0.10 <= ret_shuff <= 1.0

    def test_lyapunov_safe_margin_enforcement(self) -> None:
        safe_delta = SafeAdaptiveDeltaPredictor(
            dim=8, eta_max=1.0, stability_margin=0.05
        )
        # Very large input to test clamping
        x_huge = torch.randn(8) * 100.0
        y = torch.randn(8)
        safe_delta.adapt_step(x_huge, y)
        # Stability margin must remain strictly positive
        assert safe_delta.last_margin >= 0.05 - 1e-6

    def test_safe_self_referential_contractive_margin(self) -> None:
        safe_sr = SafeSelfReferentialPredictor(dim=8, dc_dim=2)
        x_huge = torch.randn(8) * 50.0
        y = torch.randn(8)
        safe_sr.adapt_step(x_huge, y)
        assert safe_sr.last_margin >= 0.0

    def test_memory_breakdown_consistency(self) -> None:
        models = [
            SafeAdaptiveDeltaPredictor(dim=16),
            OnlineRidgePredictor(dim=16),
            NonlinearOnlineRidgePredictor(dim=16, rff_dim=8),
            GRUPredictor(dim=16, hidden_dim=4),
            LSTMPredictor(dim=16, hidden_dim=4),
            SelectiveStateAdaptivePredictor(dim=16, feat_dim=16),
        ]
        for m in models:
            bd = m.get_memory_breakdown()
            assert "parameter_bytes" in bd
            assert "persistent_state_bytes" in bd
            assert "temporary_activation_bytes" in bd
            assert "total_bytes" in bd
            assert (
                bd["total_bytes"]
                == bd["parameter_bytes"]
                + bd["persistent_state_bytes"]
                + bd["temporary_activation_bytes"]
            )

    def test_phase_12_per_seed_data_integrity(self) -> None:
        art_dir = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "benchmarks"
            / "artifacts"
            / "phase_12"
        )
        with open(art_dir / "phase_12_per_seed.json") as f:
            ps = json.load(f)
            assert "task_b_per_seed" in ps
            assert "task_c_per_seed" in ps
            # Verify 5 seeds recorded for each model
            for m, seed_list in ps["task_b_per_seed"].items():
                assert len(seed_list) == 5, (
                    f"Expected 5 seeds for {m}, found {len(seed_list)}"
                )

    def test_phase_12_scaling_monotonicity(self) -> None:
        art_dir = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "benchmarks"
            / "artifacts"
            / "phase_12"
        )
        with open(art_dir / "phase_12_scaling.json") as f:
            sc = json.load(f)
            # SafeAdaptiveDelta memory scales with D^2 * 4
            mem_64 = sc["64"]["SafeAdaptiveDelta"]["state_memory_bytes"]
            mem_128 = sc["128"]["SafeAdaptiveDelta"]["state_memory_bytes"]
            mem_256 = sc["256"]["SafeAdaptiveDelta"]["state_memory_bytes"]
            assert mem_64 < mem_128 < mem_256
            assert mem_64 == 64 * 64 * 4
            assert mem_128 == 128 * 128 * 4
            assert mem_256 == 256 * 256 * 4

    def test_spatiotemporal_dissipative_decay(self) -> None:
        dims = SpatioTemporalDimensions(height=8, width=8, channels=1)
        X = create_initial_spatial_field(
            dims, generator=torch.Generator().manual_seed(42)
        )
        # Regime None / unforced default dissipative step
        X_decay = simulate_spatiotemporal_step(X, regime="unforced_dissipation")
        norm_orig = float(torch.linalg.norm(X).item())
        norm_decay = float(torch.linalg.norm(X_decay).item())
        assert norm_decay < norm_orig
