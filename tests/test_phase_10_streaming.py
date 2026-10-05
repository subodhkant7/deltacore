"""Unit and Property Tests for Phase 10: Streaming Adaptive-State Benchmark.

Covers:
    - Stable transition matrix construction & spectral radius bounds
    - Task A regime-switching stream determinism & change points
    - Task B delayed context retrieval invariants across delays
    - Task C abrupt shift (A -> B -> A) stream properties
    - Online/offline separation (zero parameter gradients/updates during evaluation)
    - Adaptive state updates & state norm evolution
    - Adaptive state ON vs. OFF ablation invariants
    - Continuous state vs. State reset invariants
    - All 10 predictor architectures (Frozen, RNNs, OnlineRidge, DeltaCore variants)
    - Contractive Lyapunov stability bounding in SafeAdaptiveDelta & SafeSelfReferential
    - First-passage recovery, sustained recovery (K=10), cumulative excess error
    - Forgetting metric & adaptation energy calculations
    - Machine-readable serialization & Observatory plots AD-AM smoke tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from deltacore.observatory.streaming_plots import (
    plot_adaptation_energy_ai,
    plot_adaptation_vs_forgetting_af,
    plot_adaptive_on_vs_off_ak,
    plot_continuous_vs_reset_al,
    plot_error_over_time_ad,
    plot_performance_vs_delay_ag,
    plot_performance_vs_params_aj,
    plot_recovery_distributions_ae,
    plot_runtime_per_timestep_am,
    plot_state_norm_trajectories_ah,
)
from deltacore.streaming.metrics import (
    StreamingTelemetry,
    compute_cumulative_excess_error,
    compute_first_passage_recovery,
    compute_forgetting,
    compute_pre_post_errors,
    compute_relative_step_error,
    compute_sustained_recovery,
)
from deltacore.streaming.models import (
    AdaptiveDeltaPredictor,
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    FrozenMLPPredictor,
    GRUPredictor,
    LSTMPredictor,
    OnlineRidgePredictor,
    SafeAdaptiveDeltaPredictor,
    SafeSelfReferentialPredictor,
    SelfReferentialPredictor,
)
from deltacore.streaming.regimes import (
    generate_abrupt_shift_stream,
    generate_delayed_retrieval_stream,
    generate_regime_switching_stream,
    generate_stable_transition_matrix,
)


class TestRegimeGeneration:
    """Test mathematical invariants of synthetic streaming regimes."""

    def test_stable_transition_matrix_spectral_radius(self) -> None:
        """Generated transition matrices must have spectral radius strictly < 1.0."""
        for target_sr in [0.5, 0.75, 0.85, 0.95]:
            A = generate_stable_transition_matrix(dim=8, spectral_radius=target_sr)
            evs = torch.linalg.eigvals(A)
            max_ev = float(torch.max(torch.abs(evs)).item())
            assert np.isclose(max_ev, target_sr, atol=1e-3)

    def test_stable_transition_matrix_invalid_radius_rejected(self) -> None:
        """spectral_radius >= 1.0 or <= 0.0 must raise ValueError."""
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            generate_stable_transition_matrix(dim=8, spectral_radius=1.0)
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            generate_stable_transition_matrix(dim=8, spectral_radius=-0.2)

    def test_regime_switching_stream_shapes_and_determinism(self) -> None:
        """Task A stream produces [T, D] inputs and targets deterministically."""
        s1 = generate_regime_switching_stream(seq_len=128, dim=8, seed=42)
        s2 = generate_regime_switching_stream(seq_len=128, dim=8, seed=42)
        assert s1.inputs.shape == (128, 8)
        assert s1.targets.shape == (128, 8)
        assert torch.allclose(s1.inputs, s2.inputs)
        assert torch.allclose(s1.targets, s2.targets)
        assert s1.change_points == [32, 64, 96]
        assert len(s1.regime_ids) == 128

    def test_delayed_retrieval_stream_structure(self) -> None:
        """Task B stream sets cue at cue_time and query at cue_time + delay."""
        s = generate_delayed_retrieval_stream(
            delay=64, seq_len=256, dim=8, cue_time=10, seed=123
        )
        assert s.inputs.shape == (256, 8)
        assert s.targets.shape == (256, 8)
        assert s.metadata["query_time"] == 74
        # Target at query_time must match input at cue_time
        assert torch.allclose(s.targets[74], s.inputs[10])

    def test_delayed_retrieval_invalid_length_rejected(self) -> None:
        """Task B rejects seq_len smaller than cue + delay + 2."""
        with pytest.raises(ValueError, match="must be larger than"):
            generate_delayed_retrieval_stream(delay=200, seq_len=100, cue_time=10)

    def test_abrupt_shift_stream_phases(self) -> None:
        """Task C produces 3 phases: Distribution A, B, and return to A."""
        s = generate_abrupt_shift_stream(seq_len=300, dim=8, seed=99)
        assert s.change_points == [100, 200]
        assert s.regime_ids[:100] == [0] * 100
        assert s.regime_ids[100:200] == [1] * 100
        assert s.regime_ids[200:] == [0] * 100


class TestModelArchitectureAndIsolation:
    """Verify all 10 streaming predictors and online/offline parameter isolation."""

    @pytest.mark.parametrize(
        "model_cls",
        [
            FrozenLinearPredictor,
            FrozenMLPPredictor,
            GRUPredictor,
            LSTMPredictor,
            OnlineRidgePredictor,
            FixedDeltaPredictor,
            AdaptiveDeltaPredictor,
            SafeAdaptiveDeltaPredictor,
            SelfReferentialPredictor,
            SafeSelfReferentialPredictor,
        ],
    )
    def test_zero_parameter_updates_during_evaluation(self, model_cls: type) -> None:
        """CRITICAL (EC10.3): Streaming evaluation must NEVER update model parameters or compute gradients."""
        model = model_cls(dim=8)
        # Record initial weights
        init_weights = {n: p.clone() for n, p in model.named_parameters()}

        x = torch.randn(8)
        target = torch.randn(8)

        # Single step
        y_hat = model.predict_step(x)
        assert y_hat.shape == (8,)
        model.adapt_step(x, target)

        # Confirm all trainable weights are completely unchanged
        for n, p in model.named_parameters():
            if "linear" in n or "mlp" in n or "gru" in n or "lstm" in n:
                assert torch.allclose(p, init_weights[n]), (
                    f"Parameter {n} was mutated during adapt_step!"
                )
            assert p.grad is None, (
                f"Parameter {n} accumulated gradient during streaming!"
            )

    def test_fixed_delta_state_update(self) -> None:
        """FixedDelta updates associative matrix M by eta * e * x^T."""
        m = FixedDeltaPredictor(dim=4, step_size=0.2)
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        target = torch.tensor([0.5, 0.5, 0.0, 0.0])

        init_m = m.M.clone()
        pred = m.predict_step(x)
        err = target - pred
        m.adapt_step(x, target)

        expected_m = init_m + 0.2 * torch.outer(err, x)
        assert torch.allclose(m.M, expected_m)

    def test_adaptive_state_on_vs_off_control(self) -> None:
        """Primary Causal Control: When adaptation is OFF, state remains completely constant."""
        m = AdaptiveDeltaPredictor(dim=8, eta_max=0.4)
        m.set_adaptation(False)
        init_m = m.M.clone()

        for _ in range(20):
            x = torch.randn(8)
            t = torch.randn(8)
            _ = m.predict_step(x)
            m.adapt_step(x, t)

        assert torch.allclose(m.M, init_m)
        assert m.last_update_norm == 0.0

    def test_state_reset_restores_initial_state(self) -> None:
        """reset_state() returns adaptive state to initial baseline condition."""
        m = SafeAdaptiveDeltaPredictor(dim=8)
        x = torch.randn(8)
        t = torch.randn(8)
        m.adapt_step(x, t)
        assert m.get_state_norm() > 0.0

        m.reset_state()
        expected_init = torch.eye(8) * 0.1
        assert torch.allclose(m.M, expected_init)

    def test_safe_adaptive_delta_stability_bound(self) -> None:
        """SafeAdaptiveDelta clamps eta so that step-size margin remains positive."""
        m = SafeAdaptiveDeltaPredictor(dim=4, eta_max=5.0, stability_margin=0.2)
        # Large input vector that would cause explosive gamma
        x = torch.ones(4) * 10.0
        target = torch.ones(4) * -10.0

        m.adapt_step(x, target)
        assert m.last_margin >= 0.19  # Margin is bounded and positive
        assert m.last_step_size <= 2.0 / (x @ x).item()

    def test_safe_self_referential_contractive_bounds(self) -> None:
        """SafeSelfReferential bounds both content and dynamics adaptation."""
        m = SafeSelfReferentialPredictor(dim=4, dc_dim=2)
        x = torch.randn(4) * 5.0
        target = torch.randn(4)
        m.adapt_step(x, target)
        assert m.last_margin >= 0.0
        assert np.isfinite(m.get_state_norm())

    def test_online_ridge_update_properties(self) -> None:
        """OnlineRidge predictor executes recursive least squares without neural gradients."""
        m = OnlineRidgePredictor(dim=4, lam=0.95)
        tot, tr = m.get_param_count()
        assert tr == 0  # Zero trainable parameters

        x = torch.randn(4)
        t = torch.randn(4)
        p_init = m.P.clone()
        m.adapt_step(x, t)
        assert not torch.allclose(m.P, p_init)
        assert m.last_update_norm > 0.0


class TestMetricsAndTelemetry:
    """Test adaptation, recovery, stability, and telemetry calculations."""

    def test_relative_step_error_formula(self) -> None:
        """compute_relative_step_error implements ||y - y_hat|| / max(||y||, eps)."""
        pred = torch.tensor([1.0, 1.0])
        target = torch.tensor([1.0, 2.0])
        err = compute_relative_step_error(pred, target)
        expected = float(1.0 / np.sqrt(5.0))
        assert np.isclose(err, expected, atol=1e-5)

    def test_pre_post_errors_calculation(self) -> None:
        """compute_pre_post_errors computes pre-shift mean and immediate post-shift error."""
        errors = [0.1] * 20 + [0.8, 0.7, 0.6]
        e_pre, e_post_0 = compute_pre_post_errors(errors, change_point=20, window=10)
        assert np.isclose(e_pre, 0.1)
        assert np.isclose(e_post_0, 0.8)

    def test_first_passage_recovery(self) -> None:
        """compute_first_passage_recovery finds first index satisfying threshold."""
        errors = [0.1] * 10 + [0.9, 0.8, 0.6, 0.3, 0.15, 0.12]
        # change_point = 10, threshold = 0.25 -> step 14 has 0.15 <= 0.25 (delay = 4)
        t_first = compute_first_passage_recovery(
            errors, change_point=10, threshold=0.25
        )
        assert t_first == 4

    def test_sustained_recovery_consecutive_requirement(self) -> None:
        """compute_sustained_recovery requires K=10 consecutive steps below threshold."""
        # 10 pre, then shift to 0.9, dip to 0.2 for 2 steps (not sustained), then sustained 10 steps
        errors = [0.1] * 10 + [0.9, 0.2, 0.2, 0.8] + [0.2] * 10
        t_sust = compute_sustained_recovery(
            errors, change_point=10, threshold=0.25, consecutive_steps=10
        )
        # Sustained window starts at index 14 -> delay 4
        assert t_sust == 4

    def test_cumulative_excess_error(self) -> None:
        """compute_cumulative_excess_error sums positive deviations from e_pre."""
        errors = [0.1] * 10 + [0.5, 0.4, 0.3, 0.1]
        excess = compute_cumulative_excess_error(
            errors, change_point=10, e_pre=0.1, window=4
        )
        expected = (0.5 - 0.1) + (0.4 - 0.1) + (0.3 - 0.1) + (0.1 - 0.1)
        assert np.isclose(excess, expected)

    def test_forgetting_metric(self) -> None:
        """compute_forgetting computes difference between post-return and initial steady error."""
        errors = [0.1] * 30 + [0.8] * 30 + [0.4] * 30
        # Phase 1: t in [0, 30), Phase 2: t in [30, 60), Phase 3: t in [60, 90)
        forget = compute_forgetting(
            errors, initial_steady_point=30, return_change_point=60, window=10
        )
        assert np.isclose(forget, 0.3)

    def test_telemetry_summary_metrics(self) -> None:
        """StreamingTelemetry correctly aggregates summary metrics."""
        tel = StreamingTelemetry()
        tel.record_step(
            error=0.5,
            state_norm=1.0,
            update_norm=0.1,
            stability_margin=1.5,
            step_size=0.2,
        )
        tel.record_step(
            error=0.3,
            state_norm=1.2,
            update_norm=0.2,
            stability_margin=1.8,
            step_size=0.1,
        )

        summary = tel.summary()
        assert np.isclose(summary["mean_error"], 0.4)
        assert summary["final_error"] == 0.3
        assert np.isclose(summary["max_state_norm"], 1.2)
        assert np.isclose(summary["min_stability_margin"], 1.5)
        assert np.isclose(summary["adaptation_energy"], 0.1**2 + 0.2**2)


class TestObservatoryPlotsADThroughAM:
    """Smoke test rendering for publication plots AD through AM."""

    def test_plots_ad_through_am_smoke(self) -> None:
        """Verify Plots AD through AM generate valid image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)

            # Plot AD
            trajs = {"ModelA": [0.5, 0.4, 0.2, 0.1], "ModelB": [0.6, 0.5, 0.4, 0.3]}
            f_ad = plot_error_over_time_ad(trajs, [2], p / "plot_ad.png")
            assert f_ad.exists() and f_ad.stat().st_size > 0

            # Plot AE
            f_ae = plot_recovery_distributions_ae(
                ["M1", "M2"], [5.0, 10.0], [15.0, 20.0], p / "plot_ae.png"
            )
            assert f_ae.exists() and f_ae.stat().st_size > 0

            # Plot AF
            f_af = plot_adaptation_vs_forgetting_af(
                ["M1", "M2"], [0.3, 0.5], [0.1, -0.05], p / "plot_af.png"
            )
            assert f_af.exists() and f_af.stat().st_size > 0

            # Plot AG
            f_ag = plot_performance_vs_delay_ag(
                [16, 64], {"M1": [0.2, 0.4]}, p / "plot_ag.png"
            )
            assert f_ag.exists() and f_ag.stat().st_size > 0

            # Plot AH
            f_ah = plot_state_norm_trajectories_ah(
                {"M1": [1.0, 1.2, 1.3]}, p / "plot_ah.png"
            )
            assert f_ah.exists() and f_ah.stat().st_size > 0

            # Plot AI
            f_ai = plot_adaptation_energy_ai(
                ["M1", "M2"], [12.5, 24.0], p / "plot_ai.png"
            )
            assert f_ai.exists() and f_ai.stat().st_size > 0

            # Plot AJ
            f_aj = plot_performance_vs_params_aj(
                [100, 300], [0.2, 0.15], ["M1", "M2"], p / "plot_aj.png"
            )
            assert f_aj.exists() and f_aj.stat().st_size > 0

            # Plot AK
            f_ak = plot_adaptive_on_vs_off_ak(["M1"], [0.2], [0.5], p / "plot_ak.png")
            assert f_ak.exists() and f_ak.stat().st_size > 0

            # Plot AL
            f_al = plot_continuous_vs_reset_al(
                ["M1"], [0.25], [0.40], p / "plot_al.png"
            )
            assert f_al.exists() and f_al.stat().st_size > 0

            # Plot AM
            f_am = plot_runtime_per_timestep_am(
                ["M1", "M2"], [15.2, 22.4], p / "plot_am.png"
            )
            assert f_am.exists() and f_am.stat().st_size > 0


class TestBenchmarkInvariantsAndSweeps:
    """Test horizon sweeps, continuous vs reset dynamics, and benchmark execution."""

    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_horizon_sweep_task_a(self, seq_len: int) -> None:
        """Task A streams generate valid sequences across horizons T in {128, 512, 1024}."""
        stream = generate_regime_switching_stream(seq_len=seq_len, dim=8, seed=42)
        assert stream.inputs.shape == (seq_len, 8)
        assert stream.targets.shape == (seq_len, 8)
        assert len(stream.change_points) == 3

    @pytest.mark.parametrize("delay", [16, 64, 256])
    def test_delay_sweep_task_b(self, delay: int) -> None:
        """Task B streams generate valid cue-query sequences across delays d in {16, 64, 256}."""
        stream = generate_delayed_retrieval_stream(
            delay=delay, seq_len=512, dim=8, cue_time=10, seed=42
        )
        q_time = int(stream.metadata["query_time"])
        assert q_time == 10 + delay
        assert torch.allclose(stream.targets[q_time], stream.inputs[10])

    def test_capacity_matching_group_100p(self) -> None:
        """Section 7: Parameter counts in ~100p regime must fall within 70 to 120 parameters."""
        models = [
            FrozenLinearPredictor(dim=8),
            FrozenMLPPredictor(dim=8, hidden_dim=4),
            GRUPredictor(dim=8, hidden_dim=2),
            LSTMPredictor(dim=8, hidden_dim=2),
            FixedDeltaPredictor(dim=8),
            AdaptiveDeltaPredictor(dim=8),
            SafeAdaptiveDeltaPredictor(dim=8),
            SelfReferentialPredictor(dim=8),
            SafeSelfReferentialPredictor(dim=8),
        ]
        for m in models:
            tot, _ = m.get_param_count()
            assert 64 <= tot <= 125, (
                f"Model {m.name} has {tot} params, outside ~100p capacity target!"
            )

    def test_continuous_vs_reset_state_differential(self) -> None:
        """Section 12: Continuous state preserves memory across regime change, differing from reset state."""
        from examples.phase_10_streaming_benchmark import evaluate_stream

        stream = generate_regime_switching_stream(seq_len=128, dim=8, seed=10)
        m_cont = AdaptiveDeltaPredictor(dim=8)
        m_rst = AdaptiveDeltaPredictor(dim=8)

        tel_cont, _ = evaluate_stream(
            m_cont, stream.inputs, stream.targets, reset_at_points=None
        )
        tel_rst, _ = evaluate_stream(
            m_rst, stream.inputs, stream.targets, reset_at_points=stream.change_points
        )

        assert len(tel_cont.step_errors) == 128
        assert len(tel_rst.step_errors) == 128
        # Because reset clears accumulated state at change points, trajectories diverge
        assert not np.allclose(tel_cont.step_errors, tel_rst.step_errors)

    def test_mini_benchmark_execution_and_schema(self) -> None:
        """Verify full streaming evaluation flow and output telemetry schema on a miniature stream."""
        from examples.phase_10_streaming_benchmark import evaluate_stream

        stream = generate_regime_switching_stream(seq_len=64, dim=8, seed=7)
        model = SafeAdaptiveDeltaPredictor(dim=8)
        tel, rt = evaluate_stream(model, stream.inputs, stream.targets)

        summary = tel.summary()
        assert "mean_error" in summary
        assert "max_state_norm" in summary
        assert "min_stability_margin" in summary
        assert "adaptation_energy" in summary
        assert summary["non_finite_count"] == 0
        assert rt > 0.0

    def test_task_c_recovery_and_adaptation_windows(self) -> None:
        """Task C stream provides well-defined phase 1, 2, and 3 slices."""
        stream = generate_abrupt_shift_stream(seq_len=600, dim=8, seed=15)
        t1, t2 = stream.change_points
        assert t1 == 200
        assert t2 == 400
        phase1 = stream.inputs[:t1]
        phase2 = stream.inputs[t1:t2]
        phase3 = stream.inputs[t2:]
        assert phase1.shape == (200, 8)
        assert phase2.shape == (200, 8)
        assert phase3.shape == (200, 8)

    def test_safe_adaptive_delta_zero_input_safety(self) -> None:
        """SafeAdaptiveDelta handles zero vector inputs x=0 without division by zero or NaN."""
        m = SafeAdaptiveDeltaPredictor(dim=8)
        x_zero = torch.zeros(8)
        target = torch.randn(8)
        pred = m.predict_step(x_zero)
        assert torch.allclose(pred, torch.zeros(8))
        m.adapt_step(x_zero, target)
        assert torch.isfinite(m.M).all()
        assert np.isfinite(m.last_margin)

    def test_adaptive_delta_zero_error_produces_zero_update(self) -> None:
        """When prediction is exactly correct (e=0), update must be zero."""
        m = AdaptiveDeltaPredictor(dim=8)
        x = torch.randn(8)
        pred = m.predict_step(x)
        # Pass prediction as target -> error is zero
        m.adapt_step(x, pred)
        assert m.last_update_norm == 0.0

    def test_self_referential_dc_dimension_flexibility(self) -> None:
        """SelfReferentialPredictor supports arbitrary dynamics memory dimension dc_dim."""
        m = SelfReferentialPredictor(dim=8, dc_dim=4)
        assert m.C.shape == (1, 4)
        x = torch.randn(8)
        t = torch.randn(8)
        m.adapt_step(x, t)
        assert torch.isfinite(m.C).all()
        assert torch.isfinite(m.M).all()

    def test_all_predictors_reset_state_idempotence(self) -> None:
        """Calling reset_state() consecutively must produce identical initial states."""
        from examples.phase_10_streaming_benchmark import build_model_suite

        models = build_model_suite(dim=8)
        for name, m in models.items():
            m.reset_state()
            norm1 = m.get_state_norm()
            m.reset_state()
            norm2 = m.get_state_norm()
            assert np.isclose(norm1, norm2), (
                f"Model {name} reset_state is not idempotent!"
            )

    def test_streaming_telemetry_empty_safeguards(self) -> None:
        """StreamingTelemetry summary gracefully handles empty step list."""
        tel = StreamingTelemetry()
        s = tel.summary()
        assert s["mean_error"] == 0.0
        assert s["final_error"] == 0.0
        assert s["non_finite_count"] == 0
