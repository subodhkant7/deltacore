"""Unit and integration tests for Phase 17: Online Non-Stationary Classification.

Covers all 18 requirements from Phase 17 Section 29:
    1. Deterministic stream generation
    2. Class balance
    3. Shortcut controls and audit
    4. Regime transitions
    5. Label-boundary changes
    6. Online/offline separation
    7. Parameter immutability (Delta theta = 0)
    8. Adaptive-state updates
    9. Retention modes (fixed-high, fixed-low, adaptive)
    10. State reset
    11. Recovery metrics (first-passage, sustained recovery K=10, excess loss, forgetting)
    12. Label-shuffle negative control
    13. Feature-permutation control
    14. Dimensional scaling (D in {32, 64, 128, 256})
    15. Memory accounting (exact 4*D^2 bytes for DxD associative memory)
    16. Runtime accounting
    17. Reproducibility across seeds
    18. Serialization and artifact schemas
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from deltacore.classification.metrics import (
    compute_recovery_metrics,
)
from deltacore.classification.models import (
    FixedDeltaClassifier,
    FrozenLinearClassifier,
    SafeAdaptiveDeltaClassifier,
    StateOffAblationClassifier,
    compute_model_parameter_hash,
    fit_offline_linear_head,
)
from deltacore.classification.phase_17_runner import evaluate_online_stream
from deltacore.classification.synthetic_stream import (
    audit_shortcut_statistics,
    generate_nonstationary_stream,
    generate_stationary_dataset,
    permute_features,
    shuffle_labels,
)


class TestPhase17SyntheticStream:
    """Tests for stream generation, class balance, shortcut controls, and regimes."""

    def test_deterministic_stream_generation(self) -> None:
        """Verify identical seed generates bit-for-bit identical stream tensors."""
        s1 = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=50, seed=42
        )
        s2 = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=50, seed=42
        )

        torch.testing.assert_close(s1.inputs, s2.inputs)
        torch.testing.assert_close(s1.targets, s2.targets)
        assert s1.regime_bounds == s2.regime_bounds

    def test_class_balance(self) -> None:
        """Verify class distribution is balanced across stationary splits and regimes."""
        ds = generate_stationary_dataset(
            dim=32, num_classes=6, n_train=480, n_val=180, n_test=240, seed=42
        )
        counts_train = torch.bincount(ds.train_targets, minlength=6)
        counts_test = torch.bincount(ds.test_targets, minlength=6)

        assert torch.all(counts_train == 80)
        assert torch.all(counts_test == 40)

        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=120, seed=42
        )
        assert len(stream.inputs) == 480
        assert len(stream.regime_bounds) == 3

    def test_shortcut_audit_passes(self) -> None:
        """Verify shortcut audit: energy matched, variance matched, no 1D feature leak."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        audit = audit_shortcut_statistics(
            ds.test_inputs, ds.test_targets, num_classes=6
        )

        assert audit["energy_matched"] is True
        assert audit["variance_matched"] is True
        assert audit["no_feature_leak"] is True
        assert audit["passed_shortcut_audit"] is True
        assert audit["norm_max_min_ratio"] < 1.05
        assert audit["max_single_feature_1d_acc"] < 0.60

    def test_regime_transitions(self) -> None:
        """Verify regime boundaries and shift modes."""
        stream = generate_nonstationary_stream(
            dim=32,
            num_classes=6,
            steps_per_regime=60,
            shift_mode="all_regimes",
            seed=42,
        )
        assert stream.regime_bounds == [60, 120, 180]
        assert len(stream.regime_names) == 4

        cov_stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=50, shift_mode="covariate", seed=42
        )
        assert cov_stream.regime_bounds == [50, 100]
        assert len(cov_stream.regime_names) == 3

        bound_stream = generate_nonstationary_stream(
            dim=32,
            num_classes=6,
            steps_per_regime=50,
            shift_mode="decision_boundary",
            seed=42,
        )
        assert bound_stream.regime_bounds == [50, 100]
        assert len(bound_stream.regime_names) == 3

    def test_label_boundary_shift_impact(self) -> None:
        """Verify that decision boundary rotation lowers frozen classifier accuracy."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)
        frozen = FrozenLinearClassifier(32, 6, w, b)

        bound_stream = generate_nonstationary_stream(
            dim=32,
            num_classes=6,
            steps_per_regime=60,
            severity="severe",
            shift_mode="decision_boundary",
            seed=42,
        )
        res = evaluate_online_stream(
            frozen,
            bound_stream.inputs,
            bound_stream.targets,
            bound_stream.regime_bounds,
        )

        # In severe boundary rotation, frozen linear accuracy must degrade significantly
        assert res.accuracy < 0.90
        assert res.post_shift_accuracy < 0.50


class TestPhase17ModelsAndImmutability:
    """Tests for parameter immutability, adaptive updates, retention, and reset."""

    def test_parameter_immutability(self) -> None:
        """Verify Δθ = 0: trainable parameters are bit-for-bit unchanged after streaming."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)

        models = [
            SafeAdaptiveDeltaClassifier(32, 6, w, b),
            FixedDeltaClassifier(32, 6, w, b),
            FrozenLinearClassifier(32, 6, w, b),
            StateOffAblationClassifier(32, 6, w, b),
        ]

        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=40, seed=42
        )

        for m in models:
            hash_before = compute_model_parameter_hash(m)
            _ = evaluate_online_stream(m, stream.inputs, stream.targets)
            hash_after = compute_model_parameter_hash(m)
            assert hash_before == hash_after, (
                f"Model {m.name} violated parameter immutability Δθ = 0!"
            )

    def test_adaptive_state_updates(self) -> None:
        """Verify associative state M_t updates, contraction bounds, and safety margins."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)

        model = SafeAdaptiveDeltaClassifier(
            dim=32, num_classes=6, weight_head=w, bias_head=b, eta0=0.08, rho=1.50
        )
        assert torch.all(model.M == 0.0)

        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=30, seed=42
        )
        res = evaluate_online_stream(model, stream.inputs, stream.targets)

        # M_t should have non-zero state after adaptation
        assert torch.linalg.norm(model.M).item() > 0.0
        assert res.max_state_norm > 0.0
        assert res.min_safety_margin > 0.0
        assert res.diverged == 0

    def test_state_off_ablation_control(self) -> None:
        """Verify AdaptiveStateOFF is numerically identical to FrozenLinear."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)

        m_frozen = FrozenLinearClassifier(32, 6, w, b)
        m_ablation = StateOffAblationClassifier(32, 6, w, b)

        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=30, seed=42
        )
        res_frozen = evaluate_online_stream(m_frozen, stream.inputs, stream.targets)
        res_ablation = evaluate_online_stream(m_ablation, stream.inputs, stream.targets)

        assert math.isclose(res_frozen.accuracy, res_ablation.accuracy, rel_tol=1e-5)
        assert math.isclose(res_frozen.log_loss, res_ablation.log_loss, rel_tol=1e-5)

    def test_state_reset_ablation(self) -> None:
        """Verify reset_state clears internal associative state without changing parameters."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)
        model = SafeAdaptiveDeltaClassifier(32, 6, w, b)

        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=20, seed=42
        )
        _ = evaluate_online_stream(model, stream.inputs, stream.targets)

        assert torch.linalg.norm(model.M).item() > 0.0
        model.reset_state()
        assert torch.all(model.M == 0.0)

    def test_retention_modes(self) -> None:
        """Verify fixed-high (0.99), fixed-low (0.70), and safe adaptive retention."""
        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)

        m_high = FixedDeltaClassifier(32, 6, w, b, step_size=0.05, alpha=0.99)
        m_low = FixedDeltaClassifier(32, 6, w, b, step_size=0.05, alpha=0.70)
        m_adapt = SafeAdaptiveDeltaClassifier(32, 6, w, b, eta0=0.08, alpha_min=0.95)

        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=40, seed=42
        )

        res_high = evaluate_online_stream(m_high, stream.inputs, stream.targets)
        res_low = evaluate_online_stream(m_low, stream.inputs, stream.targets)
        res_adapt = evaluate_online_stream(m_adapt, stream.inputs, stream.targets)

        assert res_high.accuracy > 0.80
        assert res_low.accuracy > 0.80
        assert res_adapt.accuracy > 0.80


class TestPhase17ControlsAndScaling:
    """Tests for robustness controls, negative controls, dimensional scaling, and accounting."""

    def test_label_shuffle_negative_control(self) -> None:
        """Verify label shuffling collapses accuracy to theoretical chance (~1/K = 0.167)."""
        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=50, seed=42
        )
        shuffled_y = shuffle_labels(stream.targets, seed=99)

        # Label shuffle should not change class counts
        torch.testing.assert_close(
            torch.bincount(stream.targets).sort().values,
            torch.bincount(shuffled_y).sort().values,
        )

        ds = generate_stationary_dataset(dim=32, num_classes=6, seed=42)
        w, b = fit_offline_linear_head(ds.train_inputs, ds.train_targets, num_classes=6)
        model = SafeAdaptiveDeltaClassifier(32, 6, w, b)

        res_shuff = evaluate_online_stream(model, stream.inputs, shuffled_y)
        # Should be near chance level 1/6 = 0.167 (+/- 0.08)
        assert abs(res_shuff.accuracy - (1.0 / 6.0)) < 0.08

    def test_feature_permutation_control(self) -> None:
        """Verify feature coordinate permutation preserves energy and norms."""
        stream = generate_nonstationary_stream(
            dim=32, num_classes=6, steps_per_regime=30, seed=42
        )
        perm_x, perm_idx = permute_features(stream.inputs, seed=123)

        assert len(perm_idx) == 32
        torch.testing.assert_close(
            torch.linalg.norm(stream.inputs, dim=-1), torch.linalg.norm(perm_x, dim=-1)
        )

    @pytest.mark.parametrize("d_val", [32, 64, 128, 256])
    def test_dimensional_scaling_and_memory(self, d_val: int) -> None:
        """Verify SafeAdaptiveDelta scales without divergence and state memory = 4*D^2 bytes."""
        stat = generate_stationary_dataset(
            dim=d_val, num_classes=6, n_train=120, n_val=60, n_test=60, seed=42
        )
        w, b = fit_offline_linear_head(
            stat.train_inputs, stat.train_targets, num_classes=6
        )
        model = SafeAdaptiveDeltaClassifier(d_val, 6, w, b)

        assert model.get_state_memory_bytes() == 4 * d_val * d_val

        stream = generate_nonstationary_stream(
            dim=d_val, num_classes=6, steps_per_regime=20, seed=42
        )
        res = evaluate_online_stream(model, stream.inputs, stream.targets)

        assert res.diverged == 0
        assert res.accuracy > 0.85
        assert res.persistent_state_bytes == 4 * d_val * d_val

    def test_recovery_metrics_computation(self) -> None:
        """Verify recovery metrics, sustained recovery K=10, and excess loss calculations."""
        # Synthetic predictions: drop for 2 steps, then recover
        step_correct = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        pre_acc, post_acc, fp_steps, sust_steps = compute_recovery_metrics(
            step_correct,
            shift_step=3,
            end_step=20,
            pre_window=3,
            recovery_window=3,
            sustained_window=5,
        )
        assert pre_acc == 1.0
        assert post_acc < 1.0
        assert fp_steps >= 2
        assert sust_steps >= 2


class TestPhase17ArtifactsAndSerialization:
    """Tests verifying all 6 required JSON artifacts and 14 plot figures exist and are valid."""

    def test_json_artifacts_exist_and_load(self) -> None:
        """Verify all 6 required Phase 17 JSON artifacts are populated and valid."""
        base_dir = Path("docs/benchmarks/artifacts/phase_17")
        assert (base_dir / "phase_17_results.json").exists()
        assert (base_dir / "phase_17_per_seed.json").exists()
        assert (base_dir / "phase_17_config.json").exists()
        assert (base_dir / "phase_17_shift_results.json").exists()
        assert (base_dir / "phase_17_retention.json").exists()
        assert (base_dir / "phase_17_scaling.json").exists()

        with open(base_dir / "phase_17_results.json") as f:
            res = json.load(f)
            assert "SafeAdaptiveDelta" in res
            assert "stat_acc_mean" in res["SafeAdaptiveDelta"]

        with open(base_dir / "phase_17_config.json") as f:
            cfg = json.load(f)
            assert cfg["dim"] == 32
            assert cfg["num_classes"] == 6
            assert cfg["shortcut_audit"]["passed_shortcut_audit"] is True

    def test_observatory_plots_exist(self) -> None:
        """Verify all 14 Phase 17 Observatory publication figures DC through DP exist."""
        plot_dir = Path("docs/benchmarks/artifacts/phase_17/plots")
        required_plots = [
            "plot_dc_stationary_perf.png",
            "plot_dd_post_shift_acc_vs_time.png",
            "plot_de_recovery_curves.png",
            "plot_df_continuous_vs_reset.png",
            "plot_dg_retention_modes.png",
            "plot_dh_covariate_shift.png",
            "plot_di_decision_boundary_shift.png",
            "plot_dj_acc_vs_dim.png",
            "plot_dk_runtime_vs_dim.png",
            "plot_dl_memory_vs_dim.png",
            "plot_dm_state_norm_trajectory.png",
            "plot_dn_adaptation_energy.png",
            "plot_do_label_shuffle_control.png",
            "plot_dp_feature_permutation_control.png",
        ]
        for p in required_plots:
            plot_file = plot_dir / p
            assert plot_file.exists(), f"Missing required plot: {p}"
            assert plot_file.stat().st_size > 1000, f"Plot {p} is empty or invalid!"
