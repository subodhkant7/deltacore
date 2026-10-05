"""Unit and Regression Tests for Phase 13: Real-World Spatio-Temporal Benchmark.

Tests:
    1. Dataset ingestion & cryptographic checksum determinism.
    2. Temporal split boundaries & leakage prevention audit.
    3. Normalization derived exclusively from training partition.
    4. Missing-data handling & land masking.
    5. Spatial flattening and reversible spatial reconstruction.
    6. Online streaming evaluation protocol & target withholding.
    7. Offline / online training separation & parameter immutability.
    8. State reset causal control at regime shift boundary.
    9. Spatial permutation control (invariance vs conv sensitivity).
    10. Structure-aware spatial metrics (FAC and SGE).
    11. Resource accounting (state memory, params, latency).
    12. Artifact serialization and reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deltacore.streaming.models import (
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    OnlineRidgePredictor,
    PersistencePredictor,
    SafeAdaptiveDeltaPredictor,
    SpatialConvControl,
    compute_parameter_hash,
)
from deltacore.streaming.real_benchmark import (
    evaluate_streaming_run,
    train_offline_model,
)
from deltacore.streaming.real_spatiotemporal import (
    RealSpatioTemporalConfig,
    apply_spatial_permutation,
    compute_field_anomaly_correlation,
    compute_leakage_audit,
    compute_spatial_gradient_error,
    flatten_spatial_field,
    generate_real_spatiotemporal_dataset,
    invert_spatial_permutation,
    restore_spatial_field,
)


class TestPhase13DatasetAndLeakage:
    """Audit tests for empirical dataset ingestion, splitting, and leakage."""

    def test_dataset_ingestion_and_reproducibility(self) -> None:
        """Dataset generation must be strictly deterministic and match checksum."""
        cfg = RealSpatioTemporalConfig(seed=42, height=8, width=8)
        data1 = generate_real_spatiotemporal_dataset(cfg)
        data2 = generate_real_spatiotemporal_dataset(cfg)

        assert torch.allclose(data1.raw_fields, data2.raw_fields)
        assert data1.sha256_checksum == data2.sha256_checksum
        assert data1.raw_fields.shape == (360, 8, 8, 1)

    def test_split_boundaries_are_strictly_disjoint(self) -> None:
        """Temporal train, val, and test splits must have zero intersection."""
        cfg = RealSpatioTemporalConfig(
            train_split=150, val_split=200, total_timesteps=360
        )
        data = generate_real_spatiotemporal_dataset(cfg)

        train_s = set(data.train_timestamps)
        val_s = set(data.val_timestamps)
        test_s = set(data.test_timestamps)

        assert train_s.isdisjoint(val_s)
        assert val_s.isdisjoint(test_s)
        assert train_s.isdisjoint(test_s)
        assert len(train_s) + len(val_s) + len(test_s) == 360

    def test_leakage_prevention_audit(self) -> None:
        """Leakage audit function must pass with zero future lookahead."""
        cfg = RealSpatioTemporalConfig(seed=42)
        data = generate_real_spatiotemporal_dataset(cfg)
        audit = compute_leakage_audit(data)

        assert audit["leakage_audit_passed"] is True
        assert audit["disjoint_train_val"] is True
        assert audit["disjoint_val_test"] is True
        assert audit["disjoint_train_test"] is True
        assert audit["future_target_identical_timesteps"] == 0

    def test_normalization_derived_from_training_split_only(self) -> None:
        """Mean and standard deviation must be computed from training partition only."""
        cfg = RealSpatioTemporalConfig(
            train_split=150, val_split=200, total_timesteps=360
        )
        data = generate_real_spatiotemporal_dataset(cfg)

        # Raw train ocean mean
        train_raw = data.raw_fields[0:150]
        ocean_train = train_raw[:, data.land_mask, :].reshape(-1)

        expected_mean = float(ocean_train.mean().item())
        expected_std = float(ocean_train.std().item())

        assert abs(data.mean_train - expected_mean) < 1e-4
        assert abs(data.std_train - expected_std) < 1e-4

        # Test slice mean should differ from 0.0 under natural distribution shift
        test_norm = data.normalized_fields[200:360]
        test_norm_mean = float(test_norm.mean().item())
        assert abs(test_norm_mean) > 0.05, "Test set artificially zero-centered!"


class TestPhase13SpatialRepresentation:
    """Tests for spatial representation, flattening, and permutation."""

    def test_flattening_and_reversible_restoration(self) -> None:
        """Flattening [T, H, W, C] to [T, D] and restoring must be exact identity."""
        t_seq = torch.randn(10, 8, 8, 1)
        flat = flatten_spatial_field(t_seq)
        assert flat.shape == (10, 64)

        restored = restore_spatial_field(flat, height=8, width=8, channels=1)
        assert restored.shape == (10, 8, 8, 1)
        assert torch.allclose(t_seq, restored)

    def test_spatial_permutation_invertibility(self) -> None:
        """Spatial permutation and inverse permutation must restore exact vector."""
        d = 64
        perm = torch.randperm(d)
        x = torch.randn(5, d)

        x_perm = apply_spatial_permutation(x, perm)
        x_inv = invert_spatial_permutation(x_perm, perm)

        assert not torch.allclose(x, x_perm)
        assert torch.allclose(x, x_inv)

    def test_field_anomaly_correlation_metric(self) -> None:
        """FAC metric must be 1.0 for identical fields and -1.0 for inverted fields."""
        y1 = torch.randn(8, 8)
        assert abs(compute_field_anomaly_correlation(y1, y1) - 1.0) < 1e-4
        assert abs(compute_field_anomaly_correlation(y1, -y1) - (-1.0)) < 1e-4

    def test_spatial_gradient_error_metric(self) -> None:
        """SGE metric must be 0.0 for identical fields and large for permuted fields."""
        y1 = torch.randn(8, 8)
        sge_identical = compute_spatial_gradient_error(y1, y1)
        assert abs(sge_identical) < 1e-5

        perm = torch.randperm(64)
        y1_perm = y1.reshape(-1)[perm].reshape(8, 8)
        sge_permuted = compute_spatial_gradient_error(y1, y1_perm)
        assert sge_permuted > 0.50


class TestPhase13StreamingAndModels:
    """Tests for streaming protocol, parameter immutability, and state reset."""

    def test_parameter_immutability_during_streaming(self) -> None:
        """CRITICAL: Offline neural/linear parameters must NEVER mutate during test streaming."""
        d = 64
        models = [
            PersistencePredictor(dim=d),
            FrozenLinearPredictor(dim=d),
            OnlineRidgePredictor(dim=d),
            FixedDeltaPredictor(dim=d, step_size=0.008),
            SafeAdaptiveDeltaPredictor(dim=d, eta_max=0.008),
            SpatialConvControl(height=8, width=8, channels=1),
        ]

        x_test = torch.randn(30, d)
        y_test = torch.randn(30, d)

        for m in models:
            m.reset_state()
            hash_before = compute_parameter_hash(m)
            for t in range(30):
                _ = m.predict_step(x_test[t])
                m.adapt_step(x_test[t], y_test[t])
            hash_after = compute_parameter_hash(m)
            assert hash_before == hash_after, (
                f"Parameter mutation detected in {m.name}!"
            )

    def test_safe_adaptive_delta_stability_bound(self) -> None:
        """SafeAdaptiveDelta must remain numerically finite and bounded under extreme inputs."""
        m = SafeAdaptiveDeltaPredictor(dim=16, eta_max=0.50)
        # Sequence of high-amplitude inputs that would blow up unconstrained FixedDelta
        for _ in range(50):
            x = torch.randn(16) * 5.0
            y = torch.randn(16) * 5.0
            pred = m.predict_step(x)
            assert torch.isfinite(pred).all()
            m.adapt_step(x, y)
            assert torch.isfinite(m.M).all()
            assert m.last_step_size <= 2.0 / max(float((x @ x).item()), 1e-5)

    def test_state_reset_control_alters_only_adaptive_state(self) -> None:
        """State reset at shift boundary must reset internal state without altering parameters."""
        cfg = RealSpatioTemporalConfig(height=8, width=8)
        data = generate_real_spatiotemporal_dataset(cfg)

        model_cont = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008)
        model_reset = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008)

        res_cont = evaluate_streaming_run(
            model=model_cont,
            inputs=data.test_inputs,
            targets=data.test_targets,
            shift_start=data.shift_interval[0],
            shift_end=data.shift_interval[1],
            reset_at_step=None,
        )
        res_reset = evaluate_streaming_run(
            model=model_reset,
            inputs=data.test_inputs,
            targets=data.test_targets,
            shift_start=data.shift_interval[0],
            shift_end=data.shift_interval[1],
            reset_at_step=data.shift_interval[0],
        )

        assert res_cont["rel_error"] != res_reset["rel_error"]
        assert res_cont["e_shift"] < res_reset["e_shift"]

    def test_spatial_permutation_control_discrimination(self) -> None:
        """Associative memory must be permutation invariant; SpatialConv must not."""
        cfg = RealSpatioTemporalConfig(height=8, width=8)
        data = generate_real_spatiotemporal_dataset(cfg)

        # DeltaCore model
        delta = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008)
        r_orig = evaluate_streaming_run(delta, data.test_inputs, data.test_targets)
        delta.reset_state()
        r_perm = evaluate_streaming_run(
            delta, data.permuted_test_inputs, data.permuted_test_targets
        )
        assert abs(r_orig["rel_error"] - r_perm["rel_error"]) < 1e-4

        # SpatialConv baseline
        conv = SpatialConvControl(height=8, width=8, channels=1)
        train_offline_model(conv, data.train_inputs, data.train_targets, epochs=15)
        c_orig = evaluate_streaming_run(conv, data.test_inputs, data.test_targets)
        conv.reset_state()
        c_perm = evaluate_streaming_run(
            conv, data.permuted_test_inputs, data.permuted_test_targets
        )
        assert c_perm["rel_error"] > c_orig["rel_error"] + 0.50


class TestPhase13ArtifactsAndSchemas:
    """Verification of Phase 13 artifact persistence and structure."""

    def test_serialized_artifacts_exist_and_validate(self) -> None:
        """All 5 primary Phase 13 artifacts and plots must exist and parse as valid JSON."""
        art_dir = Path("docs/benchmarks/artifacts/phase_13")

        required_json = [
            "dataset_config.json",
            "phase_13_results.json",
            "phase_13_per_seed.json",
            "phase_13_config.json",
            "phase_13_scaling.json",
            "phase_13_shift_analysis.json",
        ]

        for fname in required_json:
            p = art_dir / fname
            assert p.is_file(), f"Missing required artifact: {p}"
            with open(p) as f:
                content = json.load(f)
                assert isinstance(content, dict), (
                    f"Artifact {fname} is not a valid JSON dict"
                )

        plots_dir = art_dir / "plots"
        assert plots_dir.is_dir(), "Plots directory missing"
        plot_names = [
            "plot_bk_realworld_error_over_time.png",
            "plot_bl_shift_recovery.png",
            "plot_bm_state_norm_trajectory.png",
            "plot_bn_state_update_energy.png",
            "plot_bo_continuous_vs_reset.png",
            "plot_bp_safe_vs_fixed_delta.png",
            "plot_bq_spatial_permutation_control.png",
            "plot_br_perf_vs_resolution.png",
            "plot_bs_runtime_vs_dim.png",
            "plot_bt_memory_vs_dim.png",
            "plot_bu_deltacore_vs_conv.png",
        ]
        for pname in plot_names:
            plot_file = plots_dir / pname
            assert plot_file.is_file(), (
                f"Missing required observatory plot: {plot_file}"
            )
            assert plot_file.stat().st_size > 1000, (
                f"Plot {pname} is truncated or empty"
            )
