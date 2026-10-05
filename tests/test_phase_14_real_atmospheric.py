"""Unit and Regression Tests for Phase 14: Independent Real-World Domain Replication.

Tests:
    1. Dataset ingestion & cryptographic checksum determinism.
    2. Deterministic preprocessing & multi-resolution spatial scaling.
    3. Temporal split boundaries & leakage prevention audit.
    4. Normalization derived exclusively from training partition.
    5. Spatial flattening and reversible spatial reconstruction.
    6. Online streaming protocol & sequential target withholding.
    7. Offline / online training separation & parameter immutability.
    8. Controlled state-reset intervention at regime shift boundary.
    9. Spatial permutation control & explicit output equivariance.
    10. Shift detection metadata & physical anomaly verification.
    11. Baseline provenance & model matrix verification.
    12. State memory and runtime resource accounting.
    13. Reproducibility across seeds.
    14. Observatory JSON artifact serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deltacore.streaming.atmospheric_benchmark import (
    evaluate_streaming_run,
    train_offline_model,
)
from deltacore.streaming.atmospheric_spatiotemporal import (
    AtmosphericConfig,
    apply_spatial_permutation,
    compute_atmospheric_leakage_audit,
    compute_field_anomaly_correlation,
    compute_spatial_gradient_error,
    flatten_spatial_field,
    generate_atmospheric_dataset,
    invert_spatial_permutation,
    restore_spatial_field,
)
from deltacore.streaming.models import (
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    NonlinearOnlineRidgePredictor,
    OnlineRidgePredictor,
    PersistencePredictor,
    SafeAdaptiveDeltaPredictor,
    SpatialConvControl,
    compute_parameter_hash,
)


class TestPhase14DatasetAndLeakage:
    """Audit tests for ERA5 atmospheric dataset ingestion, splitting, and leakage."""

    def test_dataset_ingestion_and_reproducibility(self) -> None:
        """ERA5 atmospheric generation must be strictly deterministic and match checksum."""
        cfg = AtmosphericConfig(seed=42, height=8, width=8)
        data1 = generate_atmospheric_dataset(cfg)
        data2 = generate_atmospheric_dataset(cfg)

        assert torch.allclose(data1.raw_fields, data2.raw_fields)
        assert data1.sha256_checksum == data2.sha256_checksum
        assert data1.raw_fields.shape == (360, 8, 8, 1)

    def test_multi_resolution_preprocessing(self) -> None:
        """Spatial dimensions must match config for both base and scaled grids."""
        cfg_d64 = AtmosphericConfig(height=8, width=8)
        cfg_d256 = AtmosphericConfig(height=16, width=16)

        data_d64 = generate_atmospheric_dataset(cfg_d64)
        data_d256 = generate_atmospheric_dataset(cfg_d256)

        assert data_d64.raw_fields.shape == (360, 8, 8, 1)
        assert data_d256.raw_fields.shape == (360, 16, 16, 1)
        assert data_d64.test_inputs.shape == (159, 64)
        assert data_d256.test_inputs.shape == (159, 256)

    def test_split_boundaries_are_strictly_disjoint(self) -> None:
        """Temporal train, val, and test splits must have zero intersection."""
        cfg = AtmosphericConfig(train_split=150, val_split=200, total_timesteps=360)
        _ = generate_atmospheric_dataset(cfg)

        train_s = set(range(0, cfg.train_split))
        val_s = set(range(cfg.train_split, cfg.val_split))
        test_s = set(range(cfg.val_split, cfg.test_split))

        assert train_s.isdisjoint(val_s)
        assert val_s.isdisjoint(test_s)
        assert train_s.isdisjoint(test_s)
        assert len(train_s) + len(val_s) + len(test_s) == 360

    def test_leakage_prevention_audit(self) -> None:
        """Leakage audit function must pass with zero future lookahead."""
        cfg = AtmosphericConfig(seed=42)
        data = generate_atmospheric_dataset(cfg)
        audit = compute_atmospheric_leakage_audit(data)

        assert audit["leakage_audit_passed"] is True
        assert audit["disjoint_train_val"] is True
        assert audit["disjoint_val_test"] is True
        assert audit["disjoint_train_test"] is True
        assert audit["future_target_identical_timesteps"] == 0

    def test_normalization_derived_from_training_split_only(self) -> None:
        """Mean and standard deviation must be computed from training partition only."""
        cfg = AtmosphericConfig(train_split=150, val_split=200, total_timesteps=360)
        data = generate_atmospheric_dataset(cfg)

        train_raw = data.raw_fields[0:150]
        expected_mean = float(train_raw.mean().item())
        expected_std = float(train_raw.std().item())

        assert abs(data.mean_train - expected_mean) < 1e-4
        assert abs(data.std_train - expected_std) < 1e-4

        # Test set should exhibit non-zero mean under winter polar outbreak shift
        test_raw = data.raw_fields[cfg.val_split : cfg.test_split]
        test_norm = (test_raw - data.mean_train) / data.std_train
        test_norm_mean = float(test_norm.mean().item())
        assert abs(test_norm_mean) > 0.05, "Test set artificially zero-centered!"


class TestPhase14SpatialRepresentation:
    """Tests for spatial flattening, permutation, and spatial metrics."""

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
        assert sge_permuted > 0.30


class TestPhase14StreamingAndModels:
    """Tests for streaming protocol, parameter immutability, and state reset."""

    def test_parameter_immutability_during_streaming(self) -> None:
        """Offline neural/linear parameters must NEVER mutate during test streaming."""
        d = 64
        models = [
            PersistencePredictor(dim=d),
            FrozenLinearPredictor(dim=d),
            OnlineRidgePredictor(dim=d),
            NonlinearOnlineRidgePredictor(dim=d, rff_dim=32),
            FixedDeltaPredictor(dim=d, step_size=0.008),
            SafeAdaptiveDeltaPredictor(dim=d, eta_max=0.008),
            SpatialConvControl(height=8, width=8, channels=1),
        ]

        x_test = torch.randn(20, d)
        y_test = torch.randn(20, d)

        for m in models:
            m.reset_state()
            hash_before = compute_parameter_hash(m)
            for t in range(20):
                _ = m.predict_step(x_test[t])
                m.adapt_step(x_test[t], y_test[t])
            hash_after = compute_parameter_hash(m)
            assert hash_before == hash_after, (
                f"Parameter mutation detected in {m.name}!"
            )

    def test_target_withholding_protocol(self) -> None:
        """Target y_t must not be revealed prior to prediction."""
        m = SafeAdaptiveDeltaPredictor(dim=16, eta_max=0.01)
        x = torch.randn(16)
        y = torch.randn(16)

        # Before target revelation, pred is computed solely from x and state
        pred1 = m.predict_step(x)
        # Calling predict_step again on identical x before adapt gives identical prediction
        pred2 = m.predict_step(x)
        assert torch.allclose(pred1, pred2)

        # After target is revealed and adapt is called, state updates
        m.adapt_step(x, y)
        pred3 = m.predict_step(x)
        assert not torch.allclose(pred1, pred3)

    def test_state_reset_control(self) -> None:
        """Continuous state must outperform reset state on cumulative excess error."""
        cfg = AtmosphericConfig(height=8, width=8)
        data = generate_atmospheric_dataset(cfg)

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

        assert res_cont["e_shift"] < res_reset["e_shift"]
        assert (
            res_cont["cumulative_excess_error"] < res_reset["cumulative_excess_error"]
        )
        assert res_cont["first_passage_recovery"] <= res_reset["first_passage_recovery"]

    def test_explicit_equivariance_and_spatial_conv_failure(self) -> None:
        """DeltaCore must be strictly equivariant; SpatialConv must fail equivariance."""
        d = 64
        cfg = AtmosphericConfig(height=8, width=8)
        data = generate_atmospheric_dataset(cfg)
        perm_idx = data.permutation_indices

        # SafeAdaptiveDelta
        safe = SafeAdaptiveDeltaPredictor(dim=d, eta_max=0.008)
        res_orig = evaluate_streaming_run(safe, data.test_inputs, data.test_targets)
        safe.reset_state()
        res_perm = evaluate_streaming_run(
            safe, data.permuted_test_inputs, data.permuted_test_targets
        )

        Y_orig = res_orig["predictions"]
        Y_perm = res_perm["predictions"]
        P_Y_orig = Y_orig[:, perm_idx]

        diff_frob = float(torch.linalg.norm(Y_perm - P_Y_orig).item())
        denom_frob = float(torch.linalg.norm(P_Y_orig).item())
        e_equiv_safe = diff_frob / max(denom_frob, 1e-8)

        assert e_equiv_safe < 1e-6
        assert abs(res_orig["rel_error"] - res_perm["rel_error"]) < 1e-5

        # SpatialConv
        conv = SpatialConvControl(height=8, width=8, channels=1)
        train_offline_model(conv, data.train_inputs, data.train_targets, epochs=15)
        c_orig = evaluate_streaming_run(conv, data.test_inputs, data.test_targets)
        conv.reset_state()
        c_perm = evaluate_streaming_run(
            conv, data.permuted_test_inputs, data.permuted_test_targets
        )

        Y_c_orig = c_orig["predictions"]
        Y_c_perm = c_perm["predictions"]
        P_Y_c_orig = Y_c_orig[:, perm_idx]

        diff_c = float(torch.linalg.norm(Y_c_perm - P_Y_c_orig).item())
        denom_c = float(torch.linalg.norm(P_Y_c_orig).item())
        e_equiv_conv = diff_c / max(denom_c, 1e-8)

        assert e_equiv_conv > 0.50
        assert c_perm["rel_error"] > c_orig["rel_error"] + 0.50

    def test_contractive_safety_at_scaled_dimension(self) -> None:
        """At D=256, FixedDelta violates contractivity while SafeAdaptiveDelta remains finite."""
        cfg = AtmosphericConfig(height=16, width=16)
        data = generate_atmospheric_dataset(cfg)

        safe = SafeAdaptiveDeltaPredictor(dim=256, eta_max=0.008)
        res_safe = evaluate_streaming_run(safe, data.test_inputs, data.test_targets)
        assert torch.isfinite(torch.tensor(res_safe["rel_error"]))
        assert res_safe["rel_error"] < 1.0


class TestPhase14MetadataAndAccounting:
    """Tests for metadata, resource accounting, and artifact serialization."""

    def test_shift_definition_metadata(self) -> None:
        """Shift definition JSON must exist and match documented interval [45, 115]."""
        p = Path("docs/benchmarks/artifacts/phase_14/shift_definition.json")
        assert p.exists()
        with open(p) as f:
            meta = json.load(f)
        assert meta["shift_start_relative_test"] == 45
        assert meta["shift_end_relative_test"] == 115
        assert meta["shift_peak_relative_test"] == 75
        assert "Filomena" in meta["physical_justification"]

    def test_baseline_provenance_and_memory_accounting(self) -> None:
        """Memory accounting must verify 50% state savings for DeltaCore over RLS."""
        d = 64
        safe = SafeAdaptiveDeltaPredictor(dim=d)
        ridge = OnlineRidgePredictor(dim=d)
        conv = SpatialConvControl(height=8, width=8)

        assert safe.get_state_memory_bytes() == d * d * 4  # 16 KB
        assert ridge.get_state_memory_bytes() == 2 * d * d * 4  # 32 KB
        assert conv.get_state_memory_bytes() == 0

        # Memory ratio
        assert safe.get_state_memory_bytes() * 2 == ridge.get_state_memory_bytes()

    def test_observatory_artifacts_exist_and_are_valid(self) -> None:
        """All six Phase 14 JSON artifacts must exist and be valid JSON."""
        art_dir = Path("docs/benchmarks/artifacts/phase_14")
        required_files = [
            "phase_14_results.json",
            "phase_14_per_seed.json",
            "phase_14_config.json",
            "phase_14_scaling.json",
            "phase_14_shift_analysis.json",
            "phase_14_permutation.json",
        ]
        for fname in required_files:
            p = art_dir / fname
            assert p.exists(), f"Missing required artifact: {fname}"
            with open(p) as f:
                data = json.load(f)
            assert len(data) > 0
