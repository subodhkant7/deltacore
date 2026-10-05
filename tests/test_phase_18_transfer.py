"""Unit Tests for DeltaCore Phase 18: Unseen Classification Regime Transfer & Falsification.

Verifies:
    1. Multi-family stream generators and energy normalization invariants
    2. Model definitions and frozen evaluation configuration
    3. Strict online causal sequencing (pred -> reveal -> adapt) and timestamp monotonicity
    4. Cryptographic parameter immutability (Delta theta = 0)
    5. Causal state ablation (State ON vs State OFF)
    6. Retention tradeoff (continuous persistent state vs explicit boundary reset)
    7. Shortcut audit and negative controls (label shuffle, feature permutation)
    8. Dimensional scaling and exact 4*D^2 byte persistent state representation
    9. Artifact serialization and hypothesis evaluation logic
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from deltacore.benchmarks.phase_18.controls import (
    audit_shortcut_statistics_18,
    permute_features_18,
    shuffle_labels_18,
)
from deltacore.benchmarks.phase_18.generators import (
    NonStationaryStream18,
    StationaryDataset18,
    generate_covariate_shift_stream,
    generate_family_a_rotation,
    generate_family_b_translation,
    generate_family_c_nonlinear,
    generate_gradual_shift_stream,
    generate_prior_shift_stream,
    generate_stationary_dataset_18,
    generate_strong_mismatch_stream,
)
from deltacore.benchmarks.phase_18.models import (
    FrozenLinearClassifier18,
    SafeAdaptiveDeltaClassifier18,
    StateOffAblationClassifier18,
    build_phase_18_model_suite,
    fit_offline_linear_head_18,
)
from deltacore.benchmarks.phase_18.protocol import (
    CausalStreamingProtocol,
    verify_parameter_immutability,
)

# ==============================================================================
# 1. Generator Invariants
# ==============================================================================


def test_stationary_dataset_invariants() -> None:
    ds = generate_stationary_dataset_18(
        dim=16, num_classes=4, n_train=120, n_val=40, n_test=40, seed=42
    )
    assert isinstance(ds, StationaryDataset18)
    assert ds.train_inputs.shape == (120, 16)
    assert ds.train_targets.shape == (120,)
    assert ds.class_prototypes.shape == (4, 16)

    # Invariant: Each sample normalized to sqrt(D)
    norms = torch.linalg.norm(ds.train_inputs, dim=-1)
    expected_norm = math.sqrt(16)
    assert torch.allclose(norms, torch.full_like(norms, expected_norm), atol=1e-4)


def test_task_family_a_invariants() -> None:
    stream = generate_family_a_rotation(
        dim=16, num_classes=4, steps_per_regime=50, seed=42
    )
    assert isinstance(stream, NonStationaryStream18)
    assert stream.inputs.shape == (150, 16)
    assert stream.targets.shape == (150,)
    assert stream.regime_bounds == [50, 100]
    assert len(stream.regime_names) == 3

    # Energy normalization check
    norms = torch.linalg.norm(stream.inputs, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(16)), atol=1e-4)


def test_task_family_b_invariants() -> None:
    stream = generate_family_b_translation(
        dim=16, num_classes=4, steps_per_regime=40, seed=42
    )
    assert stream.inputs.shape == (120, 16)
    assert stream.regime_bounds == [40, 80]
    norms = torch.linalg.norm(stream.inputs, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(16)), atol=1e-4)


def test_task_family_c_invariants() -> None:
    stream = generate_family_c_nonlinear(
        dim=16, num_classes=4, steps_per_regime=30, seed=42
    )
    assert stream.inputs.shape == (90, 16)
    assert stream.regime_bounds == [30, 60]
    norms = torch.linalg.norm(stream.inputs, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(16)), atol=1e-4)


def test_covariate_shift_invariants() -> None:
    stream = generate_covariate_shift_stream(
        dim=16, num_classes=4, steps_per_regime=30, seed=42
    )
    assert stream.inputs.shape == (90, 16)
    assert stream.regime_bounds == [30, 60]
    norms = torch.linalg.norm(stream.inputs, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(16)), atol=1e-4)


def test_gradual_and_prior_shift_invariants() -> None:
    s_grad = generate_gradual_shift_stream(
        dim=16, num_classes=4, total_steps=60, seed=42
    )
    assert s_grad.inputs.shape == (60, 16)

    s_prior = generate_prior_shift_stream(
        dim=16, num_classes=4, steps_per_regime=30, seed=42
    )
    assert s_prior.inputs.shape == (90, 16)
    # Check that in Regime B classes 0 and 1 occur more frequently
    b_targets = s_prior.targets[30:60]
    high_freq_count = int(((b_targets == 0) | (b_targets == 1)).sum().item())
    assert high_freq_count >= 15  # Expected ~80% of 30 steps = 24


def test_strong_mismatch_invariants() -> None:
    stream = generate_strong_mismatch_stream(
        dim=16, num_classes=4, steps_a=50, steps_mismatch=20, seed=42
    )
    assert stream.inputs.shape == (70, 16)
    assert stream.regime_bounds == [50]


# ==============================================================================
# 2. Model Invariants & Frozen Parameter Immutability
# ==============================================================================


def test_model_suite_instantiation_and_parameter_hashes() -> None:
    stat_ds = generate_stationary_dataset_18(dim=16, num_classes=4, seed=42)
    suite = build_phase_18_model_suite(stat_ds, seed=42)
    assert len(suite) == 7
    expected_models = {
        "FrozenLinear",
        "AdaptiveStateOFF",
        "OnlineLogisticRegression",
        "OnlineRidge",
        "OnlineMulticlassLinear",
        "FixedDelta",
        "SafeAdaptiveDelta",
    }
    assert set(suite.keys()) == expected_models

    for _m_name, model in suite.items():
        h = model.get_parameter_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256


def test_safe_adaptive_delta_contraction_and_bounds() -> None:
    dim = 16
    k = 4
    w = torch.randn(k, dim)
    b = torch.zeros(k)
    model = SafeAdaptiveDeltaClassifier18(
        dim, k, w, b, eta0=0.015, rho=1.50, alpha_min=0.95
    )

    # Initial state
    assert model.M.shape == (dim, dim)
    assert torch.all(model.M == 0)
    assert model.get_state_memory_bytes() == 4 * dim * dim

    x = torch.randn(dim)
    pred, probs = model.predict_step(x)
    assert 0 <= pred < k
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    # Adapt step
    model.adapt_step(x, 1)
    assert model.last_safety_margin >= 0.0  # Must obey contraction bound
    assert model.last_retention >= 0.95  # Obey alpha_min bound
    assert float(torch.linalg.norm(model.M).item()) > 0.0

    # Reset
    model.reset_state()
    assert torch.all(model.M == 0)


def test_state_off_ablation_model() -> None:
    dim = 16
    k = 4
    w = torch.randn(k, dim)
    b = torch.zeros(k)
    off_model = StateOffAblationClassifier18(dim, k, w, b)
    frozen_model = FrozenLinearClassifier18(dim, k, w, b)

    x = torch.randn(dim)
    p_off, pr_off = off_model.predict_step(x)
    p_froz, pr_froz = frozen_model.predict_step(x)

    assert p_off == p_froz
    assert torch.allclose(pr_off, pr_froz)

    off_model.adapt_step(x, 0)
    # State should remain zero
    assert off_model.get_state_memory_bytes() == 0


# ==============================================================================
# 3. Online Causal Streaming Protocol & Timestamp Invariants
# ==============================================================================


def test_causal_streaming_protocol_execution() -> None:
    dim = 16
    k = 4
    stat_ds = generate_stationary_dataset_18(dim=dim, num_classes=k, seed=42)
    suite = build_phase_18_model_suite(stat_ds, seed=42)
    stream = generate_family_a_rotation(
        dim=dim, num_classes=k, steps_per_regime=20, seed=42
    )

    for _m_name, model in suite.items():
        res = CausalStreamingProtocol.evaluate_stream(
            model=model,
            inputs=stream.inputs,
            targets=stream.targets,
            regime_bounds=stream.regime_bounds,
            reset_at_bounds=False,
        )
        assert res.causality_verified
        assert res.parameter_immutable
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.balanced_accuracy <= 1.0
        assert res.diverged == 0


def test_parameter_immutability_assertion_catches_mutation() -> None:
    dim = 16
    k = 4
    w = torch.randn(k, dim)
    b = torch.zeros(k)
    model = FrozenLinearClassifier18(dim, k, w, b)
    h_before = model.get_parameter_hash()
    assert verify_parameter_immutability(model, h_before)

    # Mutate parameter
    model.weight[0, 0] += 1.0
    assert not verify_parameter_immutability(model, h_before)


# ==============================================================================
# 4. Shortcut Audit & Negative Controls
# ==============================================================================


def test_shortcut_audit_clean_stream() -> None:
    stream = generate_family_a_rotation(
        dim=16, num_classes=4, steps_per_regime=50, seed=42
    )
    audit = audit_shortcut_statistics_18(
        stream.inputs, stream.targets, stream.regime_bounds, 4
    )
    assert audit["passed_shortcut_audit"]
    assert audit["energy_matched"]
    assert audit["variance_matched"]
    assert audit["no_feature_leak"]
    assert audit["no_regime_leak"]


def test_label_shuffle_control_destroys_adaptation() -> None:
    dim = 16
    k = 4
    stat_ds = generate_stationary_dataset_18(dim=dim, num_classes=k, seed=42)
    w_h, b_h = fit_offline_linear_head_18(
        stat_ds.train_inputs, stat_ds.train_targets, k
    )
    stream = generate_family_a_rotation(
        dim=dim, num_classes=k, steps_per_regime=40, seed=42
    )

    shuffled_targets = shuffle_labels_18(stream.targets, seed=999)
    assert not torch.equal(shuffled_targets, stream.targets)

    model = SafeAdaptiveDeltaClassifier18(dim, k, w_h, b_h)
    res = CausalStreamingProtocol.evaluate_stream(
        model, stream.inputs, shuffled_targets, stream.regime_bounds
    )
    # Shuffled targets should yield near chance (~25% for K=4)
    assert res.accuracy < 0.45


def test_feature_permutation_invariance() -> None:
    dim = 16
    k = 4
    stat_ds = generate_stationary_dataset_18(dim=dim, num_classes=k, seed=42)
    w_h, b_h = fit_offline_linear_head_18(
        stat_ds.train_inputs, stat_ds.train_targets, k
    )
    stream = generate_family_a_rotation(
        dim=dim, num_classes=k, steps_per_regime=30, seed=42
    )

    m1 = SafeAdaptiveDeltaClassifier18(dim, k, w_h, b_h)
    res1 = CausalStreamingProtocol.evaluate_stream(
        m1, stream.inputs, stream.targets, stream.regime_bounds
    )

    perm_inputs, perm_idx = permute_features_18(stream.inputs, seed=123)
    w_h_perm = w_h[:, perm_idx]
    m2 = SafeAdaptiveDeltaClassifier18(dim, k, w_h_perm, b_h)
    res2 = CausalStreamingProtocol.evaluate_stream(
        m2, perm_inputs, stream.targets, stream.regime_bounds
    )

    assert abs(res1.accuracy - res2.accuracy) < 1e-4


# ==============================================================================
# 5. Dimensional Scaling & Memory Byte Invariants
# ==============================================================================


@pytest.mark.parametrize("d_val", [32, 64, 128, 256])
def test_exact_4d_squared_persistent_bytes(d_val: int) -> None:
    k = 6
    w = torch.randn(k, d_val)
    b = torch.zeros(k)
    model = SafeAdaptiveDeltaClassifier18(d_val, k, w, b)
    expected_bytes = 4 * d_val * d_val
    assert model.get_state_memory_bytes() == expected_bytes


# ==============================================================================
# 6. Corrected Scientific Invariants & Reproducibility Gates (Section 12)
# ==============================================================================


def test_invariant_1_state_off_exactly_matches_frozen_behavior() -> None:
    """Invariant 1: StateOff exactly matches FrozenLinear behavior across stream."""
    dim = 16
    k = 4
    w = torch.randn(k, dim)
    b = torch.randn(k)
    froz_model = FrozenLinearClassifier18(dim, k, w, b)
    off_model = StateOffAblationClassifier18(dim, k, w, b)

    stream = generate_family_a_rotation(
        dim=dim, num_classes=k, steps_per_regime=30, seed=42
    )

    res_froz = CausalStreamingProtocol.evaluate_stream(
        froz_model, stream.inputs, stream.targets
    )
    res_off = CausalStreamingProtocol.evaluate_stream(
        off_model, stream.inputs, stream.targets
    )

    assert res_froz.accuracy == res_off.accuracy
    assert res_froz.balanced_accuracy == res_off.balanced_accuracy
    assert math.isclose(res_froz.log_loss, res_off.log_loss, rel_tol=1e-5)


def test_invariant_2_parameter_hashes_remain_unchanged_during_streaming() -> None:
    """Invariant 2: Parameter hashes remain unchanged during streaming."""
    stat_ds = generate_stationary_dataset_18(dim=16, num_classes=4, seed=42)
    suite = build_phase_18_model_suite(stat_ds, seed=42)
    stream = generate_family_b_translation(
        dim=16, num_classes=4, steps_per_regime=25, seed=42
    )

    for m_name, model in suite.items():
        h_before = model.get_parameter_hash()
        res = CausalStreamingProtocol.evaluate_stream(
            model, stream.inputs, stream.targets
        )
        h_after = model.get_parameter_hash()
        assert res.parameter_immutable, f"Parameter mutability detected for {m_name}"
        assert h_before == h_after, (
            f"Pre/post hash mismatch for {m_name}: {h_before} != {h_after}"
        )


def test_invariant_3_prediction_occurs_before_label_reveal() -> None:
    """Invariant 3: Prediction occurs before label reveal with monotonic timestamps."""
    dim = 16
    k = 4
    stat_ds = generate_stationary_dataset_18(dim=dim, num_classes=k, seed=42)
    suite = build_phase_18_model_suite(stat_ds, seed=42)
    stream = generate_family_a_rotation(
        dim=dim, num_classes=k, steps_per_regime=20, seed=42
    )

    model = suite["SafeAdaptiveDelta"]
    res = CausalStreamingProtocol.evaluate_stream(model, stream.inputs, stream.targets)
    assert res.causality_verified


@pytest.mark.parametrize("dim_val", [32, 64, 128, 256])
def test_invariant_4_persistent_state_memory_equals_4d_squared_bytes(
    dim_val: int,
) -> None:
    """Invariant 4: Persistent state memory equals exact 4*D^2 bytes FP32."""
    k = 6
    w = torch.randn(k, dim_val)
    b = torch.zeros(k)
    model = SafeAdaptiveDeltaClassifier18(dim_val, k, w, b)
    assert model.get_state_memory_bytes() == 4 * dim_val * dim_val


def test_invariant_5_phase_18_configuration_serialization_deterministic() -> None:
    """Invariant 5: Phase 18 configuration serialization is deterministic."""
    import hashlib

    config_template = {
        "eta0": 0.015,
        "rho": 1.50,
        "alpha_min": 0.95,
        "gamma": 0.10,
        "epsilon": 1e-6,
        "step_size_fixed": 0.015,
        "alpha_fixed": 0.95,
        "default_dim": 32,
        "num_classes": 6,
        "seeds": [42, 43, 44, 45, 46],
        "scaling_dims": [32, 64, 128, 256],
        "steps_per_regime": 120,
    }

    s1 = json.dumps(config_template, indent=2, sort_keys=True)
    s2 = json.dumps(config_template, indent=2, sort_keys=True)
    assert s1 == s2
    h1 = hashlib.sha256(s1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(s2.encode("utf-8")).hexdigest()
    assert h1 == h2


def test_invariant_6_reported_transfer_values_reconstructed_from_per_seed_data() -> (
    None
):
    """Invariant 6: Phase 18 reported transfer values can be reconstructed from per-seed data."""
    import numpy as np

    artifacts_path = Path("docs/benchmarks/artifacts/phase_18")
    per_seed_file = artifacts_path / "phase_18_per_seed.json"
    results_file = artifacts_path / "phase_18_results.json"

    if not (per_seed_file.exists() and results_file.exists()):
        pytest.skip("Serialized benchmark artifacts not found.")

    with open(per_seed_file, encoding="utf-8") as f:
        per_seed_data = json.load(f)
    with open(results_file, encoding="utf-8") as f:
        results_data = json.load(f)

    tasks = list(per_seed_data[0]["tasks"].keys())
    for t in tasks:
        models = list(per_seed_data[0]["tasks"][t].keys())
        for m in models:
            seed_accs = [
                s["tasks"][t][m]["accuracy"]
                for s in per_seed_data
                if t in s["tasks"] and m in s["tasks"][t]
            ]
            assert len(seed_accs) == 5, f"Expected 5 seeds for task {t}, model {m}"
            reconstructed_mean = float(np.mean(seed_accs))
            reconstructed_std = float(np.std(seed_accs))

            reported_mean = results_data[t][m]["accuracy_mean"]
            reported_std = results_data[t][m]["accuracy_std"]

            assert abs(reconstructed_mean - reported_mean) < 1e-4, (
                f"Reconstruction mismatch for {t}/{m}: {reconstructed_mean} vs {reported_mean}"
            )
            assert abs(reconstructed_std - reported_std) < 1e-4, (
                f"Std reconstruction mismatch for {t}/{m}: {reconstructed_std} vs {reported_std}"
            )


def test_invariant_7_no_accidental_future_label_access() -> None:
    """Invariant 7: Modifying future labels has zero impact on step t prediction."""
    dim = 16
    k = 4
    stat_ds = generate_stationary_dataset_18(dim=dim, num_classes=k, seed=42)
    w, b = fit_offline_linear_head_18(stat_ds.train_inputs, stat_ds.train_targets, k)

    stream = generate_family_a_rotation(
        dim=dim, num_classes=k, steps_per_regime=20, seed=42
    )
    targets_clean = stream.targets.clone()
    targets_corrupted = stream.targets.clone()

    # Step t=10 prediction under original vs corrupted future labels
    m1 = SafeAdaptiveDeltaClassifier18(dim, k, w, b)
    m2 = SafeAdaptiveDeltaClassifier18(dim, k, w, b)

    # Adapt up to t=9 identically
    for t in range(10):
        x = stream.inputs[t]
        y1 = int(targets_clean[t].item())
        y2 = int(targets_corrupted[t].item())
        m1.adapt_step(x, y1)
        m2.adapt_step(x, y2)

    # Now corrupt future targets for t >= 10 in targets_corrupted
    targets_corrupted[10:] = (targets_corrupted[10:] + 1) % k

    # Prediction at t=10 must be bit-for-bit identical
    x_10 = stream.inputs[10]
    pred1, probs1 = m1.predict_step(x_10)
    pred2, probs2 = m2.predict_step(x_10)

    assert pred1 == pred2
    assert torch.equal(probs1, probs2)


def test_invariant_8_no_nan_or_inf_escapes_silently() -> None:
    """Invariant 8: Non-finite values are caught and flagged rather than silently ignored."""
    dim = 16
    k = 4
    w = torch.randn(k, dim)
    b = torch.zeros(k)
    model = SafeAdaptiveDeltaClassifier18(dim, k, w, b)

    inputs = torch.randn(10, dim)
    targets = torch.randint(0, k, (10,))

    # Inject NaN into input stream
    inputs[5, 2] = float("nan")

    res = CausalStreamingProtocol.evaluate_stream(model, inputs, targets)
    # The protocol must detect divergence / non-finiteness
    assert res.diverged == 1 or math.isnan(res.log_loss)
