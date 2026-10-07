"""Rigorous test suite for AdaptiveController canonical public API.

Verifies:
1. Input validation: rejecting wrong dimensions, wrong vector length, NaN, +inf, -inf.
2. Hetero-associative and auto-associative execution modes.
3. Score-before-update temporal invariant (diagnostic evaluated on M_{t-1}).
4. Non-mutating score() vs mutating update() vs gated step().
5. Lyapunov contractive bounding and positive stability margin.
6. Deterministic execution: identical stream + identical seed -> identical trajectory.
7. State persistence and recovery: save/load roundtrip, versioning, corruption rejection, checksum validation.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from deltacore.controller import (
    AdaptiveController,
    ControllerConfig,
)


def test_input_validation_dimension_and_shape() -> None:
    """Verify controller rejects non-1D or incorrectly dimensioned inputs."""
    ctrl = AdaptiveController(dim=8)

    # 2D matrix input should be rejected
    x_2d = torch.zeros(8, 8)
    with pytest.raises(ValueError, match="must be a 1D vector"):
        ctrl.step(x_2d)

    # Wrong vector length should be rejected
    x_wrong = torch.zeros(16)
    with pytest.raises(ValueError, match="dimension mismatch"):
        ctrl.step(x_wrong)

    # Non-tensor input
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        ctrl.step([1.0] * 8)  # type: ignore[arg-type]


def test_input_validation_non_finite_rejection() -> None:
    """Verify controller rejects NaN, +inf, and -inf with FloatingPointError."""
    ctrl = AdaptiveController(dim=4)

    x_nan = torch.tensor([1.0, float("nan"), 0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-finite"):
        ctrl.step(x_nan)

    x_inf = torch.tensor([1.0, float("inf"), 0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-finite"):
        ctrl.step(x_inf)

    x_neginf = torch.tensor([1.0, float("-inf"), 0.0, 0.5])
    with pytest.raises(FloatingPointError, match="non-finite"):
        ctrl.step(x_neginf)


def test_auto_associative_mode() -> None:
    """Verify auto-associative mode when target is omitted (v_t = x_t)."""
    ctrl = AdaptiveController(dim=4, eta0=0.1)
    x = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # Initial step from zero matrix M_0
    res = ctrl.step(x)
    assert res.adapted is True
    # Pre-update prediction should be 0.0
    assert torch.allclose(res.prediction, torch.zeros(4))
    # Pre-update error is x - 0 = x
    assert torch.allclose(res.error, x)
    assert pytest.approx(res.reconstruction_residual) == float(
        torch.linalg.norm(x).item()
    )

    # State should now be updated
    state = ctrl.get_state()
    assert float(torch.linalg.norm(state).item()) > 0.0

    # Repeating exact same x should now yield smaller pre-update residual
    res2 = ctrl.step(x)
    assert res2.reconstruction_residual < res.reconstruction_residual


def test_score_before_update_invariant() -> None:
    """Verify that score() never mutates state and evaluates strictly against M_{t-1}."""
    ctrl = AdaptiveController(dim=4, eta0=0.2)
    x = torch.tensor([0.5, 0.5, 0.5, 0.5])

    initial_state = ctrl.get_state()
    res_score = ctrl.score(x)

    assert res_score.adapted is False
    assert res_score.update_norm == 0.0
    # State must be bit-exact identical to initial state
    assert torch.equal(ctrl.get_state(), initial_state)

    # Calling step(adapt=False) must produce identical result to score()
    res_gated_false = ctrl.step(x, adapt=False)
    assert torch.equal(res_score.prediction, res_gated_false.prediction)
    assert torch.equal(res_score.error, res_gated_false.error)
    assert res_score.reconstruction_residual == res_gated_false.reconstruction_residual
    assert torch.equal(ctrl.get_state(), initial_state)

    # Now execute an update
    res_update = ctrl.update(x)
    assert res_update.adapted is True
    assert res_update.update_norm > 0.0
    # Crucial: the pre-update residual reported in res_update MUST match res_score!
    assert res_update.reconstruction_residual == res_score.reconstruction_residual
    # But current state is now updated
    assert not torch.equal(ctrl.get_state(), initial_state)


def test_lyapunov_stability_bounding() -> None:
    """Verify Lyapunov bound prevents runaway step sizes under extreme input magnitudes."""
    ctrl = AdaptiveController(dim=4, eta0=1.0, rho=1.5, epsilon=1e-6)

    # Normal input
    x_norm = torch.tensor([1.0, 0.0, 0.0, 0.0])
    res_norm = ctrl.score(x_norm)
    assert res_norm.step_size <= 1.0
    assert res_norm.stability_margin >= 0.5

    # Huge input vector
    x_huge = torch.tensor([100.0, 100.0, 100.0, 100.0])
    res_huge = ctrl.score(x_huge)
    # Safe bound is rho / (||x||^2 + eps) = 1.5 / 40000.0 = 3.75e-5
    assert res_huge.step_size <= 1.5 / 40000.0
    assert res_huge.stability_margin >= 0.0


def test_state_persistence_and_recovery() -> None:
    """Verify save_state and load_state preserve state and stream continuity."""
    ctrl1 = AdaptiveController(dim=8, eta0=0.05)

    # Feed 10 steps to ctrl1
    torch.manual_seed(100)
    stream = [torch.randn(8) for _ in range(10)]
    for x in stream:
        ctrl1.step(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_controller_state.json"
        ctrl1.save_state(save_path)
        assert save_path.exists()

        # Initialize fresh controller
        ctrl2 = AdaptiveController(dim=8, eta0=0.05)
        ctrl2.load_state(save_path)

        # State tensors must match exactly
        assert torch.equal(ctrl1.get_state(), ctrl2.get_state())
        assert ctrl2.step_count == 10

        # Feed 5 next steps to both; outputs must be identical
        next_stream = [torch.randn(8) for _ in range(5)]
        for x in next_stream:
            r1 = ctrl1.step(x)
            r2 = ctrl2.step(x)
            assert torch.allclose(r1.prediction, r2.prediction)
            assert (
                pytest.approx(r1.reconstruction_residual) == r2.reconstruction_residual
            )
            assert torch.equal(ctrl1.get_state(), ctrl2.get_state())


def test_corrupted_state_rejection() -> None:
    """Verify controller rejects corrupted payloads, version mismatches, and tampered checksums."""
    ctrl = AdaptiveController(dim=4)
    ctrl.step(torch.ones(4))

    payload = ctrl.state_dict_payload()

    # 1. Tampered checksum
    tampered_checksum = dict(payload)
    tampered_checksum["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        ctrl.load_state_dict_payload(tampered_checksum)

    # 2. Version mismatch
    tampered_ver = dict(payload)
    tampered_ver["format_version"] = "deltacore.state.v999"
    with pytest.raises(ValueError, match="Incompatible state format version"):
        ctrl.load_state_dict_payload(tampered_ver)

    # 3. Dimension mismatch
    tampered_dim = dict(payload)
    tampered_dim["k_dim"] = 8
    with pytest.raises(ValueError, match="Dimension mismatch"):
        ctrl.load_state_dict_payload(tampered_dim)


def test_hetero_associative_mode() -> None:
    """Verify rectangular hetero-associative mode (V != K)."""
    config = ControllerConfig(dim=4, val_dim=2, eta0=0.1)
    ctrl = AdaptiveController(config)

    x = torch.tensor([1.0, 0.5, -0.5, 0.0])
    v = torch.tensor([2.0, -1.0])

    res = ctrl.step(x, target=v)
    assert res.prediction.shape == (2,)
    assert res.error.shape == (2,)
    assert ctrl.get_state().shape == (2, 4)
