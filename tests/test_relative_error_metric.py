"""Unit tests for compute_relative_error metric.

Verifies:
- Exact zero error for identical tensors
- E_rel = 1.0 for zero prediction against non-zero target
- Stability when target norm is near or equal to zero (epsilon clamping)
- Shape invariance and tensor norm scaling
"""

import torch

from deltacore.benchmarks.metrics.accuracy import compute_relative_error


def test_relative_error_identical_tensors() -> None:
    """Exact identical predictions should have relative error 0.0."""
    t = torch.randn(4, 3, 8, 8)
    err = compute_relative_error(t, t)
    assert abs(err) < 1e-7


def test_relative_error_zero_prediction() -> None:
    """Zero prediction against target should have relative error ~1.0."""
    target = torch.randn(4, 3, 8, 8)
    pred = torch.zeros_like(target)
    err = compute_relative_error(pred, target)
    assert abs(err - 1.0) < 1e-5


def test_relative_error_zero_target() -> None:
    """Zero target should clamp denominator to epsilon and avoid division by zero."""
    target = torch.zeros(2, 2)
    pred = torch.ones(2, 2)
    eps = 1e-4
    err = compute_relative_error(pred, target, eps=eps)
    expected = float(torch.norm(pred).item() / eps)
    assert abs(err - expected) < 1e-3


def test_relative_error_scale_invariance() -> None:
    """Scaling both target and prediction by same constant alpha should keep E_rel unchanged."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    err1 = compute_relative_error(pred, target)
    err2 = compute_relative_error(pred * 10.0, target * 10.0)
    assert abs(err1 - err2) < 1e-5
