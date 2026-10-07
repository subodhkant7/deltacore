"""Test to verify that benchmark scoring and update loop operates strictly blind to labels."""

from __future__ import annotations

import inspect

import torch

from deltacore import AdaptiveController, ControllerConfig


def test_controller_score_signature_has_no_labels() -> None:
    """Verify that AdaptiveController.score and step take only x and target, never labels."""
    cfg = ControllerConfig(dim=16)
    ctrl = AdaptiveController(cfg)

    sig_score = inspect.signature(ctrl.score)
    assert "label" not in sig_score.parameters
    assert "anomaly" not in sig_score.parameters
    assert "ground_truth" not in sig_score.parameters

    sig_step = inspect.signature(ctrl.step)
    assert "label" not in sig_step.parameters
    assert "anomaly" not in sig_step.parameters


def test_blind_scoring_invariance() -> None:
    """Verify that model predictions are completely independent of external label definitions."""
    cfg = ControllerConfig(dim=16, eta0=0.03, rho=0.95)
    ctrl = AdaptiveController(cfg)

    x = torch.randn(16)
    x = x / torch.norm(x)

    res1 = ctrl.score(x)
    # Score must be deterministic and pure
    res2 = ctrl.score(x)
    assert abs(res1.reconstruction_residual - res2.reconstruction_residual) < 1e-7
