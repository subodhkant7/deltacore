r"""Autograd tests for adaptive step-size controllers and dynamic state transitions.

Verifies:
1. Gradients flow from memory update into controller parameters:
   input -> controller -> \eta_t -> memory update -> loss
2. Gradients reach memory, key, target, and controller parameters.
3. Numerical gradient validation using torch.autograd.gradcheck in FP64.
"""

import torch

from deltacore.scans.adaptive import adaptive_scan
from deltacore.updates.adaptive_delta import AdaptiveDeltaRule
from deltacore.updates.controllers import (
    ErrorConditionedStepSize,
    InputConditionedStepSize,
    StateConditionedStepSize,
)


class TestAdaptiveAutograd:
    """Verifies that adaptive dynamics maintain clean, non-detached PyTorch autograd graphs."""

    def test_gradients_through_input_conditioned_delta(self):
        """Verify gradients flow to w, b, memory, key, and target in InputConditionedStepSize."""
        ctrl = InputConditionedStepSize(dim=2, eta_max=1.0, bias=0.5).to(
            dtype=torch.float64
        )
        rule = AdaptiveDeltaRule(controller=ctrl)

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        res = rule.step(m, k, v)
        loss = (res.new_memory**2).sum()
        loss.backward()

        assert m.grad is not None and torch.isfinite(m.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()
        assert ctrl.weight.grad is not None and torch.isfinite(ctrl.weight.grad).all()
        assert ctrl.bias.grad is not None and torch.isfinite(ctrl.bias.grad).all()

        assert torch.norm(ctrl.weight.grad) > 0.0
        assert torch.norm(ctrl.bias.grad) > 0.0

    def test_gradients_through_error_conditioned_delta(self):
        """Verify gradients flow to scale a and bias b in ErrorConditionedStepSize."""
        ctrl = ErrorConditionedStepSize(eta_max=1.0, scale=0.5, bias=0.2).to(
            dtype=torch.float64
        )
        rule = AdaptiveDeltaRule(controller=ctrl)

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        res = rule.step(m, k, v)
        loss = (res.new_memory**2).sum()
        loss.backward()

        assert ctrl.scale.grad is not None and torch.isfinite(ctrl.scale.grad).all()
        assert ctrl.bias.grad is not None and torch.isfinite(ctrl.bias.grad).all()
        assert torch.norm(ctrl.scale.grad) > 0.0
        assert torch.norm(ctrl.bias.grad) > 0.0

    def test_gradients_through_state_conditioned_delta(self):
        """Verify gradients flow to scale a and bias b in StateConditionedStepSize."""
        ctrl = StateConditionedStepSize(eta_max=1.0, scale=0.5, bias=0.2).to(
            dtype=torch.float64
        )
        rule = AdaptiveDeltaRule(controller=ctrl)

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        res = rule.step(m, k, v)
        loss = (res.new_memory**2).sum()
        loss.backward()

        assert ctrl.scale.grad is not None and torch.isfinite(ctrl.scale.grad).all()
        assert ctrl.bias.grad is not None and torch.isfinite(ctrl.bias.grad).all()
        assert torch.norm(ctrl.scale.grad) > 0.0
        assert torch.norm(ctrl.bias.grad) > 0.0

    def test_input_conditioned_gradcheck(self):
        """Run torch.autograd.gradcheck on InputConditionedStepSize functional step."""
        ctrl = InputConditionedStepSize(dim=2, eta_max=0.5).to(dtype=torch.float64)
        rule = AdaptiveDeltaRule(controller=ctrl)

        def step_fn(m, k, v):
            return rule.step(m, k, v).new_memory

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(step_fn, (m, k, v), eps=1e-6, atol=1e-4)

    def test_error_conditioned_gradcheck(self):
        """Run torch.autograd.gradcheck on ErrorConditionedStepSize functional step."""
        ctrl = ErrorConditionedStepSize(eta_max=0.5, scale=0.3, bias=0.1).to(
            dtype=torch.float64
        )
        rule = AdaptiveDeltaRule(controller=ctrl)

        def step_fn(m, k, v):
            return rule.step(m, k, v).new_memory

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(step_fn, (m, k, v), eps=1e-6, atol=1e-4)

    def test_adaptive_scan_backward_graph(self):
        """Verify end-to-end backpropagation through unrolled adaptive scan."""
        b, t, k_dim, v_dim = 2, 3, 2, 2
        keys = torch.randn(b, t, k_dim, dtype=torch.float64, requires_grad=True)
        targets = torch.randn(b, t, v_dim, dtype=torch.float64, requires_grad=True)
        init_m = torch.zeros(b, v_dim, k_dim, dtype=torch.float64, requires_grad=True)

        ctrl = ErrorConditionedStepSize(eta_max=0.5, scale=0.2, bias=0.1).to(
            dtype=torch.float64
        )

        res = adaptive_scan(keys, targets, initial_memory=init_m, controller=ctrl)

        loss = 0.5 * torch.sum((targets - res.predictions) ** 2)
        loss.backward()

        assert init_m.grad is not None and torch.isfinite(init_m.grad).all()
        assert keys.grad is not None and torch.isfinite(keys.grad).all()
        assert targets.grad is not None and torch.isfinite(targets.grad).all()
        assert ctrl.scale.grad is not None and torch.isfinite(ctrl.scale.grad).all()
        assert ctrl.bias.grad is not None and torch.isfinite(ctrl.bias.grad).all()
