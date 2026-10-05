# ==============================================================================
# DeltaCore: tests/test_stability_autograd.py
# Autograd compatibility and gradient flow tests for Phase 4 stability controllers.
# ==============================================================================

import torch
from torch.autograd import gradcheck

from deltacore.memory.associative import AssociativeMemory
from deltacore.stability.controllers import (
    SafeDynamicsRateController,
    SafeStepSizeController,
)
from deltacore.updates.delta import DeltaRule
from deltacore.updates.dynamics_memory import DynamicsMemory
from deltacore.updates.self_referential import (
    CoupledState,
    SelfReferentialSystem,
)


class TestStabilityControllerAutograd:
    """Verifies differentiability, gradient flows, and subgradient semantics of stability controllers."""

    def test_safe_step_unclipped_gradient_flow(self) -> None:
        """When raw_step < bound, gradient flows 100% to raw_step and 0% to key."""
        controller = SafeStepSizeController(beta=1.0, eps=1e-8)
        raw_step = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
        key = torch.tensor([1.0, 0.0], requires_grad=True, dtype=torch.float64)

        # bound = 1.0 / (1.0 + 1e-8) ~ 1.0 > 0.1, so unclipped
        res = controller.safe_step(raw_step, key)
        assert not res.clipped.item()

        loss = res.safe_value * 2.0
        loss.backward()

        assert raw_step.grad is not None
        assert torch.isclose(raw_step.grad, torch.tensor(2.0, dtype=torch.float64))
        assert key.grad is not None
        assert torch.allclose(key.grad, torch.zeros(2, dtype=torch.float64))

    def test_safe_step_clipped_gradient_flow(self) -> None:
        """When raw_step > bound, gradient flows to key norm and raw_step grad is 0."""
        beta = 1.0
        controller = SafeStepSizeController(beta=beta, eps=1e-6)
        raw_step = torch.tensor(5.0, requires_grad=True, dtype=torch.float64)
        key = torch.tensor([2.0, 0.0], requires_grad=True, dtype=torch.float64)

        # key_norm_sq = 4.0; bound = 1.0 / (4.0 + 1e-6) ~ 0.25 < 5.0 -> clipped
        res = controller.safe_step(raw_step, key)
        assert res.clipped.item()

        loss = res.safe_value
        loss.backward()

        # d(safe_val)/d(raw_step) = 0
        assert raw_step.grad is not None
        assert torch.isclose(raw_step.grad, torch.tensor(0.0, dtype=torch.float64))

        # safe_val = beta / (k_norm_sq + eps)
        # d(safe_val)/d(key) = -2 * beta * key / (k_norm_sq + eps)^2
        k_norm_sq = torch.sum(key.detach() ** 2)
        denom = (k_norm_sq + 1e-6) ** 2
        expected_key_grad = -2.0 * beta * key.detach() / denom
        assert key.grad is not None
        assert torch.allclose(key.grad, expected_key_grad, atol=1e-7)

    def test_safe_step_gradcheck_unclipped(self) -> None:
        """Numerical gradcheck for safe_step in unclipped regime."""
        controller = SafeStepSizeController(beta=1.0, eps=1e-8)

        def func(raw_s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            return controller.safe_step(raw_s, k).safe_value

        raw_step = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
        key = torch.tensor([1.2, -0.8], dtype=torch.float64, requires_grad=True)

        assert gradcheck(func, (raw_step, key), eps=1e-6, atol=1e-5)

    def test_safe_step_gradcheck_clipped(self) -> None:
        """Numerical gradcheck for safe_step in clipped regime away from kink."""
        controller = SafeStepSizeController(beta=1.0, eps=1e-8)

        def func(raw_s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            return controller.safe_step(raw_s, k).safe_value

        # raw_step is 10.0, bound is ~ 1 / 2.08 = 0.48. Kink is far away.
        raw_step = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
        key = torch.tensor([1.2, -0.8], dtype=torch.float64, requires_grad=True)

        assert gradcheck(func, (raw_step, key), eps=1e-6, atol=1e-5)

    def test_safe_dynamics_rate_gradcheck(self) -> None:
        """Numerical gradcheck for SafeDynamicsRateController in both regimes."""
        controller = SafeDynamicsRateController(beta_c=1.0, eps=1e-8)

        def func_unclipped(raw_r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return controller.safe_rate(raw_r, z).safe_value

        raw_unclipped = torch.tensor([0.02], dtype=torch.float64, requires_grad=True)
        feat_unclipped = torch.tensor(
            [0.5, 0.5], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(
            func_unclipped, (raw_unclipped, feat_unclipped), eps=1e-6, atol=1e-5
        )

        raw_clipped = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)
        feat_clipped = torch.tensor(
            [1.5, -1.0], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(
            func_unclipped, (raw_clipped, feat_clipped), eps=1e-6, atol=1e-5
        )


class TestStabilityInRecurrentUpdatesAutograd:
    """Verifies gradient flow through associative memory and self-referential updates under safety control."""

    def test_gradient_flow_content_update_with_safe_controller(self) -> None:
        """Gradients propagate through delta update when safe controller is applied."""
        controller = SafeStepSizeController(beta=1.0, eps=1e-8)
        delta_rule = DeltaRule()

        M_0 = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        raw_step = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
        key = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)

        # Apply safe controller in unclipped regime (raw_step = 0.05 < bound = 0.5)
        safe_res = controller.safe_step(raw_step, key)
        step_result = delta_rule.step(M_0, key, target, step_size=safe_res.safe_value)

        loss = step_result.new_memory.sum()
        loss.backward()

        assert M_0.grad is not None
        assert key.grad is not None
        assert target.grad is not None
        assert raw_step.grad is not None
        assert torch.norm(raw_step.grad) > 0.0

    def test_gradient_flow_dynamics_update_with_safe_rate(self) -> None:
        """Gradients propagate through dynamics memory update when safe controller is applied."""
        controller = SafeDynamicsRateController(beta_c=1.0, eps=1e-8)
        c_data = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
        dyn_mem = DynamicsMemory(c_data)

        raw_rate = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
        features = torch.tensor(
            [1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        target_c = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        safe_res = controller.safe_rate(raw_rate, features)
        pred_c = dyn_mem.read(features)
        dyn_err = target_c - pred_c
        delta_c = safe_res.safe_value * torch.outer(dyn_err, features)
        new_c = c_data + delta_c

        loss = new_c.sum()
        loss.backward()

        assert c_data.grad is not None
        assert features.grad is not None
        assert target_c.grad is not None
        assert raw_rate.grad is not None

    def test_end_to_end_self_referential_step_with_safety_controllers(self) -> None:
        """Coupled single step with active safety controllers propagates gradients to all inputs."""
        system = SelfReferentialSystem(
            eta_max=2.0,
            rho=1.5,
            content_stability_controller=SafeStepSizeController(beta=1.0),
            dynamics_stability_controller=SafeDynamicsRateController(beta_c=1.0),
        ).to(dtype=torch.float64)

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        c = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        state = CoupledState(
            content_memory=AssociativeMemory(m),
            dynamics_memory=DynamicsMemory(c),
        )

        res = system.step(k, v, state=state)
        loss = (
            (res.new_content_memory.data**2).sum()
            + (res.new_dynamics_memory.data**2).sum()
            + (res.prediction**2).sum()
        )
        loss.backward()

        assert m.grad is not None and torch.isfinite(m.grad).all()
        assert c.grad is not None and torch.isfinite(c.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()

    def test_end_to_end_self_referential_scan_with_safety_controllers(self) -> None:
        """Full recurrent rollout of SelfReferentialSystem with dual safety controllers backpropagates cleanly."""
        t, k_dim, v_dim = 4, 2, 2
        system = SelfReferentialSystem(
            eta_max=2.0,
            rho=1.5,
            content_stability_controller=SafeStepSizeController(beta=1.0),
            dynamics_stability_controller=SafeDynamicsRateController(beta_c=1.0),
        ).to(dtype=torch.float64)

        keys = torch.randn(t, k_dim, dtype=torch.float64, requires_grad=True)
        targets = torch.randn(t, v_dim, dtype=torch.float64, requires_grad=True)

        m_init = torch.zeros(v_dim, k_dim, dtype=torch.float64, requires_grad=True)
        c_init = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)

        init_state = CoupledState(
            content_memory=AssociativeMemory(m_init),
            dynamics_memory=DynamicsMemory(c_init),
        )

        scan_res = system.scan(keys, targets, initial_state=init_state)

        # Loss combining predictions and final dynamics memory state
        loss = 0.5 * torch.sum((targets - scan_res.predictions) ** 2) + 0.1 * torch.sum(
            scan_res.final_state.dynamics_memory.data**2
        )
        loss.backward()

        assert m_init.grad is not None and torch.isfinite(m_init.grad).all()
        assert c_init.grad is not None and torch.isfinite(c_init.grad).all()
        assert keys.grad is not None and torch.isfinite(keys.grad).all()
        assert targets.grad is not None and torch.isfinite(targets.grad).all()
        assert torch.norm(m_init.grad) > 0.0
        assert torch.norm(c_init.grad) > 0.0
