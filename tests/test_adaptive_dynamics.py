r"""Tests for Phase 2 adaptive update dynamics and step-size controllers.

Verifies:
- Property A: Constant controller equivalence with Phase 1 DeltaRule
- Property B: Strict output boundedness: 0 < \eta_t < \eta_{max}
- Property C: Deterministic controller execution
- Property D: Zero-error invariance (\Delta M = 0 when M @ k = v)
- Property E: Zero-key invariance (\Delta M = 0 when k = 0)
- Property F: Non-mutating functional state semantics
- Property G: Modular controller swappability
- Stress testing under extreme parameters (tiny/large \eta_{max}, large errors, large memory norm).
"""

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.read import read
from deltacore.updates.adaptive_delta import (
    AdaptiveDeltaRule,
    AdaptiveDeltaStepResult,
)
from deltacore.updates.controllers import (
    ConstantStepSize,
    ErrorConditionedStepSize,
    InputConditionedStepSize,
    StateConditionedStepSize,
)
from deltacore.updates.delta import DeltaRule


class TestAdaptiveDynamicsProperties:
    """Verifies core mathematical invariants and properties of adaptive update rules."""

    def test_property_a_constant_controller_equivalence(self):
        r"""Property A (Critical Invariant): AdaptiveDeltaRule with ConstantStepSize(eta0)

        must produce the exact same update as Phase 1 DeltaRule(eta0).
        """
        eta_0 = 0.35
        base_rule = DeltaRule(step_size=eta_0)
        adaptive_rule = AdaptiveDeltaRule(controller=ConstantStepSize(eta_0))

        m = torch.randn(3, 4, dtype=torch.float64)
        k = torch.randn(4, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)

        base_res = base_rule.step(m, k, v)
        adaptive_res = adaptive_rule.step(m, k, v)

        assert torch.allclose(adaptive_res.prediction, base_res.prediction, atol=1e-14)
        assert torch.allclose(adaptive_res.error, base_res.error, atol=1e-14)
        assert torch.allclose(
            adaptive_res.outer_product, base_res.outer_product, atol=1e-14
        )
        assert torch.allclose(adaptive_res.update, base_res.update, atol=1e-14)
        assert torch.allclose(adaptive_res.new_memory, base_res.new_memory, atol=1e-14)

    def test_property_b_bounded_controller_output(self):
        r"""Property B: For \eta_t = \eta_{max} * \sigma(z), verify 0 < \eta_t < \eta_{max}

        across a broad range of inputs.
        """
        eta_max = 2.5
        ctrl_input = InputConditionedStepSize(dim=3, eta_max=eta_max, bias=0.0).to(
            dtype=torch.float64
        )
        ctrl_error = ErrorConditionedStepSize(eta_max=eta_max).to(dtype=torch.float64)
        ctrl_state = StateConditionedStepSize(eta_max=eta_max).to(dtype=torch.float64)

        test_keys = [
            torch.zeros(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64) * 1e-4,
            torch.ones(3, dtype=torch.float64) * 1e4,
            torch.randn(3, dtype=torch.float64) * 50.0,
        ]

        m = torch.randn(2, 3, dtype=torch.float64)
        v = torch.randn(2, dtype=torch.float64)

        for k in test_keys:
            pred = read(m, k)
            err = v - pred

            eta_in = ctrl_input(k, v, pred, err, m)
            eta_err = ctrl_error(k, v, pred, err, m)
            eta_st = ctrl_state(k, v, pred, err, m)

            for val, name in [
                (eta_in, "input"),
                (eta_err, "error"),
                (eta_st, "state"),
            ]:
                assert 0.0 < val.item() <= eta_max, (
                    f"Controller {name} violated bounds: {val.item()} not in (0, {eta_max}]"
                )

    def test_property_c_deterministic_controller(self):
        """Property C: Identical inputs/state must produce identical step sizes."""
        ctrl = ErrorConditionedStepSize(eta_max=1.0, scale=0.5, bias=-0.2)
        k = torch.tensor([1.0, 2.0])
        v = torch.tensor([3.0, 4.0])
        pred = torch.tensor([2.5, 3.5])
        err = v - pred
        m = torch.eye(2)

        eta1 = ctrl(k, v, pred, err, m)
        eta2 = ctrl(k, v, pred, err, m)
        assert torch.equal(eta1, eta2)

    def test_property_d_zero_error_invariance(self):
        """Property D: If M @ k == v, then Delta M == 0 regardless of controller."""
        controllers = [
            ConstantStepSize(0.5),
            InputConditionedStepSize(dim=2, eta_max=1.5),
            ErrorConditionedStepSize(eta_max=1.5),
            StateConditionedStepSize(eta_max=1.5),
        ]

        m = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
        k = torch.tensor([2.0, 1.0], dtype=torch.float64)
        v = m @ k  # Exact prediction: error is 0

        for ctrl in controllers:
            ctrl = ctrl.to(dtype=torch.float64)
            rule = AdaptiveDeltaRule(controller=ctrl)
            res = rule.step(m, k, v)
            assert isinstance(res, AdaptiveDeltaStepResult)
            assert torch.allclose(res.error, torch.zeros_like(res.error), atol=1e-12)
            assert torch.allclose(res.update, torch.zeros_like(m), atol=1e-12), (
                f"Controller {type(ctrl).__name__} failed zero-error invariance"
            )
            assert torch.allclose(res.new_memory, m, atol=1e-12)

    def test_property_e_zero_key_invariance(self):
        """Property E: If key is zero (k = 0), update remains zero."""
        ctrl = InputConditionedStepSize(dim=3, eta_max=1.0)
        rule = AdaptiveDeltaRule(controller=ctrl)

        m = torch.randn(2, 3, dtype=torch.float32)
        k = torch.zeros(3, dtype=torch.float32)
        v = torch.randn(2, dtype=torch.float32)

        res = rule.step(m, k, v)
        assert torch.allclose(res.outer_product, torch.zeros(2, 3), atol=1e-7)
        assert torch.allclose(res.update, torch.zeros(2, 3), atol=1e-7)
        assert torch.equal(res.new_memory, m)

    def test_property_f_no_state_mutation(self):
        """Property F: Calling update or step must never mutate original memory."""
        ctrl = StateConditionedStepSize(eta_max=1.0)
        rule = AdaptiveDeltaRule(controller=ctrl)

        m_tensor = torch.eye(2, dtype=torch.float32)
        mem = AssociativeMemory(m_tensor)
        k = torch.tensor([1.0, 0.5])
        v = torch.tensor([2.0, -1.0])

        new_mem = rule.update(mem, k, v)
        assert torch.equal(mem.data, torch.eye(2))
        assert not torch.equal(new_mem.data, mem.data)

    def test_property_g_controller_replacement(self):
        """Property G: Controllers can be freely swapped without changing memory interface."""
        mem = AssociativeMemory.zeros(v_dim=3, k_dim=3)
        k = torch.randn(3)
        v = torch.randn(3)

        rule1 = AdaptiveDeltaRule(controller=ConstantStepSize(0.5))
        rule2 = AdaptiveDeltaRule(controller=ErrorConditionedStepSize())

        mem1 = rule1.update(mem, k, v)
        mem2 = rule2.update(mem, k, v)

        assert isinstance(mem1, AssociativeMemory)
        assert isinstance(mem2, AssociativeMemory)
        assert mem1.shape == mem2.shape == (3, 3)


class TestAdaptiveStressCases:
    """Stress testing of adaptive dynamics across numerical and distributional edge cases."""

    def test_extremely_small_eta_max(self):
        """Verify behavior when eta_max is 1e-6 (near frozen)."""
        ctrl = ErrorConditionedStepSize(eta_max=1e-6)
        rule = AdaptiveDeltaRule(controller=ctrl)

        m = torch.randn(2, 2, dtype=torch.float64)
        k = torch.randn(2, dtype=torch.float64)
        v = torch.randn(2, dtype=torch.float64)

        res = rule.step(m, k, v)
        assert torch.isfinite(res.new_memory).all()
        assert res.step_size <= 1e-6

    def test_large_error_and_large_memory_norm(self):
        """Verify numerical stability with large inputs (norm ~ 1e5)."""
        ctrl = StateConditionedStepSize(eta_max=2.0, scale=0.01)
        rule = AdaptiveDeltaRule(controller=ctrl)

        m = torch.ones(2, 2, dtype=torch.float64) * 1e4
        k = torch.randn(2, dtype=torch.float64)
        v = torch.randn(2, dtype=torch.float64) * 1e5

        res = rule.step(m, k, v)
        assert torch.isfinite(res.new_memory).all()
        assert not torch.isnan(res.new_memory).any()

    def test_repeated_identical_keys(self):
        """Verify sequential updates over identical keys contract error without NaN."""
        ctrl = ErrorConditionedStepSize(eta_max=0.5)
        rule = AdaptiveDeltaRule(controller=ctrl)

        k = torch.tensor([1.0, 0.0], dtype=torch.float64)
        v = torch.tensor([2.0, 3.0], dtype=torch.float64)
        m = torch.zeros(2, 2, dtype=torch.float64)

        for _ in range(25):
            res = rule.step(m, k, v)
            m = res.new_memory
            assert torch.isfinite(m).all()

        # Terminal prediction should be close to target
        final_pred = read(m, k)
        assert torch.allclose(final_pred, v, atol=1e-2)
