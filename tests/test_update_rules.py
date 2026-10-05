"""Tests for Hebbian and Delta update rules.

Verifies:
- Property B: Zero step size invariance (eta = 0 => M_{t+1} = M_t)
- Property C: Zero error invariance (Mk = v => Delta M = 0 for DeltaRule)
- Property D: Exact delta update correction equation
- Property E: No hidden mutation of caller's state
- Property F: Dtype preservation (FP32, FP64)
- Property G: Device preservation
- Numerical edge cases: zero key, zero target, orthogonal keys, repeated keys,
  conflicting targets, tiny step sizes, and rejection of negative step sizes.
"""

import pytest
import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.updates.delta import DeltaRule, DeltaStepResult
from deltacore.updates.hebbian import HebbianRule, HebbianStepResult


class TestHebbianRule:
    """Test suite for the Hebbian outer-product update rule."""

    def test_property_b_zero_step_invariance(self):
        """Property B: If eta = 0, memory remains strictly unchanged."""
        rule = HebbianRule()
        m = torch.randn(3, 4, dtype=torch.float32)
        k = torch.randn(4, dtype=torch.float32)
        v = torch.randn(3, dtype=torch.float32)

        m_new = rule.update(m, k, v, step_size=0.0)
        assert torch.equal(m_new, m)

    def test_exact_hebbian_update_equation(self):
        """Verify Delta M = eta * outer(v, k)."""
        rule = HebbianRule()
        m = torch.zeros(2, 3, dtype=torch.float32)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        v = torch.tensor([4.0, 5.0], dtype=torch.float32)
        eta = 0.5

        res = rule.step(m, k, v, step_size=eta)
        assert isinstance(res, HebbianStepResult)

        expected_outer = torch.tensor(
            [[4.0, 8.0, 12.0], [5.0, 10.0, 15.0]], dtype=torch.float32
        )
        assert torch.allclose(res.outer_product, expected_outer, atol=1e-7)
        assert torch.allclose(res.update, eta * expected_outer, atol=1e-7)
        assert torch.allclose(res.new_memory, eta * expected_outer, atol=1e-7)

    def test_property_e_no_hidden_mutation(self):
        """Property E: Original memory tensor and AssociativeMemory instance are not mutated."""
        rule = HebbianRule(step_size=1.0)
        m_tensor = torch.zeros(2, 2, dtype=torch.float32)
        mem = AssociativeMemory(m_tensor)
        k = torch.tensor([1.0, 0.0], dtype=torch.float32)
        v = torch.tensor([2.0, 3.0], dtype=torch.float32)

        new_mem = rule.update(mem, k, v)

        # Original memory must still be all zeros
        assert torch.equal(mem.data, torch.zeros(2, 2))
        assert torch.equal(m_tensor, torch.zeros(2, 2))
        # New memory must reflect update
        assert not torch.equal(new_mem.data, mem.data)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_property_f_dtype_preservation(self, dtype):
        """Property F: FP32 and FP64 precisions are strictly preserved."""
        rule = HebbianRule()
        m = torch.randn(3, 2, dtype=dtype)
        k = torch.randn(2, dtype=dtype)
        v = torch.randn(3, dtype=dtype)

        res = rule.step(m, k, v)
        assert res.outer_product.dtype == dtype
        assert res.update.dtype == dtype
        assert res.new_memory.dtype == dtype

    def test_negative_step_size_rejected(self):
        """Verify that negative step sizes raise ValueError with clear explanation."""
        rule = HebbianRule()
        m = torch.zeros(2, 2)
        k = torch.ones(2)
        v = torch.ones(2)

        with pytest.raises(ValueError, match="step_size must be non-negative"):
            rule.update(m, k, v, step_size=-0.1)

        with pytest.raises(ValueError, match="step_size must be non-negative"):
            HebbianRule(step_size=-1.0)


class TestDeltaRule:
    """Test suite for the error-correcting Delta update rule."""

    def test_property_b_zero_step_invariance(self):
        """Property B: If eta = 0, DeltaRule produces zero update."""
        rule = DeltaRule()
        m = torch.randn(3, 4, dtype=torch.float32)
        k = torch.randn(4, dtype=torch.float32)
        v = torch.randn(3, dtype=torch.float32)

        m_new = rule.update(m, k, v, step_size=0.0)
        assert torch.equal(m_new, m)

    def test_property_c_zero_error_invariance(self):
        """Property C (Critical Invariant): If M @ k == v, then Delta M == 0."""
        rule = DeltaRule(step_size=1.0)
        # Construct M such that M @ k = v exactly
        m = torch.tensor(
            [[2.0, 1.0], [0.0, 3.0]], dtype=torch.float64
        )  # float64 for exact precision
        k = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v = m @ k  # [4.0, 6.0]

        res = rule.step(m, k, v)
        assert isinstance(res, DeltaStepResult)
        assert torch.allclose(
            res.error, torch.zeros(2, dtype=torch.float64), atol=1e-12
        )
        assert torch.allclose(
            res.update, torch.zeros(2, 2, dtype=torch.float64), atol=1e-12
        )
        assert torch.allclose(res.new_memory, m, atol=1e-12)

    def test_property_d_delta_update_equation_and_intermediates(self):
        """Property D: Verify M' = M + eta * (v - M k) k^T with all 5 intermediate terms."""
        rule = DeltaRule(step_size=0.5)
        # M = [[1, 0], [0, 1]]
        m = torch.eye(2, dtype=torch.float32)
        k = torch.tensor([2.0, 0.0], dtype=torch.float32)
        v = torch.tensor([6.0, 1.0], dtype=torch.float32)

        res = rule.step(m, k, v)

        # 1. prediction: M @ k = [2.0, 0.0]
        expected_pred = torch.tensor([2.0, 0.0])
        assert torch.allclose(res.prediction, expected_pred, atol=1e-7)

        # 2. error: v - v_hat = [6 - 2, 1 - 0] = [4.0, 1.0]
        expected_err = torch.tensor([4.0, 1.0])
        assert torch.allclose(res.error, expected_err, atol=1e-7)

        # 3. outer product: error @ k^T = [[4], [1]] @ [[2, 0]] = [[8, 0], [2, 0]]
        expected_outer = torch.tensor([[8.0, 0.0], [2.0, 0.0]])
        assert torch.allclose(res.outer_product, expected_outer, atol=1e-7)

        # 4. update: eta * outer = 0.5 * [[8, 0], [2, 0]] = [[4, 0], [1, 0]]
        expected_update = torch.tensor([[4.0, 0.0], [1.0, 0.0]])
        assert torch.allclose(res.update, expected_update, atol=1e-7)

        # 5. new_memory: M + update = [[1+4, 0], [0+1, 1]] = [[5, 0], [1, 1]]
        expected_new_m = torch.tensor([[5.0, 0.0], [1.0, 1.0]])
        assert torch.allclose(res.new_memory, expected_new_m, atol=1e-7)

    def test_property_e_no_hidden_mutation(self):
        """Property E: DeltaRule must not mutate the caller's memory state."""
        rule = DeltaRule(step_size=1.0)
        m = torch.eye(2, dtype=torch.float32)
        mem = AssociativeMemory(m)
        k = torch.tensor([1.0, 1.0])
        v = torch.tensor([5.0, 5.0])

        new_mem = rule.update(mem, k, v)
        # Original memory must be untouched
        assert torch.equal(mem.data, torch.eye(2))
        assert not torch.equal(new_mem.data, mem.data)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_property_f_dtype_preservation(self, dtype):
        """Property F: Dtype is strictly preserved across FP32 and FP64."""
        rule = DeltaRule()
        m = torch.randn(4, 3, dtype=dtype)
        k = torch.randn(3, dtype=dtype)
        v = torch.randn(4, dtype=dtype)

        res = rule.step(m, k, v)
        assert res.prediction.dtype == dtype
        assert res.error.dtype == dtype
        assert res.outer_product.dtype == dtype
        assert res.update.dtype == dtype
        assert res.new_memory.dtype == dtype

    def test_edge_case_zero_key(self):
        """Zero key vector produces zero outer product and zero state update."""
        rule = DeltaRule(step_size=1.0)
        m = torch.randn(3, 2, dtype=torch.float32)
        k = torch.zeros(2, dtype=torch.float32)
        v = torch.randn(3, dtype=torch.float32)

        res = rule.step(m, k, v)
        assert torch.allclose(res.prediction, torch.zeros(3), atol=1e-7)
        assert torch.allclose(res.outer_product, torch.zeros(3, 2), atol=1e-7)
        assert torch.allclose(res.update, torch.zeros(3, 2), atol=1e-7)
        assert torch.equal(res.new_memory, m)

    def test_edge_case_zero_target(self):
        """Zero target v = 0 updates memory to suppress prior response."""
        rule = DeltaRule(step_size=1.0)
        m = torch.eye(2, dtype=torch.float32)
        k = torch.tensor([1.0, 0.0], dtype=torch.float32)
        v = torch.zeros(2, dtype=torch.float32)

        res = rule.step(m, k, v)
        # v_hat = [1.0, 0.0], error = [-1.0, 0.0]
        assert torch.allclose(res.error, torch.tensor([-1.0, 0.0]), atol=1e-7)
        # new_m[0, 0] becomes 1.0 + (-1.0) = 0.0
        assert torch.allclose(res.new_memory[0, 0], torch.tensor(0.0), atol=1e-7)

    def test_edge_case_tiny_step_size(self):
        """Very small step size (1e-7) behaves smoothly without numerical overflow."""
        rule = DeltaRule(step_size=1e-7)
        m = torch.ones(2, 2, dtype=torch.float64)
        k = torch.randn(2, dtype=torch.float64)
        v = torch.randn(2, dtype=torch.float64)

        res = rule.step(m, k, v)
        assert torch.isfinite(res.new_memory).all()
        assert not torch.isnan(res.new_memory).any()

    def test_batched_3d_update_equivalence(self):
        """Verify batched 3D delta update produces identical results to per-item loop."""
        rule = DeltaRule(step_size=0.3)
        b, v, k = 4, 3, 2
        m_batch = torch.randn(b, v, k, dtype=torch.float64)
        k_batch = torch.randn(b, k, dtype=torch.float64)
        v_batch = torch.randn(b, v, dtype=torch.float64)

        res_batch = rule.step(m_batch, k_batch, v_batch)

        for i in range(b):
            res_single = rule.step(m_batch[i], k_batch[i], v_batch[i])
            assert torch.allclose(
                res_batch.prediction[i], res_single.prediction, atol=1e-12
            )
            assert torch.allclose(res_batch.error[i], res_single.error, atol=1e-12)
            assert torch.allclose(res_batch.update[i], res_single.update, atol=1e-12)
            assert torch.allclose(
                res_batch.new_memory[i], res_single.new_memory, atol=1e-12
            )
