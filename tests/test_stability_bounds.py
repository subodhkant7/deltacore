r"""Mathematical boundary and non-expansion tests for stability controllers.

Verifies:
1. Immediate-Key Residual Contraction Theorem:
   e_{t+1}^{(k_t)} = (1 - \eta_t \|k_t\|_2^2) e_t
   Non-expansion: |1 - \eta_t \|k_t\|_2^2| <= 1 <=> 0 <= \eta_t \|k_t\|_2^2 <= 2.
2. Boundary cases:
   - \eta = 0: Zero update, residual invariant.
   - 0 < \eta \|k\|^2 < 2: Strict residual contraction.
   - \eta \|k\|^2 = 2: Exact boundary, residual flips sign with identical magnitude.
   - \eta \|k\|^2 > 2: Beyond boundary, residual strictly expands.
3. Dynamics-memory residual bounds:
   d_{t+1}^{(z_t)} = (1 - \rho_t \|z_t\|_2^2) d_t
4. Multiple precisions (FP32, FP64), key norms, and safety coefficients (\beta \in {0.5, 1.0, 1.5, 1.9}).
5. Edge cases: zero key, near-zero key, zero control feature vector.
"""

import pytest
import torch

from deltacore.memory.read import read
from deltacore.stability.controllers import (
    SafeDynamicsRateController,
    SafeStepSizeController,
)
from deltacore.updates.delta import DeltaRule


class TestContentMemoryStabilityBounds:
    """Verifies content memory immediate-key residual bounds and boundary behaviors."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("norm_scale", [0.1, 1.0, 3.5, 50.0])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 1.9])
    def test_immediate_residual_contraction_bound(self, dtype, norm_scale, beta):
        r"""Verify that SafeStepSizeController strictly enforces |1 - \eta \|k\|^2| <= 1.

        For \beta < 2, the contraction factor is strictly bounded by max(|1 - \beta|, 1.0) < 1.0.
        """
        torch.manual_seed(42)
        v_dim, k_dim = 3, 4
        m = torch.randn(v_dim, k_dim, dtype=dtype)
        k = torch.randn(k_dim, dtype=dtype)
        k = (k / torch.linalg.norm(k)) * norm_scale
        v = torch.randn(v_dim, dtype=dtype)

        # Raw step size chosen to be intentionally aggressive (would diverge without control)
        raw_eta = 2.0 / norm_scale  # Large step

        controller = SafeStepSizeController(beta=beta, eps=1e-10)
        res = controller.safe_step(raw_eta, k)

        k_norm_sq = torch.sum(k**2).item()
        safe_eta = res.safe_value.item()

        # Enforced normalized step
        normalized_step = safe_eta * k_norm_sq
        assert normalized_step <= beta + 1e-6, (
            f"Normalized step {normalized_step} exceeds beta {beta}."
        )

        # Compute one-step update on content memory
        pred_t = read(m, k)
        e_t = v - pred_t

        delta_m = safe_eta * torch.outer(e_t, k)
        m_next = m + delta_m

        # Evaluate on the SAME key k
        pred_next = read(m_next, k)
        e_next = v - pred_next

        # Theoretical prediction
        expected_e_next = (1.0 - safe_eta * k_norm_sq) * e_t
        atol = 1e-5 if dtype == torch.float32 else 1e-12
        assert torch.allclose(e_next, expected_e_next, atol=atol)

        # Contraction verification
        e_next_norm = torch.linalg.norm(e_next).item()
        e_t_norm = torch.linalg.norm(e_t).item()
        contraction_factor = abs(1.0 - safe_eta * k_norm_sq)

        assert contraction_factor <= 1.0 + 1e-6
        assert e_next_norm <= (1.0 + 1e-5) * e_t_norm

    def test_boundary_zero_step(self):
        r"""Case 1: \eta = 0. Residual is invariant."""
        m = torch.randn(3, 4, dtype=torch.float64)
        k = torch.randn(4, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)

        e_t = v - read(m, k)
        # Update with eta = 0
        rule = DeltaRule(step_size=0.0)
        step_res = rule.step(m, k, v)
        e_next = v - read(step_res.new_memory, k)

        assert torch.allclose(e_next, e_t, atol=1e-15)

    def test_boundary_strict_contraction(self):
        r"""Case 2: 0 < \eta \|k\|^2 < 2. Residual strictly contracts."""
        m = torch.randn(3, 4, dtype=torch.float64)
        k = torch.randn(4, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)

        k_norm_sq = torch.sum(k**2).item()
        # Choose eta such that eta * ||k||^2 = 1.2 (in (0, 2))
        eta = 1.2 / k_norm_sq

        e_t = v - read(m, k)
        m_next = m + eta * torch.outer(e_t, k)
        e_next = v - read(m_next, k)

        assert torch.linalg.norm(e_next) < torch.linalg.norm(e_t)
        assert torch.allclose(e_next, -0.2 * e_t, atol=1e-12)

    def test_boundary_exact_boundary_sign_flip(self):
        r"""Case 3: Exact boundary \eta \|k\|^2 = 2.0.

        Residual flips sign with identical Euclidean norm: e_{t+1} = -e_t.
        """
        m = torch.randn(3, 4, dtype=torch.float64)
        k = torch.randn(4, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)

        k_norm_sq = torch.sum(k**2).item()
        eta = 2.0 / k_norm_sq

        e_t = v - read(m, k)
        m_next = m + eta * torch.outer(e_t, k)
        e_next = v - read(m_next, k)

        # Residual must exactly invert: e_{t+1} = (1 - 2) * e_t = -e_t
        assert torch.allclose(e_next, -e_t, atol=1e-12)
        assert torch.allclose(
            torch.linalg.norm(e_next), torch.linalg.norm(e_t), atol=1e-12
        )

    def test_boundary_beyond_boundary_expansion(self):
        r"""Case 4: Beyond boundary \eta \|k\|^2 > 2.0.

        Residual strictly expands: \|e_{t+1}\| > \|e_t\|.
        """
        m = torch.randn(3, 4, dtype=torch.float64)
        k = torch.randn(4, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)

        k_norm_sq = torch.sum(k**2).item()
        # Choose eta * ||k||^2 = 2.5 > 2.0
        eta = 2.5 / k_norm_sq

        e_t = v - read(m, k)
        m_next = m + eta * torch.outer(e_t, k)
        e_next = v - read(m_next, k)

        assert torch.linalg.norm(e_next) > torch.linalg.norm(e_t)
        assert torch.allclose(e_next, -1.5 * e_t, atol=1e-12)

    def test_edge_case_zero_key(self):
        """Zero query key k = 0 yields zero update and finite safe step size."""
        controller = SafeStepSizeController(beta=1.0, eps=1e-8)
        k_zero = torch.zeros(4, dtype=torch.float64)
        raw_eta = 0.5

        res = controller.safe_step(raw_eta, k_zero)
        assert torch.isfinite(res.safe_value)
        assert res.safe_value > 0.0
        assert res.normalized_value == 0.0
        assert res.stability_margin == 2.0

    def test_edge_case_near_zero_key(self):
        """Near-zero key k with norm 1e-7 remains bounded by eps without dividing by zero."""
        controller = SafeStepSizeController(beta=1.0, eps=1e-6)
        k_tiny = torch.full((4,), 1e-7, dtype=torch.float64)
        raw_eta = 100.0

        res = controller.safe_step(raw_eta, k_tiny)
        assert torch.isfinite(res.safe_value)
        # Bounded by beta / eps = 1.0 / 1e-6 = 1e6
        assert res.safe_value <= 1e6 + 1e-4

    def test_non_finite_rejection_policy(self):
        """When check_finite=True, FloatingPointError is raised on NaN or Inf."""
        controller = SafeStepSizeController(beta=1.0, check_finite=True)
        k = torch.tensor([1.0, float("nan")])
        with pytest.raises(FloatingPointError):
            controller.safe_step(0.1, k)

        k_inf = torch.tensor([1.0, float("inf")])
        with pytest.raises(FloatingPointError):
            controller.safe_step(0.1, k_inf)


class TestDynamicsMemoryStabilityBounds:
    """Verifies dynamics memory immediate-feature residual bounds and boundary behaviors."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("norm_scale", [0.2, 1.0, 5.0, 20.0])
    @pytest.mark.parametrize("beta_c", [0.5, 1.0, 1.5, 1.9])
    def test_dynamics_immediate_residual_contraction_bound(
        self, dtype, norm_scale, beta_c
    ):
        r"""Verify that SafeDynamicsRateController enforces |1 - \rho \|z\|^2| <= 1."""
        torch.manual_seed(42)
        d_c = 3
        c = torch.randn(1, d_c, dtype=dtype)
        z = torch.randn(d_c, dtype=dtype)
        z = (z / torch.linalg.norm(z)) * norm_scale
        c_target = torch.randn(1, dtype=dtype)

        raw_rho = 3.0 / norm_scale

        controller = SafeDynamicsRateController(beta_c=beta_c, eps=1e-10)
        res = controller.safe_rate(raw_rho, z)

        z_norm_sq = torch.sum(z**2).item()
        safe_rho = res.safe_value.item()

        assert res.normalized_value.item() <= beta_c + 1e-6

        # One step transition
        q_t = c @ z  # [1]
        d_t = c_target - q_t  # [1]

        delta_c = safe_rho * torch.outer(d_t, z)
        c_next = c + delta_c

        # Evaluate on the SAME feature z
        q_next = c_next @ z
        d_next = c_target - q_next

        expected_d_next = (1.0 - safe_rho * z_norm_sq) * d_t
        atol = 1e-5 if dtype == torch.float32 else 1e-12
        assert torch.allclose(d_next, expected_d_next, atol=atol)
        assert abs(d_next.item()) <= (1.0 + 1e-5) * abs(d_t.item())

    def test_dynamics_boundary_zero_rate(self):
        r"""Case 1: \rho = 0. Dynamics residual is invariant."""
        c = torch.randn(1, 3, dtype=torch.float64)
        z = torch.randn(3, dtype=torch.float64)
        c_target = torch.tensor([1.5], dtype=torch.float64)

        d_t = c_target - c @ z
        c_next = c + 0.0 * torch.outer(d_t, z)
        d_next = c_target - c_next @ z

        assert torch.allclose(d_next, d_t, atol=1e-15)

    def test_dynamics_boundary_exact_boundary(self):
        r"""Case 2: Exact boundary \rho \|z\|^2 = 2.0. Dynamics residual sign flips."""
        c = torch.randn(1, 3, dtype=torch.float64)
        z = torch.randn(3, dtype=torch.float64)
        c_target = torch.tensor([0.8], dtype=torch.float64)

        z_norm_sq = torch.sum(z**2).item()
        rho = 2.0 / z_norm_sq

        d_t = c_target - c @ z
        c_next = c + rho * torch.outer(d_t, z)
        d_next = c_target - c_next @ z

        assert torch.allclose(d_next, -d_t, atol=1e-12)
        assert torch.allclose(torch.abs(d_next), torch.abs(d_t), atol=1e-12)

    def test_dynamics_boundary_beyond_boundary_expansion(self):
        r"""Case 3: Beyond boundary \rho \|z\|^2 > 2.0. Residual magnitude expands."""
        c = torch.randn(1, 3, dtype=torch.float64)
        z = torch.randn(3, dtype=torch.float64)
        c_target = torch.tensor([0.8], dtype=torch.float64)

        z_norm_sq = torch.sum(z**2).item()
        rho = 2.8 / z_norm_sq

        d_t = c_target - c @ z
        c_next = c + rho * torch.outer(d_t, z)
        d_next = c_target - c_next @ z

        assert torch.abs(d_next) > torch.abs(d_t)
        assert torch.allclose(d_next, -1.8 * d_t, atol=1e-12)
