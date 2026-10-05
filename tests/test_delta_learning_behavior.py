"""Deterministic behavioral tests for Delta-rule learning and error contraction.

Mathematical Foundation:
Given memory M_t, key k, and target v with ||k||_2 > 0:
    1. Initial prediction: v_hat_t = M_t @ k
    2. Initial error: e_t = v - v_hat_t
    3. Delta update: M_{t+1} = M_t + eta * e_t @ k^T
    4. Immediate re-query on key k:
       v_hat_{t+1} = M_{t+1} @ k = M_t @ k + eta * e_t * (k^T k)
                   = v_hat_t + eta * ||k||_2^2 * e_t
    5. Resulting error:
       e_{t+1} = v - v_hat_{t+1} = (1 - eta * ||k||_2^2) * e_t

Therefore, the residual error scales exactly by the scalar factor:
    gamma = 1 - eta * ||k||_2^2
"""

import torch

from deltacore.memory.read import read
from deltacore.updates.delta import DeltaRule


class TestDeltaLearningBehavior:
    """Verifies mathematically derived error contraction and exact recall properties."""

    def test_exact_one_step_recall_when_eta_equals_reciprocal_norm_sq(self):
        r"""Case 1: When eta = 1 / ||k||_2^2, the error becomes EXACTLY zero in one step.

        Derivation:
            gamma = 1 - (1 / ||k||_2^2) * ||k||_2^2 = 0
            => e_{t+1} = 0 * e_t = 0.
        """
        # Set k = [2.0], so ||k||_2^2 = 4.0
        # Optimal step size: eta = 1 / 4.0 = 0.25
        k = torch.tensor([2.0], dtype=torch.float64)
        v = torch.tensor([7.0], dtype=torch.float64)
        m_0 = torch.tensor([[1.0]], dtype=torch.float64)

        # Initial prediction: 1.0 * 2.0 = 2.0, error: 7.0 - 2.0 = 5.0
        v_hat_0 = read(m_0, k)
        e_0 = v - v_hat_0
        assert torch.allclose(e_0, torch.tensor([5.0], dtype=torch.float64))

        rule = DeltaRule(step_size=0.25)
        m_1 = rule.update(m_0, k, v)

        # Immediate re-query
        v_hat_1 = read(m_1, k)
        e_1 = v - v_hat_1

        # Verify exact recall (e_1 == 0)
        assert torch.allclose(e_1, torch.zeros_like(e_1), atol=1e-12), (
            f"Expected exact zero error, got {e_1}"
        )
        assert torch.allclose(v_hat_1, v, atol=1e-12)

    def test_exact_linear_error_contraction_guarantee(self):
        r"""Case 2: When 0 < eta < 2 / ||k||_2^2, error contracts by exactly |1 - eta ||k||^2|.

        Setup:
            k = [1.0, 1.0]^T => ||k||_2^2 = 1^2 + 1^2 = 2.0
            eta = 0.25 => gamma = 1 - 0.25 * 2.0 = 0.5
            v = [4.0, -2.0]^T, M_0 = 0 (2x2)
            e_0 = [4.0, -2.0]^T
            Analytical expectation: e_1 = 0.5 * e_0 = [2.0, -1.0]^T
        """
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_norm_sq = torch.dot(k, k).item()
        eta = 0.25
        expected_gamma = 1.0 - eta * k_norm_sq
        assert expected_gamma == 0.5, "Contractual test setup error"

        v = torch.tensor([4.0, -2.0], dtype=torch.float64)
        m_0 = torch.zeros(2, 2, dtype=torch.float64)

        v_hat_0 = read(m_0, k)
        e_0 = v - v_hat_0
        e_0_norm = torch.linalg.norm(e_0)

        rule = DeltaRule(step_size=eta)
        m_1 = rule.update(m_0, k, v)

        v_hat_1 = read(m_1, k)
        e_1 = v - v_hat_1
        e_1_norm = torch.linalg.norm(e_1)

        # Verify e_1 = gamma * e_0 exactly
        expected_e_1 = expected_gamma * e_0
        assert torch.allclose(e_1, expected_e_1, atol=1e-12), (
            f"Expected {expected_e_1}, got {e_1}"
        )

        # Verify norm contraction: ||e_1|| = |gamma| * ||e_0||
        assert torch.isclose(e_1_norm, abs(expected_gamma) * e_0_norm, atol=1e-12)
        assert e_1_norm < e_0_norm

    def test_overcorrection_and_sign_flip_behavior(self):
        r"""Case 3: When 1 < eta ||k||^2 < 2, the error inverts sign but still strictly contracts.

        Setup:
            ||k||_2^2 = 2.0, eta = 0.75
            gamma = 1 - 0.75 * 2.0 = -0.5
            Analytical expectation: e_1 = -0.5 * e_0, ||e_1|| = 0.5 ||e_0||
        """
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        eta = 0.75
        expected_gamma = 1.0 - eta * 2.0
        assert expected_gamma == -0.5

        v = torch.tensor([6.0, -4.0], dtype=torch.float64)
        m_0 = torch.zeros(2, 2, dtype=torch.float64)

        e_0 = v - read(m_0, k)
        rule = DeltaRule(step_size=eta)
        m_1 = rule.update(m_0, k, v)

        e_1 = v - read(m_1, k)

        assert torch.allclose(e_1, expected_gamma * e_0, atol=1e-12)
        assert torch.isclose(
            torch.linalg.norm(e_1), 0.5 * torch.linalg.norm(e_0), atol=1e-12
        )

    def test_divergence_boundary_behavior(self):
        r"""Case 4: When eta > 2 / ||k||^2, error expands by |gamma| > 1.

        Setup:
            ||k||_2^2 = 2.0, eta = 1.5
            gamma = 1 - 1.5 * 2.0 = -2.0 (|gamma| = 2.0)
            Analytical expectation: e_1 = -2.0 * e_0, ||e_1|| = 2 * ||e_0||
        """
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        eta = 1.5
        expected_gamma = 1.0 - eta * 2.0
        assert expected_gamma == -2.0

        v = torch.tensor([2.0, 3.0], dtype=torch.float64)
        m_0 = torch.zeros(2, 2, dtype=torch.float64)

        e_0 = v - read(m_0, k)
        rule = DeltaRule(step_size=eta)
        m_1 = rule.update(m_0, k, v)

        e_1 = v - read(m_1, k)

        assert torch.allclose(e_1, expected_gamma * e_0, atol=1e-12)
        assert torch.isclose(
            torch.linalg.norm(e_1), 2.0 * torch.linalg.norm(e_0), atol=1e-12
        )
