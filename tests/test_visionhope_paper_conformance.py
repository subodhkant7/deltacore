# ==============================================================================
# DeltaCore: tests/test_visionhope_paper_conformance.py
# Independent equation-level tests and deterministic regression fixtures
# auditing exact mathematical correspondences vs engineering differences.
# ==============================================================================

from __future__ import annotations

import math

import torch

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem

# ------------------------------------------------------------------------------
# 1. Independent Mathematical Reference Functions (VisionHOPE Paper Eq. 19-21)
# ------------------------------------------------------------------------------


def paper_soft_injection_cap(
    raw_eta: float,
    alpha: float,
    delta: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    r"""Independently coded reference for VisionHOPE Eq. (19):
    r_t = sqrt(||delta_t||^2 + eps^2) + eps
    eta_t^{inj} = (1 - alpha_t) / r_t
    bar_eta_t = eta_t^{inj} * (1 - exp(-raw_eta / eta_t^{inj}))
    """
    delta_sq = float(torch.sum(delta * delta).item())
    r_t = math.sqrt(delta_sq + eps * eps) + eps
    eta_inj = (1.0 - alpha) / r_t
    bar_eta = eta_inj * (1.0 - math.exp(-raw_eta / eta_inj))
    return r_t, eta_inj, bar_eta


def paper_spectral_clamp(
    bar_eta: float,
    alpha: float,
    k: torch.Tensor,
) -> tuple[float, float]:
    r"""Independently coded reference for VisionHOPE Eq. (20):
    eta_t^{spec} = 2 * alpha_t / ||k_t||^2
    tilde_eta_t = min(bar_eta_t, eta_t^{spec})
    """
    k_sq = float(torch.sum(k * k).item())
    eta_spec = 2.0 * alpha / k_sq if k_sq > 0.0 else float("inf")
    tilde_eta = min(bar_eta, eta_spec)
    return eta_spec, tilde_eta


def paper_sr_dgd_transition(
    m_prev: torch.Tensor,
    alpha: float,
    tilde_eta: float,
    k: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Independently coded reference for VisionHOPE Eq. (21):
    A_t = alpha * I_d - tilde_eta * k * k^T
    B_t = -tilde_eta * delta * k^T
    T_t = A_t + B_t
    M_t = M_{t-1} * T_t
    """
    d = k.shape[0]
    eye_d = torch.eye(d, dtype=torch.float64)
    a_t = alpha * eye_d - tilde_eta * torch.outer(k, k)
    b_t = -tilde_eta * torch.outer(delta, k)
    t_t = a_t + b_t
    m_t = torch.matmul(m_prev, t_t)
    return a_t, b_t, t_t, m_t


# ------------------------------------------------------------------------------
# 2. Paper-Specific Regression Fixtures (Item 15)
# ------------------------------------------------------------------------------


def test_fixture_stability_control_step():
    r"""This fixture tests the implementation against independently derived instances of Proposition 1 operator bounds in FP64:
    ||A_t||_2 <= alpha_t
    ||B_t||_2 < 1 - alpha_t
    ||T_t||_2 < 1
    """
    dtype = torch.float64
    d = 4
    alpha = 0.85
    raw_eta = 0.50
    k = torch.tensor([0.4, -0.3, 0.2, 0.1], dtype=dtype)
    # Ensure ||k|| <= 1 as specified by paper key normalization
    assert torch.linalg.norm(k) <= 1.0

    v = torch.tensor([0.1, 0.2, -0.1, 0.3], dtype=dtype)
    delta = k - v

    r_t, eta_inj, bar_eta = paper_soft_injection_cap(raw_eta, alpha, delta)
    eta_spec, tilde_eta = paper_spectral_clamp(bar_eta, alpha, k)

    # 1. Injection cap satisfies bar_eta < eta_inj
    assert bar_eta < eta_inj
    assert 0.0 < tilde_eta <= eta_spec

    # 2. Evaluate operator matrices
    a_t, b_t, t_t, _ = paper_sr_dgd_transition(
        torch.eye(d, dtype=dtype), alpha, tilde_eta, k, delta
    )

    # 3. Proposition 1 operator bounds
    norm_a = float(torch.linalg.matrix_norm(a_t, ord=2).item())
    norm_b = float(torch.linalg.matrix_norm(b_t, ord=2).item())
    norm_t = float(torch.linalg.matrix_norm(t_t, ord=2).item())

    # Bound 1: ||A_t||_2 <= alpha_t
    assert norm_a <= alpha + 1e-12

    # Bound 2: ||B_t||_2 < 1 - alpha_t
    assert norm_b < (1.0 - alpha) + 1e-12

    # Combined: ||T_t||_2 < 1.0 (Strict token-wise contraction)
    assert norm_t < 1.0


def test_fixture_content_memory_paper_update():
    r"""Deterministic paper fixture for content memory M^m update via Eq. (13, 17)."""
    dtype = torch.float64
    d = 3
    alpha = 0.90
    tilde_eta = 0.20
    k = torch.tensor([0.5, 0.0, 0.5], dtype=dtype)
    v = torch.tensor([0.2, 0.4, 0.1], dtype=dtype)
    delta = k - v

    m_prev = torch.tensor(
        [[1.0, 0.2, 0.0], [0.1, 0.8, -0.1], [0.0, 0.3, 0.9]], dtype=dtype
    )

    a_t, b_t, t_t, m_t = paper_sr_dgd_transition(m_prev, alpha, tilde_eta, k, delta)

    # Verify token-wise Frobenius non-expansion: ||M_t||_F <= ||M_{t-1}||_F
    f_prev = float(torch.linalg.matrix_norm(m_prev, ord="fro").item())
    f_next = float(torch.linalg.matrix_norm(m_t, ord="fro").item())
    assert f_next <= f_prev + 1e-12

    # Check against explicit manual algebraic calculation:
    # M_t = M_prev * (alpha * I - tilde_eta * k * k^T) - tilde_eta * (M_prev * delta) * k^T
    g_m = torch.matmul(m_prev, delta)
    manual_m = torch.matmul(
        m_prev, alpha * torch.eye(d, dtype=dtype) - tilde_eta * torch.outer(k, k)
    ) - tilde_eta * torch.outer(g_m, k)
    diff = float(torch.max(torch.abs(m_t - manual_m)).item())
    assert diff < 1e-15


def test_fixture_key_and_value_paper_updates():
    r"""Deterministic paper fixture for key memory M^k and value memory M^v updates via Eq. (13, 17)."""
    dtype = torch.float64
    alpha = 0.95
    tilde_eta = 0.10
    k = torch.tensor([0.6, -0.4], dtype=dtype)
    v = torch.tensor([0.3, 0.1], dtype=dtype)
    delta = k - v

    m_k_prev = torch.tensor([[0.8, -0.2], [0.1, 0.9]], dtype=dtype)
    m_v_prev = torch.tensor([[0.5, 0.3], [-0.2, 0.7]], dtype=dtype)

    _, _, _, m_k_t = paper_sr_dgd_transition(m_k_prev, alpha, tilde_eta, k, delta)
    _, _, _, m_v_t = paper_sr_dgd_transition(m_v_prev, alpha, tilde_eta, k, delta)

    assert torch.linalg.matrix_norm(m_k_t, ord="fro") <= torch.linalg.matrix_norm(
        m_k_prev, ord="fro"
    )
    assert torch.linalg.matrix_norm(m_v_t, ord="fro") <= torch.linalg.matrix_norm(
        m_v_prev, ord="fro"
    )


def test_fixture_scalar_memories_paper_updates():
    r"""Deterministic paper fixture for 1 x d row-vector memories m^eta and m^alpha (Eq. 8, 13)."""
    dtype = torch.float64
    alpha = 0.88
    tilde_eta = 0.15
    k = torch.tensor([0.3, 0.4, 0.0], dtype=dtype)
    v = torch.tensor([0.1, 0.2, 0.2], dtype=dtype)
    delta = k - v

    m_eta_prev = torch.tensor([[0.1, -0.3, 0.5]], dtype=dtype)  # [1, d]
    m_alpha_prev = torch.tensor([[0.4, 0.2, -0.1]], dtype=dtype)  # [1, d]

    _, _, _, m_eta_t = paper_sr_dgd_transition(m_eta_prev, alpha, tilde_eta, k, delta)
    _, _, _, m_alpha_t = paper_sr_dgd_transition(
        m_alpha_prev, alpha, tilde_eta, k, delta
    )

    # 1 x d linear maps preserve norm bound: ||m_t||_2 <= ||m_{t-1}||_2
    assert torch.linalg.norm(m_eta_t) <= torch.linalg.norm(m_eta_prev) + 1e-12
    assert torch.linalg.norm(m_alpha_t) <= torch.linalg.norm(m_alpha_prev) + 1e-12


def test_fixture_complete_coupled_paper_transition():
    r"""Deterministic multi-step coupled five-memory transition following VisionHOPE Eq. (9, 10, 12, 13, 19, 20).
    Demonstrates that Corollary 1 holds: memory norms are monotonically non-increasing along sequence.
    """
    dtype = torch.float64
    d = 3
    gamma_eta = 0.025
    alpha_init = 0.90
    bias_alpha = math.log(alpha_init / (1.0 - alpha_init))

    # Initial states
    m_c = torch.eye(d, dtype=dtype) * 0.5
    m_k = torch.eye(d, dtype=dtype) * 0.4
    m_v = torch.eye(d, dtype=dtype) * 0.6
    m_eta = torch.zeros(1, d, dtype=dtype)
    m_ret = torch.zeros(1, d, dtype=dtype)

    tokens = [
        torch.tensor([0.5, 0.2, -0.1], dtype=dtype),
        torch.tensor([-0.3, 0.4, 0.2], dtype=dtype),
        torch.tensor([0.1, -0.2, 0.6], dtype=dtype),
    ]

    for x_t in tokens:
        # Eq. (9): Key, value, LR, retention generation
        k_raw = torch.matmul(m_k, x_t)
        v_t = torch.matmul(m_v, x_t)

        # Stabilized key normalization
        k_t = k_raw / math.sqrt(float(torch.sum(k_raw * k_raw).item()) + 1e-6)

        s_eta = float(torch.matmul(m_eta, x_t).item())
        raw_eta = gamma_eta * math.log(1.0 + math.exp(s_eta))  # softplus

        s_ret = float(torch.matmul(m_ret, x_t).item())
        alpha_t = 1.0 / (1.0 + math.exp(-(s_ret + bias_alpha)))  # sigmoid

        delta_t = k_t - v_t

        # Eq. (19, 20): Stability control
        _, _, bar_eta = paper_soft_injection_cap(raw_eta, alpha_t, delta_t)
        _, tilde_eta = paper_spectral_clamp(bar_eta, alpha_t, k_t)

        # Norms before update
        nc_pre = float(torch.linalg.matrix_norm(m_c, ord="fro").item())
        nk_pre = float(torch.linalg.matrix_norm(m_k, ord="fro").item())
        nv_pre = float(torch.linalg.matrix_norm(m_v, ord="fro").item())
        nlr_pre = float(torch.linalg.norm(m_eta).item())
        nret_pre = float(torch.linalg.norm(m_ret).item())

        # Eq. (13, 21): SR-DGD coupled update
        _, _, _, m_c = paper_sr_dgd_transition(m_c, alpha_t, tilde_eta, k_t, delta_t)
        _, _, _, m_k = paper_sr_dgd_transition(m_k, alpha_t, tilde_eta, k_t, delta_t)
        _, _, _, m_v = paper_sr_dgd_transition(m_v, alpha_t, tilde_eta, k_t, delta_t)
        _, _, _, m_eta = paper_sr_dgd_transition(
            m_eta, alpha_t, tilde_eta, k_t, delta_t
        )
        _, _, _, m_ret = paper_sr_dgd_transition(
            m_ret, alpha_t, tilde_eta, k_t, delta_t
        )

        # Verify Corollary 1 token-wise non-expansion for all 5 states
        assert float(torch.linalg.matrix_norm(m_c, ord="fro").item()) <= nc_pre + 1e-12
        assert float(torch.linalg.matrix_norm(m_k, ord="fro").item()) <= nk_pre + 1e-12
        assert float(torch.linalg.matrix_norm(m_v, ord="fro").item()) <= nv_pre + 1e-12
        assert float(torch.linalg.norm(m_eta).item()) <= nlr_pre + 1e-12
        assert float(torch.linalg.norm(m_ret).item()) <= nret_pre + 1e-12


# ------------------------------------------------------------------------------
# 3. Equation-Level Comparison with DeltaCore (Item 14)
# ------------------------------------------------------------------------------


def test_deltacore_content_update_independent_reference_fp64():
    r"""Verify that DeltaCore's content update M_{t+1} = lambda * M_t + eta * (v_t - M_t * k_t) * k_t^T
    matches an independently evaluated reference formula to machine precision (< 1e-15).
    """
    dtype = torch.float64
    v_dim, k_dim, in_dim = 3, 4, 5
    config = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        apply_stability_control=False,
    )
    system = FiveMemorySystem(config)

    gen = torch.Generator().manual_seed(999)
    c0 = torch.randn(v_dim, k_dim, generator=gen, dtype=dtype)
    k0 = torch.randn(k_dim, in_dim, generator=gen, dtype=dtype)
    v0 = torch.randn(v_dim, in_dim, generator=gen, dtype=dtype)
    lr0 = torch.randn(1, in_dim, generator=gen, dtype=dtype)
    ret0 = torch.randn(1, in_dim, generator=gen, dtype=dtype)

    state = FiveMemoryState(
        content=c0, key=k0, value=v0, learning_rate=lr0, retention=ret0
    )
    x = torch.randn(in_dim, generator=gen, dtype=dtype)

    res = system.step(x, state)

    # Independent evaluation of content memory update
    k_ref = torch.matmul(k0, x)
    v_ref = torch.matmul(v0, x)
    v_pred_ref = torch.matmul(c0, k_ref)
    e_ref = v_ref - v_pred_ref

    eta_ref = float(res.safe_learning_rate.item())
    lambda_ref = float(res.safe_retention.item())

    # Independent mathematical update equation
    c_next_ref = lambda_ref * c0 + eta_ref * torch.outer(e_ref, k_ref)

    max_err = float(torch.max(torch.abs(res.new_state.content - c_next_ref)).item())
    assert max_err < 1e-14, (
        f"Content update mismatch vs independent reference: {max_err}"
    )


def test_audit_demonstrates_stability_engineering_difference():
    r"""Demonstrate empirically why DeltaCore Phase 4 controller is an ENGINEERING DIFFERENCE
    from VisionHOPE's spectral clamp.
    When retention lambda_t < 1, DeltaCore's beta=1.9 clamp allows ||A_t||_2 to exceed 1.0,
    whereas VisionHOPE's spectral clamp (eta_spec = 2*alpha/||k||^2) guarantees ||A_t||_2 <= alpha_t.
    """
    dtype = torch.float64
    k = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    k_sq = float(torch.sum(k * k).item())  # 1.0
    alpha = 0.70  # retention < 1.0

    # DeltaCore Phase 4 safe step size with beta = 1.9:
    beta = 1.9
    eta_deltacore = beta / k_sq  # 1.9

    # VisionHOPE spectral limit:
    eta_visionhope = 2.0 * alpha / k_sq  # 1.4

    # Evaluate homogeneous operator A_t = alpha * I - eta * k * k^T
    d = 3
    a_deltacore = alpha * torch.eye(d, dtype=dtype) - eta_deltacore * torch.outer(k, k)
    a_visionhope = alpha * torch.eye(d, dtype=dtype) - eta_visionhope * torch.outer(
        k, k
    )

    norm_a_deltacore = float(torch.linalg.matrix_norm(a_deltacore, ord=2).item())
    norm_a_visionhope = float(torch.linalg.matrix_norm(a_visionhope, ord=2).item())

    # In DeltaCore: alpha - eta * k_sq = 0.7 - 1.9 = -1.2 -> ||A_t||_2 = 1.2 > 1.0 (Expansive!)
    assert norm_a_deltacore > 1.0, (
        f"Expected DeltaCore A_t to be expansive, got {norm_a_deltacore}"
    )
    assert abs(norm_a_deltacore - 1.2) < 1e-14, (
        f"Expected ||A_t|| = 1.2, got {norm_a_deltacore}"
    )

    # In VisionHOPE: alpha - eta * k_sq = 0.7 - 1.4 = -0.7 -> ||A_t||_2 = 0.7 <= alpha (Contractive!)
    assert norm_a_visionhope <= alpha + 1e-14, (
        f"Expected VisionHOPE A_t <= {alpha}, got {norm_a_visionhope}"
    )
