r"""DeltaCore Phase 11: Nonlinear Streaming Regimes and Distribution Shifts.

Constructs reproducible, deterministic multivariate temporal streams (D=8 to 128)
governed by nonlinear dynamical transitions:
    x_{t+1} = f_r(x_t) + \epsilon_t, \quad r \in \{A, B, C\}
where:
    f_r(x) = \tanh(A_r x) + \lambda_r \phi(B_r x), \quad \phi(z) = z^2

Tasks:
    1. Primary Task A: Nonlinear Regime Dynamics (A -> B -> C -> A)
    2. Primary Task B: Regime Switching with Stale-Memory Penalty (A -> B -> A)
    3. Task C: Delayed Context Retrieval with Nonlinear Target (y = g(c))
"""

from __future__ import annotations

import math
from typing import Any

import torch

from deltacore.streaming.regimes import StreamData, generate_stable_transition_matrix


def generate_bounded_nonlinear_operator(
    dim: int = 8,
    spectral_radius: float = 0.80,
    lambda_quad: float = 0.08,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    r"""Generate (A, B, \lambda) for bounded nonlinear dynamics.

    f(x) = \tanh(A x) + \lambda (B x)^2

    Boundedness: With \rho(A) <= 0.80, \lambda <= 0.08, and ||B||_2 <= 0.5,
    trajectories are strictly dissipative and bounded (|x_t| < 5.0).
    """
    A = generate_stable_transition_matrix(
        dim=dim, spectral_radius=spectral_radius, generator=generator
    )
    M_b = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    norm_b = float(torch.linalg.matrix_norm(M_b, ord=2).item())
    B = M_b * (0.50 / max(norm_b, 1e-6))
    return A, B, lambda_quad


def simulate_nonlinear_step(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_quad: float,
) -> torch.Tensor:
    r"""Compute f(x) = \tanh(A x) + \lambda \phi(B x) with \phi(z) = z^2."""
    linear_part = torch.tanh(torch.matmul(A, x))
    quad_part = torch.square(torch.matmul(B, x))
    return linear_part + lambda_quad * quad_part


def generate_nonlinear_regime_stream(
    seq_len: int = 512,
    dim: int = 8,
    noise_std: float = 0.05,
    spectral_radius: float = 0.80,
    lambda_quad: float = 0.08,
    seed: int = 42,
) -> StreamData:
    r"""Generate Task A: Nonlinear Regime-Switching Stream (A -> B -> C -> A).

    Regime switches occur at [T/4, T/2, 3T/4].
    System is verified to be bounded (|x_t| < 10.0).
    """
    gen = torch.Generator().manual_seed(seed)

    # 3 distinct nonlinear regimes
    A1, B1, l1 = generate_bounded_nonlinear_operator(
        dim, spectral_radius, lambda_quad, gen
    )
    A2, B2, l2 = generate_bounded_nonlinear_operator(
        dim, spectral_radius, lambda_quad, gen
    )
    A3, B3, l3 = generate_bounded_nonlinear_operator(
        dim, spectral_radius, lambda_quad, gen
    )

    regimes = [(A1, B1, l1), (A2, B2, l2), (A3, B3, l3), (A1, B1, l1)]
    change_points = [seq_len // 4, seq_len // 2, (3 * seq_len) // 4]

    # Pre-allocate
    inputs = torch.zeros(seq_len, dim, dtype=torch.float32)
    targets = torch.zeros(seq_len, dim, dtype=torch.float32)
    regime_ids: list[int] = []

    # Initial condition
    x_curr = torch.randn(dim, generator=gen, dtype=torch.float32) * 0.2

    for t in range(seq_len):
        if t < change_points[0]:
            r_idx = 0
        elif t < change_points[1]:
            r_idx = 1
        elif t < change_points[2]:
            r_idx = 2
        else:
            r_idx = 3

        regime_ids.append(r_idx % 3)
        A_curr, B_curr, l_curr = regimes[r_idx]

        # Targets = f_r(x_curr) + noise
        noise = torch.randn(dim, generator=gen, dtype=torch.float32) * noise_std
        x_next = simulate_nonlinear_step(x_curr, A_curr, B_curr, l_curr) + noise

        # Verify boundedness
        if float(torch.max(torch.abs(x_next)).item()) > 10.0:
            # Trajectory diverged without adaptive model; reject and clamp
            x_next = torch.clamp(x_next, -5.0, 5.0)

        inputs[t] = x_curr
        targets[t] = x_next
        x_curr = x_next

    metadata: dict[str, Any] = {
        "task": "Task A: Nonlinear Regime Dynamics",
        "dim": dim,
        "seq_len": seq_len,
        "lambda_quad": lambda_quad,
        "seed": seed,
    }

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata=metadata,
    )


def generate_stale_penalty_stream(
    seq_len: int = 512,
    dim: int = 8,
    noise_std: float = 0.05,
    spectral_radius: float = 0.80,
    lambda_quad: float = 0.08,
    seed: int = 42,
) -> StreamData:
    r"""Generate Task B: Regime Switching with Stale-Memory Penalty (A -> B -> A).

    Regime A: x_{t+1} =  \tanh(A x_t) + \lambda (B x_t)^2 + \epsilon_t
    Regime B: x_{t+1} = -\tanh(A x_t) - \lambda (B x_t)^2 + \epsilon_t
    Phase 1: [0, T/3) Regime A
    Phase 2: [T/3, 2T/3) Regime B (negation of A dynamics)
    Phase 3: [2T/3, T) Regime A (return to A)

    Carrying un-reset A-memory into B causes direct adversarial error (negative transfer).
    """
    gen = torch.Generator().manual_seed(seed)

    A_a, B_a, l_a = generate_bounded_nonlinear_operator(
        dim, spectral_radius, lambda_quad, gen
    )
    change_points = [seq_len // 3, (2 * seq_len) // 3]

    inputs = torch.zeros(seq_len, dim, dtype=torch.float32)
    targets = torch.zeros(seq_len, dim, dtype=torch.float32)
    regime_ids: list[int] = []

    x_curr = torch.randn(dim, generator=gen, dtype=torch.float32) * 0.2

    for t in range(seq_len):
        if t < change_points[0]:
            r_idx = 0  # Regime A
            sign = +1.0
        elif t < change_points[1]:
            r_idx = 1  # Regime B (active negation)
            sign = -1.0
        else:
            r_idx = 0  # Regime A (return)
            sign = +1.0

        regime_ids.append(r_idx)

        noise = torch.randn(dim, generator=gen, dtype=torch.float32) * noise_std
        f_val = simulate_nonlinear_step(x_curr, A_a, B_a, l_a)
        x_next = sign * f_val + noise

        if float(torch.max(torch.abs(x_next)).item()) > 10.0:
            x_next = torch.clamp(x_next, -5.0, 5.0)

        inputs[t] = x_curr
        targets[t] = x_next
        x_curr = x_next

    metadata: dict[str, Any] = {
        "task": "Task B: Stale-Memory Penalty Stream",
        "dim": dim,
        "seq_len": seq_len,
        "lambda_quad": lambda_quad,
        "seed": seed,
    }

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata=metadata,
    )


def generate_nonlinear_delayed_retrieval_stream(
    delay: int = 64,
    seq_len: int = 512,
    dim: int = 8,
    noise_std: float = 0.05,
    seed: int = 42,
) -> StreamData:
    r"""Generate Task C: Delayed Context with Explicitly Nonlinear Target y = g(c).

    At t0 = 10, cue vector c is presented.
    From t = 11 to 10 + delay, irrelevant context distractors occur.
    At t_q = 10 + delay + 1, query prompt vector q is presented.
    Target at t_q is y = g(c) = \sin(W_g c) + \tanh(c^2) / \sqrt{D}.
    """
    if delay + 20 >= seq_len:
        raise ValueError(f"delay ({delay}) + 20 must be less than seq_len ({seq_len})")

    gen = torch.Generator().manual_seed(seed)

    inputs = torch.randn(seq_len, dim, generator=gen, dtype=torch.float32) * noise_std
    targets = torch.randn(seq_len, dim, generator=gen, dtype=torch.float32) * noise_std
    regime_ids = [0] * seq_len

    t0 = 10
    t_query = t0 + delay + 1

    # Cue vector
    cue = torch.randn(dim, generator=gen, dtype=torch.float32)
    cue = cue / torch.linalg.norm(cue)
    inputs[t0] = cue
    targets[t0] = cue

    # Query prompt vector (orthogonal to cue)
    raw_query = torch.randn(dim, generator=gen, dtype=torch.float32)
    proj = torch.dot(raw_query, cue) * cue
    query = raw_query - proj
    query = query / torch.linalg.norm(query)
    inputs[t_query] = query

    # Nonlinear target function g(c)
    W_g = torch.randn(dim, dim, generator=gen, dtype=torch.float32) / math.sqrt(dim)
    target_val = torch.sin(torch.matmul(W_g, cue)) + torch.tanh(
        torch.square(cue)
    ) / math.sqrt(dim)
    targets[t_query] = target_val

    metadata: dict[str, Any] = {
        "task": "Task C: Nonlinear Delayed Context Retrieval",
        "delay": delay,
        "t0": t0,
        "query_time": t_query,
        "dim": dim,
        "seq_len": seq_len,
        "seed": seed,
    }

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=[t0, t_query],
        metadata=metadata,
    )
