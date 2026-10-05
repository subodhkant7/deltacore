"""DeltaCore Streaming Regimes & Synthetic Distribution Shift Generators.

Constructs reproducible, deterministic multivariate temporal streams (D=8)
with controlled distribution shifts across three core tasks:
    1. Task A: Regime-Switching Prediction (A1 -> A2 -> A3 -> A1)
    2. Task B: Delayed Context Retrieval (delay d in {16, 64, 256})
    3. Task C: Abrupt Distribution Shift (A -> B -> A)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def generate_stable_transition_matrix(
    dim: int = 8,
    spectral_radius: float = 0.85,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Generate a strictly stable transition matrix A with target spectral radius.

    Stability requirement: \rho(A) = \max_i |\lambda_i| < 1.0.
    """
    if spectral_radius >= 1.0 or spectral_radius <= 0.0:
        raise ValueError(
            f"spectral_radius must be in (0, 1) for asymptotic stability, got {spectral_radius}"
        )

    # Random matrix
    M = torch.randn(dim, dim, generator=generator, dtype=torch.float32)

    # Compute eigenvalues and rescale
    eigenvals = torch.linalg.eigvals(M)
    max_ev = float(torch.max(torch.abs(eigenvals)).item())
    if max_ev < 1e-7:
        return torch.eye(dim) * spectral_radius

    A = M * (spectral_radius / max_ev)
    return A


@dataclass(frozen=True)
class StreamData:
    """Container for streaming task sequence."""

    inputs: torch.Tensor  # [T, D]
    targets: torch.Tensor  # [T, D]
    regime_ids: list[int]  # [T] regime identifier per timestep
    change_points: list[int]  # timesteps where regime switches occurred
    metadata: dict[str, Any]  # task-specific metadata


def generate_regime_switching_stream(
    seq_len: int = 512,
    dim: int = 8,
    noise_std: float = 0.05,
    spectral_radius: float = 0.85,
    seed: int = 42,
) -> StreamData:
    r"""Task A: Synthetic dynamical process with deterministic regime changes.

    System dynamics:
        x_{t+1} = A_r x_t + \epsilon_t, \quad r \in {1, 2, 3}

    Regime Schedule:
        Regime 1 -> Regime 2 -> Regime 3 -> Regime 1
    """
    g = torch.Generator().manual_seed(seed)

    # Create 3 distinct stable transition matrices
    A1 = generate_stable_transition_matrix(
        dim, spectral_radius=spectral_radius, generator=g
    )
    A2 = generate_stable_transition_matrix(
        dim, spectral_radius=spectral_radius, generator=g
    )
    A3 = generate_stable_transition_matrix(
        dim, spectral_radius=spectral_radius, generator=g
    )

    matrices = [A1, A2, A3]

    # Quarter change points
    q1 = seq_len // 4
    q2 = seq_len // 2
    q3 = (3 * seq_len) // 4
    change_points = [q1, q2, q3]

    x_list = [torch.randn(dim, generator=g, dtype=torch.float32)]
    regime_ids = []

    for t in range(seq_len):
        if t < q1:
            r = 0
        elif t < q2:
            r = 1
        elif t < q3:
            r = 2
        else:
            r = 0
        regime_ids.append(r)

        A_curr = matrices[r]
        noise = torch.randn(dim, generator=g, dtype=torch.float32) * noise_std
        x_next = A_curr @ x_list[-1] + noise
        x_list.append(x_next)

    # Inputs x_t, Targets x_{t+1}
    inputs = torch.stack(x_list[:-1], dim=0)  # [T, D]
    targets = torch.stack(x_list[1:], dim=0)  # [T, D]

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata={
            "task": "Task A: Regime-Switching Prediction",
            "spectral_radius": spectral_radius,
            "noise_std": noise_std,
            "seed": seed,
        },
    )


def generate_delayed_retrieval_stream(
    delay: int = 64,
    seq_len: int = 512,
    dim: int = 8,
    cue_time: int = 10,
    seed: int = 42,
) -> StreamData:
    r"""Task B: Delayed context retrieval across controlled temporal gap d.

    Structure:
        t < cue_time: Background Gaussian noise
        t = cue_time: Cue vector presented (target signal)
        cue_time < t < cue_time + delay: Irrelevant distracting tokens
        t = cue_time + delay: Query token presented
        t > cue_time + delay: Target retrieval evaluation
    """
    if cue_time + delay + 2 > seq_len:
        raise ValueError(
            f"seq_len ({seq_len}) must be larger than cue_time + delay + 2 ({cue_time + delay + 2})"
        )

    g = torch.Generator().manual_seed(seed)

    # Informative cue vector
    cue = torch.sign(torch.randn(dim, generator=g, dtype=torch.float32))
    cue = cue / torch.norm(cue)

    # Distinct query trigger
    query = torch.ones(dim, dtype=torch.float32) * 2.0

    inputs = torch.randn(seq_len, dim, generator=g, dtype=torch.float32) * 0.1
    targets = torch.zeros(seq_len, dim, dtype=torch.float32)
    regime_ids = [0] * seq_len

    # Present cue
    inputs[cue_time] = cue
    regime_ids[cue_time] = 1  # Cue marker

    # Query step
    query_time = cue_time + delay
    inputs[query_time] = query
    regime_ids[query_time] = 2  # Query marker

    # Target: The model must predict cue at query_time
    targets[query_time] = cue

    change_points = [cue_time, query_time]

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata={
            "task": "Task B: Delayed Context Retrieval",
            "delay": delay,
            "cue_time": cue_time,
            "query_time": query_time,
            "seed": seed,
        },
    )


def generate_abrupt_shift_stream(
    seq_len: int = 512,
    dim: int = 8,
    noise_std: float = 0.05,
    seed: int = 42,
) -> StreamData:
    r"""Task C: Abrupt Distribution Shift (A -> B -> A).

    Phase 1 (t in [0, T/3)):
        Distribution A: Zero mean \mu_A = 0, diagonal covariance \Sigma_A = I, dynamics A_1.
    Phase 2 (t in [T/3, 2T/3)):
        Distribution B: Mean shift \mu_B = 1.5, correlated features (\rho = 0.6), dynamics A_2.
    Phase 3 (t in [2T/3, T)):
        Return to Distribution A: Tests forgetting and recovery speed.
    """
    g = torch.Generator().manual_seed(seed)

    A1 = generate_stable_transition_matrix(dim, spectral_radius=0.80, generator=g)
    A2 = generate_stable_transition_matrix(dim, spectral_radius=0.85, generator=g)

    # Correlated covariance for Regime B
    corr_matrix = torch.eye(dim) * 0.4 + 0.6 * torch.ones(dim, dim)
    L_B = torch.linalg.cholesky(corr_matrix)
    mu_B = torch.ones(dim) * 1.5

    t1 = seq_len // 3
    t2 = (2 * seq_len) // 3
    change_points = [t1, t2]

    x_curr = torch.randn(dim, generator=g, dtype=torch.float32) * 0.1
    x_list = [x_curr]
    regime_ids = []

    for t in range(seq_len):
        if t < t1 or t >= t2:
            r = 0  # Distribution A
            A_curr = A1
            noise = torch.randn(dim, generator=g, dtype=torch.float32) * noise_std
            x_next = A_curr @ x_list[-1] + noise
        else:
            r = 1  # Distribution B
            A_curr = A2
            raw_noise = torch.randn(dim, generator=g, dtype=torch.float32)
            noise = L_B @ raw_noise * noise_std
            x_next = mu_B + A_curr @ (x_list[-1] - mu_B) + noise

        regime_ids.append(r)
        x_list.append(x_next)

    inputs = torch.stack(x_list[:-1], dim=0)
    targets = torch.stack(x_list[1:], dim=0)

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata={
            "task": "Task C: Abrupt Distribution Shift (A -> B -> A)",
            "t1": t1,
            "t2": t2,
            "mean_shift": 1.5,
            "seed": seed,
        },
    )
