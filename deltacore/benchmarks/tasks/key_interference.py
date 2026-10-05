# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/key_interference.py
# Benchmark 3: Key interference with controlled cosine similarity.
# ==============================================================================

import math
from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.metrics.accuracy import compute_final_error
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.base import BaseTask

DEFAULT_SIMILARITIES = [0.00, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]


def construct_correlated_key_pair(
    k_dim: int,
    cosine_sim: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Construct two unit keys k_1, k_2 with exact cosine similarity <k_1, k_2> = \rho."""
    rho = max(0.0, min(1.0, float(cosine_sim)))
    u = torch.randn(k_dim, generator=generator, dtype=dtype)
    u = torch.nn.functional.normalize(u, p=2, dim=-1)

    v = torch.randn(k_dim, generator=generator, dtype=dtype)
    # Gram-Schmidt orthogonalize v w.r.t u
    v = v - torch.dot(v, u) * u
    v = torch.nn.functional.normalize(v, p=2, dim=-1)

    k1 = u
    k2 = rho * u + math.sqrt(max(0.0, 1.0 - rho * rho)) * v
    k2 = torch.nn.functional.normalize(k2, p=2, dim=-1)
    return k1, k2


class KeyInterferenceTask(BaseTask):
    r"""Benchmark 3: Key Interference under Controlled Cosine Similarity.

    Constructs pairs of keys with controlled geometric overlap:
    \rho \in [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    and evaluates retrieval error, cross-talk, and state growth.
    """

    def __init__(self) -> None:
        super().__init__(name="key_interference")

    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        gen = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        rho = config.task_params.get("cosine_similarity", 0.5)
        k1, k2 = construct_correlated_key_pair(
            config.key_dim, rho, generator=gen, dtype=dtype
        )

        v1 = torch.randn(config.value_dim, generator=gen, dtype=dtype)
        v2 = torch.randn(config.value_dim, generator=gen, dtype=dtype)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)
        v2 = torch.nn.functional.normalize(v2, p=2, dim=-1)

        t_steps = config.sequence_length
        # Interleave presentations of k1 and k2
        keys = torch.zeros((t_steps, config.key_dim), dtype=dtype)
        targets = torch.zeros((t_steps, config.value_dim), dtype=dtype)

        # Alternating presentation
        for t in range(t_steps):
            if t % 2 == 0:
                keys[t] = k1
                targets[t] = v1
            else:
                keys[t] = k2
                targets[t] = v2

        metadata = {
            "cosine_similarity": rho,
            "k1": k1,
            "k2": k2,
            "v1": v1,
            "v2": v2,
        }
        return keys, targets, metadata

    def compute_metrics(
        self,
        trajectory: BaselineTrajectoryResult,
        metadata: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        k1 = metadata["k1"].to(trajectory.final_memory.device)
        k2 = metadata["k2"].to(trajectory.final_memory.device)
        v1 = metadata["v1"].to(trajectory.final_memory.device)
        v2 = metadata["v2"].to(trajectory.final_memory.device)

        m_final = trajectory.final_memory
        pred1 = m_final @ k1
        pred2 = m_final @ k2

        err1 = float(torch.linalg.norm(v1 - pred1).item())
        err2 = float(torch.linalg.norm(v2 - pred2).item())
        mean_err = 0.5 * (err1 + err2)
        seq_final_err = compute_final_error(trajectory.errors)

        return {
            "cosine_similarity": metadata["cosine_similarity"],
            "retrieval_error_k1": err1,
            "retrieval_error_k2": err2,
            "mean_retrieval_error": mean_err,
            "sequence_final_error": seq_final_err,
            "final_state_norm": float(torch.linalg.norm(m_final).item()),
            "max_state_norm": trajectory.max_state_norm,
            "all_states_finite": trajectory.all_states_finite,
            "terminal_state_finite": trajectory.terminal_state_finite,
            "first_nonfinite_step": trajectory.first_nonfinite_step,
        }
