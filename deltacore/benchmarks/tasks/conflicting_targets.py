# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/conflicting_targets.py
# Benchmark 4: Contradictory associations on identical keys.
# ==============================================================================

from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.base import BaseTask


class ConflictingTargetsTask(BaseTask):
    """Benchmark 4: Conflicting Associations.

    Presents k -> v1 for T1 steps, then switches to k -> v2 with controlled
    conflict magnitude ||v1 - v2||. Measures adaptation speed, catastrophic
    forgetting (old retention), oscillation, and update magnitude.
    """

    def __init__(self) -> None:
        super().__init__(name="conflicting_targets")

    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        gen = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        t_steps = config.sequence_length
        split_step = (
            config.shift_position if config.shift_position is not None else t_steps // 2
        )
        conflict_magnitude = config.task_params.get("conflict_magnitude", 2.0)

        # Single key or small key set
        k = torch.randn(config.key_dim, generator=gen, dtype=dtype)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        v1 = torch.randn(config.value_dim, generator=gen, dtype=dtype)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)

        # Construct orthogonal direction for controlled conflict magnitude
        w = torch.randn(config.value_dim, generator=gen, dtype=dtype)
        w = w - torch.dot(w, v1) * v1
        w = torch.nn.functional.normalize(w, p=2, dim=-1)

        # v2 satisfies ||v2 - v1|| = conflict_magnitude
        v2 = v1 + conflict_magnitude * w

        keys = k.unsqueeze(0).repeat(t_steps, 1)
        targets = torch.zeros((t_steps, config.value_dim), dtype=dtype)
        targets[:split_step] = v1
        targets[split_step:] = v2

        actual_conflict = float(torch.linalg.norm(v2 - v1).item())

        metadata = {
            "split_step": split_step,
            "conflict_magnitude": actual_conflict,
            "k": k,
            "v1": v1,
            "v2": v2,
            "tau": config.task_params.get("tau", 0.2),
        }
        return keys, targets, metadata

    def compute_metrics(
        self,
        trajectory: BaselineTrajectoryResult,
        metadata: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        split_step = metadata["split_step"]
        conflict_mag = metadata["conflict_magnitude"]
        tau = metadata["tau"]
        k = metadata["k"].to(trajectory.final_memory.device)
        v1 = metadata["v1"].to(trajectory.final_memory.device)
        v2 = metadata["v2"].to(trajectory.final_memory.device)

        m_final = trajectory.final_memory
        final_pred = m_final @ k

        old_retention = float(torch.linalg.norm(final_pred - v1).item())
        new_error = float(torch.linalg.norm(final_pred - v2).item())

        # Adaptation speed: steps after split_step until error <= tau * conflict_mag
        adapt_speed = None
        target_thresh = tau * conflict_mag
        for j in range(trajectory.errors.shape[0] - split_step):
            err_norm = float(
                torch.linalg.norm(trajectory.errors[split_step + j]).item()
            )
            if err_norm <= target_thresh:
                adapt_speed = j
                break

        # Oscillation metric: variance of update norms post-split
        post_updates = trajectory.update_norms[split_step:]
        if len(post_updates) > 1:
            mean_upd = sum(post_updates) / len(post_updates)
            oscillation = float(
                sum((u - mean_upd) ** 2 for u in post_updates) / len(post_updates)
            )
        else:
            oscillation = 0.0

        max_post_update = max(post_updates) if post_updates else 0.0

        return {
            "conflict_magnitude": conflict_mag,
            "adaptation_speed": adapt_speed,
            "old_memory_retention": old_retention,
            "new_target_error": new_error,
            "oscillation": oscillation,
            "max_update_norm": max_post_update,
            "final_state_norm": float(torch.linalg.norm(m_final).item()),
            "all_states_finite": trajectory.all_states_finite,
            "terminal_state_finite": trajectory.terminal_state_finite,
            "first_nonfinite_step": trajectory.first_nonfinite_step,
        }
