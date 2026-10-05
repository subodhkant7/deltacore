# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/distribution_shift.py
# Benchmark 2: Distribution shift and test-time adaptation recovery.
# ==============================================================================

from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.metrics.accuracy import (
    compute_final_error,
    compute_mean_error,
)
from deltacore.benchmarks.metrics.energy import (
    compute_state_growth_ratio,
    compute_update_energy,
)
from deltacore.benchmarks.metrics.recovery import (
    compute_first_passage_recovery,
    compute_sustained_recovery,
)
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.base import BaseTask


class DistributionShiftTask(BaseTask):
    """Benchmark 2: Distribution Shift & Online Recovery.

    Evaluates adaptation speed, stability, and recovery metrics when the
    associative mapping undergoes an abrupt distribution shift at t_s.
    """

    def __init__(self) -> None:
        super().__init__(name="distribution_shift")

    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        gen = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        t_steps = config.sequence_length
        shift_pos = (
            config.shift_position if config.shift_position is not None else t_steps // 2
        )
        shift_type = config.task_params.get(
            "shift_type", "target_inversion"
        )  # 'target_inversion', 'target_permutation', 'mapping_change', 'partial_change'
        num_patterns = config.task_params.get("num_patterns", 8)

        base_keys = torch.randn(
            num_patterns, config.key_dim, generator=gen, dtype=dtype
        )
        base_keys = torch.nn.functional.normalize(base_keys, p=2, dim=-1)

        base_targets = torch.randn(
            num_patterns, config.value_dim, generator=gen, dtype=dtype
        )

        # Build pre-shift and post-shift target maps
        if shift_type == "target_inversion":
            shifted_targets = -base_targets
        elif shift_type == "target_permutation":
            perm = torch.randperm(num_patterns, generator=gen)
            shifted_targets = base_targets[perm]
        elif shift_type == "mapping_change":
            shifted_targets = torch.randn(
                num_patterns, config.value_dim, generator=gen, dtype=dtype
            )
        elif shift_type == "partial_change":
            shifted_targets = base_targets.clone()
            frac = config.task_params.get("partial_fraction", 0.5)
            num_changed = max(1, int(num_patterns * frac))
            changed_indices = torch.randperm(num_patterns, generator=gen)[:num_changed]
            shifted_targets[changed_indices] = torch.randn(
                num_changed, config.value_dim, generator=gen, dtype=dtype
            )
        else:
            raise ValueError(
                f"Unknown shift_type '{shift_type}'. Expected one of: "
                f"['target_inversion', 'target_permutation', 'mapping_change', 'partial_change']."
            )

        # Sample sequence indices
        seq_indices = torch.randint(0, num_patterns, (t_steps,), generator=gen)

        keys = base_keys[seq_indices]
        targets = torch.zeros((t_steps, config.value_dim), dtype=dtype)

        # Pre-shift mapping
        targets[:shift_pos] = base_targets[seq_indices[:shift_pos]]
        # Post-shift mapping
        targets[shift_pos:] = shifted_targets[seq_indices[shift_pos:]]

        metadata = {
            "shift_position": shift_pos,
            "shift_type": shift_type,
            "tau": config.task_params.get("tau", 0.5),
            "window": config.task_params.get("window", 4),
        }
        return keys, targets, metadata

    def compute_metrics(
        self,
        trajectory: BaselineTrajectoryResult,
        metadata: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        shift_pos = metadata["shift_position"]
        tau = metadata["tau"]
        window = metadata["window"]

        # Pre-shift and shock errors
        pre_shift_err = compute_mean_error(
            trajectory.errors,
            start_idx=max(0, shift_pos - 10),
            end_idx=shift_pos,
        )
        shock_err = (
            float(torch.linalg.norm(trajectory.errors[shift_pos]).item())
            if shift_pos < len(trajectory.errors)
            else 0.0
        )
        final_err = compute_final_error(trajectory.errors)

        # Recovery metrics
        fp_rec = compute_first_passage_recovery(
            trajectory.errors, shift_index=shift_pos, tau=tau
        )
        sustained_rec = compute_sustained_recovery(
            trajectory.errors, shift_index=shift_pos, window=window, tau=tau
        )

        growth_ratio = compute_state_growth_ratio(
            trajectory.memory_norms, shift_index=shift_pos
        )
        upd_energy = compute_update_energy(trajectory.update_norms)

        return {
            "pre_shift_error": pre_shift_err,
            "shock_error": shock_err,
            "final_error": final_err,
            "first_passage_recovery": fp_rec,
            "sustained_recovery": sustained_rec,
            "recovery_failure": fp_rec is None,
            "state_growth_ratio": growth_ratio,
            "update_energy": upd_energy,
            "all_states_finite": trajectory.all_states_finite,
            "terminal_state_finite": trajectory.terminal_state_finite,
            "first_nonfinite_step": trajectory.first_nonfinite_step,
        }
