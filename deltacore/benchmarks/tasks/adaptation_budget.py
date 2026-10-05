# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/adaptation_budget.py
# Benchmark 6: Adaptation budget and adaptation-energy trade-offs.
# ==============================================================================

from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.metrics.accuracy import compute_final_error
from deltacore.benchmarks.metrics.energy import (
    compute_step_energy,
    compute_update_energy,
)
from deltacore.benchmarks.metrics.recovery import compute_first_passage_recovery
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.base import BaseTask


class AdaptationBudgetTask(BaseTask):
    """Benchmark 6: Adaptation Budget Trade-Offs.

    Measures the update energy and step-size budget expended to achieve a
    given recovery threshold following an abrupt distribution shift.
    Exposes the fundamental trade-off: faster adaptation vs. larger state modification.
    """

    def __init__(self) -> None:
        super().__init__(name="adaptation_budget")

    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        gen = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        t_steps = config.sequence_length
        shift_pos = (
            config.shift_position if config.shift_position is not None else t_steps // 3
        )
        num_patterns = config.task_params.get("num_patterns", 6)

        base_keys = torch.randn(
            num_patterns, config.key_dim, generator=gen, dtype=dtype
        )
        base_keys = torch.nn.functional.normalize(base_keys, p=2, dim=-1)

        base_targets = torch.randn(
            num_patterns, config.value_dim, generator=gen, dtype=dtype
        )
        shifted_targets = -base_targets  # Invert targets for clean shock

        seq_indices = torch.randint(0, num_patterns, (t_steps,), generator=gen)

        keys = base_keys[seq_indices]
        targets = torch.zeros((t_steps, config.value_dim), dtype=dtype)
        targets[:shift_pos] = base_targets[seq_indices[:shift_pos]]
        targets[shift_pos:] = shifted_targets[seq_indices[shift_pos:]]

        metadata = {
            "shift_position": shift_pos,
            "tau": config.task_params.get("tau", 0.5),
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

        fp_rec = compute_first_passage_recovery(
            trajectory.errors, shift_index=shift_pos, tau=tau
        )
        tot_upd_energy = compute_update_energy(trajectory.update_norms)
        tot_step_energy = compute_step_energy(trajectory.step_sizes)

        # Budget consumed strictly up to recovery passage
        if fp_rec is not None:
            rec_end = shift_pos + fp_rec + 1
            upd_to_rec = float(sum(trajectory.update_norms[shift_pos:rec_end]))
            step_to_rec = float(sum(trajectory.step_sizes[shift_pos:rec_end]))
            shock_err = float(torch.linalg.norm(trajectory.errors[shift_pos]).item())
            rec_err = float(torch.linalg.norm(trajectory.errors[rec_end - 1]).item())
            delta_err = max(0.0, shock_err - rec_err)
            efficiency = float(delta_err / max(upd_to_rec, 1e-6))
        else:
            upd_to_rec = None
            step_to_rec = None
            efficiency = None

        final_err = compute_final_error(trajectory.errors)
        post_updates = trajectory.update_norms[shift_pos:]
        peak_upd = max(post_updates) if post_updates else 0.0

        return {
            "recovery_steps": fp_rec,
            "recovery_failure": fp_rec is None,
            "update_energy_to_recovery": upd_to_rec,
            "step_energy_to_recovery": step_to_rec,
            "total_update_energy": tot_upd_energy,
            "total_step_energy": tot_step_energy,
            "adaptation_efficiency": efficiency,
            "peak_post_shift_update": peak_upd,
            "final_error": final_err,
            "all_states_finite": trajectory.all_states_finite,
            "terminal_state_finite": trajectory.terminal_state_finite,
            "first_nonfinite_step": trajectory.first_nonfinite_step,
        }
