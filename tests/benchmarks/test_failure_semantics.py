# ==============================================================================
# DeltaCore: tests/benchmarks/test_failure_semantics.py
# Unit tests verifying first-class failure semantics and non-finite step tracking.
# ==============================================================================

import torch

from deltacore.benchmarks.baselines.base import (
    BaseBaseline,
    BaselineTrajectoryResult,
)
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.schemas.result import RunStatus
from deltacore.benchmarks.tasks.stationary_recall import StationaryRecallTask


class SyntheticFailingBaseline(BaseBaseline):
    """Synthetic baseline that injects an Inf/NaN at a pre-specified step index."""

    def __init__(
        self, fail_step: int = 5, key_dim: int = 8, value_dim: int = 8
    ) -> None:
        super().__init__(name="FailingBaseline", key_dim=key_dim, value_dim=value_dim)
        self.fail_step = fail_step

    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        pass

    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        t_steps = keys.shape[0]
        preds = []
        errors = []
        mem_norms = []
        all_finite = True
        first_nonfinite = None

        m = torch.zeros((self.value_dim, self.key_dim))
        for t in range(t_steps):
            if t == self.fail_step:
                pred = torch.tensor([float("inf")] * self.value_dim)
            else:
                pred = m @ keys[t]

            err = targets[t] - pred
            preds.append(pred)
            errors.append(err)

            if not (torch.isfinite(pred).all() and torch.isfinite(err).all()):
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = t

            mem_norms.append(float(torch.linalg.norm(m).item()))

        preds_t = torch.stack(preds, dim=0)
        errors_t = torch.stack(errors, dim=0)

        return BaselineTrajectoryResult(
            predictions=preds_t,
            errors=errors_t,
            final_memory=m,
            memory_norms=mem_norms,
            update_norms=[0.0] * t_steps,
            step_sizes=[0.1] * t_steps,
            stability_margins=[1.9] * t_steps,
            normalized_steps=[0.1] * t_steps,
            clip_count=0,
            all_states_finite=all_finite,
            terminal_state_finite=all_finite,
            first_nonfinite_step=first_nonfinite,
            max_state_norm=0.0,
            max_update_norm=0.0,
            final_error_norm=0.0,
        )


def test_failure_is_first_class_run_status() -> None:
    """When a baseline produces a non-finite state, status must be NUMERICAL_FAILURE."""
    task = StationaryRecallTask()
    cfg = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=42,
        sequence_length=10,
        key_dim=8,
        value_dim=8,
    )
    fail_model = SyntheticFailingBaseline(fail_step=3, key_dim=8, value_dim=8)

    result = task.run(fail_model, cfg)
    assert result.status == RunStatus.NUMERICAL_FAILURE
    assert result.error_step == 3
    assert result.error_message is not None
    assert "step 3" in result.error_message


def test_invalid_configuration_status() -> None:
    """If configuration is invalid, task.run returns INVALID_CONFIGURATION without crash."""
    task = StationaryRecallTask()
    # Construct invalid config by bypassing post_init or triggering validation
    cfg = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=42,
        sequence_length=10,
        key_dim=8,
        value_dim=8,
    )
    # Mutate to an invalid sequence_length using object.__setattr__ on frozen dataclass
    object.__setattr__(cfg, "sequence_length", -5)

    fail_model = SyntheticFailingBaseline(fail_step=3, key_dim=8, value_dim=8)
    res = task.run(fail_model, cfg)
    assert res.status == RunStatus.INVALID_CONFIGURATION
    assert res.error_message is not None
