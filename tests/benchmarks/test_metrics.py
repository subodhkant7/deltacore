# ==============================================================================
# DeltaCore: tests/benchmarks/test_metrics.py
# Unit tests for benchmark metrics: error, recovery, energy, gain, and margins.
# ==============================================================================

import torch

from deltacore.benchmarks.metrics.accuracy import (
    compute_adaptation_gain,
    compute_final_error,
    compute_mean_error,
)
from deltacore.benchmarks.metrics.energy import (
    compute_max_normalized_step,
    compute_min_stability_margin,
    compute_state_growth_ratio,
    compute_step_energy,
    compute_update_energy,
)
from deltacore.benchmarks.metrics.recovery import (
    compute_first_passage_recovery,
    compute_sustained_recovery,
)


def test_final_error_metric() -> None:
    """Verify final Euclidean error norm calculation."""
    errors = torch.tensor([[1.0, 0.0], [0.0, 3.0], [3.0, 4.0]])
    final_err = compute_final_error(errors)
    assert abs(final_err - 5.0) < 1e-6


def test_mean_error_metric() -> None:
    """Verify mean Euclidean error norm calculation over slices."""
    errors = torch.tensor([[3.0, 4.0], [0.0, 5.0], [1.0, 0.0]])  # norms: 5, 5, 1
    mean_all = compute_mean_error(errors)
    assert abs(mean_all - (11.0 / 3.0)) < 1e-6

    mean_sub = compute_mean_error(errors, start_idx=0, end_idx=2)
    assert abs(mean_sub - 5.0) < 1e-6


def test_first_passage_recovery() -> None:
    """Verify first-passage recovery returns exact step hitting tau * shock_err."""
    # Shock at t=2 with norm 10.0. Target threshold tau=0.5 -> norm <= 5.0
    errors = torch.tensor(
        [
            [1.0, 0.0],  # t=0
            [1.0, 0.0],  # t=1
            [10.0, 0.0],  # t=2 (shock)
            [8.0, 0.0],  # t=3 (j=1)
            [6.0, 0.0],  # t=4 (j=2)
            [4.0, 0.0],  # t=5 (j=3, norm=4 <= 5 -> recovery at j=3!)
            [2.0, 0.0],  # t=6 (j=4)
        ]
    )
    rec_step = compute_first_passage_recovery(errors, shift_index=2, tau=0.5)
    assert rec_step == 3


def test_first_passage_recovery_failure() -> None:
    """If error never drops below threshold, return None."""
    errors = torch.tensor([[10.0, 0.0], [9.0, 0.0], [8.0, 0.0]])
    rec_step = compute_first_passage_recovery(errors, shift_index=0, tau=0.5)
    assert rec_step is None


def test_sustained_recovery() -> None:
    """Sustained recovery requires W consecutive steps below threshold."""
    # Shock at t=0 with norm 10.0, tau=0.5 -> threshold 5.0, window W=3
    errors = torch.tensor(
        [
            [10.0, 0.0],  # t=0 (shock)
            [4.0, 0.0],  # t=1 (below, w=0)
            [6.0, 0.0],  # t=2 (bounces up! not sustained)
            [4.0, 0.0],  # t=3 (below, w=0)
            [3.0, 0.0],  # t=4 (below, w=1)
            [2.0, 0.0],  # t=5 (below, w=2 -> sustained window of 3 reached!)
        ]
    )
    # For window W=3: starting at j=3 (t=3), steps t=3,4,5 are 4, 3, 2 (all <= 5)
    sustained_step = compute_sustained_recovery(
        errors, shift_index=0, window=3, tau=0.5
    )
    assert sustained_step == 3


def test_adaptation_gain_sign_convention() -> None:
    """Positive gain indicates improvement (lower eval error); negative indicates degradation."""
    # Lower error is better
    gain_improve = compute_adaptation_gain(eval_error=2.0, baseline_error=4.0)
    assert gain_improve == 0.5  # 50% improvement

    gain_degrade = compute_adaptation_gain(eval_error=6.0, baseline_error=4.0)
    assert gain_degrade == -0.5  # 50% degradation


def test_energy_and_margin_metrics() -> None:
    """Verify update energy, step energy, state growth, and stability margins."""
    updates = [0.5, 1.2, 0.8]
    assert abs(compute_update_energy(updates) - 2.5) < 1e-6

    steps = [0.1, 0.2, 0.3]
    assert abs(compute_step_energy(steps) - 0.6) < 1e-6

    margins = [1.5, 0.8, 1.9]
    assert compute_min_stability_margin(margins) == 0.8

    norm_steps = [0.5, 1.2, 0.1]
    assert compute_max_normalized_step(norm_steps) == 1.2

    mem_norms = [2.0, 2.5, 4.0, 3.0]
    growth = compute_state_growth_ratio(mem_norms, shift_index=1)
    assert abs(growth - (4.0 / 2.5)) < 1e-6
