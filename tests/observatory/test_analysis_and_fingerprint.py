# ==============================================================================
# DeltaCore: tests/observatory/test_analysis_and_fingerprint.py
# Unit tests for shift response, stability events, and run comparisons.
# ==============================================================================

import pytest

from deltacore.observatory.analysis import (
    analyze_shift_response,
    analyze_stability_events,
    compare_runs,
)
from deltacore.observatory.fingerprint import (
    compare_fingerprints,
    generate_fingerprint,
)
from deltacore.observatory.schema import StateTrajectory, TrajectoryStep


@pytest.fixture
def sample_trajectories():
    """Build a pair of comparative trajectories for testing deltas and invariants."""
    # Fast recovering model (Model A)
    steps_a = [
        TrajectoryStep(
            step=0,
            error_norm=0.1,
            step_size=0.2,
            update_norm=0.1,
            memory_norm=1.0,
            stability_margin=1.5,
        ),
        TrajectoryStep(
            step=1,
            error_norm=2.0,
            step_size=0.4,
            update_norm=0.8,
            memory_norm=1.5,
            stability_margin=1.0,
        ),  # Shock
        TrajectoryStep(
            step=2,
            error_norm=0.8,
            step_size=0.3,
            update_norm=0.4,
            memory_norm=1.6,
            stability_margin=1.2,
        ),  # Recov at step 1
        TrajectoryStep(
            step=3,
            error_norm=0.2,
            step_size=0.1,
            update_norm=0.1,
            memory_norm=1.6,
            stability_margin=1.7,
        ),
    ]
    traj_a = StateTrajectory(
        model="ModelA_Adaptive",
        task="distribution_shift",
        seed=10,
        steps=steps_a,
        metadata={"shift_position": 1, "tau": 0.5, "window": 2},
    )

    # Slow recovering baseline (Model B)
    steps_b = [
        TrajectoryStep(
            step=0,
            error_norm=0.1,
            step_size=0.1,
            update_norm=0.05,
            memory_norm=1.0,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=1,
            error_norm=2.0,
            step_size=0.1,
            update_norm=0.2,
            memory_norm=1.1,
            stability_margin=1.8,
        ),  # Shock
        TrajectoryStep(
            step=2,
            error_norm=1.8,
            step_size=0.1,
            update_norm=0.2,
            memory_norm=1.2,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=3,
            error_norm=1.5,
            step_size=0.1,
            update_norm=0.2,
            memory_norm=1.3,
            stability_margin=1.8,
        ),  # Never reaches 1.0
    ]
    traj_b = StateTrajectory(
        model="ModelB_Fixed",
        task="distribution_shift",
        seed=10,
        steps=steps_b,
        metadata={"shift_position": 1, "tau": 0.5, "window": 2},
    )

    return traj_a, traj_b


def test_analyze_shift_response(sample_trajectories):
    traj_a, traj_b = sample_trajectories

    res_a = analyze_shift_response(traj_a)
    assert res_a.shock_error == 2.0
    assert res_a.first_passage_recovery == 1  # 1 step from shift
    assert res_a.final_error == 0.2
    assert res_a.peak_update_after_shift == 0.8
    assert res_a.update_energy_after_shift == pytest.approx(0.8 + 0.4 + 0.1)

    res_b = analyze_shift_response(traj_b)
    assert res_b.shock_error == 2.0
    assert res_b.first_passage_recovery is None  # Never reached 0.5 * 2.0 = 1.0
    assert res_b.final_error == 1.5


def test_analyze_stability_events():
    steps = [
        TrajectoryStep(
            step=0,
            error_norm=1.0,
            memory_norm=1.0,
            stability_margin=1.0,
            normalized_step=1.0,
            clip_event=True,
        ),
        TrajectoryStep(
            step=1,
            error_norm=2.0,
            memory_norm=2.5,
            stability_margin=-0.5,
            normalized_step=2.5,
            clip_event=False,
        ),
    ]
    traj = StateTrajectory(
        model="test", task="stress", steps=steps, all_states_finite=True
    )

    stab_res = analyze_stability_events(traj)
    assert stab_res.minimum_content_margin == -0.5
    assert stab_res.maximum_normalized_step == 2.5
    assert stab_res.clip_count == 1
    assert stab_res.all_states_finite is True
    assert stab_res.terminal_state_finite is True
    assert stab_res.maximum_memory_norm == 2.5


def test_stability_events_failure_visibility():
    """Verify that intermediate numerical failure makes terminal_state_finite False and unobserved steps are handled."""
    steps = [
        TrajectoryStep(
            step=0,
            observed=True,
            error_norm=1.0,
            memory_norm=1.0,
            stability_margin=1.0,
            finite_state=True,
        ),
        TrajectoryStep(
            step=1,
            observed=True,
            error_norm=None,
            memory_norm=None,
            stability_margin=None,
            finite_state=False,
        ),
        TrajectoryStep(
            step=2,
            observed=False,
            error_norm=None,
            memory_norm=None,
            stability_margin=None,
            finite_state=None,
        ),
    ]
    traj = StateTrajectory(
        model="failed_model",
        task="stress",
        steps=steps,
        all_states_finite=False,
        first_nonfinite_step=1,
        terminal_state_finite=False,
    )

    stab_res = analyze_stability_events(traj)
    assert stab_res.all_states_finite is False
    assert stab_res.first_nonfinite_step == 1
    assert stab_res.terminal_state_finite is False


def test_compare_runs_sign_conventions(sample_trajectories):
    traj_a, traj_b = sample_trajectories

    # Compare Model A (eval) vs Model B (baseline): Delta = A - B
    cmp = compare_runs(eval_trajectory=traj_a, baseline_trajectory=traj_b)

    # Error delta: final_error(A) - final_error(B) = 0.2 - 1.5 = -1.3 (negative means A achieved lower error)
    assert cmp.error_delta == pytest.approx(0.2 - 1.5)

    # Recovery delta: B never recovered (first_passage is None) -> recovery_delta must be None
    assert cmp.recovery_delta is None

    # Energy delta: update_energy(A) > update_energy(B) -> positive
    assert cmp.energy_delta is not None and cmp.energy_delta > 0

    # Failure delta: both survived -> 0
    assert cmp.failure_delta == 0


def test_generate_and_compare_fingerprints(sample_trajectories):
    traj_a, traj_b = sample_trajectories

    fp_a = generate_fingerprint(traj_a)
    fp_b = generate_fingerprint(traj_b)

    assert fp_a.model == "ModelA_Adaptive"
    assert fp_a.recovery_latency == 1
    assert fp_a.mean_eta is not None

    assert fp_b.model == "ModelB_Fixed"
    assert fp_b.recovery_latency is None

    comp = compare_fingerprints([fp_a, fp_b])
    assert comp["num_models"] == 2
    assert not comp["has_custom_weights"]
    for row in comp["comparison_table"]:
        # No automated winner score by default
        assert "weighted_score" not in row
