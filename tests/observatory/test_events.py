# ==============================================================================
# DeltaCore: tests/observatory/test_events.py
# Unit tests for non-causal trajectory event extraction.
# ==============================================================================

from deltacore.observatory.events import extract_events
from deltacore.observatory.schema import StateTrajectory, TrajectoryStep


def test_extract_events_shift_and_recovery():
    """Verify detection of shift boundary, first threshold crossing, and sustained recovery."""
    # Construct a synthetic trajectory:
    # Pre-shift: steps 0-4 (low error 0.1)
    # Shift at step 5: shock error 2.0 (threshold at tau=0.5 is 1.0)
    # Post-shift:
    #   step 5: 2.0
    #   step 6: 1.5
    #   step 7: 0.9 (first passage crossing: <= 1.0)
    #   step 8: 0.8
    #   step 9: 0.7
    #   step 10: 0.6 (window of 4 sustained: 7, 8, 9, 10)
    steps = [
        TrajectoryStep(
            step=0,
            error_norm=0.1,
            update_norm=0.05,
            memory_norm=1.0,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=1,
            error_norm=0.1,
            update_norm=0.05,
            memory_norm=1.0,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=2,
            error_norm=0.1,
            update_norm=0.05,
            memory_norm=1.0,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=3,
            error_norm=0.1,
            update_norm=0.05,
            memory_norm=1.0,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=4,
            error_norm=0.1,
            update_norm=0.05,
            memory_norm=1.0,
            stability_margin=1.8,
        ),
        TrajectoryStep(
            step=5,
            error_norm=2.0,
            update_norm=0.80,
            memory_norm=1.2,
            stability_margin=1.5,
        ),
        TrajectoryStep(
            step=6,
            error_norm=1.5,
            update_norm=0.60,
            memory_norm=1.3,
            stability_margin=1.6,
        ),
        TrajectoryStep(
            step=7,
            error_norm=0.9,
            update_norm=0.40,
            memory_norm=1.4,
            stability_margin=1.7,
        ),
        TrajectoryStep(
            step=8,
            error_norm=0.8,
            update_norm=0.20,
            memory_norm=1.4,
            stability_margin=1.7,
        ),
        TrajectoryStep(
            step=9,
            error_norm=0.7,
            update_norm=0.15,
            memory_norm=1.4,
            stability_margin=1.7,
        ),
        TrajectoryStep(
            step=10,
            error_norm=0.6,
            update_norm=0.10,
            memory_norm=1.4,
            stability_margin=1.8,
        ),
    ]

    traj = StateTrajectory(
        model="test_model",
        task="distribution_shift",
        seed=1,
        steps=steps,
        metadata={"shift_position": 5, "tau": 0.5, "window": 4},
    )

    events = extract_events(traj)
    ev_types = [e.event_type for e in events]

    assert "shift_boundary" in ev_types
    shift_ev = next(e for e in events if e.event_type == "shift_boundary")
    assert shift_ev.step == 5
    assert shift_ev.value == 2.0

    assert "first_threshold_crossing" in ev_types
    cross_ev = next(e for e in events if e.event_type == "first_threshold_crossing")
    assert cross_ev.step == 7
    assert cross_ev.value == 0.9
    assert cross_ev.context["latency_from_shift"] == 2

    assert "sustained_recovery" in ev_types
    sust_ev = next(e for e in events if e.event_type == "sustained_recovery")
    assert sust_ev.step == 7  # sustained window starts at step 7

    assert "maximum_update" in ev_types
    max_upd_ev = next(e for e in events if e.event_type == "maximum_update")
    assert max_upd_ev.step == 5
    assert max_upd_ev.value == 0.80

    assert "maximum_state_norm" in ev_types
    max_norm_ev = next(e for e in events if e.event_type == "maximum_state_norm")
    assert max_norm_ev.step in (7, 8, 9, 10)
    assert max_norm_ev.value == 1.4

    assert "minimum_stability_margin" in ev_types
    min_marg_ev = next(e for e in events if e.event_type == "minimum_stability_margin")
    assert min_marg_ev.step == 5
    assert min_marg_ev.value == 1.5


def test_extract_events_nonfinite_and_regime_change():
    """Verify detection of non-finite failure and controller regime change across S_t = 0."""
    steps = [
        TrajectoryStep(
            step=0,
            error_norm=1.0,
            update_norm=0.1,
            memory_norm=1.0,
            stability_margin=1.0,
        ),
        TrajectoryStep(
            step=1,
            error_norm=2.0,
            update_norm=0.5,
            memory_norm=1.5,
            stability_margin=-0.5,
        ),  # crosses into expansive
        TrajectoryStep(
            step=2,
            error_norm=None,
            update_norm=None,
            memory_norm=None,
            stability_margin=None,
            finite_state=False,
        ),
    ]

    traj = StateTrajectory(
        model="unstable_model",
        task="stress_task",
        steps=steps,
        all_states_finite=False,
        first_nonfinite_step=2,
    )

    events = extract_events(traj)
    ev_types = [e.event_type for e in events]

    assert "first_nonfinite_state" in ev_types
    fail_ev = next(e for e in events if e.event_type == "first_nonfinite_state")
    assert fail_ev.step == 2

    assert "controller_regime_change" in ev_types
    regime_ev = next(e for e in events if e.event_type == "controller_regime_change")
    assert regime_ev.step == 1
    assert regime_ev.context["to_regime"] == "expansive"
