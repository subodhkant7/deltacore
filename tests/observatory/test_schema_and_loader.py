# ==============================================================================
# DeltaCore: tests/observatory/test_schema_and_loader.py
# Unit tests for StateTrajectory schema, serialization, and loader validation.
# ==============================================================================

import json
from pathlib import Path

import pytest
import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.observatory.loader import (
    convert_baseline_trajectory,
    load_artifact,
    load_run_result,
    load_trajectory,
)
from deltacore.observatory.schema import StateTrajectory, TrajectoryStep


def test_trajectory_step_missing_values_preserved():
    """Verify that None values are strictly preserved and not coerced to 0.0 or NaN."""
    step = TrajectoryStep(
        step=0, error_norm=None, step_size=None, dynamics_memory_norm=None
    )
    d = step.to_dict()

    assert d["error_norm"] is None
    assert d["step_size"] is None
    assert d["dynamics_memory_norm"] is None

    reconstructed = TrajectoryStep.from_dict(d)
    assert reconstructed.error_norm is None
    assert reconstructed.step_size is None
    assert reconstructed.dynamics_memory_norm is None


def test_state_trajectory_serialization_roundtrip(tmp_path: Path):
    """Verify JSON serialization and deserialization preserves all fields and types."""
    steps = [
        TrajectoryStep(
            step=0,
            observed=True,
            error_norm=1.5,
            step_size=0.1,
            memory_norm=2.0,
            finite_state=True,
        ),
        TrajectoryStep(
            step=1,
            observed=True,
            error_norm=0.8,
            step_size=0.08,
            memory_norm=2.2,
            finite_state=True,
        ),
        TrajectoryStep(
            step=2,
            observed=True,
            error_norm=None,
            step_size=None,
            memory_norm=None,
            finite_state=False,
        ),
        TrajectoryStep(
            step=3,
            observed=False,
            error_norm=None,
            step_size=None,
            memory_norm=None,
            finite_state=None,
        ),
    ]
    traj = StateTrajectory(
        model="test_model",
        task="test_task",
        seed=42,
        steps=steps,
        all_states_finite=False,
        first_nonfinite_step=2,
        terminal_state_finite=False,
    )

    save_path = tmp_path / "traj.json"
    traj.save(save_path)
    assert save_path.is_file()

    loaded = StateTrajectory.load(save_path)
    assert loaded.model == "test_model"
    assert loaded.task == "test_task"
    assert loaded.seed == 42
    assert len(loaded) == 4
    assert not loaded.all_states_finite
    assert loaded.first_nonfinite_step == 2
    assert loaded.terminal_state_finite is False

    assert loaded.steps[0].error_norm == 1.5
    assert loaded.steps[0].observed is True
    assert loaded.steps[0].finite_state is True

    assert loaded.steps[2].observed is True
    assert loaded.steps[2].error_norm is None
    assert loaded.steps[2].finite_state is False

    assert loaded.steps[3].observed is False
    assert loaded.steps[3].error_norm is None
    assert loaded.steps[3].finite_state is None


def test_convert_baseline_trajectory_frozen():
    """Verify that Frozen model trajectory sets step size and adaptive metrics to None."""
    t_steps = 5
    baseline_res = BaselineTrajectoryResult(
        predictions=torch.zeros(t_steps, 4),
        errors=torch.ones(t_steps, 4),
        final_memory=torch.zeros(4, 4),
        memory_norms=[1.0] * t_steps,
        update_norms=[0.0] * t_steps,
        step_sizes=[0.0] * t_steps,
        stability_margins=[2.0] * t_steps,
        normalized_steps=[0.0] * t_steps,
        clip_count=0,
        all_states_finite=True,
        terminal_state_finite=True,
        first_nonfinite_step=None,
        max_state_norm=1.0,
        max_update_norm=0.0,
        final_error_norm=2.0,
    )

    traj = convert_baseline_trajectory(baseline_res, model="Frozen", task="test_task")
    assert len(traj) == t_steps
    for s in traj.steps:
        # Frozen models do not have an adaptive step size or stability margin
        assert s.step_size is None
        assert s.step_size_change is None
        assert s.normalized_step is None
        assert s.stability_margin is None
        assert s.dynamics_memory_norm is None


def test_convert_baseline_trajectory_failure_visibility():
    """Verify that post-failure steps are marked as unobserved rather than observed non-finite."""
    t_steps = 5
    fail_step = 2
    baseline_res = BaselineTrajectoryResult(
        predictions=torch.zeros(t_steps, 4),
        errors=torch.ones(t_steps, 4),
        final_memory=torch.zeros(4, 4),
        memory_norms=[1.0, 1.5, float("inf"), float("nan"), float("inf")],
        update_norms=[0.1, 0.2, float("inf"), 0.0, 0.0],
        step_sizes=[0.1, 0.1, 0.1, 0.1, 0.1],
        stability_margins=[1.8, 1.8, -10.0, 0.0, 0.0],
        normalized_steps=[0.2, 0.2, 12.0, 0.0, 0.0],
        clip_count=0,
        all_states_finite=False,
        terminal_state_finite=False,
        first_nonfinite_step=fail_step,
        max_state_norm=1.5,
        max_update_norm=0.2,
        final_error_norm=float("nan"),
    )

    traj = convert_baseline_trajectory(baseline_res, model="adaptive", task="test_task")
    assert not traj.all_states_finite
    assert traj.first_nonfinite_step == fail_step

    # Pre-failure steps are valid and observed
    assert traj.steps[0].observed is True
    assert traj.steps[0].finite_state is True
    assert traj.steps[0].memory_norm == 1.0

    # Failure step itself is marked observed=True, finite_state=False
    assert traj.steps[fail_step].observed is True
    assert traj.steps[fail_step].finite_state is False

    # Strictly post-failure steps must have observed=False, finite_state=None, and unavailable metrics (None)
    assert traj.steps[3].observed is False
    assert traj.steps[3].finite_state is None
    assert traj.steps[3].error_norm is None
    assert traj.steps[3].step_size is None
    assert traj.steps[3].memory_norm is None


def test_loader_rejects_malformed_artifacts(tmp_path: Path):
    """Verify that malformed artifacts raise clear, descriptive ValueErrors."""
    # 1. Malformed JSON syntax
    bad_json_file = tmp_path / "bad.json"
    bad_json_file.write_text("{ unquoted_key: 123", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_trajectory(bad_json_file)

    # 2. Missing required fields in StateTrajectory
    missing_fields_file = tmp_path / "missing_fields.json"
    missing_fields_file.write_text(json.dumps({"model": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized artifact schema"):
        load_trajectory(missing_fields_file)

    # 3. Missing required fields in RunResult
    bad_run_res_file = tmp_path / "bad_run.json"
    bad_run_res_file.write_text(json.dumps({"experiment": "task1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_run_result(bad_run_res_file)


def test_load_artifact_polymorphic(tmp_path: Path):
    """Verify load_artifact correctly identifies StateTrajectory and RunResult."""
    # StateTrajectory
    traj = StateTrajectory(model="m", task="t", steps=[TrajectoryStep(step=0)])
    p_traj = tmp_path / "t.json"
    traj.save(p_traj)
    loaded_t = load_artifact(p_traj)
    assert isinstance(loaded_t, StateTrajectory)

    # RunResult
    p_run = tmp_path / "run.json"
    p_run.write_text(
        json.dumps(
            {
                "experiment": "exp",
                "seed": 0,
                "model": "m",
                "status": "SUCCESS",
                "config": {},
                "metrics": {},
                "device": "cpu",
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    loaded_r = load_artifact(p_run)
    assert hasattr(loaded_r, "experiment")
