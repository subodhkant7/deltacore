# ==============================================================================
# DeltaCore: deltacore/observatory/loader.py
# Schema-validating loaders for trajectories, RunResults, and AggregateResults.
# ==============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.schemas.result import AggregateResult, RunResult
from deltacore.observatory.schema import StateTrajectory, TrajectoryStep


def convert_baseline_trajectory(
    traj: BaselineTrajectoryResult,
    model: str,
    task: str,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> StateTrajectory:
    """Convert an internal BaselineTrajectoryResult into a standardized StateTrajectory.

    Enforces strict missing-data and failure visibility semantics:
        1. Non-adaptive models (e.g. 'frozen') have None for step_size, normalized_step, etc.
        2. Models lacking dynamics memory have None for dynamics_memory_norm and dynamics_update_norm.
        3. For steps strictly after first_nonfinite_step, invalid values are set to None.
    """
    model_lower = model.lower()
    is_frozen = "frozen" in model_lower
    has_dynamics = "self_referential" in model_lower or "selfreferential" in model_lower

    meta = metadata or {}
    mets = metrics or {}

    err_norms: list[float] = []
    errors_val = getattr(traj, "errors", None)
    if errors_val is not None:
        if isinstance(errors_val, torch.Tensor):
            for t in range(errors_val.shape[0]):
                val = float(torch.linalg.norm(errors_val[t].float()).item())
                err_norms.append(val)
        elif isinstance(errors_val, (list, tuple)):
            err_norms = [float(e) for e in errors_val]

    t_steps = (
        len(err_norms)
        if err_norms
        else (len(traj.step_sizes) if traj.step_sizes else len(traj.memory_norms))
    )
    steps: list[TrajectoryStep] = []

    dyn_mem_norms = getattr(traj, "dynamics_memory_norms", None)
    dyn_upd_norms = getattr(traj, "dynamics_update_norms", None)
    dyn_margins = getattr(traj, "dynamics_stability_margins", None)

    first_fail = traj.first_nonfinite_step

    has_offset_m = len(traj.memory_norms) == t_steps + 1
    has_offset_c = dyn_mem_norms is not None and len(dyn_mem_norms) == t_steps + 1

    for t in range(t_steps):
        # Step is strictly after numerical failure (unobserved post-failure)
        if first_fail is not None and t > first_fail:
            steps.append(
                TrajectoryStep(
                    step=t,
                    observed=False,
                    error_norm=None,
                    step_size=None,
                    step_size_change=None,
                    memory_norm=None,
                    dynamics_memory_norm=None,
                    update_norm=None,
                    dynamics_update_norm=None,
                    normalized_step=None,
                    stability_margin=None,
                    dynamics_stability_margin=None,
                    finite_state=None,
                    clip_event=False,
                )
            )
            continue

        is_fail_step = first_fail is not None and t == first_fail
        e_norm = err_norms[t] if t < len(err_norms) else None

        if is_frozen:
            s_size = None
            s_change = None
            u_norm = 0.0
            norm_s = None
            margin = None
        else:
            s_size = traj.step_sizes[t] if t < len(traj.step_sizes) else None
            if t == 0 or s_size is None or traj.step_sizes[t - 1] is None:
                s_change = None
            else:
                s_change = s_size - traj.step_sizes[t - 1]
            u_norm = traj.update_norms[t] if t < len(traj.update_norms) else None
            norm_s = (
                traj.normalized_steps[t] if t < len(traj.normalized_steps) else None
            )
            margin = (
                traj.stability_margins[t] if t < len(traj.stability_margins) else None
            )

        if has_offset_m:
            m_norm = traj.memory_norms[t + 1]
        else:
            m_norm = traj.memory_norms[t] if t < len(traj.memory_norms) else None

        if has_dynamics and dyn_mem_norms is not None:
            c_norm = (
                dyn_mem_norms[t + 1]
                if has_offset_c
                else (dyn_mem_norms[t] if t < len(dyn_mem_norms) else None)
            )
        else:
            c_norm = None

        if has_dynamics and dyn_upd_norms is not None and t < len(dyn_upd_norms):
            c_upd = dyn_upd_norms[t]
        else:
            c_upd = None

        if has_dynamics and dyn_margins is not None and t < len(dyn_margins):
            c_marg = dyn_margins[t]
        else:
            c_marg = None

        steps.append(
            TrajectoryStep(
                step=t,
                error_norm=e_norm,
                step_size=s_size,
                step_size_change=s_change,
                memory_norm=m_norm,
                dynamics_memory_norm=c_norm,
                update_norm=u_norm,
                dynamics_update_norm=c_upd,
                normalized_step=norm_s,
                stability_margin=margin,
                dynamics_stability_margin=c_marg,
                finite_state=not is_fail_step,
                clip_event=False,
            )
        )

    return StateTrajectory(
        model=model,
        task=task,
        seed=seed,
        steps=steps,
        all_states_finite=traj.all_states_finite,
        first_nonfinite_step=traj.first_nonfinite_step,
        terminal_state_finite=traj.terminal_state_finite,
        metadata=meta,
        metrics=mets,
    )


def load_trajectory(
    filepath: str | Path, replay_if_missing: bool = True
) -> StateTrajectory:
    """Load a StateTrajectory from disk, validating schema and handling provenance.

    Supported input types:
        1. Standalone StateTrajectory JSON artifact (has 'steps' array).
        2. RunResult JSON artifact with embedded trajectory in extra_metadata['trajectory'].
        3. RunResult JSON artifact with sibling trajectory artifact.
        4. RunResult JSON artifact replayed deterministically if replay_if_missing is True.

    Raises:
        ValueError: If file is malformed, unrecognized, or lacks trajectory data without replay.
        FileNotFoundError: If filepath does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON in artifact {path}: {err}") from err

    if not isinstance(data, dict):
        raise ValueError(
            f"Artifact at {path} must be a JSON object, got {type(data).__name__}"
        )

    # Case 1: Standalone StateTrajectory JSON
    if "steps" in data and "model" in data and "task" in data:
        return StateTrajectory.from_dict(data)

    # Case 2: RunResult JSON artifact
    if "experiment" in data and "seed" in data and "model" in data and "status" in data:
        run_res = RunResult.from_dict(data)

        # Check if trajectory is in extra_metadata
        if "trajectory" in run_res.extra_metadata:
            traj_data = run_res.extra_metadata["trajectory"]
            if isinstance(traj_data, dict) and "steps" in traj_data:
                return StateTrajectory.from_dict(traj_data)

        # Check sibling file: <name>_trajectory.json or trajectory.json
        sibling_candidates = [
            path.parent / f"{path.stem}_trajectory.json",
            path.parent / "trajectory.json",
            path.parent / f"seed-{run_res.seed}_trajectory.json",
        ]
        for sib in sibling_candidates:
            if sib.is_file():
                try:
                    return load_trajectory(sib, replay_if_missing=False)
                except Exception:
                    pass

        # Case 4: Replay to reconstruct trajectory
        if replay_if_missing:
            from deltacore.observatory.replay import replay_run_result

            return replay_run_result(run_res)

        raise ValueError(
            f"RunResult at {path} does not contain embedded trajectory data and replay_if_missing=False."
        )

    raise ValueError(
        f"Unrecognized artifact schema at {path}. Expected StateTrajectory or RunResult."
    )


def load_run_result(filepath: str | Path) -> RunResult:
    """Load and validate a RunResult JSON artifact.

    Raises:
        ValueError: If schema is invalid or required fields are missing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"RunResult file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON in RunResult artifact {path}: {err}") from err

    if not isinstance(data, dict):
        raise ValueError(
            f"RunResult artifact must be a JSON object, got {type(data).__name__}"
        )

    required = ["experiment", "seed", "model", "status", "config", "metrics"]
    for req in required:
        if req not in data:
            raise ValueError(
                f"Malformed RunResult schema in {path}: missing required field '{req}'"
            )

    return RunResult.from_dict(data)


def load_aggregate_result(filepath: str | Path) -> AggregateResult:
    """Load and validate an AggregateResult JSON artifact.

    Raises:
        ValueError: If schema is invalid or required fields are missing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"AggregateResult file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Malformed JSON in AggregateResult artifact {path}: {err}"
        ) from err

    if not isinstance(data, dict):
        raise ValueError(
            f"AggregateResult artifact must be a JSON object, got {type(data).__name__}"
        )

    required = ["experiment", "model", "num_runs", "num_success", "metric_aggregates"]
    for req in required:
        if req not in data:
            raise ValueError(
                f"Malformed AggregateResult schema in {path}: missing required field '{req}'"
            )

    return AggregateResult.from_dict(data)


def load_artifact(
    filepath: str | Path,
) -> StateTrajectory | RunResult | AggregateResult:
    """Inspect and load any valid DeltaCore benchmark or observatory artifact.

    Determines artifact type from structure and returns typed object.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Artifact must be a JSON object, got {type(data).__name__}")

    if "steps" in data and "model" in data and "task" in data:
        return StateTrajectory.from_dict(data)
    if "metric_aggregates" in data and "num_runs" in data:
        return AggregateResult.from_dict(data)
    if "experiment" in data and "seed" in data and "status" in data:
        return RunResult.from_dict(data)

    raise ValueError(f"Unrecognized artifact schema at {path}")
