# ==============================================================================
# DeltaCore: deltacore/observatory/replay.py
# Deterministic replay and numerical verification of stored benchmark results.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deltacore.benchmarks.baselines.base import BaseBaseline
from deltacore.benchmarks.baselines.factory import get_baseline
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.schemas.result import RunResult
from deltacore.benchmarks.tasks import get_task
from deltacore.observatory.loader import convert_baseline_trajectory
from deltacore.observatory.schema import StateTrajectory


@dataclass(frozen=True)
class MetricComparison:
    """Numerical delta assessment for a single scalar metric during replay."""

    metric_name: str
    stored_value: Any
    regenerated_value: Any
    absolute_diff: float | None
    relative_diff: float | None
    status: (
        str  # 'exact match', 'within numerical tolerance', 'different', 'type mismatch'
    )


@dataclass
class ReplayResult:
    """Complete diagnostic report produced by re-executing a stored benchmark run.

    Distinguishes:
        - exact match: All metrics match bit-for-bit (diff == 0.0).
        - within numerical tolerance: All metrics agree within (rtol, atol).
        - different: At least one metric diverged beyond tolerance.
        - cannot replay: Experiment definition, task, or model cannot be replayed.
    """

    status: str
    task: str
    model: str
    seed: int
    metric_comparisons: list[MetricComparison] = field(default_factory=list)
    max_absolute_diff: float = 0.0
    max_relative_diff: float = 0.0
    regenerated_trajectory: StateTrajectory | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.regenerated_trajectory is not None:
            d["regenerated_trajectory"] = self.regenerated_trajectory.to_dict()
        return d


def replay_run_result(
    run_result: RunResult,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> StateTrajectory:
    """Deterministic trajectory reconstruction from a RunResult.

    Re-runs the exact task and model using the stored seed and configuration.
    """
    replay_out = execute_replay(run_result, rtol=rtol, atol=atol)
    if replay_out.regenerated_trajectory is None:
        raise RuntimeError(
            f"Cannot replay run {run_result.experiment} ({run_result.model}, seed {run_result.seed}): "
            f"{replay_out.error_message}"
        )
    return replay_out.regenerated_trajectory


def execute_replay(
    artifact: RunResult | str | Path,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ReplayResult:
    """Load configuration, recover seed, re-run experiment, and verify numerical agreement.

    Never modifies or overwrites the original artifact on disk.
    """
    if isinstance(artifact, (str, Path)):
        from deltacore.observatory.loader import load_run_result

        run_res = load_run_result(artifact)
    else:
        run_res = artifact

    cfg_dict = run_res.config
    task_name = run_res.experiment
    model_name = run_res.model
    seed = run_res.seed

    try:
        task = get_task(task_name)
    except Exception as err:
        return ReplayResult(
            status="cannot replay",
            task=task_name,
            model=model_name,
            seed=seed,
            error_message=f"Failed to instantiate task '{task_name}': {err}",
        )

    def _normalize_controller_name(name: str) -> str:
        low = name.lower().replace("-", "_")
        if "safe" in low:
            return "safe_self_referential"
        if "self" in low:
            return "self_referential"
        if "adapt" in low:
            return "adaptive"
        if "frozen" in low:
            return "frozen"
        if "fix" in low:
            return "fixed"
        return low

    ctrl_name = cfg_dict.get("controller") or _normalize_controller_name(model_name)

    try:
        cfg = BenchmarkConfig(
            experiment_name=task_name,
            seed=seed,
            controller=ctrl_name,
            key_dim=cfg_dict.get("key_dim", 32),
            value_dim=cfg_dict.get("value_dim", 32),
            sequence_length=cfg_dict.get("sequence_length", 50),
            device=cfg_dict.get("device", "cpu"),
            dtype=cfg_dict.get("dtype", "float32"),
            warmup_count=0,
            shift_position=cfg_dict.get("shift_position"),
            task_params=cfg_dict.get("task_params", {}),
        )
        model: BaseBaseline = get_baseline(
            name=ctrl_name,
            key_dim=cfg.key_dim,
            value_dim=cfg.value_dim,
            dtype=cfg.dtype,  # type: ignore[arg-type]
            device=cfg.device,
        )
    except Exception as err:
        return ReplayResult(
            status="cannot replay",
            task=task_name,
            model=model_name,
            seed=seed,
            error_message=f"Failed to instantiate model '{model_name}': {err}",
        )

    # Execute deterministic rerun
    keys, targets, task_meta = task.generate_data(cfg)
    traj_res = model.run_sequence(keys, targets)
    regen_metrics = task.compute_metrics(traj_res, task_meta, cfg)

    # Build StateTrajectory
    regen_traj = convert_baseline_trajectory(
        traj=traj_res,
        model=model_name,
        task=task_name,
        seed=seed,
        metadata=task_meta,
        metrics=regen_metrics,
    )

    # Compare regenerated scalar metrics against stored metrics
    stored_metrics = run_res.metrics or {}
    comparisons: list[MetricComparison] = []

    all_exact = True
    all_within_tol = True
    max_abs = 0.0
    max_rel = 0.0

    all_keys = set(stored_metrics.keys()).union(regen_metrics.keys())
    for k in sorted(all_keys):
        v_stored = stored_metrics.get(k)
        v_regen = regen_metrics.get(k)

        if v_stored is None and v_regen is None:
            comparisons.append(
                MetricComparison(
                    metric_name=k,
                    stored_value=None,
                    regenerated_value=None,
                    absolute_diff=0.0,
                    relative_diff=0.0,
                    status="exact match",
                )
            )
            continue

        if v_stored is None or v_regen is None:
            all_exact = False
            all_within_tol = False
            comparisons.append(
                MetricComparison(
                    metric_name=k,
                    stored_value=v_stored,
                    regenerated_value=v_regen,
                    absolute_diff=None,
                    relative_diff=None,
                    status="different",
                )
            )
            continue

        # If numerical comparison
        if isinstance(v_stored, (int, float)) and isinstance(v_regen, (int, float)):
            f_stored = float(v_stored)
            f_regen = float(v_regen)

            if math.isnan(f_stored) and math.isnan(f_regen):
                comparisons.append(
                    MetricComparison(
                        metric_name=k,
                        stored_value=f_stored,
                        regenerated_value=f_regen,
                        absolute_diff=0.0,
                        relative_diff=0.0,
                        status="exact match",
                    )
                )
                continue

            abs_diff = abs(f_regen - f_stored)
            denom = max(abs(f_stored), 1e-12)
            rel_diff = abs_diff / denom

            max_abs = max(max_abs, abs_diff)
            max_rel = max(max_rel, rel_diff)

            if abs_diff == 0.0:
                stat = "exact match"
            elif abs_diff <= atol + rtol * abs(f_stored):
                stat = "within numerical tolerance"
                all_exact = False
            else:
                stat = "different"
                all_exact = False
                all_within_tol = False

            comparisons.append(
                MetricComparison(
                    metric_name=k,
                    stored_value=v_stored,
                    regenerated_value=v_regen,
                    absolute_diff=abs_diff,
                    relative_diff=rel_diff,
                    status=stat,
                )
            )
        else:
            # Categorical / string / bool comparison
            if v_stored == v_regen:
                comparisons.append(
                    MetricComparison(
                        metric_name=k,
                        stored_value=v_stored,
                        regenerated_value=v_regen,
                        absolute_diff=0.0,
                        relative_diff=0.0,
                        status="exact match",
                    )
                )
            else:
                all_exact = False
                all_within_tol = False
                comparisons.append(
                    MetricComparison(
                        metric_name=k,
                        stored_value=v_stored,
                        regenerated_value=v_regen,
                        absolute_diff=None,
                        relative_diff=None,
                        status="different",
                    )
                )

    if all_exact:
        overall_status = "exact match"
    elif all_within_tol:
        overall_status = "within numerical tolerance"
    else:
        overall_status = "different"

    return ReplayResult(
        status=overall_status,
        task=task_name,
        model=model_name,
        seed=seed,
        metric_comparisons=comparisons,
        max_absolute_diff=max_abs,
        max_relative_diff=max_rel,
        regenerated_trajectory=regen_traj,
    )
