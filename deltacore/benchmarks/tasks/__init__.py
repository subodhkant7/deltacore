# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/__init__.py
# Standardized synthetic benchmark tasks.
# ==============================================================================

from deltacore.benchmarks.tasks.adaptation_budget import AdaptationBudgetTask
from deltacore.benchmarks.tasks.base import BaseTask
from deltacore.benchmarks.tasks.conflicting_targets import ConflictingTargetsTask
from deltacore.benchmarks.tasks.distribution_shift import DistributionShiftTask
from deltacore.benchmarks.tasks.key_interference import KeyInterferenceTask
from deltacore.benchmarks.tasks.stability_stress import StabilityStressTask
from deltacore.benchmarks.tasks.stationary_recall import StationaryRecallTask

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "stationary_recall": StationaryRecallTask,
    "distribution_shift": DistributionShiftTask,
    "key_interference": KeyInterferenceTask,
    "conflicting_targets": ConflictingTargetsTask,
    "stability_stress": StabilityStressTask,
    "adaptation_budget": AdaptationBudgetTask,
}


def get_task(name: str) -> BaseTask:
    """Retrieve an instantiated benchmark task by name."""
    key = name.lower().replace("-", "_")
    if key not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available tasks: {sorted(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[key]()


__all__ = [
    "BaseTask",
    "StationaryRecallTask",
    "DistributionShiftTask",
    "KeyInterferenceTask",
    "ConflictingTargetsTask",
    "StabilityStressTask",
    "AdaptationBudgetTask",
    "TASK_REGISTRY",
    "get_task",
]
