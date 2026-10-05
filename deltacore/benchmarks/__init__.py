# ==============================================================================
# DeltaCore: deltacore/benchmarks/__init__.py
# Top-level exports for the DeltaCore Adaptive-State Benchmark Framework.
# ==============================================================================

from deltacore.benchmarks.baselines import (
    BaseBaseline,
    BaselineTrajectoryResult,
    get_baseline,
)
from deltacore.benchmarks.reporting import (
    aggregate_runs,
    compute_paired_differences,
    generate_markdown_report,
)
from deltacore.benchmarks.runners.determinism import get_environment_info, set_seed
from deltacore.benchmarks.runners.suite import (
    run_benchmark_suite,
    run_single_model_seeds,
)
from deltacore.benchmarks.runners.timing import measure_execution_time
from deltacore.benchmarks.schemas import (
    AggregateResult,
    BenchmarkConfig,
    RunResult,
    RunStatus,
)
from deltacore.benchmarks.tasks import (
    TASK_REGISTRY,
    BaseTask,
    get_task,
)

__all__ = [
    "BenchmarkConfig",
    "RunResult",
    "AggregateResult",
    "RunStatus",
    "BaseBaseline",
    "BaselineTrajectoryResult",
    "get_baseline",
    "BaseTask",
    "get_task",
    "TASK_REGISTRY",
    "set_seed",
    "get_environment_info",
    "measure_execution_time",
    "run_benchmark_suite",
    "run_single_model_seeds",
    "aggregate_runs",
    "compute_paired_differences",
    "generate_markdown_report",
]
