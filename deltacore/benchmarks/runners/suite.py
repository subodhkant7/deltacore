# ==============================================================================
# DeltaCore: deltacore/benchmarks/runners/suite.py
# Benchmark suite execution engine managing repeated trials and artifact layout.
# ==============================================================================

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deltacore.benchmarks.baselines.base import BaseBaseline
from deltacore.benchmarks.baselines.factory import get_baseline
from deltacore.benchmarks.reporting.report import generate_markdown_report
from deltacore.benchmarks.reporting.statistics import aggregate_runs
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.schemas.result import AggregateResult, RunResult
from deltacore.benchmarks.tasks import get_task


def run_single_model_seeds(
    task_name: str,
    model_name: str,
    seeds: list[int],
    base_config: BenchmarkConfig,
    task_params: dict[str, Any] | None = None,
) -> tuple[list[RunResult], AggregateResult]:
    """Execute a task across multiple random seeds for a single model baseline.

    Args:
        task_name: Registered task name (e.g. 'distribution_shift').
        model_name: Baseline identifier (e.g. 'fixed', 'safe_self_referential').
        seeds: List of integer seeds.
        base_config: Base configuration template.
        task_params: Optional task-specific hyperparameters.

    Returns:
        Tuple of (list of individual RunResults, computed AggregateResult).
    """
    task = get_task(task_name)
    runs: list[RunResult] = []

    for seed in seeds:
        cfg = replace(
            base_config,
            experiment_name=task_name,
            seed=seed,
            controller=model_name,
            task_params=task_params
            if task_params is not None
            else base_config.task_params,
        )
        model: BaseBaseline = get_baseline(
            name=model_name,
            key_dim=cfg.key_dim,
            value_dim=cfg.value_dim,
            dtype=cfg.dtype,  # type: ignore[arg-type]
            device=cfg.device,
        )
        res = task.run(model, cfg)
        runs.append(res)

    aggregate = aggregate_runs(runs)
    return runs, aggregate


def run_benchmark_suite(
    task_name: str,
    models: list[str],
    seeds: list[int],
    base_config: BenchmarkConfig,
    output_dir: Path | str | None = None,
    task_params: dict[str, Any] | None = None,
) -> dict[str, AggregateResult]:
    """Execute a full comparative benchmark across multiple models and repeated seeds.

    Artifact layout on disk (if output_dir provided):
        output_dir/
        └── <task_name>/
            └── <YYYY-MM-DD>/
                ├── config.json
                ├── report.md
                └── <model_name>/
                    ├── aggregate.json
                    └── runs/
                        ├── seed-0.json
                        ├── seed-1.json
                        └── ...

    Args:
        task_name: Benchmark task identifier.
        models: List of model identifiers to evaluate.
        seeds: List of random seeds for repeated evaluation.
        base_config: Base BenchmarkConfig.
        output_dir: Root directory for artifact output.
        task_params: Optional task hyperparameters.

    Returns:
        Dictionary mapping model_name -> AggregateResult.
    """
    aggregates: dict[str, AggregateResult] = {}
    all_runs_by_model: dict[str, list[RunResult]] = {}

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    task_out_dir: Path | None = (
        Path(output_dir) / task_name / date_str if output_dir is not None else None
    )

    for model_name in models:
        runs, agg = run_single_model_seeds(
            task_name=task_name,
            model_name=model_name,
            seeds=seeds,
            base_config=base_config,
            task_params=task_params,
        )
        aggregates[model_name] = agg
        all_runs_by_model[model_name] = runs

        if task_out_dir is not None:
            model_dir = task_out_dir / model_name
            runs_dir = model_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)

            agg.save(model_dir / "aggregate.json")
            for run_res in runs:
                run_res.save(runs_dir / f"seed-{run_res.seed}.json")

    if task_out_dir is not None:
        task_out_dir.mkdir(parents=True, exist_ok=True)
        # Save base config
        import json

        (task_out_dir / "config.json").write_text(
            json.dumps(base_config.to_dict(), indent=2), encoding="utf-8"
        )
        # Generate Markdown report
        report_md = generate_markdown_report(
            config=base_config,
            aggregate_results=list(aggregates.values()),
        )
        (task_out_dir / "report.md").write_text(report_md, encoding="utf-8")

    return aggregates
