# ==============================================================================
# DeltaCore: deltacore/benchmarks/cli.py
# Command-line interface for executing DeltaCore benchmark suites.
# ==============================================================================

import argparse
import sys
from pathlib import Path

from deltacore.benchmarks.runners.suite import run_benchmark_suite
from deltacore.benchmarks.schemas.config import (
    VALID_CONTROLLERS,
    BenchmarkConfig,
)
from deltacore.benchmarks.tasks import TASK_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for benchmark commands."""
    parser = argparse.ArgumentParser(
        prog="python3 -m deltacore.benchmarks",
        description="DeltaCore Adaptive-State Benchmark Framework CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: list
    subparsers.add_parser("list", help="List registered tasks and available baselines.")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Execute a benchmark task.")
    run_parser.add_argument(
        "task",
        type=str,
        choices=sorted(TASK_REGISTRY.keys()),
        help="Benchmark task identifier to execute.",
    )
    run_parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "frozen",
            "fixed",
            "adaptive",
            "self_referential",
            "safe_self_referential",
        ],
        help="Baseline models to evaluate.",
    )
    run_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="List of random seeds for repeated evaluation.",
    )
    run_parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length T (default 128).",
    )
    run_parser.add_argument(
        "--key-dim",
        type=int,
        default=16,
        help="Key dimension K (default 16).",
    )
    run_parser.add_argument(
        "--val-dim",
        type=int,
        default=16,
        help="Value dimension V (default 16).",
    )
    run_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Numerical precision (default float32).",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda', default 'cpu').",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory to save JSON artifacts and report (default 'results').",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output summaries.",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """CLI entrypoint for running benchmark suites."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.command == "list":
        print("DeltaCore Registered Benchmark Tasks:")
        for t in sorted(TASK_REGISTRY.keys()):
            print(f"  - {t}")
        print("\nAvailable Baselines:")
        for b in sorted(VALID_CONTROLLERS):
            print(f"  - {b}")
        return 0

    if parsed.command == "run":
        task_name = parsed.task
        models = parsed.models
        seeds = parsed.seeds
        output_dir = Path(parsed.output)

        config = BenchmarkConfig(
            experiment_name=task_name,
            seed=seeds[0] if seeds else 42,
            sequence_length=parsed.seq_len,
            key_dim=parsed.key_dim,
            value_dim=parsed.val_dim,
            dtype=parsed.dtype,
            device=parsed.device,
        )

        if not parsed.quiet:
            print("=" * 72)
            print(f"DeltaCore Benchmark: {task_name}")
            print(f"Models: {models}")
            print(
                f"Seeds: {seeds} | T={parsed.seq_len} | Dtype={parsed.dtype} | Device={parsed.device}"
            )
            print("=" * 72)

        aggregates = run_benchmark_suite(
            task_name=task_name,
            models=models,
            seeds=seeds,
            base_config=config,
            output_dir=output_dir,
        )

        if not parsed.quiet:
            print("\nAggregate Results Summary:")
            for m, agg in aggregates.items():
                print(f"\nModel: {m} (Success: {agg.num_success}/{agg.num_runs})")
                for k, stats in sorted(agg.metric_aggregates.items()):
                    print(f"  {k:28s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

            print(f"\nArtifacts saved to: {output_dir / task_name}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
