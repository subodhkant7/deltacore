# ==============================================================================
# DeltaCore: deltacore/observatory/cli.py
# Command-line interface for the Adaptive State Observatory.
# ==============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deltacore.benchmarks.schemas.result import AggregateResult, RunResult
from deltacore.observatory.events import extract_events
from deltacore.observatory.fingerprint import generate_fingerprint
from deltacore.observatory.loader import (
    load_artifact,
    load_run_result,
    load_trajectory,
)
from deltacore.observatory.plots import generate_all_plots
from deltacore.observatory.replay import execute_replay
from deltacore.observatory.report import generate_observatory_report
from deltacore.observatory.schema import StateTrajectory


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a benchmark result, trajectory, or aggregate summary artifact."""
    filepath = Path(args.artifact)
    if not filepath.is_file():
        print(f"Error: Artifact file not found: {filepath}", file=sys.stderr)
        return 1

    artifact = load_artifact(filepath)

    if isinstance(artifact, StateTrajectory):
        print(
            f"=== DeltaCore State Trajectory: {artifact.model} on {artifact.task} ==="
        )
        print(
            f"Seed: {artifact.seed} | Steps: {len(artifact.steps)} | Finite: {artifact.all_states_finite}"
        )
        if artifact.first_nonfinite_step is not None:
            print(
                f"Numerical Failure Encountered at Step: {artifact.first_nonfinite_step}"
            )

        fp = generate_fingerprint(artifact)
        print("\n--- Fingerprint ---")
        print(
            f"  Mean Error:          {fp.mean_error:.4f}"
            if fp.mean_error is not None
            else "  Mean Error:          N/A"
        )
        print(
            f"  Final Error:         {fp.final_error:.4f}"
            if fp.final_error is not None
            else "  Final Error:         N/A"
        )
        print(
            f"  Recovery Latency:    {fp.recovery_latency}"
            if fp.recovery_latency is not None
            else "  Recovery Latency:    N/A"
        )
        print(
            f"  Update Energy:       {fp.update_energy:.4f}"
            if fp.update_energy is not None
            else "  Update Energy:       N/A"
        )
        print(
            f"  Min Stability Margin:{fp.min_stability_margin:.4f}"
            if fp.min_stability_margin is not None
            else "  Min Stability Margin:N/A"
        )

        events = extract_events(artifact)
        print(f"\n--- Detected Events ({len(events)}) ---")
        for ev in events:
            val_str = (
                f"{ev.value:.4f}" if isinstance(ev.value, float) else str(ev.value)
            )
            print(
                f"  [Step {ev.step:2d}] {ev.event_type:<26} val={val_str} {ev.context}"
            )

        if args.report:
            rep_path = Path(args.report)
            generate_observatory_report(artifact, output_path=rep_path)
            print(f"\nGenerated report written to: {rep_path}")

    elif isinstance(artifact, RunResult):
        print(
            f"=== DeltaCore Benchmark Run Result: {artifact.model} on {artifact.experiment} ==="
        )
        print(
            f"Seed: {artifact.seed} | Status: {artifact.status.value} | Device: {artifact.device} | Dtype: {artifact.dtype}"
        )
        if artifact.error_message:
            print(
                f"Error Message: {artifact.error_message} (at step {artifact.error_step})"
            )
        print("\n--- Key Metrics ---")
        for k, v in artifact.metrics.items():
            if isinstance(v, float):
                print(f"  {k:<28}: {v:.4f}")
            else:
                print(f"  {k:<28}: {v}")

        # If report requested, load trajectory (with replay if needed)
        if args.report:
            traj = load_trajectory(filepath, replay_if_missing=True)
            rep_path = Path(args.report)
            generate_observatory_report(traj, output_path=rep_path)
            print(f"\nGenerated report written to: {rep_path}")

    elif isinstance(artifact, AggregateResult):
        print(
            f"=== DeltaCore Aggregate Result: {artifact.model} on {artifact.experiment} ==="
        )
        print(
            f"Runs: {artifact.num_runs} | Success: {artifact.num_success} | Failures: {artifact.num_failures}"
        )
        print("\n--- Metric Aggregates ---")
        for m, stats in artifact.metric_aggregates.items():
            mean_val = stats.get("mean", 0.0)
            std_val = stats.get("std", 0.0)
            print(f"  {m:<28}: {mean_val:.4f} ± {std_val:.4f}")

    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    """Generate static publication-quality plots from a trajectory or RunResult."""
    filepath = Path(args.artifact)
    if not filepath.is_file():
        print(f"Error: Artifact file not found: {filepath}", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else filepath.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        traj = load_trajectory(filepath, replay_if_missing=True)
    except Exception as err:
        print(
            f"Error: Failed to extract trajectory from {filepath}: {err}",
            file=sys.stderr,
        )
        return 1

    plot_paths = generate_all_plots(traj, output_dir=out_dir)
    print(f"Successfully generated {len(plot_paths)} plots in: {out_dir}")
    for name, p in plot_paths.items():
        print(f"  - {name}: {p.name}")

    if args.report:
        rep_path = Path(args.report)
        generate_observatory_report(traj, plot_paths=plot_paths, output_path=rep_path)
        print(f"Reproducibility report generated: {rep_path}")

    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Re-run experiment deterministically and verify stored metrics."""
    filepath = Path(args.artifact)
    if not filepath.is_file():
        print(f"Error: Artifact file not found: {filepath}", file=sys.stderr)
        return 1

    try:
        run_res = load_run_result(filepath)
    except Exception as err:
        print(f"Error: Replay requires a RunResult artifact: {err}", file=sys.stderr)
        return 1

    print(
        f"Replaying experiment: {run_res.experiment} ({run_res.model}, seed={run_res.seed})..."
    )
    res = execute_replay(run_res, rtol=args.rtol, atol=args.atol)

    print(f"Replay Status: {res.status.upper()}")
    print(f"Max Absolute Metric Delta: {res.max_absolute_diff:.2e}")
    print(f"Max Relative Metric Delta: {res.max_relative_diff:.2e}")

    if res.error_message:
        print(f"Diagnostic Error: {res.error_message}")

    print("\n--- Metric Verification Breakdown ---")
    for cmp in res.metric_comparisons:
        print(
            f"  {cmp.metric_name:<26} stored={str(cmp.stored_value):<10} regen={str(cmp.regenerated_value):<10} status={cmp.status}"
        )

    if res.status in ("exact match", "within numerical tolerance"):
        print("\nVerification PASSED.")
        return 0
    else:
        print(f"\nVerification FAILED with status: {res.status}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for DeltaCore Observatory."""
    parser = argparse.ArgumentParser(
        prog="python3 -m deltacore.observatory",
        description="DeltaCore Adaptive State Observatory: scientific analysis and diagnostics for adaptive neural state.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. inspect
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect an artifact's metrics, events, and invariants."
    )
    inspect_parser.add_argument(
        "artifact", help="Path to RunResult, StateTrajectory, or AggregateResult JSON."
    )
    inspect_parser.add_argument(
        "--report", help="Optional path to output a Markdown report."
    )

    # 2. plot
    plot_parser = subparsers.add_parser(
        "plot", help="Generate static plots (Plots A through E) for a trajectory."
    )
    plot_parser.add_argument(
        "artifact", help="Path to RunResult or StateTrajectory JSON."
    )
    plot_parser.add_argument(
        "--output", "-o", default=None, help="Directory to save generated plot PNGs."
    )
    plot_parser.add_argument(
        "--report",
        help="Optional path to output a Markdown report embedding the plots.",
    )

    # 3. replay
    replay_parser = subparsers.add_parser(
        "replay", help="Deterministically re-execute a RunResult and verify metrics."
    )
    replay_parser.add_argument("artifact", help="Path to RunResult JSON.")
    replay_parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for metric verification.",
    )
    replay_parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for metric verification.",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "inspect":
        return _cmd_inspect(args)
    elif args.subcommand == "plot":
        return _cmd_plot(args)
    elif args.subcommand == "replay":
        return _cmd_replay(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
