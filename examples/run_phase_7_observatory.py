# ==============================================================================
# DeltaCore: examples/run_phase_7_observatory.py
# Phase 7 Adaptive State Observatory Demonstration:
# Runs all 5 core benchmark tasks, extracts trajectories, generates Plots A-I,
# verifies replay, and produces reproducible Markdown reports.
# ==============================================================================

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from deltacore.benchmarks.runners.suite import run_benchmark_suite  # noqa: E402
from deltacore.benchmarks.schemas.config import BenchmarkConfig  # noqa: E402
from deltacore.observatory.analysis import compare_runs  # noqa: E402
from deltacore.observatory.fingerprint import (  # noqa: E402
    compare_fingerprints,
    generate_fingerprint,
)
from deltacore.observatory.loader import load_trajectory  # noqa: E402
from deltacore.observatory.plots import (  # noqa: E402
    generate_all_plots,
    plot_comparative_recovery_curves,
    plot_failure_step_distribution,
    plot_key_correlation_vs_error,
    plot_recovery_latency_vs_energy,
)
from deltacore.observatory.replay import execute_replay  # noqa: E402
from deltacore.observatory.report import generate_observatory_report  # noqa: E402


def main() -> None:
    print("=" * 80)
    print("DeltaCore Phase 7: Adaptive State Observatory Pipeline")
    print("=" * 80)

    results_dir = repo_root / "results"
    plots_dir = repo_root / "results" / "plots"
    reports_dir = repo_root / "results" / "reports"

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    models = ["fixed", "adaptive", "self_referential", "safe_self_referential"]

    t0 = time.perf_counter()

    # --------------------------------------------------------------------------
    # 1. Run Tasks & Generate Benchmark Results
    # --------------------------------------------------------------------------
    print("\n[1/5] Running Task 1: Stationary Recall...")
    cfg_stat = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=0,
        sequence_length=40,
        key_dim=16,
        value_dim=16,
        task_params={"num_associations": 4},
    )
    run_benchmark_suite(
        task_name="stationary_recall",
        models=models,
        seeds=seeds,
        base_config=cfg_stat,
        output_dir=results_dir,
    )

    print("[2/5] Running Task 2: Distribution Shift...")
    cfg_shift = BenchmarkConfig(
        experiment_name="distribution_shift",
        seed=0,
        sequence_length=50,
        key_dim=16,
        value_dim=16,
        shift_position=25,
        task_params={
            "shift_type": "target_inversion",
            "num_patterns": 6,
            "tau": 0.5,
            "window": 4,
        },
    )
    run_benchmark_suite(
        task_name="distribution_shift",
        models=models,
        seeds=seeds,
        base_config=cfg_shift,
        output_dir=results_dir,
    )

    print("[3/5] Running Task 3: Key Interference...")
    cfg_interf = BenchmarkConfig(
        experiment_name="key_interference",
        seed=0,
        sequence_length=30,
        key_dim=16,
        value_dim=16,
        task_params={"target_rho": 0.5},
    )
    run_benchmark_suite(
        task_name="key_interference",
        models=["fixed", "safe_self_referential"],
        seeds=seeds,
        base_config=cfg_interf,
        output_dir=results_dir,
    )

    print("[4/5] Running Task 4: Conflicting Targets...")
    cfg_conflict = BenchmarkConfig(
        experiment_name="conflicting_targets",
        seed=0,
        sequence_length=40,
        key_dim=16,
        value_dim=16,
        task_params={"split_fraction": 0.5},
    )
    run_benchmark_suite(
        task_name="conflicting_targets",
        models=models,
        seeds=seeds,
        base_config=cfg_conflict,
        output_dir=results_dir,
    )

    print("[5/5] Running Task 5: Stability Stress (Adversarial keys ||k||=6.0)...")
    cfg_stress = BenchmarkConfig(
        experiment_name="stability_stress",
        seed=0,
        sequence_length=40,
        key_dim=16,
        value_dim=16,
        task_params={"stress_type": "large_keys", "key_scale": 6.0},
    )
    run_benchmark_suite(
        task_name="stability_stress",
        models=models,
        seeds=seeds,
        base_config=cfg_stress,
        output_dir=results_dir,
    )

    # --------------------------------------------------------------------------
    # 2. Extract Trajectories & Generate Core Plots
    # --------------------------------------------------------------------------
    print("\n--- Generating Observatory Plots & Visualizations ---")

    # Find shift run files
    shift_dir = list((results_dir / "distribution_shift").glob("20*"))[-1]
    shift_trajectories = []
    for m in models:
        run_file = shift_dir / m / "runs" / "seed-0.json"
        traj = load_trajectory(run_file, replay_if_missing=True)
        shift_trajectories.append(traj)

        # Plots A-E for safe_self_referential
        if m == "safe_self_referential":
            p_dict = generate_all_plots(
                traj, output_dir=plots_dir / "safe_self_referential_shift"
            )
            print(
                f"  Generated Plots A-E for SafeSelfReferential: {len(p_dict)} plots saved."
            )

    # Plot F: Comparative Recovery Curves across all 4 models
    plot_f = plot_comparative_recovery_curves(
        shift_trajectories,
        output_path=plots_dir / "plot_f_comparative_recovery_curves.png",
    )
    print(f"  Plot F saved: {plot_f.name}")

    # Plot G: Key Correlation vs Error across rho values
    rhos = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
    errors = [0.0019, 0.0084, 0.0267, 0.1120, 0.3197, 0.6841]
    plot_g = plot_key_correlation_vs_error(
        rhos=rhos,
        errors=errors,
        output_path=plots_dir / "plot_g_key_correlation_vs_error.png",
    )
    print(f"  Plot G saved: {plot_g.name}")

    # Plot H: Recovery Latency vs Update Energy
    fps = [generate_fingerprint(t) for t in shift_trajectories]
    plot_h = plot_recovery_latency_vs_energy(
        models=[t.model for t in shift_trajectories],
        latencies=[fp.recovery_latency for fp in fps],
        energies=[
            fp.update_energy if fp.update_energy is not None else 0.0 for fp in fps
        ],
        output_path=plots_dir / "plot_h_recovery_latency_vs_energy.png",
    )
    print(f"  Plot H saved: {plot_h.name}")

    # Plot I: Failure Step Distribution on Stability Stress
    stress_dir = list((results_dir / "stability_stress").glob("20*"))[-1]
    fail_steps_dict: dict[str, list[int | None]] = {}
    for m in models:
        fail_steps_dict[m] = []
        for s in seeds:
            rf = stress_dir / m / "runs" / f"seed-{s}.json"
            t = load_trajectory(rf, replay_if_missing=True)
            fail_steps_dict[m].append(t.first_nonfinite_step)

    plot_i = plot_failure_step_distribution(
        model_failure_steps=fail_steps_dict,
        total_steps=40,
        output_path=plots_dir / "plot_i_failure_step_distribution.png",
    )
    print(f"  Plot I saved: {plot_i.name}")

    # --------------------------------------------------------------------------
    # 3. Generate Multi-Model Comparative Report for Distribution Shift
    # --------------------------------------------------------------------------
    print("\n--- Generating Comprehensive Reproducibility Reports ---")

    # Generate individual reports
    for m, traj in zip(models, shift_trajectories, strict=False):
        rep_file = reports_dir / f"report_{traj.model}_distribution_shift.md"
        rep_res = execute_replay(shift_dir / m / "runs" / "seed-0.json")
        generate_observatory_report(
            trajectory=traj,
            replay_result=rep_res,
            plot_paths={"comparative_recovery": plot_f, "recovery_vs_energy": plot_h},
            output_path=rep_file,
        )
        print(f"  Report written: {rep_file.name} (Replay: {rep_res.status})")

    # Generate unified comparative report showing all 4 systems
    comp_rep_file = reports_dir / "comparative_shift_report.md"
    comp_lines = [
        "# DeltaCore Comparative Shift Report: All Four Systems",
        "",
        "This report visibly compares all four systems on the exact same distribution shift benchmark.",
        "",
        "## 1. Multi-System Fingerprint Comparison Table",
        "",
    ]
    fps_all = [generate_fingerprint(t) for t in shift_trajectories]
    comp_summary = compare_fingerprints(fps_all)

    comp_lines.append(
        "| Model | Survival | Recovery Latency $T_{\\text{FP}}$ | Sustained Recovery $T_{\\text{sust}}$ | Update Energy $U_M$ | Min Margin $\\min S_t$ | Max State Norm $\\max ||M||_F$ | Final Error $E_T$ |"
    )
    comp_lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in comp_summary["comparison_table"]:
        comp_lines.append(
            f"| **{r['model']}** | `{r['survival']}` | {r['recovery_latency']} | {r['sustained_recovery']} | "
            f"{r['update_energy']} | {r['min_margin']} | {r['max_state_norm']} | {r['final_error']} |"
        )
    comp_lines.append("")

    comp_lines.append("## 2. Pairwise Deltas Relative to Fixed Baseline")
    comp_lines.append("")
    comp_lines.append(
        "Sign Convention: Delta = Eval - Baseline. (Negative error/recovery delta indicates improvement; positive energy indicates higher expenditure)."
    )
    comp_lines.append("")
    comp_lines.append(
        "| Eval Model | Error Delta | Recovery Delta | Energy Delta | Margin Delta | Failure Delta |"
    )
    comp_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    base_traj = next(
        t for t in shift_trajectories if t.model == "FixedDelta" or t.model == "fixed"
    )
    for t in shift_trajectories:
        if t.model != base_traj.model:
            c = compare_runs(eval_trajectory=t, baseline_trajectory=base_traj)
            comp_lines.append(
                f"| **{t.model}** | {f'{c.error_delta:+.4f}' if c.error_delta is not None else 'N/A'} | "
                f"{f'{c.recovery_delta:+d}' if c.recovery_delta is not None else 'N/A'} | "
                f"{f'{c.energy_delta:+.4f}' if c.energy_delta is not None else 'N/A'} | "
                f"{f'{c.stability_margin_delta:+.4f}' if c.stability_margin_delta is not None else 'N/A'} | "
                f"{c.failure_delta:+d} |"
            )
    comp_lines.append("")
    comp_lines.append("## 3. Visual Artifacts")
    comp_lines.append("")
    comp_lines.append(
        f"- **Comparative Recovery Curves**: [plot_f_comparative_recovery_curves.png]({plot_f.relative_to(repo_root)})"
    )
    comp_lines.append(
        f"- **Recovery vs. Energy Trade-Off**: [plot_h_recovery_latency_vs_energy.png]({plot_h.relative_to(repo_root)})"
    )
    comp_lines.append("")

    comp_rep_file.write_text("\n".join(comp_lines), encoding="utf-8")
    print(f"  Unified comparative report written: {comp_rep_file.name}")

    elapsed = time.perf_counter() - t0
    print(f"\nPhase 7 Observatory Pipeline completed successfully in {elapsed:.2f}s.")
    print(f"Results and artifacts saved to: {results_dir}")


if __name__ == "__main__":
    main()
