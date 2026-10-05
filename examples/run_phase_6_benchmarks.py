# ==============================================================================
# DeltaCore: examples/run_phase_6_benchmarks.py
# Phase 6 Reference Benchmark Suite: Runs all core tasks across baselines on CPU.
# ==============================================================================

import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from deltacore.benchmarks.runners.suite import run_benchmark_suite  # noqa: E402
from deltacore.benchmarks.schemas.config import BenchmarkConfig  # noqa: E402


def format_table(headers: list[str], rows: list[list[str]], title: str) -> str:
    """Format an ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = (
        "| "
        + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False))
        + " |"
    )
    table_lines = [f"\n=== {title} ===", sep, hdr, sep]
    for row in rows:
        row_str = (
            "| "
            + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths, strict=False))
            + " |"
        )
        table_lines.append(row_str)
    table_lines.append(sep)
    return "\n".join(table_lines)


def run_all() -> None:
    """Execute the standard reference suite across all 5 core tasks."""
    print("=" * 80)
    print("DeltaCore Phase 6: Adaptive-State Benchmark Framework Reference Suite")
    print("=" * 80)

    seeds = [0, 1, 2, 3, 4]
    models = [
        "frozen",
        "fixed",
        "adaptive",
        "self_referential",
        "safe_self_referential",
    ]
    device = "cpu"
    dtype = "float32"

    t_suite_start = time.perf_counter()

    # --------------------------------------------------------------------------
    # 1. Stationary Recall
    # --------------------------------------------------------------------------
    cfg_stat = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=seeds[0],
        sequence_length=64,
        key_dim=16,
        value_dim=16,
        dtype=dtype,
        device=device,
        task_params={"num_associations": 6},
    )
    stat_aggs = run_benchmark_suite(
        task_name="stationary_recall",
        models=models,
        seeds=seeds,
        base_config=cfg_stat,
    )

    stat_headers = [
        "Model",
        "Success",
        "Final Error",
        "Mean Error",
        "Final ||M||_F",
        "Update Energy",
    ]
    stat_rows = []
    for m in models:
        agg = stat_aggs[m]
        f_err = agg.metric_aggregates.get("final_error", {"mean": 0.0, "std": 0.0})
        m_err = agg.metric_aggregates.get("mean_error", {"mean": 0.0, "std": 0.0})
        m_norm = agg.metric_aggregates.get(
            "final_state_norm", {"mean": 0.0, "std": 0.0}
        )
        u_eng = agg.metric_aggregates.get("update_energy", {"mean": 0.0, "std": 0.0})
        stat_rows.append(
            [
                m,
                f"{agg.num_success}/{agg.num_runs}",
                f"{f_err['mean']:.4f} ± {f_err['std']:.3f}",
                f"{m_err['mean']:.4f} ± {m_err['std']:.3f}",
                f"{m_norm['mean']:.4f} ± {m_norm['std']:.3f}",
                f"{u_eng['mean']:.2f} ± {u_eng['std']:.2f}",
            ]
        )
    print(format_table(stat_headers, stat_rows, "Task 1: Stationary Recall (T=64)"))

    # --------------------------------------------------------------------------
    # 2. Distribution Shift
    # --------------------------------------------------------------------------
    cfg_shift = BenchmarkConfig(
        experiment_name="distribution_shift",
        seed=seeds[0],
        sequence_length=64,
        key_dim=16,
        value_dim=16,
        shift_position=32,
        dtype=dtype,
        device=device,
        task_params={
            "shift_type": "target_inversion",
            "num_patterns": 6,
            "tau": 0.5,
            "window": 4,
        },
    )
    shift_aggs = run_benchmark_suite(
        task_name="distribution_shift",
        models=models,
        seeds=seeds,
        base_config=cfg_shift,
    )

    shift_headers = [
        "Model",
        "Success",
        "Pre-Shift Err",
        "Shock Err",
        "Final Err",
        "T_FP (Steps)",
        "Growth Ratio",
    ]
    shift_rows = []
    for m in models:
        agg = shift_aggs[m]
        pre = agg.metric_aggregates.get("pre_shift_error", {"mean": 0.0, "std": 0.0})
        shk = agg.metric_aggregates.get("shock_error", {"mean": 0.0, "std": 0.0})
        fin = agg.metric_aggregates.get("final_error", {"mean": 0.0, "std": 0.0})
        tfp = agg.metric_aggregates.get("first_passage_recovery")
        tfp_str = (
            f"{tfp['mean']:.1f} ± {tfp['std']:.1f}"
            if tfp and tfp["count"] > 0
            else "Fail / N/A"
        )
        grw = agg.metric_aggregates.get("state_growth_ratio", {"mean": 1.0, "std": 0.0})
        shift_rows.append(
            [
                m,
                f"{agg.num_success}/{agg.num_runs}",
                f"{pre['mean']:.3f} ± {pre['std']:.2f}",
                f"{shk['mean']:.3f} ± {shk['std']:.2f}",
                f"{fin['mean']:.3f} ± {fin['std']:.2f}",
                tfp_str,
                f"{grw['mean']:.3f} ± {grw['std']:.2f}",
            ]
        )
    print(
        format_table(
            shift_headers,
            shift_rows,
            "Task 2: Distribution Shift (Target Inversion at t=32)",
        )
    )

    # --------------------------------------------------------------------------
    # 3. Key Interference (sweep similarity rho = 0.0, 0.5, 0.9)
    # --------------------------------------------------------------------------
    print("\n=== Task 3: Key Interference vs. Cosine Similarity ===")
    for rho in [0.00, 0.50, 0.90]:
        cfg_interf = BenchmarkConfig(
            experiment_name="key_interference",
            seed=seeds[0],
            sequence_length=40,
            key_dim=16,
            value_dim=16,
            dtype=dtype,
            device=device,
            task_params={"cosine_similarity": rho},
        )
        interf_aggs = run_benchmark_suite(
            task_name="key_interference",
            models=["fixed", "adaptive", "self_referential", "safe_self_referential"],
            seeds=seeds,
            base_config=cfg_interf,
            task_params={"cosine_similarity": rho},
        )
        int_headers = ["Model", "Mean Retr Err", "Final Error", "Final ||M||_F"]
        int_rows = []
        for m in ["fixed", "adaptive", "self_referential", "safe_self_referential"]:
            agg = interf_aggs[m]
            m_err = agg.metric_aggregates.get(
                "mean_retrieval_error", {"mean": 0.0, "std": 0.0}
            )
            f_err = agg.metric_aggregates.get(
                "sequence_final_error", {"mean": 0.0, "std": 0.0}
            )
            m_n = agg.metric_aggregates.get(
                "final_state_norm", {"mean": 0.0, "std": 0.0}
            )
            int_rows.append(
                [
                    m,
                    f"{m_err['mean']:.4f} ± {m_err['std']:.3f}",
                    f"{f_err['mean']:.4f} ± {f_err['std']:.3f}",
                    f"{m_n['mean']:.4f} ± {m_n['std']:.3f}",
                ]
            )
        print(format_table(int_headers, int_rows, f"Key Similarity rho={rho:.2f}"))

    # --------------------------------------------------------------------------
    # 4. Conflicting Targets
    # --------------------------------------------------------------------------
    cfg_conf = BenchmarkConfig(
        experiment_name="conflicting_targets",
        seed=seeds[0],
        sequence_length=40,
        key_dim=16,
        value_dim=16,
        shift_position=20,
        dtype=dtype,
        device=device,
        task_params={"conflict_magnitude": 2.0},
    )
    conf_aggs = run_benchmark_suite(
        task_name="conflicting_targets",
        models=["fixed", "adaptive", "self_referential", "safe_self_referential"],
        seeds=seeds,
        base_config=cfg_conf,
    )
    conf_headers = [
        "Model",
        "Success",
        "Old Retention",
        "New Err",
        "Adapt Speed",
        "Oscillation",
    ]
    conf_rows = []
    for m in ["fixed", "adaptive", "self_referential", "safe_self_referential"]:
        agg = conf_aggs[m]
        old_r = agg.metric_aggregates.get(
            "old_memory_retention", {"mean": 0.0, "std": 0.0}
        )
        new_e = agg.metric_aggregates.get("new_target_error", {"mean": 0.0, "std": 0.0})
        spd = agg.metric_aggregates.get("adaptation_speed")
        spd_str = (
            f"{spd['mean']:.1f} ± {spd['std']:.1f}"
            if spd and spd["count"] > 0
            else "Fail / N/A"
        )
        osc = agg.metric_aggregates.get("oscillation", {"mean": 0.0, "std": 0.0})
        conf_rows.append(
            [
                m,
                f"{agg.num_success}/{agg.num_runs}",
                f"{old_r['mean']:.3f} ± {old_r['std']:.2f}",
                f"{new_e['mean']:.3f} ± {new_e['std']:.2f}",
                spd_str,
                f"{osc['mean']:.4f} ± {osc['std']:.4f}",
            ]
        )
    print(
        format_table(
            conf_headers,
            conf_rows,
            "Task 4: Conflicting Targets (k -> v1 then k -> v2, ||v1-v2||=2.0)",
        )
    )

    # --------------------------------------------------------------------------
    # 5. Stability Stress
    # --------------------------------------------------------------------------
    cfg_stress = BenchmarkConfig(
        experiment_name="stability_stress",
        seed=seeds[0],
        sequence_length=50,
        key_dim=16,
        value_dim=16,
        dtype=dtype,
        device=device,
        task_params={"regime": "large_key_norms", "key_scale": 6.0},
    )
    stress_aggs = run_benchmark_suite(
        task_name="stability_stress",
        models=["fixed", "self_referential", "safe_self_referential"],
        seeds=seeds,
        base_config=cfg_stress,
    )
    stress_headers = [
        "Model",
        "Status",
        "Success",
        "Failures",
        "Max ||M||_F",
        "Min Margin S_t",
        "Clips",
    ]
    stress_rows = []
    for m in ["fixed", "self_referential", "safe_self_referential"]:
        agg = stress_aggs[m]
        status = (
            "ALL SUCCESS"
            if agg.num_failures == 0
            else f"{agg.num_failures} NUMERICAL_FAILURE"
        )
        m_norm = agg.metric_aggregates.get("max_state_norm", {"mean": 0.0, "std": 0.0})
        margin = agg.metric_aggregates.get(
            "min_stability_margin", {"mean": 2.0, "std": 0.0}
        )
        clips = agg.metric_aggregates.get("clip_count", {"mean": 0.0, "std": 0.0})
        stress_rows.append(
            [
                m,
                status,
                f"{agg.num_success}/{agg.num_runs}",
                str(agg.num_failures),
                f"{m_norm['mean']:.2e} ± {m_norm['std']:.2e}",
                f"{margin['mean']:.3f} ± {margin['std']:.2f}",
                f"{clips['mean']:.1f}",
            ]
        )
    print(
        format_table(
            stress_headers,
            stress_rows,
            "Task 5: Stability Stress (Large Keys ||k||=6.0, ||k||^2=36.0)",
        )
    )

    t_suite_total = time.perf_counter() - t_suite_start
    print("\n" + "=" * 80)
    print(f"Reference Suite Completed in {t_suite_total:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
