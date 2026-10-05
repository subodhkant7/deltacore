"""DeltaCore Phase 16: Unseen Shift Robustness & Adaptive Safety Benchmark.

Executes complete Phase 16 benchmark evaluating the frozen pooled SafeAdaptiveDelta
configuration (eta0=0.015, rho=1.5, alpha_min=0.95) against unseen shifts across
5 shift types (Mean, Variance, Temporal speed, Noise, Combined), 3 severities
(Mild, Moderate, Severe), 2 real-world domains (NOAA OISST SST & ECMWF ERA5 T2m),
and 5 deterministic seeds (42, 43, 44, 45, 46).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deltacore.streaming.models import (
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    OnlineRidgePredictor,
    PersistencePredictor,
    SafeAdaptiveDeltaPredictor,
    SpatialConvControl,
    StreamingPredictor,
)
from deltacore.streaming.real_benchmark import train_offline_model
from deltacore.streaming.regime_transfer import load_cross_domain_datasets
from deltacore.streaming.unseen_shifts import (
    SEVERITY_SCALES,
    apply_unseen_shift,
    check_permutation_equivariance,
    evaluate_perturbation_robustness,
    evaluate_shift_reset_ablation,
    evaluate_shift_streaming_run,
    generate_retention_stress_stream,
    map_failure_boundary_grid,
)


def instantiate_phase_16_models(
    dim: int,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    seed: int = 42,
    pooled_eta: float = 0.015,
    pooled_rho: float = 1.5,
    pooled_alpha: float = 0.95,
) -> dict[str, StreamingPredictor]:
    """Instantiate primary 5 models + secondary spatial control."""
    models: dict[str, StreamingPredictor] = {}
    models["Persistence"] = PersistencePredictor(dim=dim)

    fl = FrozenLinearPredictor(dim=dim)
    train_offline_model(fl, train_inputs, train_targets, epochs=25, seed=seed)
    models["FrozenLinear"] = fl

    models["OnlineRidge"] = OnlineRidgePredictor(dim=dim, lam=1.0)
    models["FixedDelta"] = FixedDeltaPredictor(dim=dim, step_size=pooled_eta)
    models["SafeAdaptiveDelta"] = SafeAdaptiveDeltaPredictor(
        dim=dim,
        eta_max=pooled_eta,
        rho=pooled_rho,
        alpha_min=pooled_alpha,
        gamma=0.05,
    )

    h = int(math.isqrt(dim))
    if h * h == dim:
        sc = SpatialConvControl(height=h, width=h, channels=1, seed=seed)
        train_offline_model(sc, train_inputs, train_targets, epochs=25, seed=seed)
        models["SpatialConv"] = sc

    return models


def run_phase_16_benchmark(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: str | Path = "docs/benchmarks/artifacts/phase_16",
) -> dict[str, Any]:
    """Execute complete Phase 16 benchmark suite and serialize all JSON artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DELTACORE PHASE 16: UNSEEN SHIFT ROBUSTNESS & ADAPTIVE SAFETY BENCHMARK")
    print(
        "Primary Configuration: Pooled SafeAdaptiveDelta (eta0=0.015, rho=1.5, alpha_min=0.95)"
    )
    print("Domains: Domain A (OISST SST) <---> Domain B (ERA5 T2m)")
    print(f"Seeds: {seeds}")
    print("=" * 88)

    shift_types = ["mean", "variance", "temporal_speed", "noise", "combined"]
    severities = ["mild", "moderate", "severe"]
    model_names = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "FixedDelta",
        "SafeAdaptiveDelta",
        "SpatialConv",
    ]

    # Data structures for results
    per_seed_results: dict[str, Any] = {
        "domain_a": {m: {} for m in model_names},
        "domain_b": {m: {} for m in model_names},
    }
    per_seed_shift_matrix: dict[str, Any] = {
        s_type: {
            sev: {
                "domain_a": {m: {} for m in model_names},
                "domain_b": {m: {} for m in model_names},
            }
            for sev in severities
        }
        for s_type in shift_types
    }

    # Load base seed 42 dataset for reference and config
    base_data = load_cross_domain_datasets(seed=42, resolution="small")
    data_a = base_data.domain_a
    data_b = base_data.domain_b

    # 1. Sensitivity Analysis on Validation Splits (Validation-only, no test shift leakage)
    print("\n[Step 1] Running Sensitivity Analysis on Validation Splits...")
    sensitivity_grid = [
        {"eta_max": eta, "alpha_min": alpha, "rho": 1.5, "gamma": 0.05}
        for eta in [0.010, 0.015, 0.020]
        for alpha in [0.90, 0.95]
    ]
    sensitivity_results: list[dict[str, Any]] = []
    for cfg in sensitivity_grid:
        mA = SafeAdaptiveDeltaPredictor(
            dim=64,
            eta_max=cfg["eta_max"],
            rho=cfg["rho"],
            alpha_min=cfg["alpha_min"],
            gamma=cfg["gamma"],
        )
        mB = SafeAdaptiveDeltaPredictor(
            dim=64,
            eta_max=cfg["eta_max"],
            rho=cfg["rho"],
            alpha_min=cfg["alpha_min"],
            gamma=cfg["gamma"],
        )
        res_val_a = evaluate_shift_streaming_run(
            mA, data_a.val_inputs, data_a.val_targets
        )
        res_val_b = evaluate_shift_streaming_run(
            mB, data_b.val_inputs, data_b.val_targets
        )
        sensitivity_results.append(
            {
                "config": cfg,
                "val_error_domain_a": res_val_a["rel_error"],
                "val_error_domain_b": res_val_b["rel_error"],
                "val_error_mean": 0.5
                * (res_val_a["rel_error"] + res_val_b["rel_error"]),
            }
        )

    # 2. Multi-Seed Robustness Evaluation across Shifts and Severities
    print("\n[Step 2] Executing Multi-Seed Robustness Matrix across 5 Seeds...")
    for seed in seeds:
        print(f"  --- Running Seed {seed} ---")
        ds = load_cross_domain_datasets(seed=seed, resolution="small")
        cur_a = ds.domain_a
        cur_b = ds.domain_b

        models_a = instantiate_phase_16_models(
            dim=64,
            train_inputs=cur_a.train_inputs,
            train_targets=cur_a.train_targets,
            seed=seed,
        )
        models_b = instantiate_phase_16_models(
            dim=64,
            train_inputs=cur_b.train_inputs,
            train_targets=cur_b.train_targets,
            seed=seed,
        )

        # Natural (unperturbed) stream evaluation
        for m_name in model_names:
            rA = evaluate_shift_streaming_run(
                models_a[m_name],
                cur_a.test_inputs,
                cur_a.test_targets,
                shift_start=cur_a.shift_interval[0],
                shift_end=cur_a.shift_interval[1],
            )
            rB = evaluate_shift_streaming_run(
                models_b[m_name],
                cur_b.test_inputs,
                cur_b.test_targets,
                shift_start=cur_b.shift_interval[0],
                shift_end=cur_b.shift_interval[1],
            )
            per_seed_results["domain_a"][m_name][f"seed_{seed}"] = rA
            per_seed_results["domain_b"][m_name][f"seed_{seed}"] = rB

        # Shifted streams evaluation
        for s_type in shift_types:
            for sev in severities:
                shift_in_a, shift_tgt_a = apply_unseen_shift(
                    cur_a.test_inputs,
                    cur_a.test_targets,
                    shift_type=s_type,
                    severity=sev,
                    shift_start=cur_a.shift_interval[0],
                    shift_end=cur_a.shift_interval[1],
                    seed=seed,
                )
                shift_in_b, shift_tgt_b = apply_unseen_shift(
                    cur_b.test_inputs,
                    cur_b.test_targets,
                    shift_type=s_type,
                    severity=sev,
                    shift_start=cur_b.shift_interval[0],
                    shift_end=cur_b.shift_interval[1],
                    seed=seed,
                )

                for m_name in model_names:
                    res_s_a = evaluate_shift_streaming_run(
                        models_a[m_name],
                        shift_in_a,
                        shift_tgt_a,
                        shift_start=cur_a.shift_interval[0],
                        shift_end=cur_a.shift_interval[1],
                    )
                    res_s_b = evaluate_shift_streaming_run(
                        models_b[m_name],
                        shift_in_b,
                        shift_tgt_b,
                        shift_start=cur_b.shift_interval[0],
                        shift_end=cur_b.shift_interval[1],
                    )
                    per_seed_shift_matrix[s_type][sev]["domain_a"][m_name][
                        f"seed_{seed}"
                    ] = res_s_a
                    per_seed_shift_matrix[s_type][sev]["domain_b"][m_name][
                        f"seed_{seed}"
                    ] = res_s_b

    # Aggregate results across seeds
    print("\n[Step 3] Aggregating Robustness Matrix across Seeds...")
    aggregated_results: dict[str, Any] = {"domain_a": {}, "domain_b": {}}
    for dom in ["domain_a", "domain_b"]:
        for m_name in model_names:
            seeds_list = [per_seed_results[dom][m_name][f"seed_{s}"] for s in seeds]
            rel_errs = [r["rel_error"] for r in seeds_list if not r["diverged"]]
            shift_errs = [r["e_shift"] for r in seeds_list if not r["diverged"]]
            latencies = [r["mean_latency_us"] for r in seeds_list]
            energies = [r["adaptation_energy"] for r in seeds_list]
            max_norms = [r["max_state_norm"] for r in seeds_list]

            aggregated_results[dom][m_name] = {
                "rel_error_mean": float(np.mean(rel_errs))
                if rel_errs
                else float("nan"),
                "rel_error_std": float(np.std(rel_errs)) if rel_errs else 0.0,
                "shift_error_mean": float(np.mean(shift_errs))
                if shift_errs
                else float("nan"),
                "shift_error_std": float(np.std(shift_errs)) if shift_errs else 0.0,
                "runtime_us_mean": float(np.mean(latencies)),
                "adaptation_energy_mean": float(np.mean(energies)),
                "max_state_norm_mean": float(np.mean(max_norms)),
                "persistent_state_bytes": seeds_list[0]["persistent_state_bytes"],
            }

    # Aggregated Shift Matrix (5 shifts x 3 severities x 2 domains)
    aggregated_shift_matrix: dict[str, Any] = {
        s_type: {sev: {"domain_a": {}, "domain_b": {}} for sev in severities}
        for s_type in shift_types
    }
    for s_type in shift_types:
        for sev in severities:
            for dom in ["domain_a", "domain_b"]:
                for m_name in model_names:
                    s_runs = [
                        per_seed_shift_matrix[s_type][sev][dom][m_name][f"seed_{s}"]
                        for s in seeds
                    ]
                    r_errs = [r["rel_error"] for r in s_runs if not r["diverged"]]
                    s_errs = [r["e_shift"] for r in s_runs if not r["diverged"]]
                    fp_rec = [r["first_passage_recovery"] for r in s_runs]
                    sus_rec = [r["sustained_recovery"] for r in s_runs]
                    cum_err = [r["cumulative_excess_error"] for r in s_runs]
                    energies = [r["adaptation_energy"] for r in s_runs]
                    max_norms = [r["max_state_norm"] for r in s_runs]
                    margins = [r["min_safety_margin"] for r in s_runs]

                    aggregated_shift_matrix[s_type][sev][dom][m_name] = {
                        "rel_error_mean": float(np.mean(r_errs))
                        if r_errs
                        else float("nan"),
                        "rel_error_std": float(np.std(r_errs)) if r_errs else 0.0,
                        "shift_error_mean": float(np.mean(s_errs))
                        if s_errs
                        else float("nan"),
                        "shift_error_std": float(np.std(s_errs)) if s_errs else 0.0,
                        "first_passage_recovery_mean": float(np.mean(fp_rec)),
                        "sustained_recovery_mean": float(np.mean(sus_rec)),
                        "cumulative_excess_error_mean": float(np.mean(cum_err)),
                        "adaptation_energy_mean": float(np.mean(energies)),
                        "max_state_norm_mean": float(np.mean(max_norms)),
                        "min_safety_margin_mean": float(np.mean(margins)),
                        "diverged_count": sum(1 for r in s_runs if r["diverged"]),
                    }

    # 3. State Reset Ablation across Shifts
    print("\n[Step 4] Running State Reset Ablation...")
    reset_ablation_results: dict[str, Any] = {"domain_a": {}, "domain_b": {}}
    model_safe_a = SafeAdaptiveDeltaPredictor(
        dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95, gamma=0.05
    )
    model_safe_b = SafeAdaptiveDeltaPredictor(
        dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95, gamma=0.05
    )

    for s_type in shift_types:
        reset_ablation_results["domain_a"][s_type] = {}
        reset_ablation_results["domain_b"][s_type] = {}
        for sev in severities:
            s_in_a, s_tgt_a = apply_unseen_shift(
                data_a.test_inputs, data_a.test_targets, s_type, sev, 40, 110, seed=42
            )
            s_in_b, s_tgt_b = apply_unseen_shift(
                data_b.test_inputs, data_b.test_targets, s_type, sev, 45, 115, seed=42
            )

            ab_a = evaluate_shift_reset_ablation(model_safe_a, s_in_a, s_tgt_a, 40, 110)
            ab_b = evaluate_shift_reset_ablation(model_safe_b, s_in_b, s_tgt_b, 45, 115)

            reset_ablation_results["domain_a"][s_type][sev] = {
                "continuous_rel_error": ab_a["continuous"]["rel_error"],
                "reset_rel_error": ab_a["reset"]["rel_error"],
                "delta_e_rel": ab_a["delta_e_rel"],
                "delta_cum_excess": ab_a["delta_cum_excess"],
                "continuous_advantage": ab_a["continuous_advantage"],
            }
            reset_ablation_results["domain_b"][s_type][sev] = {
                "continuous_rel_error": ab_b["continuous"]["rel_error"],
                "reset_rel_error": ab_b["reset"]["rel_error"],
                "delta_e_rel": ab_b["delta_e_rel"],
                "delta_cum_excess": ab_b["delta_cum_excess"],
                "continuous_advantage": ab_b["continuous_advantage"],
            }

    # 4. Retention Stress Multi-Regime Histories
    print("\n[Step 5] Running Retention Stress Histories...")
    retention_histories = ["A_to_B", "A_to_B_to_A", "A_to_B_to_C", "A_to_severeB_to_A"]
    retention_stress_results: dict[str, Any] = {}

    for h_type in retention_histories:
        stress_in, stress_tgt, bounds = generate_retention_stress_stream(
            data_a.test_inputs,
            data_a.test_targets,
            data_b.test_inputs,
            data_b.test_targets,
            history_type=h_type,
            seg_len=40,
        )
        m_cont = SafeAdaptiveDeltaPredictor(
            dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95, gamma=0.05
        )
        m_reset = SafeAdaptiveDeltaPredictor(
            dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95, gamma=0.05
        )

        run_cont = evaluate_shift_streaming_run(m_cont, stress_in, stress_tgt)

        # Reset at first boundary
        run_reset = evaluate_shift_streaming_run(
            m_reset, stress_in, stress_tgt, reset_at_step=bounds[0]
        )

        retention_stress_results[h_type] = {
            "bounds": bounds,
            "continuous_rel_error": run_cont["rel_error"],
            "reset_rel_error": run_reset["rel_error"],
            "continuous_step_errors": run_cont["step_errors"],
            "reset_step_errors": run_reset["step_errors"],
            "continuous_state_norms": run_cont["state_norm_history"],
            "reset_state_norms": run_reset["state_norm_history"],
            "continuous_retention": run_cont["retention_history"],
            "max_state_norm": run_cont["max_state_norm"],
            "adaptation_energy": run_cont["adaptation_energy"],
        }

    # 5. Failure Boundary Mapping at D=256
    print("\n[Step 6] Running Failure Boundary Mapping at D=256...")
    scaled_data = load_cross_domain_datasets(seed=42, resolution="medium")
    data_a_256 = scaled_data.domain_a
    data_b_256 = scaled_data.domain_b

    failure_boundary_results: dict[str, Any] = {
        "domain_a_oisst_256": map_failure_boundary_grid(
            data_a_256.test_inputs, data_a_256.test_targets, dim=256
        ),
        "domain_b_era5_256": map_failure_boundary_grid(
            data_b_256.test_inputs, data_b_256.test_targets, dim=256
        ),
    }

    # 6. Perturbation Robustness (0%, 1%, 5%, 10%)
    print("\n[Step 7] Running Perturbation Robustness Evaluations...")
    pert_results: dict[str, Any] = {
        "domain_a": {
            "SafeAdaptiveDelta": evaluate_perturbation_robustness(
                SafeAdaptiveDeltaPredictor(
                    dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95
                ),
                data_a.test_inputs,
                data_a.test_targets,
            ),
            "FixedDelta": evaluate_perturbation_robustness(
                FixedDeltaPredictor(dim=64, step_size=0.015),
                data_a.test_inputs,
                data_a.test_targets,
            ),
        },
        "domain_b": {
            "SafeAdaptiveDelta": evaluate_perturbation_robustness(
                SafeAdaptiveDeltaPredictor(
                    dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95
                ),
                data_b.test_inputs,
                data_b.test_targets,
            ),
            "FixedDelta": evaluate_perturbation_robustness(
                FixedDeltaPredictor(dim=64, step_size=0.015),
                data_b.test_inputs,
                data_b.test_targets,
            ),
        },
    }

    # 7. Permutation Equivariance Check
    print("\n[Step 8] Verifying Permutation Equivariance Under Perturbation...")
    equiv_a = check_permutation_equivariance(
        SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95),
        data_a.test_inputs,
    )
    equiv_b = check_permutation_equivariance(
        SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.015, rho=1.5, alpha_min=0.95),
        data_b.test_inputs,
    )
    print(
        f"  Domain A Equivariance Error: {equiv_a:.2e} | Domain B Equivariance Error: {equiv_b:.2e}"
    )

    # 8. Serialization of all 6 Required JSON Artifacts
    print("\n[Step 9] Serializing all Phase 16 JSON Artifacts...")

    config_artifact = {
        "domain_a": asdict(data_a.config),
        "domain_b": asdict(data_b.config),
        "primary_frozen_config": {
            "model": "SafeAdaptiveDelta",
            "eta_max": 0.015,
            "rho": 1.5,
            "alpha_min": 0.95,
            "gamma": 0.05,
        },
        "severity_scales": {
            k: {sk: asdict(sv) for sk, sv in v.items()}
            for k, v in SEVERITY_SCALES.items()
        },
        "sensitivity_grid": sensitivity_results,
        "seeds": list(seeds),
    }

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean(x) for x in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj) if math.isfinite(obj) else None
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return None
        return obj

    with open(out_path / "phase_16_results.json", "w") as f:
        json.dump(_clean(aggregated_results), f, indent=2)

    with open(out_path / "phase_16_per_seed.json", "w") as f:
        json.dump(_clean(per_seed_results), f, indent=2)

    with open(out_path / "phase_16_config.json", "w") as f:
        json.dump(_clean(config_artifact), f, indent=2)

    shift_matrix_artifact = {
        "aggregated_shift_matrix": aggregated_shift_matrix,
        "per_seed_shift_matrix": per_seed_shift_matrix,
        "perturbation_robustness": pert_results,
        "permutation_equivariance": {"domain_a": equiv_a, "domain_b": equiv_b},
    }
    with open(out_path / "phase_16_shift_matrix.json", "w") as f:
        json.dump(_clean(shift_matrix_artifact), f, indent=2)

    with open(out_path / "phase_16_safety_boundary.json", "w") as f:
        json.dump(_clean(failure_boundary_results), f, indent=2)

    retention_artifact = {
        "reset_ablation": reset_ablation_results,
        "retention_stress_histories": retention_stress_results,
    }
    with open(out_path / "phase_16_retention.json", "w") as f:
        json.dump(_clean(retention_artifact), f, indent=2)

    print("  Successfully written all 6 JSON artifacts:")
    print("    - docs/benchmarks/artifacts/phase_16/phase_16_results.json")
    print("    - docs/benchmarks/artifacts/phase_16/phase_16_per_seed.json")
    print("    - docs/benchmarks/artifacts/phase_16/phase_16_config.json")
    print("    - docs/benchmarks/artifacts/phase_16/phase_16_shift_matrix.json")
    print("    - docs/benchmarks/artifacts/phase_16/phase_16_safety_boundary.json")
    print("    - docs/benchmarks/artifacts/phase_16/phase_16_retention.json")

    return {
        "aggregated_results": aggregated_results,
        "shift_matrix": aggregated_shift_matrix,
        "failure_boundary": failure_boundary_results,
        "retention_stress": retention_stress_results,
    }
