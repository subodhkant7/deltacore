"""Phase 15: Adaptive Regime Transfer & Robustness Benchmark Suite.

Executes comprehensive cross-domain transfer benchmark comparing:
    - Domain A: NOAA OISST Sea Surface Temperature (slow thermal regime)
    - Domain B: ECMWF ERA5 2m Air Temperature (fast synoptic advection regime)

Evaluates:
    - Experiment A: Cross-Domain Parameter Transfer (A -> B, B -> A)
    - Experiment B: Zero-Retuning Comparison (Config 1, Config 2, Config 3 pooled)
    - Experiment C: Controlled Aggressiveness Sweep (alpha_min in {0.70, 0.85, 0.95}, rho in {1.5, 1.9})
    - Experiment D: Adaptation / Stability Pareto Frontier
    - Experiment E: Controlled Regime-Switch Transfer (OISST -> ERA5 -> OISST)
    - Experiment F: State Retention Under Transfer (Continuous vs Reset vs Retain-High vs Adaptive-Safe)
    - Experiment G: State Initialization Sensitivity (Zero, Random, Transferred State)
    - Experiment H: Safe vs Unsafe Failure Boundary at D=256
    - Strong Baselines across 5 seeds on both domains
    - Serialization of artifacts to docs/benchmarks/artifacts/phase_15/
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deltacore.streaming.atmospheric_benchmark import (
    evaluate_streaming_run,
    train_offline_model,
)
from deltacore.streaming.models import (
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    NonlinearOnlineRidgePredictor,
    OnlineRidgePredictor,
    PersistencePredictor,
    SafeAdaptiveDeltaPredictor,
    SpatialConvControl,
    StreamingPredictor,
)
from deltacore.streaming.regime_transfer import (
    SafeAdaptiveDeltaConfig,
    calibrate_validation_grid,
    evaluate_failure_boundary,
    evaluate_state_initialization,
    generate_regime_switch_stream,
    load_cross_domain_datasets,
)


def instantiate_domain_model_suite(
    dim: int, height: int, width: int, seed: int = 42
) -> dict[str, StreamingPredictor]:
    """Instantiate standard 7-model matrix for a given domain dimension."""
    return {
        "Persistence": PersistencePredictor(dim=dim),
        "FrozenLinear": FrozenLinearPredictor(dim=dim),
        "OnlineRidge": OnlineRidgePredictor(dim=dim, lam=0.98),
        "NonlinearOnlineRidge": NonlinearOnlineRidgePredictor(
            dim=dim, rff_dim=32, seed=seed
        ),
        "SpatialConv": SpatialConvControl(height=height, width=width, channels=1),
        "FixedDelta": FixedDeltaPredictor(dim=dim, step_size=0.008),
        "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(
            dim=dim, eta_max=0.008, rho=1.90
        ),
    }


def run_phase_15_benchmark(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 15 Cross-Domain Transfer Benchmark."""
    if output_dir is None:
        out_path = Path("docs/benchmarks/artifacts/phase_15")
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DELTACORE PHASE 15: ADAPTIVE REGIME TRANSFER & ROBUSTNESS BENCHMARK")
    print("Domains: Domain A (OISST SST) <---> Domain B (ERA5 T2m)")
    print("=" * 88)

    # 1. Load Datasets for Seed 42 (Base Resolution D=64)
    print("\n[Step 1] Loading base datasets (seed 42)...")
    base_data = load_cross_domain_datasets(seed=42, resolution="small")
    data_a = base_data.domain_a
    data_b = base_data.domain_b

    # 2. Validation Calibration Sweep (Experiment B & C validation phase)
    print("\n[Step 2] Executing validation grid calibration...")
    val_calib = calibrate_validation_grid(
        val_inputs_a=data_a.val_inputs,
        val_targets_a=data_a.val_targets,
        val_inputs_b=data_b.val_inputs,
        val_targets_b=data_b.val_targets,
        dim=64,
    )
    best_cfg_a = SafeAdaptiveDeltaConfig(**val_calib["best_domain_a"]["config"])
    best_cfg_b = SafeAdaptiveDeltaConfig(**val_calib["best_domain_b"]["config"])
    best_cfg_pooled = SafeAdaptiveDeltaConfig(**val_calib["best_pooled"]["config"])

    print(
        f"  Best Domain A (OISST Val): eta={best_cfg_a.eta_max}, rho={best_cfg_a.rho}, a_min={best_cfg_a.alpha_min} (ValErr: {val_calib['best_domain_a']['val_error']:.4f})"
    )
    print(
        f"  Best Domain B (ERA5 Val):  eta={best_cfg_b.eta_max}, rho={best_cfg_b.rho}, a_min={best_cfg_b.alpha_min} (ValErr: {val_calib['best_domain_b']['val_error']:.4f})"
    )
    print(
        f"  Best Pooled Val:           eta={best_cfg_pooled.eta_max}, rho={best_cfg_pooled.rho}, a_min={best_cfg_pooled.alpha_min} (ValErr: {val_calib['best_pooled']['val_error']:.4f})"
    )

    # 3. Experiment A & B: Parameter Transfer & Zero-Retuning Evaluation
    print(
        "\n[Step 3] Running Experiment A & B (Cross-Domain Parameter Transfer & Zero-Retuning)..."
    )
    cfg_frozen13 = SafeAdaptiveDeltaConfig(
        eta_max=0.008, rho=1.90, alpha_min=0.85, gamma=0.05
    )

    transfer_results: dict[str, Any] = {
        "validation_calibration": val_calib,
        "configs": {
            "config_1_frozen13": asdict(cfg_frozen13),
            "config_2_domain_b_calibrated": asdict(best_cfg_b),
            "config_3_pooled_calibrated": asdict(best_cfg_pooled),
            "domain_a_calibrated": asdict(best_cfg_a),
        },
        "test_evaluations": {},
    }

    eval_configs = {
        "Config1_Frozen13": cfg_frozen13,
        "Config2_DomainB_Calibrated": best_cfg_b,
        "Config3_Pooled": best_cfg_pooled,
        "DomainA_Calibrated": best_cfg_a,
    }

    for c_name, c_cfg in eval_configs.items():
        # Evaluate on Domain A Test
        mod_a = c_cfg.instantiate(dim=64)
        res_a = evaluate_streaming_run(
            mod_a,
            data_a.test_inputs,
            data_a.test_targets,
            shift_start=data_a.shift_interval[0],
            shift_end=data_a.shift_interval[1],
        )

        # Evaluate on Domain B Test
        mod_b = c_cfg.instantiate(dim=64)
        res_b = evaluate_streaming_run(
            mod_b,
            data_b.test_inputs,
            data_b.test_targets,
            shift_start=data_b.shift_interval[0],
            shift_end=data_b.shift_interval[1],
        )

        transfer_results["test_evaluations"][c_name] = {
            "domain_a_oisst": {
                "rel_error": res_a["rel_error"],
                "e_shift": res_a["e_shift"],
                "first_passage_recovery": res_a["first_passage_recovery"],
                "sustained_recovery": res_a["sustained_recovery"],
                "max_state_norm": max(res_a["state_norms"])
                if res_a.get("state_norms")
                else 0.0,
                "adaptation_energy": res_a["adaptation_energy"],
                "runtime_us": res_a.get("runtime_us_per_token", 0.0),
            },
            "domain_b_era5": {
                "rel_error": res_b["rel_error"],
                "e_shift": res_b["e_shift"],
                "first_passage_recovery": res_b["first_passage_recovery"],
                "sustained_recovery": res_b["sustained_recovery"],
                "max_state_norm": max(res_b["state_norms"])
                if res_b.get("state_norms")
                else 0.0,
                "adaptation_energy": res_b["adaptation_energy"],
                "runtime_us": res_b.get("runtime_us_per_token", 0.0),
            },
        }
        print(
            f"  {c_name:<28} | A (OISST) Err: {res_a['rel_error']:.4f} ShiftErr: {res_a['e_shift']:.4f} | "
            f"B (ERA5) Err: {res_b['rel_error']:.4f} ShiftErr: {res_b['e_shift']:.4f}"
        )

    # 4. Multi-Seed Baselines Evaluation across both domains (Seeds 42..46)
    print(
        "\n[Step 4] Running 7-Model Primary Matrix across 5 Seeds on Domain A and Domain B..."
    )
    model_names = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "SpatialConv",
        "FixedDelta",
        "SafeAdaptiveDelta",
    ]

    per_seed_results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "domain_a": {m: {} for m in model_names},
        "domain_b": {m: {} for m in model_names},
    }

    for s in seeds:
        print(f"  --- Running Seed {s} ---")
        cur_datasets = load_cross_domain_datasets(seed=s, resolution="small")
        cur_a = cur_datasets.domain_a
        cur_b = cur_datasets.domain_b

        # Domain A models
        models_a = instantiate_domain_model_suite(
            cur_a.config.dim, height=8, width=8, seed=s
        )
        train_offline_model(
            models_a["FrozenLinear"],
            cur_a.train_inputs,
            cur_a.train_targets,
            seed=s,
        )
        train_offline_model(
            models_a["SpatialConv"],
            cur_a.train_inputs,
            cur_a.train_targets,
            seed=s,
        )

        for m_name in model_names:
            m = models_a[m_name]
            res = evaluate_streaming_run(
                model=m,
                inputs=cur_a.test_inputs,
                targets=cur_a.test_targets,
                shift_start=cur_a.shift_interval[0],
                shift_end=cur_a.shift_interval[1],
            )
            per_seed_results["domain_a"][m_name][f"seed_{s}"] = res

        # Domain B models
        models_b = instantiate_domain_model_suite(
            cur_b.config.dim, height=8, width=8, seed=s
        )
        train_offline_model(
            models_b["FrozenLinear"],
            cur_b.train_inputs,
            cur_b.train_targets,
            seed=s,
        )
        train_offline_model(
            models_b["SpatialConv"],
            cur_b.train_inputs,
            cur_b.train_targets,
            seed=s,
        )

        for m_name in model_names:
            m = models_b[m_name]
            res = evaluate_streaming_run(
                model=m,
                inputs=cur_b.test_inputs,
                targets=cur_b.test_targets,
                shift_start=cur_b.shift_interval[0],
                shift_end=cur_b.shift_interval[1],
            )
            per_seed_results["domain_b"][m_name][f"seed_{s}"] = res

    # Compute aggregate stats across 5 seeds
    aggregated_results: dict[str, dict[str, dict[str, float]]] = {
        "domain_a": {},
        "domain_b": {},
    }

    for dom in ["domain_a", "domain_b"]:
        for m_name in model_names:
            seed_dict = per_seed_results[dom][m_name]
            rel_errs = [v["rel_error"] for v in seed_dict.values()]
            shift_errs = [v["e_shift"] for v in seed_dict.values()]
            runtimes = [v.get("runtime_us_per_token", 0.0) for v in seed_dict.values()]
            energies = [v["adaptation_energy"] for v in seed_dict.values()]
            max_norms = [
                max(v["state_norms"]) if v.get("state_norms") else 0.0
                for v in seed_dict.values()
            ]
            sample_model = models_a[m_name] if dom == "domain_a" else models_b[m_name]

            aggregated_results[dom][m_name] = {
                "rel_error_mean": float(np.mean(rel_errs)),
                "rel_error_std": float(np.std(rel_errs)),
                "shift_error_mean": float(np.mean(shift_errs)),
                "shift_error_std": float(np.std(shift_errs)),
                "runtime_us_mean": float(np.mean(runtimes)),
                "adaptation_energy_mean": float(np.mean(energies)),
                "max_state_norm_mean": float(np.mean(max_norms)),
                "total_params": sum(p.numel() for p in sample_model.parameters()),
                "persistent_state_bytes": sample_model.get_state_memory_bytes(),
            }

    # 5. Experiment C & D: Controlled Aggressiveness Sweep & Pareto Frontier
    print("\n[Step 5] Running Experiment C & D (Aggressiveness & Pareto Analysis)...")
    sweep_grid = [
        SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.5, alpha_min=0.70),
        SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.5, alpha_min=0.85),
        SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.5, alpha_min=0.95),
        SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.9, alpha_min=0.70),
        SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.9, alpha_min=0.85),
        SafeAdaptiveDeltaConfig(eta_max=0.008, rho=1.9, alpha_min=0.95),
        SafeAdaptiveDeltaConfig(eta_max=0.015, rho=1.9, alpha_min=0.85),
        SafeAdaptiveDeltaConfig(eta_max=0.015, rho=1.9, alpha_min=0.95),
    ]

    safety_pareto_results: dict[str, Any] = {
        "configurations": [],
        "domain_a_pareto": [],
        "domain_b_pareto": [],
    }

    for cfg in sweep_grid:
        cfg_dict = asdict(cfg)
        safety_pareto_results["configurations"].append(cfg_dict)

        # Domain A evaluation
        mA = cfg.instantiate(dim=64)
        resA = evaluate_streaming_run(
            mA,
            data_a.test_inputs,
            data_a.test_targets,
            shift_start=data_a.shift_interval[0],
            shift_end=data_a.shift_interval[1],
        )

        # Domain B evaluation
        mB = cfg.instantiate(dim=64)
        resB = evaluate_streaming_run(
            mB,
            data_b.test_inputs,
            data_b.test_targets,
            shift_start=data_b.shift_interval[0],
            shift_end=data_b.shift_interval[1],
        )

        normA = max(resA["state_norms"]) if resA.get("state_norms") else 0.0
        normB = max(resB["state_norms"]) if resB.get("state_norms") else 0.0
        riskA = normA * (cfg.eta_max / cfg.rho)
        riskB = normB * (cfg.eta_max / cfg.rho)

        entryA = {
            "config": cfg_dict,
            "rel_error": resA["rel_error"],
            "e_shift": resA["e_shift"],
            "recovery_time": resA["sustained_recovery"],
            "adaptation_energy": resA["adaptation_energy"],
            "max_state_norm": normA,
            "numerical_risk": riskA,
        }
        entryB = {
            "config": cfg_dict,
            "rel_error": resB["rel_error"],
            "e_shift": resB["e_shift"],
            "recovery_time": resB["sustained_recovery"],
            "adaptation_energy": resB["adaptation_energy"],
            "max_state_norm": normB,
            "numerical_risk": riskB,
        }

        safety_pareto_results["domain_a_pareto"].append(entryA)
        safety_pareto_results["domain_b_pareto"].append(entryB)

    # 6. Experiment E: Controlled Regime-Switch Transfer Stream
    print("\n[Step 6] Running Experiment E (Regime-Switch Transfer Stream)...")
    mix_in, mix_tgt, boundaries = generate_regime_switch_stream(
        test_inputs_a=data_a.test_inputs,
        test_targets_a=data_a.test_targets,
        test_inputs_b=data_b.test_inputs,
        test_targets_b=data_b.test_targets,
        segment_len=70,
    )

    regime_switch_results: dict[str, Any] = {
        "stream_boundaries": boundaries,
        "segment_length": 70,
        "total_timesteps": len(mix_in),
        "models": {},
    }

    # Evaluate SafeAdaptiveDelta (Continuous vs Reset) on Regime-Switch
    for reset_mode in ["continuous", "reset_at_boundaries"]:
        m_mix = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
        reset_steps = list(boundaries) if reset_mode == "reset_at_boundaries" else []

        step_errs: list[float] = []
        state_norms: list[float] = []
        update_energies: list[float] = []

        m_mix.reset_state()
        for t in range(len(mix_in)):
            if t in reset_steps:
                m_mix.reset_state()

            xt = mix_in[t]
            yt = mix_tgt[t]
            pred = m_mix.predict_step(xt)
            err = float(
                torch.linalg.norm(yt - pred).item()
                / max(torch.linalg.norm(yt).item(), 1e-8)
            )
            step_errs.append(err)

            m_mix.adapt_step(xt, yt)
            state_norms.append(m_mix.get_state_norm())
            update_energies.append(m_mix.last_update_norm**2)

        # Split phase errors
        p1_err = float(np.mean(step_errs[: boundaries[0]]))
        p2_err = float(np.mean(step_errs[boundaries[0] : boundaries[1]]))
        p3_err = float(np.mean(step_errs[boundaries[1] :]))
        overall_err = float(np.mean(step_errs))

        # Boundary transient error (5 steps right after boundary)
        transient_b1 = float(np.mean(step_errs[boundaries[0] : boundaries[0] + 5]))
        transient_b2 = float(np.mean(step_errs[boundaries[1] : boundaries[1] + 5]))

        regime_switch_results["models"][reset_mode] = {
            "overall_error": overall_err,
            "phase1_oisst_error": p1_err,
            "phase2_era5_error": p2_err,
            "phase3_return_oisst_error": p3_err,
            "boundary_1_transient": transient_b1,
            "boundary_2_transient": transient_b2,
            "total_adaptation_energy": float(np.sum(update_energies)),
            "step_errors": step_errs,
            "state_norms": state_norms,
        }
        print(
            f"  MixedStream {reset_mode:<22} | P1: {p1_err:.4f} | P2: {p2_err:.4f} | P3: {p3_err:.4f} | "
            f"TransB1: {transient_b1:.4f} | TransB2: {transient_b2:.4f}"
        )

    # 7. Experiment F: State Retention Under Transfer Ablation
    print("\n[Step 7] Running Experiment F (State Retention Under Transfer)...")
    retention_experiments: dict[str, Any] = {
        "domain_a": {},
        "domain_b": {},
        "regime_switch": regime_switch_results["models"],
    }

    # Evaluate on Domain A
    m_cont_a = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    m_reset_a = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    m_high_a = SafeAdaptiveDeltaPredictor(
        dim=64, eta_max=0.008, rho=1.90, alpha_min=0.99, gamma=0.0
    )

    r_cont_a = evaluate_streaming_run(
        m_cont_a,
        data_a.test_inputs,
        data_a.test_targets,
        shift_start=data_a.shift_interval[0],
        shift_end=data_a.shift_interval[1],
        reset_at_step=None,
    )
    r_reset_a = evaluate_streaming_run(
        m_reset_a,
        data_a.test_inputs,
        data_a.test_targets,
        shift_start=data_a.shift_interval[0],
        shift_end=data_a.shift_interval[1],
        reset_at_step=data_a.shift_interval[0],
    )
    r_high_a = evaluate_streaming_run(
        m_high_a,
        data_a.test_inputs,
        data_a.test_targets,
        shift_start=data_a.shift_interval[0],
        shift_end=data_a.shift_interval[1],
    )

    retention_experiments["domain_a"] = {
        "continuous": {
            "rel_error": r_cont_a["rel_error"],
            "e_shift": r_cont_a["e_shift"],
            "cumulative_excess_error": r_cont_a["cumulative_excess_error"],
            "adaptation_energy": r_cont_a["adaptation_energy"],
        },
        "reset": {
            "rel_error": r_reset_a["rel_error"],
            "e_shift": r_reset_a["e_shift"],
            "cumulative_excess_error": r_reset_a["cumulative_excess_error"],
            "adaptation_energy": r_reset_a["adaptation_energy"],
        },
        "retain_high": {
            "rel_error": r_high_a["rel_error"],
            "e_shift": r_high_a["e_shift"],
            "cumulative_excess_error": r_high_a["cumulative_excess_error"],
            "adaptation_energy": r_high_a["adaptation_energy"],
        },
    }

    # Evaluate on Domain B
    m_cont_b = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    m_reset_b = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    m_high_b = SafeAdaptiveDeltaPredictor(
        dim=64, eta_max=0.008, rho=1.90, alpha_min=0.99, gamma=0.0
    )

    r_cont_b = evaluate_streaming_run(
        m_cont_b,
        data_b.test_inputs,
        data_b.test_targets,
        shift_start=data_b.shift_interval[0],
        shift_end=data_b.shift_interval[1],
        reset_at_step=None,
    )
    r_reset_b = evaluate_streaming_run(
        m_reset_b,
        data_b.test_inputs,
        data_b.test_targets,
        shift_start=data_b.shift_interval[0],
        shift_end=data_b.shift_interval[1],
        reset_at_step=data_b.shift_interval[0],
    )
    r_high_b = evaluate_streaming_run(
        m_high_b,
        data_b.test_inputs,
        data_b.test_targets,
        shift_start=data_b.shift_interval[0],
        shift_end=data_b.shift_interval[1],
    )

    retention_experiments["domain_b"] = {
        "continuous": {
            "rel_error": r_cont_b["rel_error"],
            "e_shift": r_cont_b["e_shift"],
            "cumulative_excess_error": r_cont_b["cumulative_excess_error"],
            "adaptation_energy": r_cont_b["adaptation_energy"],
        },
        "reset": {
            "rel_error": r_reset_b["rel_error"],
            "e_shift": r_reset_b["e_shift"],
            "cumulative_excess_error": r_reset_b["cumulative_excess_error"],
            "adaptation_energy": r_reset_b["adaptation_energy"],
        },
        "retain_high": {
            "rel_error": r_high_b["rel_error"],
            "e_shift": r_high_b["e_shift"],
            "cumulative_excess_error": r_high_b["cumulative_excess_error"],
            "adaptation_energy": r_high_b["adaptation_energy"],
        },
    }

    # 8. Experiment G: State Initialization Sensitivity (Zero, Random, Transferred)
    print("\n[Step 8] Running Experiment G (State Initialization Sensitivity)...")
    # First get final states from baseline streaming runs
    m_seed_a = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    _ = evaluate_streaming_run(m_seed_a, data_a.test_inputs, data_a.test_targets)
    final_state_a = m_seed_a.M.data.clone()

    m_seed_b = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    _ = evaluate_streaming_run(m_seed_b, data_b.test_inputs, data_b.test_targets)
    final_state_b = m_seed_b.M.data.clone()

    init_sensitivity: dict[str, Any] = {
        "eval_on_domain_b": {},
        "eval_on_domain_a": {},
    }

    # Evaluated on Domain B (ERA5)
    model_eval_b = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    init_b_zero = evaluate_state_initialization(
        model=model_eval_b,
        inputs=data_b.test_inputs,
        targets=data_b.test_targets,
        initial_state=None,
    )
    init_b_rand = evaluate_state_initialization(
        model=model_eval_b,
        inputs=data_b.test_inputs,
        targets=data_b.test_targets,
        noise_std=0.01,
        seed=42,
    )
    init_b_from_a = evaluate_state_initialization(
        model=model_eval_b,
        inputs=data_b.test_inputs,
        targets=data_b.test_targets,
        initial_state=final_state_a,
    )

    init_sensitivity["eval_on_domain_b"] = {
        "zero_state": {
            "rel_error": init_b_zero["rel_error"],
            "early_rel_error_first_10": init_b_zero["early_rel_error_first_10"],
        },
        "small_random_state": {
            "rel_error": init_b_rand["rel_error"],
            "early_rel_error_first_10": init_b_rand["early_rel_error_first_10"],
        },
        "transferred_from_domain_a": {
            "rel_error": init_b_from_a["rel_error"],
            "early_rel_error_first_10": init_b_from_a["early_rel_error_first_10"],
        },
    }

    # Evaluated on Domain A (OISST)
    model_eval_a = SafeAdaptiveDeltaPredictor(dim=64, eta_max=0.008, rho=1.90)
    init_a_zero = evaluate_state_initialization(
        model=model_eval_a,
        inputs=data_a.test_inputs,
        targets=data_a.test_targets,
        initial_state=None,
    )
    init_a_rand = evaluate_state_initialization(
        model=model_eval_a,
        inputs=data_a.test_inputs,
        targets=data_a.test_targets,
        noise_std=0.01,
        seed=42,
    )
    init_a_from_b = evaluate_state_initialization(
        model=model_eval_a,
        inputs=data_a.test_inputs,
        targets=data_a.test_targets,
        initial_state=final_state_b,
    )

    init_sensitivity["eval_on_domain_a"] = {
        "zero_state": {
            "rel_error": init_a_zero["rel_error"],
            "early_rel_error_first_10": init_a_zero["early_rel_error_first_10"],
        },
        "small_random_state": {
            "rel_error": init_a_rand["rel_error"],
            "early_rel_error_first_10": init_a_rand["early_rel_error_first_10"],
        },
        "transferred_from_domain_b": {
            "rel_error": init_a_from_b["rel_error"],
            "early_rel_error_first_10": init_a_from_b["early_rel_error_first_10"],
        },
    }

    print(
        f"  Eval on B: Zero Err: {init_b_zero['rel_error']:.4f} (Early: {init_b_zero['early_rel_error_first_10']:.4f}) | "
        f"Transferred from A: {init_b_from_a['rel_error']:.4f} (Early: {init_b_from_a['early_rel_error_first_10']:.4f})"
    )
    print(
        f"  Eval on A: Zero Err: {init_a_zero['rel_error']:.4f} (Early: {init_a_zero['early_rel_error_first_10']:.4f}) | "
        f"Transferred from B: {init_a_from_b['rel_error']:.4f} (Early: {init_a_from_b['early_rel_error_first_10']:.4f})"
    )

    # 9. Experiment H: Safe vs Unsafe Failure Boundary at D=256
    print(
        "\n[Step 9] Running Experiment H (Safe vs Unsafe Failure Boundary at D=256)..."
    )
    scaled_data = load_cross_domain_datasets(seed=42, resolution="medium")
    data_a_256 = scaled_data.domain_a
    data_b_256 = scaled_data.domain_b

    failure_boundary_results: dict[str, Any] = {
        "domain_a_oisst_256": {
            "FixedDelta": evaluate_failure_boundary(
                dim=256,
                inputs=data_a_256.test_inputs,
                targets=data_a_256.test_targets,
                is_safe=False,
                step_size=0.008,
            ),
            "SafeAdaptiveDelta": evaluate_failure_boundary(
                dim=256,
                inputs=data_a_256.test_inputs,
                targets=data_a_256.test_targets,
                is_safe=True,
                step_size=0.008,
                rho=1.90,
            ),
        },
        "domain_b_era5_256": {
            "FixedDelta": evaluate_failure_boundary(
                dim=256,
                inputs=data_b_256.test_inputs,
                targets=data_b_256.test_targets,
                is_safe=False,
                step_size=0.008,
            ),
            "SafeAdaptiveDelta": evaluate_failure_boundary(
                dim=256,
                inputs=data_b_256.test_inputs,
                targets=data_b_256.test_targets,
                is_safe=True,
                step_size=0.008,
                rho=1.90,
            ),
        },
    }

    for dom_k, dom_res in failure_boundary_results.items():
        fd = dom_res["FixedDelta"]
        sd = dom_res["SafeAdaptiveDelta"]
        print(
            f"  {dom_k:<20} | FixedDelta diverged: {fd['diverged']} (time: {fd['time_to_nonfinite']}) | "
            f"SafeAdaptiveDelta diverged: {sd['diverged']} (RelErr: {sd['rel_error']:.4f}, MinMargin: {sd['min_safety_margin']:.4f})"
        )

    # 10. Multi-Resolution Resource Scaling Data
    scaling_resource_results = {
        "small_D64": {
            "D": 64,
            "domain_a_runtimes_us": {
                m: aggregated_results["domain_a"][m]["runtime_us_mean"]
                for m in model_names
            },
            "domain_b_runtimes_us": {
                m: aggregated_results["domain_b"][m]["runtime_us_mean"]
                for m in model_names
            },
            "state_memory_bytes": {
                m: aggregated_results["domain_a"][m]["persistent_state_bytes"]
                for m in model_names
            },
        },
        "medium_D256": {
            "D": 256,
            "SafeAdaptiveDelta": {
                "domain_a_rel_error": failure_boundary_results["domain_a_oisst_256"][
                    "SafeAdaptiveDelta"
                ]["rel_error"],
                "domain_b_rel_error": failure_boundary_results["domain_b_era5_256"][
                    "SafeAdaptiveDelta"
                ]["rel_error"],
                "persistent_state_bytes": 256 * 256 * 4,
            },
            "OnlineRidge": {
                "persistent_state_bytes": 2 * 256 * 256 * 4,
            },
        },
    }

    # 11. Serialization of all 6 Required Artifacts
    print("\n[Step 11] Serializing all Phase 15 JSON Artifacts...")

    config_artifact = {
        "domain_a": asdict(data_a.config),
        "domain_b": asdict(data_b.config),
        "calibration_sweep": val_calib,
        "seeds": list(seeds),
    }

    # Clean per_seed_results for JSON serialization
    def _clean_dict(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean_dict(x) for x in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj) if math.isfinite(obj) else None
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return None  # tensors not serialized to primary JSON
        return obj

    with open(out_path / "phase_15_results.json", "w") as f:
        json.dump(_clean_dict(aggregated_results), f, indent=2)

    with open(out_path / "phase_15_per_seed.json", "w") as f:
        json.dump(_clean_dict(per_seed_results), f, indent=2)

    with open(out_path / "phase_15_config.json", "w") as f:
        json.dump(_clean_dict(config_artifact), f, indent=2)

    transfer_artifact = {
        "experiment_a_parameter_transfer": transfer_results,
        "experiment_e_regime_switch": regime_switch_results,
        "experiment_g_state_initialization": init_sensitivity,
    }
    with open(out_path / "phase_15_transfer.json", "w") as f:
        json.dump(_clean_dict(transfer_artifact), f, indent=2)

    safety_artifact = {
        "experiment_c_sweep": safety_pareto_results["configurations"],
        "experiment_d_pareto": {
            "domain_a": safety_pareto_results["domain_a_pareto"],
            "domain_b": safety_pareto_results["domain_b_pareto"],
        },
        "experiment_h_failure_boundary": failure_boundary_results,
        "scaling_resources": scaling_resource_results,
    }
    with open(out_path / "phase_15_safety.json", "w") as f:
        json.dump(_clean_dict(safety_artifact), f, indent=2)

    with open(out_path / "phase_15_retention.json", "w") as f:
        json.dump(_clean_dict(retention_experiments), f, indent=2)

    print("  Successfully written all 6 JSON artifacts:")
    print("    - docs/benchmarks/artifacts/phase_15/phase_15_results.json")
    print("    - docs/benchmarks/artifacts/phase_15/phase_15_per_seed.json")
    print("    - docs/benchmarks/artifacts/phase_15/phase_15_config.json")
    print("    - docs/benchmarks/artifacts/phase_15/phase_15_transfer.json")
    print("    - docs/benchmarks/artifacts/phase_15/phase_15_safety.json")
    print("    - docs/benchmarks/artifacts/phase_15/phase_15_retention.json")

    return {
        "aggregated_results": aggregated_results,
        "transfer_results": transfer_results,
        "safety_pareto_results": safety_pareto_results,
        "regime_switch_results": regime_switch_results,
        "retention_experiments": retention_experiments,
        "init_sensitivity": init_sensitivity,
        "failure_boundary_results": failure_boundary_results,
    }
