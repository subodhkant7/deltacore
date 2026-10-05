"""Phase 13: Real-World Spatio-Temporal Adaptive State Benchmark Suite.

Executes full empirical evaluation of DeltaCore vs classical and neural baselines
on the NOAA Equatorial Pacific SST dataset.

Evaluates:
    - Main model matrix across 5 deterministic seeds.
    - Parameter immutability audit (SHA-256 hash checks).
    - Shift-aware evaluation & recovery under the 2015-2016 Super El Niño event.
    - Causal controls: State Reset Control & Spatial Permutation Control.
    - Multi-resolution scaling: D=64 (8x8) vs D=256 (16x16).
    - Resource accounting: parameter, state, activation memory & runtime per token.
    - Serialization of reproducibility artifacts to docs/benchmarks/artifacts/phase_13/.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from deltacore.streaming.models import (
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    GRUPredictor,
    NonlinearOnlineRidgePredictor,
    OnlineRidgePredictor,
    PersistencePredictor,
    SafeAdaptiveDeltaPredictor,
    SelectiveRetentionPredictor,
    SpatialConvControl,
    StreamingPredictor,
    compute_parameter_hash,
)
from deltacore.streaming.real_spatiotemporal import (
    RealSpatioTemporalConfig,
    compute_field_anomaly_correlation,
    compute_leakage_audit,
    compute_relative_frobenius_error,
    compute_spatial_gradient_error,
    generate_real_spatiotemporal_dataset,
    restore_spatial_field,
)


def train_offline_model(
    model: StreamingPredictor,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-2,
    seed: int = 42,
) -> None:
    """Train static baseline offline strictly on training data."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_samples = train_inputs.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, 16):
            batch_idx = perm[i : i + 16]
            bx = train_inputs[batch_idx]
            by = train_targets[batch_idx]

            opt.zero_grad()
            preds = [model.predict_step(bx[k]) for k in range(len(bx))]
            batch_pred = torch.stack(preds)
            loss = nn.functional.mse_loss(batch_pred, by)
            loss.backward()
            opt.step()

    # Strictly freeze parameters after offline training
    for p in model.parameters():
        p.requires_grad = False
    model.reset_state()


def evaluate_streaming_run(
    model: StreamingPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    shift_start: int = 40,
    shift_end: int = 110,
    reset_at_step: int | None = None,
    track_fields: bool = True,
    height: int = 8,
    width: int = 8,
) -> dict[str, Any]:
    """Execute online streaming evaluation on a test sequence."""
    model.reset_state()
    h_before = compute_parameter_hash(model)

    t_steps = inputs.shape[0]
    preds: list[torch.Tensor] = []
    step_errors: list[float] = []
    state_norms: list[float] = []
    update_norms: list[float] = []
    step_latencies: list[float] = []

    for t in range(t_steps):
        # State reset control intervention at regime shift onset
        if reset_at_step is not None and t == reset_at_step:
            model.reset_state()

        xt = inputs[t]
        yt = targets[t]

        t0 = time.perf_counter()
        y_hat = model.predict_step(xt)
        t_eval = time.perf_counter() - t0
        step_latencies.append(t_eval * 1e6)

        preds.append(y_hat.detach().clone())
        err_rel_t = compute_relative_frobenius_error(yt, y_hat)
        step_errors.append(err_rel_t)

        # State adaptation
        model.adapt_step(xt, yt)
        state_norms.append(model.get_state_norm())
        update_norms.append(model.last_update_norm)

    h_after = compute_parameter_hash(model)
    assert h_before == h_after, (
        f"Parameter mutation violation in {model.name}! Hashes differ."
    )

    preds_tensor = torch.stack(preds)
    overall_rel_err = compute_relative_frobenius_error(targets, preds_tensor)
    overall_mae = float(torch.mean(torch.abs(targets - preds_tensor)).item())
    overall_rmse = float(torch.sqrt(torch.mean((targets - preds_tensor) ** 2)).item())

    # Pre-shift error (t in [0, shift_start))
    pre_shift_errors = step_errors[:shift_start]
    e_pre = float(np.mean(pre_shift_errors)) if pre_shift_errors else 0.0

    # Immediate post-shift degradation E_{post, 0}
    e_post_0 = step_errors[shift_start] if shift_start < t_steps else 0.0

    # Shift period error (t in [shift_start, shift_end])
    shift_errors = step_errors[shift_start : min(shift_end, t_steps)]
    e_shift = float(np.mean(shift_errors)) if shift_errors else 0.0

    # Recovery metrics
    # First-passage recovery: steps after shift_start until error returns to <= e_pre + 0.05
    recovery_threshold = e_pre + 0.05
    fp_recovery = None
    for step_offset, err in enumerate(step_errors[shift_start:]):
        if err <= recovery_threshold:
            fp_recovery = step_offset
            break
    if fp_recovery is None:
        fp_recovery = t_steps - shift_start

    # Sustained recovery: 5 consecutive steps below threshold
    sustained_recovery = None
    window_sz = 5
    for step_offset in range(len(step_errors[shift_start:]) - window_sz):
        window = step_errors[
            shift_start + step_offset : shift_start + step_offset + window_sz
        ]
        if all(e <= recovery_threshold for e in window):
            sustained_recovery = step_offset
            break
    if sustained_recovery is None:
        sustained_recovery = t_steps - shift_start

    # Cumulative excess error during shift
    cum_excess_error = float(
        sum(max(0.0, err - e_pre) for err in step_errors[shift_start:shift_end])
    )

    # Structure-aware spatial metrics
    fac_scores = []
    sge_scores = []
    if track_fields and height * width == inputs.shape[-1]:
        spatial_targets = restore_spatial_field(targets, height, width, 1)
        spatial_preds = restore_spatial_field(preds_tensor, height, width, 1)
        for t in range(t_steps):
            fac_scores.append(
                compute_field_anomaly_correlation(spatial_targets[t], spatial_preds[t])
            )
            sge_scores.append(
                compute_spatial_gradient_error(spatial_targets[t], spatial_preds[t])
            )

    mean_fac = float(np.mean(fac_scores)) if fac_scores else 0.0
    mean_sge = float(np.mean(sge_scores)) if sge_scores else 0.0

    # Resource accounting
    total_params, trainable_params = model.get_param_count()
    memory_breakdown = model.get_memory_breakdown()
    adaptation_energy = float(sum(u**2 for u in update_norms))

    return {
        "model_name": model.name,
        "rel_error": overall_rel_err,
        "mae": overall_mae,
        "rmse": overall_rmse,
        "e_pre": e_pre,
        "e_post_0": e_post_0,
        "e_shift": e_shift,
        "first_passage_recovery": fp_recovery,
        "sustained_recovery": sustained_recovery,
        "cumulative_excess_error": cum_excess_error,
        "fac": mean_fac,
        "sge": mean_sge,
        "runtime_us_per_token": float(np.mean(step_latencies)),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "persistent_state_bytes": memory_breakdown["persistent_state_bytes"],
        "parameter_bytes": memory_breakdown["parameter_bytes"],
        "temporary_activation_bytes": memory_breakdown["temporary_activation_bytes"],
        "adaptation_energy": adaptation_energy,
        "final_state_norm": state_norms[-1] if state_norms else 0.0,
        "step_errors": step_errors,
        "state_norms": state_norms,
        "update_norms": update_norms,
    }


def instantiate_model_suite(
    dim: int, height: int, width: int, seed: int = 42
) -> dict[str, StreamingPredictor]:
    """Instantiate the primary benchmark models."""
    return {
        "Persistence": PersistencePredictor(dim=dim),
        "FrozenLinear": FrozenLinearPredictor(dim=dim),
        "OnlineRidge": OnlineRidgePredictor(dim=dim, lam=0.98),
        "NonlinearOnlineRidge": NonlinearOnlineRidgePredictor(
            dim=dim, rff_dim=32, lam=0.98, seed=seed
        ),
        "GRU": GRUPredictor(dim=dim, hidden_dim=8),
        "FixedDelta": FixedDeltaPredictor(dim=dim, step_size=0.008),
        "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(
            dim=dim, eta_max=0.008, stability_margin=0.10
        ),
        "SelectiveRetention": SelectiveRetentionPredictor(
            dim=dim,
            feat_dim=dim,
            retention_mode="fixed_high",
            use_nonlinear_features=False,
            alpha_high=0.99,
            eta_max=0.008,
            seed=seed,
        ),
        "SpatialConv": SpatialConvControl(
            height=height, width=width, channels=1, seed=seed
        ),
    }


def run_phase_13_benchmark(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 13 Real-World Spatio-Temporal Benchmark."""
    if output_dir is None:
        out_path = Path("docs/benchmarks/artifacts/phase_13")
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DeltaCore Phase 13: Real-World Spatio-Temporal Benchmark Suite")
    print("Dataset: NOAA OISST v2.1 Equatorial Pacific Sea Surface Temperature")
    print("=" * 88)

    # 1. Dataset Generation & Ingestion
    cfg_small = RealSpatioTemporalConfig(resolution="small", height=8, width=8, seed=42)
    data_small = generate_real_spatiotemporal_dataset(cfg_small)
    leakage_audit = compute_leakage_audit(data_small)
    assert leakage_audit["leakage_audit_passed"], "Mandatory leakage audit failed!"
    print(
        f"[Audit] Leakage audit passed! SHA-256: {data_small.sha256_checksum[:16]}..."
    )

    # Save dataset config
    with open(out_path / "dataset_config.json", "w") as f:
        json.dump(asdict(cfg_small), f, indent=2)

    # 2. Multi-Seed Benchmark Evaluation
    model_names = [
        "Persistence",
        "FrozenLinear",
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "GRU",
        "FixedDelta",
        "SafeAdaptiveDelta",
        "SelectiveRetention",
        "SpatialConv",
    ]

    per_seed_results: dict[str, dict[str, dict[str, Any]]] = {
        m: {} for m in model_names
    }

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        models = instantiate_model_suite(
            data_small.config.dim, height=8, width=8, seed=seed
        )

        # Offline training for static/neural models on train split
        train_offline_model(
            models["FrozenLinear"],
            data_small.train_inputs,
            data_small.train_targets,
            epochs=25,
            seed=seed,
        )
        train_offline_model(
            models["GRU"],
            data_small.train_inputs,
            data_small.train_targets,
            epochs=25,
            seed=seed,
        )
        train_offline_model(
            models["SpatialConv"],
            data_small.train_inputs,
            data_small.train_targets,
            epochs=25,
            seed=seed,
        )

        # Sequential test evaluation
        for m_name in model_names:
            model = models[m_name]
            res = evaluate_streaming_run(
                model=model,
                inputs=data_small.test_inputs,
                targets=data_small.test_targets,
                shift_start=data_small.shift_interval[0],
                shift_end=data_small.shift_interval[1],
                track_fields=True,
                height=8,
                width=8,
            )
            # Exclude full trajectories from per-seed compact summary
            compact_res = {k: v for k, v in res.items() if not isinstance(v, list)}
            per_seed_results[m_name][str(seed)] = compact_res
            print(
                f"{m_name:<20} | RelErr: {res['rel_error']:.4f} | ShiftErr: {res['e_shift']:.4f} | "
                f"MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f} | FAC: {res['fac']:.4f} | Latency: {res['runtime_us_per_token']:.1f}us"
            )

    # 3. Aggregate Statistical Results
    aggregated_results: dict[str, dict[str, Any]] = {}
    for m_name in model_names:
        seed_dicts = list(per_seed_results[m_name].values())
        rel_errs = [d["rel_error"] for d in seed_dicts]
        shift_errs = [d["e_shift"] for d in seed_dicts]
        maes = [d["mae"] for d in seed_dicts]
        rmses = [d["rmse"] for d in seed_dicts]
        facs = [d["fac"] for d in seed_dicts]
        sges = [d["sge"] for d in seed_dicts]
        latencies = [d["runtime_us_per_token"] for d in seed_dicts]
        recoveries = [d["first_passage_recovery"] for d in seed_dicts]

        aggregated_results[m_name] = {
            "rel_error_mean": float(np.mean(rel_errs)),
            "rel_error_std": float(np.std(rel_errs)),
            "shift_error_mean": float(np.mean(shift_errs)),
            "shift_error_std": float(np.std(shift_errs)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "fac_mean": float(np.mean(facs)),
            "sge_mean": float(np.mean(sges)),
            "runtime_us_mean": float(np.mean(latencies)),
            "first_passage_recovery_mean": float(np.mean(recoveries)),
            "total_params": seed_dicts[0]["total_params"],
            "persistent_state_bytes": seed_dicts[0]["persistent_state_bytes"],
        }

    # 4. State Reset Control & Retention Ablation (Section 13 & 14)
    print("\n--- Running State Reset Control & Retention Ablation ---")
    retention_ablation: dict[str, Any] = {}
    test_models = {
        "SafeAdaptiveDelta_Continuous": (
            SafeAdaptiveDeltaPredictor(dim=data_small.config.dim, eta_max=0.008),
            None,
        ),
        "SafeAdaptiveDelta_Reset": (
            SafeAdaptiveDeltaPredictor(dim=data_small.config.dim, eta_max=0.008),
            data_small.shift_interval[0],
        ),
        "FixedDelta_Continuous": (
            FixedDeltaPredictor(dim=data_small.config.dim, step_size=0.008),
            None,
        ),
        "FixedDelta_Reset": (
            FixedDeltaPredictor(dim=data_small.config.dim, step_size=0.008),
            data_small.shift_interval[0],
        ),
        "SelectiveRetention_Continuous": (
            SelectiveRetentionPredictor(
                dim=data_small.config.dim,
                feat_dim=data_small.config.dim,
                retention_mode="fixed_high",
                use_nonlinear_features=False,
                eta_max=0.008,
            ),
            None,
        ),
        "SelectiveRetention_Reset": (
            SelectiveRetentionPredictor(
                dim=data_small.config.dim,
                feat_dim=data_small.config.dim,
                retention_mode="fixed_high",
                use_nonlinear_features=False,
                eta_max=0.008,
            ),
            data_small.shift_interval[0],
        ),
    }

    for variant_name, (mod, reset_step) in test_models.items():
        res = evaluate_streaming_run(
            model=mod,
            inputs=data_small.test_inputs,
            targets=data_small.test_targets,
            shift_start=data_small.shift_interval[0],
            shift_end=data_small.shift_interval[1],
            reset_at_step=reset_step,
            track_fields=False,
        )
        retention_ablation[variant_name] = {
            "rel_error": res["rel_error"],
            "e_shift": res["e_shift"],
            "first_passage_recovery": res["first_passage_recovery"],
            "cumulative_excess_error": res["cumulative_excess_error"],
            "adaptation_energy": res["adaptation_energy"],
        }
        print(
            f"{variant_name:<32} | RelErr: {res['rel_error']:.4f} | ShiftErr: {res['e_shift']:.4f} | "
            f"Recovery: {res['first_passage_recovery']} | ExcessErr: {res['cumulative_excess_error']:.4f}"
        )

    # 5. Spatial Permutation Control (Section 15)
    print("\n--- Running Spatial Permutation Control ---")
    spatial_control_results: dict[str, Any] = {}
    control_models = {
        "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(
            dim=data_small.config.dim, eta_max=0.008
        ),
        "FixedDelta": FixedDeltaPredictor(dim=data_small.config.dim, step_size=0.008),
        "OnlineRidge": OnlineRidgePredictor(dim=data_small.config.dim, lam=0.98),
        "SpatialConv": SpatialConvControl(height=8, width=8, channels=1),
    }
    train_offline_model(
        control_models["SpatialConv"],
        data_small.train_inputs,
        data_small.train_targets,
        epochs=25,
        seed=42,
    )

    for c_name, c_mod in control_models.items():
        # Evaluate on original spatial sequence
        res_orig = evaluate_streaming_run(
            model=c_mod,
            inputs=data_small.test_inputs,
            targets=data_small.test_targets,
            track_fields=True,
            height=8,
            width=8,
        )
        # Evaluate on fixed permuted spatial sequence
        c_mod.reset_state()
        res_perm = evaluate_streaming_run(
            model=c_mod,
            inputs=data_small.permuted_test_inputs,
            targets=data_small.permuted_test_targets,
            track_fields=True,
            height=8,
            width=8,
        )
        delta_err = res_perm["rel_error"] - res_orig["rel_error"]
        spatial_control_results[c_name] = {
            "original_rel_error": res_orig["rel_error"],
            "permuted_rel_error": res_perm["rel_error"],
            "rel_error_delta": delta_err,
            "original_sge": res_orig["sge"],
            "permuted_sge": res_perm["sge"],
            "permutation_invariant": bool(abs(delta_err) < 0.01),
        }
        print(
            f"{c_name:<18} | OrigErr: {res_orig['rel_error']:.4f} | PermErr: {res_perm['rel_error']:.4f} | "
            f"Delta: {delta_err:+.4f} | Invariant: {abs(delta_err) < 0.01}"
        )

    # 6. Multi-Resolution Scaling Experiment (Section 20)
    print("\n--- Running Multi-Resolution Scaling Experiment ---")
    cfg_medium = RealSpatioTemporalConfig(
        resolution="medium", height=16, width=16, seed=42
    )
    data_medium = generate_real_spatiotemporal_dataset(cfg_medium)

    scaling_results: dict[str, Any] = {}
    for res_name, d_obj in [("small_D64", data_small), ("medium_D256", data_medium)]:
        d_dim = d_obj.config.dim
        h_dim = d_obj.config.height
        w_dim = d_obj.config.width
        scale_models = {
            "Persistence": PersistencePredictor(dim=d_dim),
            "OnlineRidge": OnlineRidgePredictor(dim=d_dim, lam=0.98),
            "FixedDelta": FixedDeltaPredictor(dim=d_dim, step_size=0.008),
            "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(dim=d_dim, eta_max=0.008),
            "SpatialConv": SpatialConvControl(height=h_dim, width=w_dim, channels=1),
        }
        train_offline_model(
            scale_models["SpatialConv"],
            d_obj.train_inputs,
            d_obj.train_targets,
            epochs=20,
            seed=42,
        )

        scaling_results[res_name] = {
            "D": d_dim,
            "height": h_dim,
            "width": w_dim,
            "models": {},
        }
        for sm_name, sm_mod in scale_models.items():
            res_scale = evaluate_streaming_run(
                model=sm_mod,
                inputs=d_obj.test_inputs,
                targets=d_obj.test_targets,
                track_fields=False,
            )
            scaling_results[res_name]["models"][sm_name] = {
                "rel_error": res_scale["rel_error"],
                "runtime_us": res_scale["runtime_us_per_token"],
                "state_memory_bytes": res_scale["persistent_state_bytes"],
            }
            print(
                f"[{res_name}] {sm_name:<18} | D={d_dim} | RelErr: {res_scale['rel_error']:.4f} | "
                f"Latency: {res_scale['runtime_us_per_token']:.1f}us | StateMem: {res_scale['persistent_state_bytes']} B"
            )

    # 7. Shift Analysis Trajectories Artifact
    # Extract representative trajectories from SafeAdaptiveDelta, FixedDelta, OnlineRidge, Persistence
    rep_models = instantiate_model_suite(
        data_small.config.dim, height=8, width=8, seed=42
    )
    shift_trajectories: dict[str, Any] = {}
    for r_name in ["Persistence", "FixedDelta", "SafeAdaptiveDelta", "OnlineRidge"]:
        r_res = evaluate_streaming_run(
            model=rep_models[r_name],
            inputs=data_small.test_inputs,
            targets=data_small.test_targets,
            shift_start=data_small.shift_interval[0],
            shift_end=data_small.shift_interval[1],
            track_fields=False,
        )
        shift_trajectories[r_name] = {
            "step_errors": r_res["step_errors"],
            "state_norms": r_res["state_norms"],
            "update_norms": r_res["update_norms"],
        }

    # 8. Persist All Required Artifacts (Section 23)
    with open(out_path / "phase_13_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=2)

    with open(out_path / "phase_13_per_seed.json", "w") as f:
        json.dump(per_seed_results, f, indent=2)

    with open(out_path / "phase_13_config.json", "w") as f:
        json.dump(
            {
                "seeds": list(seeds),
                "dataset_config": asdict(cfg_small),
                "leakage_audit": leakage_audit,
            },
            f,
            indent=2,
        )

    with open(out_path / "phase_13_scaling.json", "w") as f:
        json.dump(scaling_results, f, indent=2)

    with open(out_path / "phase_13_shift_analysis.json", "w") as f:
        json.dump(
            {
                "shift_interval": list(data_small.shift_interval),
                "shift_peak": data_small.shift_peak,
                "retention_ablation": retention_ablation,
                "spatial_permutation_control": spatial_control_results,
                "representative_trajectories": shift_trajectories,
            },
            f,
            indent=2,
        )

    print(f"\n[Artifacts] Successfully written all Phase 13 artifacts to {out_path}!")

    return {
        "aggregated_results": aggregated_results,
        "per_seed_results": per_seed_results,
        "retention_ablation": retention_ablation,
        "spatial_permutation_control": spatial_control_results,
        "scaling_results": scaling_results,
        "leakage_audit": leakage_audit,
    }
