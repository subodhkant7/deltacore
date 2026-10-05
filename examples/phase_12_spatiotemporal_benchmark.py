"""DeltaCore Phase 12: Controlled Spatio-Temporal Adaptive State Benchmark.

Scientific Evaluation of DeltaCore's adaptive state and selective retention
under spatially structured, temporally evolving, distribution-shifting data:

Tasks:
    - Task A: Moving Spatial Field Prediction
    - Task B: Spatio-Temporal Regime Switching (A -> B -> C -> A)
    - Task C: Stale-Memory Challenge (A1 -> B -> A2) and Return Recovery (A1 -> B -> A1)

Evaluations & Controls:
    - Causal Control A: Adaptive retention ON vs. OFF
    - Causal Control B: Continuous state vs. State reset
    - Causal Control C: Fixed-high vs. Fixed-low vs. Adaptive retention
    - Causal Control D: True adaptive retention vs. Shuffled retention controller
    - Spatial Structure Ablation: True 2D order vs. Shuffled permutation P_{spatial}
    - Spatial Representation Controls: Raw flattened vs. Local pooling vs. Small Conv2d
    - Shift Frequency Ablation: Frequent (64), Medium (128), Infrequent (256)
    - Dimensional Scaling: D in {64, 128, 256} (HWC explicitly verified)

Generates 13 Observatory Plots (AX through BJ) and 5 JSON Artifacts into:
    docs/benchmarks/artifacts/phase_12/
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deltacore.observatory.spatiotemporal_plots import (
    plot_a1_b_a1_recovery_bb,
    plot_adaptation_energy_vs_perf_bh,
    plot_adaptation_negative_transfer_pareto_ay,
    plot_continuous_vs_reset_bj,
    plot_error_vs_state_adaptive_bc,
    plot_memory_vs_dim_bg,
    plot_perf_vs_dim_be,
    plot_retention_ablation_ba,
    plot_retention_trajectories_az,
    plot_runtime_vs_dim_bf,
    plot_safe_vs_unsafe_bi,
    plot_spatial_vs_shuffled_bd,
    plot_spatiotemporal_error_over_time_ax,
)
from deltacore.streaming.metrics import (
    StreamingTelemetry,
    compute_cumulative_excess_error,
    compute_first_passage_recovery,
    compute_forgetting,
    compute_negative_transfer,
    compute_pre_post_errors,
    compute_relative_step_error,
    compute_sustained_recovery,
)
from deltacore.streaming.models import (
    AdaptiveDeltaPredictor,
    FixedDeltaPredictor,
    FrozenLinearPredictor,
    FrozenMLPPredictor,
    GRUPredictor,
    LSTMPredictor,
    NaivePredictor,
    NonlinearOnlineRidgePredictor,
    OnlineRidgePredictor,
    SafeAdaptiveDeltaPredictor,
    SafeSelfReferentialPredictor,
    SelectiveRetentionPredictor,
    SelectiveStateAdaptivePredictor,
    SelfReferentialPredictor,
    SpatialConvControl,
    SpatialDownsampleControl,
    StreamingPredictor,
)
from deltacore.streaming.spatiotemporal_regimes import (
    apply_spatial_permutation,
    generate_spatiotemporal_stale_stream,
    generate_spatiotemporal_stream,
)


def evaluate_stream(
    model: StreamingPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reset_at_points: list[int] | None = None,
) -> tuple[StreamingTelemetry, float]:
    """Run sequential streaming evaluation without parameter updates."""
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.reset_state()
    telemetry = StreamingTelemetry()

    T = inputs.shape[0]
    reset_set = set(reset_at_points or [])

    start_time = time.perf_counter()

    for t in range(T):
        if t in reset_set:
            model.reset_state()

        x_t = inputs[t]
        y_t = targets[t]

        with torch.no_grad():
            y_hat = model.predict_step(x_t)

        step_err = compute_relative_step_error(y_hat, y_t)

        with torch.no_grad():
            model.adapt_step(x_t, y_t)

        state_norm = model.get_state_norm()
        upd_norm = model.last_update_norm
        is_finite = bool(torch.isfinite(y_hat).all().item() and np.isfinite(state_norm))

        margin = getattr(model, "last_margin", 2.0)
        step_size = getattr(model, "last_step_size", 0.0)
        retention = getattr(model, "last_retention", 1.0)

        telemetry.record_step(
            error=step_err,
            state_norm=state_norm,
            update_norm=upd_norm,
            stability_margin=margin,
            step_size=step_size,
            retention=retention,
            is_finite=is_finite,
        )

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return telemetry, runtime_ms


def build_phase_12_model_suite(
    height: int = 8,
    width: int = 8,
    channels: int = 1,
    oracle_change_points: list[int] | None = None,
    seed: int = 42,
) -> dict[str, StreamingPredictor]:
    """Instantiate Phase 12 model suite covering all specification models."""
    dim = height * width * channels

    return {
        "Naive_persistence": NaivePredictor(dim=dim, mode="persistence"),
        "FrozenLinear": FrozenLinearPredictor(dim=dim),
        "FrozenMLP": FrozenMLPPredictor(dim=dim, hidden_dim=max(4, dim // 8)),
        "SpatialConvControl": SpatialConvControl(
            height=height,
            width=width,
            channels=channels,
            conv_channels=4,
            seed=seed,
        ),
        "SpatialDownsampleControl": SpatialDownsampleControl(
            height=height, width=width, channels=channels, seed=seed
        ),
        "GRU": GRUPredictor(dim=dim, hidden_dim=max(2, dim // 16)),
        "LSTM": LSTMPredictor(dim=dim, hidden_dim=max(2, dim // 16)),
        "OnlineRidge": OnlineRidgePredictor(dim=dim, lam=0.98),
        "NonlinearOnlineRidge": NonlinearOnlineRidgePredictor(
            dim=dim, rff_dim=max(16, dim // 2), lam=0.98, seed=seed
        ),
        "FixedDelta": FixedDeltaPredictor(dim=dim, step_size=0.15),
        "AdaptiveDelta": AdaptiveDeltaPredictor(dim=dim, eta_max=0.40),
        "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(dim=dim, eta_max=0.50),
        "SelfReferential": SelfReferentialPredictor(dim=dim, dc_dim=2),
        "SafeSelfReferential": SafeSelfReferentialPredictor(dim=dim, dc_dim=2),
        "Selective_fixed_high": SelectiveRetentionPredictor(
            dim=dim,
            retention_mode="fixed_high",
            use_nonlinear_features=True,
            seed=seed,
        ),
        "Selective_fixed_low": SelectiveRetentionPredictor(
            dim=dim,
            retention_mode="fixed_low",
            use_nonlinear_features=True,
            seed=seed,
        ),
        "Selective_adaptive": SelectiveRetentionPredictor(
            dim=dim,
            retention_mode="adaptive",
            use_nonlinear_features=True,
            seed=seed,
        ),
        "Selective_state_adaptive": SelectiveStateAdaptivePredictor(
            dim=dim,
            use_nonlinear_features=True,
            shuffled_control=False,
            seed=seed,
        ),
        "Selective_shuffled_control": SelectiveStateAdaptivePredictor(
            dim=dim,
            use_nonlinear_features=True,
            shuffled_control=True,
            seed=seed,
        ),
        "Selective_oracle": SelectiveRetentionPredictor(
            dim=dim,
            retention_mode="oracle",
            use_nonlinear_features=True,
            oracle_change_points=oracle_change_points,
            seed=seed,
        ),
    }


def run_phase_12_benchmark(
    seeds: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 12 Controlled Spatio-Temporal Benchmark."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "benchmarks"
            / "artifacts"
            / "phase_12"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" DeltaCore Phase 12: Controlled Spatio-Temporal Benchmark")
    print(f" Seeds: {seeds} | Output: {output_dir}")
    print("==================================================================")

    # --------------------------------------------------------------------------
    # 1. Primary Spatio-Temporal Regime Switching Stream (Task B: A -> B -> C -> A)
    # --------------------------------------------------------------------------
    seq_len = 512
    h_base, w_base, c_base = 8, 8, 1
    d_base = h_base * w_base * c_base
    change_points_b = [seq_len // 4, seq_len // 2, (3 * seq_len) // 4]

    task_b_per_seed: dict[str, list[dict[str, Any]]] = {}
    trajectories_by_model: dict[str, list[float]] = {}
    retentions_by_model: dict[str, list[float]] = {}

    for s_idx, seed in enumerate(seeds):
        stream_b = generate_spatiotemporal_stream(
            seq_len=seq_len,
            height=h_base,
            width=w_base,
            channels=c_base,
            seed=seed,
        )

        models = build_phase_12_model_suite(
            height=h_base,
            width=w_base,
            channels=c_base,
            oracle_change_points=change_points_b,
            seed=seed,
        )

        for name, model in models.items():
            if name not in task_b_per_seed:
                task_b_per_seed[name] = []

            # 1. Continuous state execution
            telem_cont, runtime_ms = evaluate_stream(
                model=model,
                inputs=stream_b.inputs,
                targets=stream_b.targets,
                reset_at_points=None,
            )

            # 2. Reset state execution at shift points (for negative transfer calculation)
            model_reset = build_phase_12_model_suite(
                height=h_base,
                width=w_base,
                channels=c_base,
                oracle_change_points=change_points_b,
                seed=seed,
            )[name]
            telem_reset, _ = evaluate_stream(
                model=model_reset,
                inputs=stream_b.inputs,
                targets=stream_b.targets,
                reset_at_points=change_points_b,
            )

            # Compute transition metrics at first regime shift (t = change_points_b[0])
            shift_0 = change_points_b[0]
            e_pre, e_post_0 = compute_pre_post_errors(
                telem_cont.step_errors, change_point=shift_0, window=15
            )
            tau_fp = compute_first_passage_recovery(
                telem_cont.step_errors,
                change_point=shift_0,
                threshold=e_pre * 1.25,
            )
            tau_sr = compute_sustained_recovery(
                telem_cont.step_errors,
                change_point=shift_0,
                threshold=e_pre * 1.25,
                consecutive_steps=10,
            )
            excess_err = compute_cumulative_excess_error(
                telem_cont.step_errors,
                change_point=shift_0,
                e_pre=e_pre,
                window=50,
            )

            # Negative transfer on phase B: continuous vs reset
            shift_end = change_points_b[1]
            b_slice_cont = telem_cont.step_errors[shift_0:shift_end]
            b_slice_reset = telem_reset.step_errors[shift_0:shift_end]
            mean_b_cont = float(np.mean(b_slice_cont))
            mean_b_reset = float(np.mean(b_slice_reset))
            neg_transfer = compute_negative_transfer(mean_b_cont, mean_b_reset)

            summary = telem_cont.summary()
            total_params, trainable_params = model.get_param_count()
            state_bytes = model.get_state_memory_bytes()
            memory_breakdown = model.get_memory_breakdown()

            record = {
                "seed": seed,
                "mean_error": summary["mean_error"],
                "e_pre": e_pre,
                "e_post_0": e_post_0,
                "tau_fp": tau_fp,
                "tau_sr": tau_sr,
                "excess_err": excess_err,
                "negative_transfer": neg_transfer,
                "adaptation_energy": summary["adaptation_energy"],
                "max_state_norm": summary["max_state_norm"],
                "min_stability_margin": summary["min_stability_margin"],
                "non_finite_count": summary["non_finite_count"],
                "mean_retention": summary["mean_retention"],
                "runtime_ms": runtime_ms,
                "runtime_us_token": (runtime_ms * 1000.0) / seq_len,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "state_memory_bytes": state_bytes,
                "memory_breakdown": memory_breakdown,
                "error_b_cont": mean_b_cont,
                "error_b_reset": mean_b_reset,
            }
            task_b_per_seed[name].append(record)

            if s_idx == 0:
                trajectories_by_model[name] = telem_cont.step_errors
                retentions_by_model[name] = telem_cont.retentions

    # --------------------------------------------------------------------------
    # 2. Stale-Memory Challenge (Task C: A1 -> B -> A2 and A1 -> B -> A1)
    # --------------------------------------------------------------------------
    task_c_per_seed: dict[str, list[dict[str, Any]]] = {}

    for seed in seeds:
        stream_c = generate_spatiotemporal_stale_stream(
            seq_len=seq_len,
            height=h_base,
            width=w_base,
            channels=c_base,
            seed=seed,
        )
        cp_c = stream_c.change_points

        models = build_phase_12_model_suite(
            height=h_base,
            width=w_base,
            channels=c_base,
            oracle_change_points=cp_c,
            seed=seed,
        )

        for name, model in models.items():
            if name not in task_c_per_seed:
                task_c_per_seed[name] = []

            telem, _ = evaluate_stream(
                model=model,
                inputs=stream_c.inputs,
                targets=stream_c.targets,
                reset_at_points=None,
            )

            # Forgetting metric across shift points [cp_c[0], cp_c[1]]
            e_forget = compute_forgetting(
                errors=telem.step_errors,
                initial_steady_point=cp_c[0],
                return_change_point=cp_c[1],
                window=15,
            )

            # Phase B error
            b_errors = telem.step_errors[cp_c[0] : cp_c[1]]
            mean_b = float(np.mean(b_errors)) if b_errors else 0.0

            # Phase A2 / return error
            a2_errors = telem.step_errors[cp_c[1] :]
            mean_a2 = float(np.mean(a2_errors)) if a2_errors else 0.0

            task_c_per_seed[name].append(
                {
                    "seed": seed,
                    "mean_error": telem.summary()["mean_error"],
                    "phase_b_error": mean_b,
                    "phase_a2_error": mean_a2,
                    "forgetting": e_forget,
                }
            )

    # --------------------------------------------------------------------------
    # 3. Spatial Structure Ablation (Section 12: Order vs P_{spatial})
    # --------------------------------------------------------------------------
    spatial_ablation_results: dict[str, dict[str, float]] = {}
    selected_ablation_models = [
        "SpatialConvControl",
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "SafeAdaptiveDelta",
        "Selective_adaptive",
        "Selective_state_adaptive",
    ]

    for model_name in selected_ablation_models:
        orig_errs = []
        perm_errs = []

        for seed in seeds:
            stream_orig = generate_spatiotemporal_stream(
                seq_len=seq_len,
                height=h_base,
                width=w_base,
                channels=c_base,
                seed=seed,
            )
            stream_perm = apply_spatial_permutation(
                stream_orig, permutation_seed=seed + 999
            )

            m_orig = build_phase_12_model_suite(
                height=h_base,
                width=w_base,
                channels=c_base,
                seed=seed,
            )[model_name]
            telem_orig, _ = evaluate_stream(
                m_orig, stream_orig.inputs, stream_orig.targets
            )
            orig_errs.append(telem_orig.summary()["mean_error"])

            m_perm = build_phase_12_model_suite(
                height=h_base,
                width=w_base,
                channels=c_base,
                seed=seed,
            )[model_name]
            telem_perm, _ = evaluate_stream(
                m_perm, stream_perm.inputs, stream_perm.targets
            )
            perm_errs.append(telem_perm.summary()["mean_error"])

        spatial_ablation_results[model_name] = {
            "original_mean": float(np.mean(orig_errs)),
            "original_std": float(np.std(orig_errs)),
            "permuted_mean": float(np.mean(perm_errs)),
            "permuted_std": float(np.std(perm_errs)),
            "diff_mean": float(np.mean(np.array(perm_errs) - np.array(orig_errs))),
        }

    # --------------------------------------------------------------------------
    # 4. Temporal Shift Frequency Ablation (Section 14: 64, 128, 256)
    # --------------------------------------------------------------------------
    frequency_results: dict[int, dict[str, float]] = {}
    shift_intervals = [64, 128, 256]

    for interval in shift_intervals:
        freq_errs: dict[str, list[float]] = {m: [] for m in selected_ablation_models}
        for seed in seeds:
            stream_freq = generate_spatiotemporal_stream(
                seq_len=seq_len,
                height=h_base,
                width=w_base,
                channels=c_base,
                shift_interval=interval,
                seed=seed,
            )
            suite = build_phase_12_model_suite(
                height=h_base,
                width=w_base,
                channels=c_base,
                seed=seed,
            )
            for m in selected_ablation_models:
                telem_f, _ = evaluate_stream(
                    suite[m], stream_freq.inputs, stream_freq.targets
                )
                freq_errs[m].append(telem_f.summary()["mean_error"])

        frequency_results[interval] = {
            m: float(np.mean(errs)) for m, errs in freq_errs.items()
        }

    # --------------------------------------------------------------------------
    # 5. Dimensional Scaling (Section 11: D in {64, 128, 256})
    # --------------------------------------------------------------------------
    dim_configs = [
        {"height": 8, "width": 8, "channels": 1, "dim": 64},
        {"height": 8, "width": 8, "channels": 2, "dim": 128},
        {"height": 16, "width": 16, "channels": 1, "dim": 256},
    ]

    scaling_models = [
        "FrozenLinear",
        "SpatialConvControl",
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "SafeAdaptiveDelta",
        "Selective_adaptive",
        "Selective_state_adaptive",
    ]

    scaling_results: dict[int, dict[str, dict[str, float]]] = {}

    for cfg in dim_configs:
        d = cfg["dim"]
        h = cfg["height"]
        w = cfg["width"]
        c = cfg["channels"]
        assert d == h * w * c, f"D={d} != HWC={h * w * c}"

        scaling_results[d] = {}
        for m_name in scaling_models:
            errs = []
            runtimes = []
            state_mems = []

            for seed in seeds:
                stream_d = generate_spatiotemporal_stream(
                    seq_len=seq_len,
                    height=h,
                    width=w,
                    channels=c,
                    seed=seed,
                )
                m_inst = build_phase_12_model_suite(
                    height=h,
                    width=w,
                    channels=c,
                    seed=seed,
                )[m_name]
                telem_d, rt_ms = evaluate_stream(
                    m_inst, stream_d.inputs, stream_d.targets
                )

                errs.append(telem_d.summary()["mean_error"])
                runtimes.append((rt_ms * 1000.0) / seq_len)
                state_mems.append(m_inst.get_state_memory_bytes())

            scaling_results[d][m_name] = {
                "error_mean": float(np.mean(errs)),
                "error_std": float(np.std(errs)),
                "runtime_us_token": float(np.mean(runtimes)),
                "state_memory_bytes": int(state_mems[0]),
            }

    # --------------------------------------------------------------------------
    # 6. Aggregate Main Results (Means, STDs, Tables)
    # --------------------------------------------------------------------------
    aggregated_b: dict[str, dict[str, Any]] = {}
    for name, records in task_b_per_seed.items():
        errs = [r["mean_error"] for r in records]
        pre_errs = [r["e_pre"] for r in records]
        post_errs = [r["e_post_0"] for r in records]
        tau_fps = [r["tau_fp"] for r in records]
        tau_srs = [r["tau_sr"] for r in records]
        excesses = [r["excess_err"] for r in records]
        neg_trans = [r["negative_transfer"] for r in records]
        energies = [r["adaptation_energy"] for r in records]
        rts = [r["runtime_us_token"] for r in records]
        max_norms = [r["max_state_norm"] for r in records]
        min_margins = [r["min_stability_margin"] for r in records]
        non_finites = [r["non_finite_count"] for r in records]
        retentions = [r["mean_retention"] for r in records]
        c_errs = [task_c_per_seed[name][i]["forgetting"] for i in range(len(records))]
        b_adapts = [
            task_c_per_seed[name][i]["phase_b_error"] for i in range(len(records))
        ]

        aggregated_b[name] = {
            "mean_error": float(np.mean(errs)),
            "std_error": float(np.std(errs)),
            "pre_error_mean": float(np.mean(pre_errs)),
            "post_error_mean": float(np.mean(post_errs)),
            "tau_fp_mean": float(np.mean(tau_fps)),
            "tau_sr_mean": float(np.mean(tau_srs)),
            "excess_error_mean": float(np.mean(excesses)),
            "negative_transfer_mean": float(np.mean(neg_trans)),
            "adaptation_energy_mean": float(np.mean(energies)),
            "runtime_us_token": float(np.mean(rts)),
            "max_state_norm_mean": float(np.mean(max_norms)),
            "min_stability_margin_mean": float(np.mean(min_margins)),
            "non_finite_count_total": int(np.sum(non_finites)),
            "mean_retention": float(np.mean(retentions)),
            "forgetting_mean": float(np.mean(c_errs)),
            "phase_b_adapt_mean": float(np.mean(b_adapts)),
            "total_params": records[0]["total_params"],
            "trainable_params": records[0]["trainable_params"],
            "state_memory_bytes": records[0]["state_memory_bytes"],
            "memory_breakdown": records[0]["memory_breakdown"],
        }

    # --------------------------------------------------------------------------
    # 7. Render 13 Observatory Plots (AX through BJ)
    # --------------------------------------------------------------------------
    print("Generating Observatory Plots AX through BJ...")

    # Plot AX: Error over time
    plot_spatiotemporal_error_over_time_ax(
        trajectories=trajectories_by_model,
        change_points=change_points_b,
        output_path=output_dir / "plot_ax_error_over_time.png",
    )

    # Plot AY: Pareto frontier (post-shift error vs negative transfer)
    models_ay = list(aggregated_b.keys())
    ps_ay = [aggregated_b[m]["post_error_mean"] for m in models_ay]
    nt_ay = [aggregated_b[m]["negative_transfer_mean"] for m in models_ay]
    plot_adaptation_negative_transfer_pareto_ay(
        models=models_ay,
        post_shift_errors=ps_ay,
        negative_transfers=nt_ay,
        output_path=output_dir / "plot_ay_pareto_frontier.png",
    )

    # Plot AZ: Retention trajectories
    retention_series = {
        "Selective_fixed_high": retentions_by_model["Selective_fixed_high"],
        "Selective_fixed_low": retentions_by_model["Selective_fixed_low"],
        "Selective_adaptive": retentions_by_model["Selective_adaptive"],
        "Selective_state_adaptive": retentions_by_model["Selective_state_adaptive"],
        "Selective_oracle": retentions_by_model["Selective_oracle"],
    }
    plot_retention_trajectories_az(
        retention_series=retention_series,
        change_points=change_points_b,
        output_path=output_dir / "plot_az_retention_trajectories.png",
    )

    # Plot BA: Retention ablation
    ret_modes = [
        "Selective_fixed_high",
        "Selective_fixed_low",
        "Selective_adaptive",
        "Selective_state_adaptive",
        "Selective_oracle",
    ]
    ps_ba = [aggregated_b[m]["post_error_mean"] for m in ret_modes]
    nt_ba = [aggregated_b[m]["negative_transfer_mean"] for m in ret_modes]
    plot_retention_ablation_ba(
        retention_modes=ret_modes,
        post_shift_errors=ps_ba,
        negative_transfers=nt_ba,
        output_path=output_dir / "plot_ba_retention_ablation.png",
    )

    # Plot BB: A1 -> B -> A1 recovery and forgetting
    models_bb = [
        "FrozenLinear",
        "OnlineRidge",
        "SafeAdaptiveDelta",
        "Selective_fixed_high",
        "Selective_adaptive",
        "Selective_state_adaptive",
        "Selective_oracle",
    ]
    b_err_bb = [aggregated_b[m]["phase_b_adapt_mean"] for m in models_bb]
    fg_bb = [aggregated_b[m]["forgetting_mean"] for m in models_bb]
    plot_a1_b_a1_recovery_bb(
        models=models_bb,
        adapt_b_errors=b_err_bb,
        forgetting_metrics=fg_bb,
        output_path=output_dir / "plot_bb_a1_b_a1_recovery.png",
    )

    # Plot BC: Error-only vs state-conditioned retention
    bc_series = {
        "Selective_adaptive (Error-Only)": trajectories_by_model["Selective_adaptive"],
        "Selective_state_adaptive (State-Conditioned)": trajectories_by_model[
            "Selective_state_adaptive"
        ],
        "Selective_shuffled_control (Causal Control D)": trajectories_by_model[
            "Selective_shuffled_control"
        ],
    }
    plot_error_vs_state_adaptive_bc(
        error_trajectories=bc_series,
        change_points=change_points_b,
        output_path=output_dir / "plot_bc_state_adaptive_comparison.png",
    )

    # Plot BD: Spatial structure ablation
    models_bd = selected_ablation_models
    orig_bd = [spatial_ablation_results[m]["original_mean"] for m in models_bd]
    perm_bd = [spatial_ablation_results[m]["permuted_mean"] for m in models_bd]
    plot_spatial_vs_shuffled_bd(
        models=models_bd,
        original_errors=orig_bd,
        shuffled_errors=perm_bd,
        output_path=output_dir / "plot_bd_spatial_vs_shuffled.png",
    )

    # Plot BE: Performance vs dimensionality
    dims_scaling = [64, 128, 256]
    model_errs_be = {
        m: [scaling_results[d][m]["error_mean"] for d in dims_scaling]
        for m in scaling_models
    }
    plot_perf_vs_dim_be(
        dimensions=dims_scaling,
        model_errors=model_errs_be,
        output_path=output_dir / "plot_be_perf_vs_dim.png",
    )

    # Plot BF: Runtime/token vs dimensionality
    model_rts_bf = {
        m: [scaling_results[d][m]["runtime_us_token"] for d in dims_scaling]
        for m in scaling_models
    }
    plot_runtime_vs_dim_bf(
        dimensions=dims_scaling,
        model_runtimes=model_rts_bf,
        output_path=output_dir / "plot_bf_runtime_vs_dim.png",
    )

    # Plot BG: State memory vs dimensionality
    model_mems_bg = {
        m: [float(scaling_results[d][m]["state_memory_bytes"]) for d in dims_scaling]
        for m in scaling_models
    }
    plot_memory_vs_dim_bg(
        dimensions=dims_scaling,
        model_memories=model_mems_bg,
        output_path=output_dir / "plot_bg_memory_vs_dim.png",
    )

    # Plot BH: Adaptation energy vs performance
    models_bh = list(aggregated_b.keys())
    eng_bh = [aggregated_b[m]["adaptation_energy_mean"] for m in models_bh]
    err_bh = [aggregated_b[m]["mean_error"] for m in models_bh]
    plot_adaptation_energy_vs_perf_bh(
        models=models_bh,
        energies=eng_bh,
        errors=err_bh,
        output_path=output_dir / "plot_bh_energy_vs_performance.png",
    )

    # Plot BI: Safe vs unsafe variants
    safe_comp_models = [
        "AdaptiveDelta",
        "SafeAdaptiveDelta",
        "SelfReferential",
        "SafeSelfReferential",
    ]
    max_norms_bi = [aggregated_b[m]["max_state_norm_mean"] for m in safe_comp_models]
    min_margins_bi = [
        aggregated_b[m]["min_stability_margin_mean"] for m in safe_comp_models
    ]
    plot_safe_vs_unsafe_bi(
        models=safe_comp_models,
        max_norms=max_norms_bi,
        min_margins=min_margins_bi,
        output_path=output_dir / "plot_bi_safe_vs_unsafe.png",
    )

    # Plot BJ: Continuous state vs state reset
    models_bj = [
        "OnlineRidge",
        "SafeAdaptiveDelta",
        "Selective_fixed_high",
        "Selective_fixed_low",
        "Selective_adaptive",
        "Selective_state_adaptive",
    ]
    cont_bj = [
        float(np.mean([r["error_b_cont"] for r in task_b_per_seed[m]]))
        for m in models_bj
    ]
    reset_bj = [
        float(np.mean([r["error_b_reset"] for r in task_b_per_seed[m]]))
        for m in models_bj
    ]
    plot_continuous_vs_reset_bj(
        models=models_bj,
        continuous_errors=cont_bj,
        reset_errors=reset_bj,
        output_path=output_dir / "plot_bj_continuous_vs_reset.png",
    )

    # --------------------------------------------------------------------------
    # 8. Save Machine-Readable Artifacts
    # --------------------------------------------------------------------------
    print("Saving JSON artifacts...")

    # phase_12_config.json
    config_artifact = {
        "phase": 12,
        "seeds": seeds,
        "seq_len": seq_len,
        "base_dimensions": {
            "height": h_base,
            "width": w_base,
            "channels": c_base,
            "dim": d_base,
        },
        "change_points": change_points_b,
        "shift_intervals": shift_intervals,
        "scaling_dimensions": dim_configs,
        "models": list(aggregated_b.keys()),
    }
    with open(output_dir / "phase_12_config.json", "w") as f:
        json.dump(config_artifact, f, indent=2)

    # phase_12_results.json
    results_artifact = {
        "aggregated_task_b": aggregated_b,
        "spatial_ablation": spatial_ablation_results,
        "frequency_ablation": frequency_results,
    }
    with open(output_dir / "phase_12_results.json", "w") as f:
        json.dump(results_artifact, f, indent=2)

    # phase_12_per_seed.json
    per_seed_artifact = {
        "task_b_per_seed": task_b_per_seed,
        "task_c_per_seed": task_c_per_seed,
    }
    with open(output_dir / "phase_12_per_seed.json", "w") as f:
        json.dump(per_seed_artifact, f, indent=2)

    # phase_12_scaling.json
    with open(output_dir / "phase_12_scaling.json", "w") as f:
        json.dump(scaling_results, f, indent=2)

    # phase_12_retention.json
    retention_artifact = {
        mode: {
            "post_shift_error": aggregated_b[mode]["post_error_mean"],
            "negative_transfer": aggregated_b[mode]["negative_transfer_mean"],
            "forgetting": aggregated_b[mode]["forgetting_mean"],
            "adaptation_energy": aggregated_b[mode]["adaptation_energy_mean"],
            "mean_retention": aggregated_b[mode]["mean_retention"],
        }
        for mode in ret_modes
    }
    with open(output_dir / "phase_12_retention.json", "w") as f:
        json.dump(retention_artifact, f, indent=2)

    print(
        "Phase 12 benchmark complete! All plots and artifacts successfully generated."
    )
    return results_artifact


if __name__ == "__main__":
    run_phase_12_benchmark()
