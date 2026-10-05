"""DeltaCore Phase 11: Nonlinear Adaptive State & Selective Retention Benchmark.

Evaluates online adaptation, recovery, retention, forgetting, negative transfer,
stability, and computational scaling across nonlinear streaming regimes:

Tasks:
    - Primary Task A: Nonlinear Regime Dynamics (A -> B -> C -> A)
    - Primary Task B: Regime Switching with Stale-Memory Penalty (A -> B -> A)
    - Task C: Delayed Context Retrieval with Nonlinear Target (y = g(c), d in {16, 64, 256})

Causal Ablations:
    - A. Adaptive State ON vs. OFF
    - B. Adaptive Retention ON vs. OFF
    - C. Continuous State vs. Reset
    - D. Learned/Adaptive Retention vs. Fixed High / Fixed Low / Oracle
    - E. Nonlinear Feature Map ON vs. OFF

Dimensional Scaling:
    - D in {8, 16, 32, 64, 128} (memory footprint, runtime/token, T_D / T_8)

Generates Observatory Plots AN through AW into docs/benchmarks/artifacts/phase_11/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deltacore.observatory.nonlinear_plots import (
    plot_adaptation_efficiency_av,
    plot_adaptation_forgetting_ao,
    plot_memory_vs_dim_au,
    plot_model_comparison_ar,
    plot_nonlinear_error_over_time_an,
    plot_perf_vs_dim_as,
    plot_retention_ablation_aq,
    plot_retention_trajectory_ap,
    plot_runtime_vs_dim_at,
    plot_safe_vs_unsafe_aw,
)
from deltacore.streaming.metrics import (
    StreamingTelemetry,
    compute_adaptation_efficiency,
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
    SelfReferentialPredictor,
    StreamingPredictor,
)
from deltacore.streaming.nonlinear_regimes import (
    generate_nonlinear_delayed_retrieval_stream,
    generate_nonlinear_regime_stream,
    generate_stale_penalty_stream,
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


def build_phase_11_model_suite(
    dim: int = 8,
    change_points_b: list[int] | None = None,
    seed: int = 42,
) -> dict[str, StreamingPredictor]:
    """Instantiate Phase 11 model suite covering baselines, DeltaCore, and selective retention."""
    return {
        "Naive": NaivePredictor(dim=dim, mode="persistence"),
        "FrozenLinear": FrozenLinearPredictor(dim=dim),
        "FrozenMLP": FrozenMLPPredictor(dim=dim, hidden_dim=4),
        "GRU": GRUPredictor(dim=dim, hidden_dim=2),
        "LSTM": LSTMPredictor(dim=dim, hidden_dim=2),
        "OnlineRidge": OnlineRidgePredictor(dim=dim, lam=0.98),
        "NonlinearOnlineRidge": NonlinearOnlineRidgePredictor(
            dim=dim, rff_dim=16, lam=0.98, seed=seed
        ),
        "FixedDelta": FixedDeltaPredictor(dim=dim, step_size=0.15),
        "AdaptiveDelta": AdaptiveDeltaPredictor(dim=dim, eta_max=0.40),
        "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(dim=dim, eta_max=0.50),
        "SelfReferential": SelfReferentialPredictor(dim=dim, dc_dim=2),
        "SafeSelfReferential": SafeSelfReferentialPredictor(dim=dim, dc_dim=2),
        "Selective_fixed_high": SelectiveRetentionPredictor(
            dim=dim, retention_mode="fixed_high", use_nonlinear_features=True, seed=seed
        ),
        "Selective_fixed_low": SelectiveRetentionPredictor(
            dim=dim, retention_mode="fixed_low", use_nonlinear_features=True, seed=seed
        ),
        "Selective_adaptive": SelectiveRetentionPredictor(
            dim=dim, retention_mode="adaptive", use_nonlinear_features=True, seed=seed
        ),
        "Selective_oracle": SelectiveRetentionPredictor(
            dim=dim,
            retention_mode="oracle",
            use_nonlinear_features=True,
            oracle_change_points=change_points_b,
            seed=seed,
        ),
    }


def compute_state_memory_bytes(model: StreamingPredictor) -> int:
    """Compute exact byte size of internal dynamic state variables."""
    total_bytes = 0
    # State buffers / tensors in model
    for name, buf in model.named_buffers():
        if name in ("M", "C", "P", "W_out", "last_x", "hidden", "cell"):
            total_bytes += buf.nelement() * buf.element_size()
    # Check recurrent internal state
    hidden_val = getattr(model, "hidden", None)
    if isinstance(hidden_val, torch.Tensor):
        total_bytes += hidden_val.nelement() * hidden_val.element_size()
    cell_val = getattr(model, "cell", None)
    if isinstance(cell_val, torch.Tensor):
        total_bytes += cell_val.nelement() * cell_val.element_size()
    if total_bytes == 0:
        # Fallback to state dimension floats * 4
        tot_p, _ = model.get_param_count()
        total_bytes = tot_p * 4
    return total_bytes


def run_phase_11_benchmark() -> dict[str, Any]:
    print("=" * 88)
    print(
        "DeltaCore Phase 11: Nonlinear Adaptive State & Selective Retention Benchmark"
    )
    print("=" * 88)

    seeds = [0, 1, 2, 3, 4]
    dim = 8
    seq_len = 512
    out_dir = Path("docs/benchmarks/artifacts/phase_11")
    out_dir.mkdir(parents=True, exist_ok=True)

    change_points_a = [seq_len // 4, seq_len // 2, (3 * seq_len) // 4]
    change_points_b = [seq_len // 3, (2 * seq_len) // 3]

    # Sample model suite for accounting
    sample_models = build_phase_11_model_suite(dim=dim, change_points_b=change_points_b)
    param_counts: dict[str, int] = {}
    state_mem_bytes: dict[str, int] = {}

    print("\n[1. Model Capacity & State Accounting (D=8)]")
    print(f"{'Model':<24} | {'Total Params':<14} | {'State Memory (Bytes)':<20}")
    print("-" * 64)
    for name, m in sample_models.items():
        tot, _ = m.get_param_count()
        mem = compute_state_memory_bytes(m)
        param_counts[name] = tot
        state_mem_bytes[name] = mem
        print(f"{name:<24} | {tot:<14} | {mem:<20}")

    # --------------------------------------------------------------------------
    # 2. Task A: Nonlinear Regime Dynamics (A -> B -> C -> A)
    # --------------------------------------------------------------------------
    print("\n[2. Task A: Nonlinear Regime Dynamics (A -> B -> C -> A)]")
    task_a_seed_errors: dict[str, list[float]] = {m: [] for m in sample_models}
    task_a_first_passage: dict[str, list[int]] = {m: [] for m in sample_models}
    task_a_sustained: dict[str, list[int]] = {m: [] for m in sample_models}
    task_a_energies: dict[str, list[float]] = {m: [] for m in sample_models}
    task_a_runtimes: dict[str, list[float]] = {m: [] for m in sample_models}
    sample_trajectories: dict[str, list[float]] = {}

    for seed in seeds:
        stream_a = generate_nonlinear_regime_stream(seq_len=seq_len, dim=dim, seed=seed)
        models = build_phase_11_model_suite(
            dim=dim, change_points_b=change_points_b, seed=seed
        )

        for name, model in models.items():
            telemetry, rt = evaluate_stream(model, stream_a.inputs, stream_a.targets)
            task_a_seed_errors[name].append(float(np.mean(telemetry.step_errors)))
            task_a_runtimes[name].append(rt)
            task_a_energies[name].append(telemetry.adaptation_energy)

            cp = stream_a.change_points[0]
            e_pre, _ = compute_pre_post_errors(telemetry.step_errors, cp)
            thresh = max(e_pre * 1.3, 0.25)
            t_fp = compute_first_passage_recovery(
                telemetry.step_errors, cp, threshold=thresh
            )
            t_st = compute_sustained_recovery(
                telemetry.step_errors, cp, threshold=thresh, consecutive_steps=10
            )

            task_a_first_passage[name].append(t_fp)
            task_a_sustained[name].append(t_st)

            if seed == 0:
                sample_trajectories[name] = telemetry.step_errors

    print(
        f"{'Model':<24} | {'Mean Error E_rel':<18} | {'1st-Passage':<14} | {'Sustained':<14} | {'Energy':<10}"
    )
    print("-" * 88)
    for name in sample_models:
        m_e, s_e = (
            float(np.mean(task_a_seed_errors[name])),
            float(np.std(task_a_seed_errors[name])),
        )
        m_fp = float(np.mean(task_a_first_passage[name]))
        m_st = float(np.mean(task_a_sustained[name]))
        m_en = float(np.mean(task_a_energies[name]))
        print(
            f"{name:<24} | {m_e:>7.4f} ± {s_e:<8.4f} | {m_fp:>6.1f} steps   | {m_st:>6.1f} steps   | {m_en:>8.2f}"
        )

    # --------------------------------------------------------------------------
    # 3. Task B: Stale-Memory Penalty Stream (A -> B -> A)
    # --------------------------------------------------------------------------
    print("\n[3. Task B: Regime Switching with Stale-Memory Penalty (A -> B -> A)]")
    task_b_adapt_errors: dict[str, list[float]] = {m: [] for m in sample_models}
    task_b_forgetting: dict[str, list[float]] = {m: [] for m in sample_models}
    task_b_return_recovery: dict[str, list[int]] = {m: [] for m in sample_models}
    task_b_negative_transfers: dict[str, list[float]] = {m: [] for m in sample_models}
    sample_retentions: dict[str, list[float]] = {}

    t1, t2 = seq_len // 3, (2 * seq_len) // 3

    for seed in seeds:
        stream_b = generate_stale_penalty_stream(seq_len=seq_len, dim=dim, seed=seed)

        # 1. Continuous run
        models_cont = build_phase_11_model_suite(
            dim=dim, change_points_b=change_points_b, seed=seed
        )
        cont_b_errs: dict[str, float] = {}

        for name, model in models_cont.items():
            telemetry_cont, _ = evaluate_stream(
                model, stream_b.inputs, stream_b.targets
            )
            err_b = float(np.mean(telemetry_cont.step_errors[t1:t2]))
            cont_b_errs[name] = err_b
            task_b_adapt_errors[name].append(err_b)

            fg = compute_forgetting(
                telemetry_cont.step_errors,
                initial_steady_point=t1,
                return_change_point=t2,
            )
            task_b_forgetting[name].append(fg)

            # Return recovery in Phase 3
            e_pre_a, _ = compute_pre_post_errors(telemetry_cont.step_errors, t1)
            t_ret = compute_sustained_recovery(
                telemetry_cont.step_errors, t2, threshold=max(e_pre_a * 1.3, 0.25)
            )
            task_b_return_recovery[name].append(t_ret)

            if seed == 0 and "Selective" in name:
                sample_retentions[name] = telemetry_cont.retentions

        # 2. Reset run (to compute negative transfer)
        models_reset = build_phase_11_model_suite(
            dim=dim, change_points_b=change_points_b, seed=seed
        )
        for name, model in models_reset.items():
            telemetry_reset, _ = evaluate_stream(
                model,
                stream_b.inputs,
                stream_b.targets,
                reset_at_points=change_points_b,
            )
            err_b_reset = float(np.mean(telemetry_reset.step_errors[t1:t2]))
            neg_transfer = compute_negative_transfer(cont_b_errs[name], err_b_reset)
            task_b_negative_transfers[name].append(neg_transfer)

    print(
        f"{'Model':<24} | {'Phase B Adapt':<16} | {'Forgetting':<14} | {'Negative Transfer':<18}"
    )
    print("-" * 78)
    for name in sample_models:
        m_ad = float(np.mean(task_b_adapt_errors[name]))
        m_fg = float(np.mean(task_b_forgetting[name]))
        m_nt = float(np.mean(task_b_negative_transfers[name]))
        print(
            f"{name:<24} | {m_ad:>7.4f}          | {m_fg:>+7.4f}      | {m_nt:>+7.4f}"
        )

    # --------------------------------------------------------------------------
    # 4. Task C: Delayed Context with Nonlinear Target y = g(c)
    # --------------------------------------------------------------------------
    print("\n[4. Task C: Nonlinear Delayed Context Retrieval (d in {16, 64, 256})]")
    delays = [16, 64, 256]
    task_c_results: dict[str, dict[int, list[float]]] = {
        m: {d: [] for d in delays} for m in sample_models
    }

    for d in delays:
        for seed in seeds:
            stream_c = generate_nonlinear_delayed_retrieval_stream(
                delay=d, seq_len=seq_len, dim=dim, seed=seed
            )
            models = build_phase_11_model_suite(
                dim=dim, change_points_b=change_points_b, seed=seed
            )
            q_time = stream_c.metadata["query_time"]

            for name, model in models.items():
                telemetry, _ = evaluate_stream(model, stream_c.inputs, stream_c.targets)
                query_err = telemetry.step_errors[q_time]
                task_c_results[name][d].append(query_err)

    print(
        f"{'Model':<24} | {'d=16 Error':<14} | {'d=64 Error':<14} | {'d=256 Error':<14}"
    )
    print("-" * 72)
    for name in sample_models:
        e16 = float(np.mean(task_c_results[name][16]))
        e64 = float(np.mean(task_c_results[name][64]))
        e256 = float(np.mean(task_c_results[name][256]))
        print(f"{name:<24} | {e16:>6.4f}         | {e64:>6.4f}         | {e256:>6.4f}")

    # --------------------------------------------------------------------------
    # 5. Causal Ablations A through E
    # --------------------------------------------------------------------------
    print("\n[5. Causal Ablations A through E]")
    # Ablation A: Adaptive State ON vs OFF (Selective_adaptive)
    sel_on_runs, sel_off_runs = [], []
    # Ablation E: Nonlinear feature map ON vs OFF
    sel_nonlin_runs, sel_linear_runs = [], []

    for seed in seeds:
        stream_a = generate_nonlinear_regime_stream(seq_len=seq_len, dim=dim, seed=seed)

        # ON
        m_on = SelectiveRetentionPredictor(
            dim=dim, retention_mode="adaptive", use_nonlinear_features=True, seed=seed
        )
        t_on, _ = evaluate_stream(m_on, stream_a.inputs, stream_a.targets)
        sel_on_runs.append(float(np.mean(t_on.step_errors)))

        # OFF
        m_off = SelectiveRetentionPredictor(
            dim=dim, retention_mode="adaptive", use_nonlinear_features=True, seed=seed
        )
        m_off.set_adaptation(False)
        t_off, _ = evaluate_stream(m_off, stream_a.inputs, stream_a.targets)
        sel_off_runs.append(float(np.mean(t_off.step_errors)))

        # Nonlinear Features (already computed in m_on)
        sel_nonlin_runs.append(float(np.mean(t_on.step_errors)))

        # Linear Features
        m_lin = SelectiveRetentionPredictor(
            dim=dim, retention_mode="adaptive", use_nonlinear_features=False, seed=seed
        )
        t_lin, _ = evaluate_stream(m_lin, stream_a.inputs, stream_a.targets)
        sel_linear_runs.append(float(np.mean(t_lin.step_errors)))

    m_on_val, m_off_val = float(np.mean(sel_on_runs)), float(np.mean(sel_off_runs))
    m_nl_val, m_lin_val = (
        float(np.mean(sel_nonlin_runs)),
        float(np.mean(sel_linear_runs)),
    )

    print(
        f"Ablation A: Adaptive State ON ({m_on_val:.4f}) vs OFF ({m_off_val:.4f}) -> Delta: {m_on_val - m_off_val:+.4f}"
    )
    print(
        f"Ablation E: Nonlinear Features ON ({m_nl_val:.4f}) vs OFF ({m_lin_val:.4f}) -> Delta: {m_nl_val - m_lin_val:+.4f}"
    )

    # --------------------------------------------------------------------------
    # 6. Dimensional Scaling Sweep: D in {8, 16, 32, 64, 128}
    # --------------------------------------------------------------------------
    print("\n[6. Dimensional Scaling Sweep: D in {8, 16, 32, 64, 128}]")
    dims = [8, 16, 32, 64, 128]
    scaling_models = [
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "SafeAdaptiveDelta",
        "Selective_adaptive",
    ]

    scaling_perf: dict[str, list[float]] = {m: [] for m in scaling_models}
    scaling_runtimes: dict[str, list[float]] = {m: [] for m in scaling_models}
    scaling_memory: dict[str, list[int]] = {m: [] for m in scaling_models}

    for d in dims:
        stream_d = generate_nonlinear_regime_stream(seq_len=256, dim=d, seed=42)

        inst_models = {
            "OnlineRidge": OnlineRidgePredictor(dim=d, lam=0.98),
            "NonlinearOnlineRidge": NonlinearOnlineRidgePredictor(
                dim=d, rff_dim=2 * d, lam=0.98, seed=42
            ),
            "SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(dim=d, eta_max=0.50),
            "Selective_adaptive": SelectiveRetentionPredictor(
                dim=d,
                retention_mode="adaptive",
                use_nonlinear_features=True,
                feat_dim=d,
                seed=42,
            ),
        }

        print(f"\n--- Dimension D = {d} ---")
        for name in scaling_models:
            m = inst_models[name]
            tel, rt_ms = evaluate_stream(m, stream_d.inputs, stream_d.targets)
            us_per_tok = (rt_ms / 256.0) * 1000.0
            mem_b = compute_state_memory_bytes(m)
            mean_err = float(np.mean(tel.step_errors))

            scaling_perf[name].append(mean_err)
            scaling_runtimes[name].append(us_per_tok)
            scaling_memory[name].append(mem_b)

            print(
                f"{name:<22} | Error: {mean_err:>6.4f} | Latency: {us_per_tok:>7.1f} us/tok | State: {mem_b:>7d} B"
            )

    # Compute scaling ratios T_D / T_8
    scaling_ratios: dict[str, list[float]] = {}
    for name in scaling_models:
        t8 = scaling_runtimes[name][0]
        scaling_ratios[name] = [float(t / t8) for t in scaling_runtimes[name]]

    # --------------------------------------------------------------------------
    # 7. Adaptation Efficiency Metric
    # --------------------------------------------------------------------------
    print("\n[7. Adaptation Efficiency Calculation]")
    baseline_naive_err = float(np.mean(task_a_seed_errors["Naive"]))
    adaptation_efficiencies: dict[str, float] = {}

    for name in sample_models:
        m_err = float(np.mean(task_a_seed_errors[name]))
        err_reduction = max(0.0, baseline_naive_err - m_err)
        energy = float(np.mean(task_a_energies[name]))
        eff = compute_adaptation_efficiency(err_reduction, energy)
        adaptation_efficiencies[name] = eff
        print(
            f"{name:<24} | Error Reduction: {err_reduction:.4f} | Energy: {energy:>8.2f} | Efficiency: {eff:.4f}"
        )

    # --------------------------------------------------------------------------
    # 8. Generate Observatory Publication Plots AN through AW
    # --------------------------------------------------------------------------
    print("\nGenerating Phase 11 Observatory Plots AN through AW...")

    # Plot AN: Nonlinear Error Over Time
    plot_nonlinear_error_over_time_an(
        sample_trajectories,
        change_points=change_points_a,
        output_path=out_dir / "plot_an_nonlinear_error_over_time.png",
    )

    # Plot AO: Adaptation vs Forgetting
    m_list = list(sample_models.keys())
    ad_list = [float(np.mean(task_b_adapt_errors[m])) for m in m_list]
    fg_list = [float(np.mean(task_b_forgetting[m])) for m in m_list]
    plot_adaptation_forgetting_ao(
        m_list,
        ad_list,
        fg_list,
        output_path=out_dir / "plot_ao_adaptation_vs_forgetting.png",
    )

    # Plot AP: Retention Trajectories
    plot_retention_trajectory_ap(
        sample_retentions,
        change_points=change_points_b,
        output_path=out_dir / "plot_ap_retention_trajectories.png",
    )

    # Plot AQ: Retention Modes Comparison
    ret_modes = [
        "Selective_fixed_high",
        "Selective_fixed_low",
        "Selective_adaptive",
        "Selective_oracle",
    ]
    aq_errs = [float(np.mean(task_a_seed_errors[m])) for m in ret_modes]
    aq_nts = [float(np.mean(task_b_negative_transfers[m])) for m in ret_modes]
    plot_retention_ablation_aq(
        ret_modes, aq_errs, aq_nts, output_path=out_dir / "plot_aq_retention_modes.png"
    )

    # Plot AR: DeltaCore vs Linear RLS vs Nonlinear RLS
    ar_models = [
        "OnlineRidge",
        "NonlinearOnlineRidge",
        "FixedDelta",
        "SafeAdaptiveDelta",
        "Selective_adaptive",
    ]
    ar_errs = [float(np.mean(task_a_seed_errors[m])) for m in ar_models]
    plot_model_comparison_ar(
        ar_models, ar_errs, output_path=out_dir / "plot_ar_delta_vs_rls.png"
    )

    # Plot AS: Performance vs Dimensionality
    plot_perf_vs_dim_as(
        dims, scaling_perf, output_path=out_dir / "plot_as_perf_vs_dim.png"
    )

    # Plot AT: Runtime vs Dimensionality
    plot_runtime_vs_dim_at(
        dims, scaling_runtimes, output_path=out_dir / "plot_at_runtime_vs_dim.png"
    )

    # Plot AU: Memory vs Dimensionality
    plot_memory_vs_dim_au(
        dims, scaling_memory, output_path=out_dir / "plot_au_memory_vs_dim.png"
    )

    # Plot AV: Adaptation Efficiency
    av_effs = [adaptation_efficiencies[m] for m in m_list]
    plot_adaptation_efficiency_av(
        m_list, av_effs, output_path=out_dir / "plot_av_adaptation_efficiency.png"
    )

    # Plot AW: Safe vs Unsafe
    aw_pairs = [
        ("AdaptiveDelta", "SafeAdaptiveDelta"),
        ("SelfReferential", "SafeSelfReferential"),
    ]
    aw_u = [
        float(np.mean(task_a_seed_errors["AdaptiveDelta"])),
        float(np.mean(task_a_seed_errors["SelfReferential"])),
    ]
    aw_s = [
        float(np.mean(task_a_seed_errors["SafeAdaptiveDelta"])),
        float(np.mean(task_a_seed_errors["SafeSelfReferential"])),
    ]
    plot_safe_vs_unsafe_aw(
        aw_pairs, aw_u, aw_s, output_path=out_dir / "plot_aw_safe_vs_unsafe.png"
    )

    print(
        "Phase 11 Observatory Publication Plots AN through AW generated successfully."
    )

    # --------------------------------------------------------------------------
    # 9. Serialize Machine-Readable Artifacts
    # --------------------------------------------------------------------------
    config_artifact = {
        "phase": "11",
        "title": "Nonlinear Adaptive State & Selective Retention Benchmark",
        "date": "October 2026",
        "dimension": dim,
        "sequence_length": seq_len,
        "seeds": seeds,
        "delays": delays,
        "scaling_dims": dims,
        "models": m_list,
    }
    with open(out_dir / "phase_11_config.json", "w") as f:
        json.dump(config_artifact, f, indent=2)

    per_seed_artifact = {
        "task_a_seed_errors": task_a_seed_errors,
        "task_a_first_passage": task_a_first_passage,
        "task_a_sustained": task_a_sustained,
        "task_b_adapt_errors": task_b_adapt_errors,
        "task_b_forgetting": task_b_forgetting,
        "task_b_negative_transfers": task_b_negative_transfers,
        "task_c_delay_results": task_c_results,
    }
    with open(out_dir / "phase_11_per_seed.json", "w") as f:
        json.dump(per_seed_artifact, f, indent=2)

    scaling_artifact = {
        "dimensions": dims,
        "performance": scaling_perf,
        "runtimes_us_per_token": scaling_runtimes,
        "scaling_ratios_td_over_t8": scaling_ratios,
        "state_memory_bytes": scaling_memory,
    }
    with open(out_dir / "phase_11_scaling.json", "w") as f:
        json.dump(scaling_artifact, f, indent=2)

    summary_artifact = {
        "metadata": config_artifact,
        "param_counts": param_counts,
        "state_mem_bytes": state_mem_bytes,
        "task_a": {
            m: {
                "mean_error": float(np.mean(task_a_seed_errors[m])),
                "std_error": float(np.std(task_a_seed_errors[m])),
                "first_passage": float(np.mean(task_a_first_passage[m])),
                "sustained": float(np.mean(task_a_sustained[m])),
                "adaptation_energy": float(np.mean(task_a_energies[m])),
                "latency_us_per_token": (float(np.mean(task_a_runtimes[m])) / seq_len)
                * 1000.0,
                "adaptation_efficiency": adaptation_efficiencies[m],
            }
            for m in m_list
        },
        "task_b": {
            m: {
                "adapt_error_b": float(np.mean(task_b_adapt_errors[m])),
                "forgetting": float(np.mean(task_b_forgetting[m])),
                "negative_transfer": float(np.mean(task_b_negative_transfers[m])),
            }
            for m in m_list
        },
        "task_c": {
            m: {d: float(np.mean(task_c_results[m][d])) for d in delays} for m in m_list
        },
        "ablations": {
            "adaptive_state_on": m_on_val,
            "adaptive_state_off": m_off_val,
            "nonlinear_features_on": m_nl_val,
            "nonlinear_features_off": m_lin_val,
        },
    }
    with open(out_dir / "phase_11_results.json", "w") as f:
        json.dump(summary_artifact, f, indent=2)

    print("=" * 88)
    print(f"Phase 11 Benchmark Complete. Artifacts saved to {out_dir.absolute()}")
    print("=" * 88)
    return summary_artifact


if __name__ == "__main__":
    run_phase_11_benchmark()
