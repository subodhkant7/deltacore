"""Phase 17: Online Non-Stationary Classification Benchmark Pipeline.

Executes:
    - Task A: Stationary classification (useful prediction gate)
    - Task B: Abrupt label-preserving distribution shift (A -> B -> A covariate shift)
    - Task C: Decision-boundary shift (A -> C -> A hyperplane rotation)
    - Task D: Retention stress & stale state ablation (continuous, reset, fixed-high, fixed-low, adaptive)
    - Dimensional scaling: D in {32, 64, 128, 256}
    - Robustness controls: Label shuffle (negative control) and feature permutation
    - Multi-seed execution across seeds = [42, 43, 44, 45, 46]
    - Serializes all 6 required Phase 17 JSON artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deltacore.classification.metrics import (
    ClassificationMetrics,
    compute_classification_telemetry,
)
from deltacore.classification.models import (
    FixedDeltaClassifier,
    FrozenLinearClassifier,
    GRUClassifier,
    LSTMClassifier,
    OnlineClassificationPredictor,
    OnlineLogisticRegression,
    OnlineMulticlassLinear,
    OnlineRidgeClassifier,
    SafeAdaptiveDeltaClassifier,
    SmallMLPClassifier,
    StateOffAblationClassifier,
    fit_offline_gru_cell,
    fit_offline_linear_head,
    fit_offline_lstm_cell,
    fit_offline_small_mlp,
)
from deltacore.classification.synthetic_stream import (
    SEVERITY_CONFIGS,
    StationaryDataset,
    audit_shortcut_statistics,
    generate_nonstationary_stream,
    generate_stationary_dataset,
    permute_features,
    shuffle_labels,
)


def evaluate_online_stream(
    model: OnlineClassificationPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    regime_bounds: list[int] | None = None,
    reset_at_bounds: bool = False,
) -> ClassificationMetrics:
    """Evaluate predictor online sample-by-sample, updating state after label revelation."""
    model.reset_state()
    h_before = model.get_parameter_hash()

    if regime_bounds is None:
        regime_bounds = []

    preds: list[int] = []
    probs: list[torch.Tensor] = []
    targs: list[int] = []
    state_norms: list[float] = []
    etas: list[float] = []
    safety_margins: list[float] = []
    update_norms: list[float] = []
    latencies: list[float] = []

    for t in range(len(inputs)):
        # Optional state reset at regime boundaries
        if reset_at_bounds and t in regime_bounds:
            model.reset_state()

        x_t = inputs[t]
        y_t = int(targets[t].item())

        # 1. Prediction step (label not visible)
        t0 = time.perf_counter()
        pred_y, prob_y = model.predict_step(x_t)
        dt = (time.perf_counter() - t0) * 1e6
        latencies.append(dt)

        preds.append(pred_y)
        probs.append(prob_y.detach().cpu())
        targs.append(y_t)

        # 2. Online adaptation step (label revealed)
        model.adapt_step(x_t, y_t)

        # Record telemetry
        if isinstance(model, (SafeAdaptiveDeltaClassifier, FixedDeltaClassifier)):
            m_norm = float(torch.linalg.norm(model.M).item())
        elif isinstance(model, (GRUClassifier, LSTMClassifier)):
            m_norm = float(torch.linalg.norm(model.h).item())
        elif isinstance(model, OnlineRidgeClassifier):
            m_norm = float(torch.linalg.norm(model.W).item())
        elif isinstance(model, (OnlineMulticlassLinear, OnlineLogisticRegression)):
            m_norm = float(torch.linalg.norm(model.weight).item())
        else:
            m_norm = 0.0

        state_norms.append(m_norm)

        etas.append(getattr(model, "last_eta", 0.0))
        safety_margins.append(getattr(model, "last_safety_margin", 2.0))
        update_norms.append(getattr(model, "last_update_norm", 0.0))

    # Strict check: offline parameters must NOT mutate during streaming
    h_after = model.get_parameter_hash()
    if h_before != h_after:
        raise RuntimeError(
            f"Parameter mutation detected in {model.name}! Hash before != after."
        )

    num_classes = int(torch.max(targets).item()) + 1
    return compute_classification_telemetry(
        predictions=preds,
        probabilities=probs,
        targets=targs,
        regime_bounds=regime_bounds,
        state_norms=state_norms,
        etas=etas,
        safety_margins=safety_margins,
        update_norms=update_norms,
        latencies=latencies,
        persistent_bytes=model.get_state_memory_bytes(),
        parameter_count=model.get_parameter_count(),
        model_name=model.name,
        num_classes=num_classes,
    )


def build_model_suite(
    stat_ds: StationaryDataset,
    seed: int = 42,
) -> dict[str, OnlineClassificationPredictor]:
    """Train offline base models and instantiate full classification benchmark suite."""
    dim = stat_ds.dim
    num_classes = stat_ds.num_classes

    # 1. Offline fit linear head
    w_head, b_head = fit_offline_linear_head(
        stat_ds.train_inputs, stat_ds.train_targets, num_classes, epochs=40, lr=0.02
    )

    # 2. Offline fit small MLP
    mlp_net = fit_offline_small_mlp(
        stat_ds.train_inputs,
        stat_ds.train_targets,
        num_classes,
        hidden_dim=16,
        epochs=40,
        lr=0.01,
    )

    # 3. Offline fit GRU cell
    gru_cell, gru_head = fit_offline_gru_cell(
        stat_ds.train_inputs,
        stat_ds.train_targets,
        num_classes,
        hidden_dim=16,
        epochs=35,
        lr=0.01,
    )

    # 4. Offline fit LSTM cell
    lstm_cell, lstm_head = fit_offline_lstm_cell(
        stat_ds.train_inputs,
        stat_ds.train_targets,
        num_classes,
        hidden_dim=16,
        epochs=35,
        lr=0.01,
    )

    suite: dict[str, OnlineClassificationPredictor] = {
        "FrozenLinear": FrozenLinearClassifier(dim, num_classes, w_head, b_head),
        "SmallMLP": SmallMLPClassifier(dim, 16, num_classes, mlp_net),
        "OnlineLogisticRegression": OnlineLogisticRegression(dim, num_classes, lr=0.05),
        "OnlineRidge": OnlineRidgeClassifier(dim, num_classes, lam=0.99, delta=50.0),
        "OnlineMulticlassLinear": OnlineMulticlassLinear(
            dim, num_classes, step_size=0.05
        ),
        "GRU": GRUClassifier(dim, 16, num_classes, gru_cell, gru_head),
        "LSTM": LSTMClassifier(dim, 16, num_classes, lstm_cell, lstm_head),
        "FixedDelta": FixedDeltaClassifier(
            dim, num_classes, w_head, b_head, step_size=0.05, alpha=0.99
        ),
        "SafeAdaptiveDelta": SafeAdaptiveDeltaClassifier(
            dim,
            num_classes,
            w_head,
            b_head,
            eta0=0.08,
            rho=1.50,
            alpha_min=0.95,
            gamma=0.10,
        ),
        "AdaptiveStateOFF": StateOffAblationClassifier(
            dim, num_classes, w_head, b_head
        ),
    }

    return suite


def run_phase_17_benchmark(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    dim: int = 32,
    num_classes: int = 6,
    output_dir: Path | str = "docs/benchmarks/artifacts/phase_17",
) -> dict[str, Any]:
    """Execute complete Phase 17 benchmark suite and serialize all 6 required artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DELTACORE PHASE 17: ONLINE NON-STATIONARY CLASSIFICATION BENCHMARK")
    print(f"Configuration: D={dim}, K={num_classes}, Seeds={seeds}")
    print("=" * 80)

    # 1. Audit shortcut statistics on stationary base distribution
    stat_sample = generate_stationary_dataset(dim=dim, num_classes=num_classes, seed=42)
    shortcut_audit = audit_shortcut_statistics(
        stat_sample.test_inputs, stat_sample.test_targets, num_classes
    )
    print(
        f"[Shortcut Audit] Energy Matched: {shortcut_audit['energy_matched']}, No Leak: {shortcut_audit['no_feature_leak']}"
    )

    per_seed_results: dict[str, Any] = {f"seed_{s}": {} for s in seeds}
    shift_results: dict[str, Any] = {}
    retention_results: dict[str, Any] = {}
    scaling_results: dict[str, Any] = {}

    # Initialize aggregators
    stationary_accs: dict[str, list[float]] = {}
    covariate_accs: dict[str, list[float]] = {}
    boundary_accs: dict[str, list[float]] = {}
    runtimes: dict[str, list[float]] = {}
    state_memories: dict[str, int] = {}
    param_counts: dict[str, int] = {}

    for s in seeds:
        print(f"\n---> Executing Benchmark on Seed {s}...")
        stat_ds = generate_stationary_dataset(dim=dim, num_classes=num_classes, seed=s)
        models = build_model_suite(stat_ds, seed=s)

        # -------------------------------------------------------------
        # Task A: Stationary Evaluation (Useful-Prediction Gate)
        # -------------------------------------------------------------
        seed_stat = {}
        for m_name, m_inst in models.items():
            m_inst.reset_state()
            metrics = evaluate_online_stream(
                m_inst, stat_ds.test_inputs, stat_ds.test_targets
            )
            seed_stat[m_name] = {
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "log_loss": metrics.log_loss,
                "latency_us": metrics.mean_latency_us,
                "state_bytes": metrics.persistent_state_bytes,
                "params": metrics.parameter_count,
            }
            stationary_accs.setdefault(m_name, []).append(metrics.accuracy)
            runtimes.setdefault(m_name, []).append(metrics.mean_latency_us)
            state_memories[m_name] = metrics.persistent_state_bytes
            param_counts[m_name] = metrics.parameter_count

        per_seed_results[f"seed_{s}"]["task_A_stationary"] = seed_stat

        # -------------------------------------------------------------
        # Task B: Abrupt Covariate Shift (A -> B -> A)
        # -------------------------------------------------------------
        cov_stream = generate_nonstationary_stream(
            dim=dim,
            num_classes=num_classes,
            steps_per_regime=120,
            severity="moderate",
            shift_mode="covariate",
            seed=s,
        )
        seed_cov = {}
        for m_name, m_inst in models.items():
            m_inst.reset_state()
            metrics = evaluate_online_stream(
                m_inst, cov_stream.inputs, cov_stream.targets, cov_stream.regime_bounds
            )
            seed_cov[m_name] = {
                "accuracy": metrics.accuracy,
                "post_shift_acc": metrics.post_shift_accuracy,
                "fp_recovery": metrics.first_passage_recovery,
                "sust_recovery": metrics.sustained_recovery,
                "excess_loss": metrics.cumulative_excess_loss,
                "adapt_energy": metrics.adaptation_energy,
                "max_norm": metrics.max_state_norm,
                "rolling_acc": metrics.rolling_accuracy_history,
            }
            covariate_accs.setdefault(m_name, []).append(metrics.accuracy)

        per_seed_results[f"seed_{s}"]["task_B_covariate"] = seed_cov

        # -------------------------------------------------------------
        # Task C: Decision-Boundary Shift (A -> C -> A)
        # -------------------------------------------------------------
        bound_stream = generate_nonstationary_stream(
            dim=dim,
            num_classes=num_classes,
            steps_per_regime=120,
            severity="moderate",
            shift_mode="decision_boundary",
            seed=s,
        )
        seed_bound = {}
        for m_name, m_inst in models.items():
            m_inst.reset_state()
            metrics = evaluate_online_stream(
                m_inst,
                bound_stream.inputs,
                bound_stream.targets,
                bound_stream.regime_bounds,
            )

            seed_bound[m_name] = {
                "accuracy": metrics.accuracy,
                "post_shift_acc": metrics.post_shift_accuracy,
                "forgetting": metrics.forgetting,
                "sust_recovery": metrics.sustained_recovery,
                "excess_loss": metrics.cumulative_excess_loss,
                "adapt_energy": metrics.adaptation_energy,
            }
            boundary_accs.setdefault(m_name, []).append(metrics.accuracy)

        per_seed_results[f"seed_{s}"]["task_C_boundary"] = seed_bound

    # -------------------------------------------------------------
    # Task D: Retention Stress & Reset Ablation (Seed 42)
    # -------------------------------------------------------------
    print("\n---> Running Task D: Retention Stress & Reset Ablation...")
    stat_42 = generate_stationary_dataset(dim=dim, num_classes=num_classes, seed=42)
    w_head_42, b_head_42 = fit_offline_linear_head(
        stat_42.train_inputs, stat_42.train_targets, num_classes
    )
    stress_stream = generate_nonstationary_stream(
        dim=dim,
        num_classes=num_classes,
        steps_per_regime=120,
        severity="moderate",
        shift_mode="all_regimes",
        seed=42,
    )

    # 1. Continuous vs Reset State
    m_cont = SafeAdaptiveDeltaClassifier(dim, num_classes, w_head_42, b_head_42)
    res_cont = evaluate_online_stream(
        m_cont,
        stress_stream.inputs,
        stress_stream.targets,
        stress_stream.regime_bounds,
        reset_at_bounds=False,
    )

    m_reset = SafeAdaptiveDeltaClassifier(dim, num_classes, w_head_42, b_head_42)
    res_reset = evaluate_online_stream(
        m_reset,
        stress_stream.inputs,
        stress_stream.targets,
        stress_stream.regime_bounds,
        reset_at_bounds=True,
    )

    # 2. Retention modes: Fixed-High (0.99), Fixed-Low (0.70), Safe Adaptive
    m_high = FixedDeltaClassifier(
        dim, num_classes, w_head_42, b_head_42, step_size=0.08, alpha=0.99
    )
    res_high = evaluate_online_stream(
        m_high, stress_stream.inputs, stress_stream.targets, stress_stream.regime_bounds
    )

    m_low = FixedDeltaClassifier(
        dim, num_classes, w_head_42, b_head_42, step_size=0.08, alpha=0.70
    )
    res_low = evaluate_online_stream(
        m_low, stress_stream.inputs, stress_stream.targets, stress_stream.regime_bounds
    )

    retention_results = {
        "continuous": {
            "accuracy": res_cont.accuracy,
            "loss": res_cont.log_loss,
            "excess_loss": res_cont.cumulative_excess_loss,
            "rolling_acc": res_cont.rolling_accuracy_history,
            "state_norms": res_cont.state_norm_history,
        },
        "reset_at_bounds": {
            "accuracy": res_reset.accuracy,
            "loss": res_reset.log_loss,
            "excess_loss": res_reset.cumulative_excess_loss,
            "rolling_acc": res_reset.rolling_accuracy_history,
            "state_norms": res_reset.state_norm_history,
        },
        "fixed_high_retention": {
            "alpha": 0.99,
            "accuracy": res_high.accuracy,
            "excess_loss": res_high.cumulative_excess_loss,
            "rolling_acc": res_high.rolling_accuracy_history,
        },
        "fixed_low_retention": {
            "alpha": 0.70,
            "accuracy": res_low.accuracy,
            "excess_loss": res_low.cumulative_excess_loss,
            "rolling_acc": res_low.rolling_accuracy_history,
        },
        "adaptive_retention": {
            "accuracy": res_cont.accuracy,
            "excess_loss": res_cont.cumulative_excess_loss,
            "rolling_acc": res_cont.rolling_accuracy_history,
        },
        "regime_bounds": stress_stream.regime_bounds,
    }

    # -------------------------------------------------------------
    # Shift Severity Experiment (Mild, Moderate, Severe)
    # -------------------------------------------------------------
    print("\n---> Running Shift Severity Grid...")
    shift_results = {"covariate": {}, "decision_boundary": {}}
    for smode in ["covariate", "decision_boundary"]:
        for sev in ["mild", "moderate", "severe"]:
            st_stream = generate_nonstationary_stream(
                dim=dim,
                num_classes=num_classes,
                steps_per_regime=120,
                severity=sev,
                shift_mode=smode,
                seed=42,
            )
            shift_results[smode][sev] = {}
            for m_test in [
                "FrozenLinear",
                "OnlineLogisticRegression",
                "OnlineRidge",
                "SafeAdaptiveDelta",
            ]:
                m_obj = build_model_suite(stat_42, seed=42)[m_test]
                res_sev = evaluate_online_stream(
                    m_obj, st_stream.inputs, st_stream.targets, st_stream.regime_bounds
                )
                shift_results[smode][sev][m_test] = {
                    "accuracy": res_sev.accuracy,
                    "post_shift_acc": res_sev.post_shift_accuracy,
                    "sust_recovery": res_sev.sustained_recovery,
                    "excess_loss": res_sev.cumulative_excess_loss,
                    "adapt_energy": res_sev.adaptation_energy,
                    "max_norm": res_sev.max_state_norm,
                }

    # -------------------------------------------------------------
    # Task Robustness Controls (Label Shuffle & Feature Permutation)
    # -------------------------------------------------------------
    print("\n---> Running Robustness Controls (Label-Shuffle & Permutation)...")
    ctrl_stream = generate_nonstationary_stream(
        dim=dim, num_classes=num_classes, steps_per_regime=100, seed=42
    )
    # Negative Control: Shuffled Labels
    shuffled_y = shuffle_labels(ctrl_stream.targets, seed=99)
    # Permutation Control: Permuted Features
    perm_x, perm_idx = permute_features(ctrl_stream.inputs, seed=101)

    controls = {}
    for m_test in ["FrozenLinear", "OnlineRidge", "SafeAdaptiveDelta"]:
        m_normal = build_model_suite(stat_42, seed=42)[m_test]
        res_normal = evaluate_online_stream(
            m_normal, ctrl_stream.inputs, ctrl_stream.targets
        )

        m_shuff = build_model_suite(stat_42, seed=42)[m_test]
        res_shuff = evaluate_online_stream(m_shuff, ctrl_stream.inputs, shuffled_y)

        # Permutation equivariance check on SafeAdaptiveDelta
        m_perm = build_model_suite(stat_42, seed=42)[m_test]
        res_perm = evaluate_online_stream(m_perm, perm_x, ctrl_stream.targets)

        controls[m_test] = {
            "normal_accuracy": res_normal.accuracy,
            "label_shuffled_accuracy": res_shuff.accuracy,
            "feature_permuted_accuracy": res_perm.accuracy,
            "chance_level": 1.0 / num_classes,
        }

    # -------------------------------------------------------------
    # Dimensional Scaling (D = 32, 64, 128, 256)
    # -------------------------------------------------------------
    print("\n---> Running Dimensional Scaling Sweep (D in {32, 64, 128, 256})...")
    dims = [32, 64, 128, 256]
    scaling_results = {"dimensions": dims, "models": {}}
    for d_val in dims:
        stat_d = generate_stationary_dataset(
            dim=d_val, num_classes=num_classes, seed=42
        )
        stream_d = generate_nonstationary_stream(
            dim=d_val,
            num_classes=num_classes,
            steps_per_regime=100,
            severity="moderate",
            seed=42,
        )
        suite_d = build_model_suite(stat_d, seed=42)
        for m_name in [
            "FrozenLinear",
            "OnlineRidge",
            "FixedDelta",
            "SafeAdaptiveDelta",
        ]:
            m_inst = suite_d[m_name]
            res_d = evaluate_online_stream(
                m_inst, stream_d.inputs, stream_d.targets, stream_d.regime_bounds
            )
            scaling_results["models"].setdefault(m_name, []).append(
                {
                    "dim": d_val,
                    "accuracy": res_d.accuracy,
                    "latency_us": res_d.mean_latency_us,
                    "state_memory_bytes": res_d.persistent_state_bytes,
                    "max_state_norm": res_d.max_state_norm,
                    "min_safety_margin": res_d.min_safety_margin,
                    "diverged": res_d.diverged,
                }
            )

    # -------------------------------------------------------------
    # Aggregate Results Across Seeds
    # -------------------------------------------------------------
    results_agg = {}
    for m in stationary_accs:
        results_agg[m] = {
            "stat_acc_mean": float(np.mean(stationary_accs[m])),
            "stat_acc_std": float(np.std(stationary_accs[m])),
            "cov_acc_mean": float(np.mean(covariate_accs[m])),
            "cov_acc_std": float(np.std(covariate_accs[m])),
            "bound_acc_mean": float(np.mean(boundary_accs[m])),
            "bound_acc_std": float(np.std(boundary_accs[m])),
            "latency_us_mean": float(np.mean(runtimes[m])),
            "latency_us_std": float(np.std(runtimes[m])),
            "state_bytes": state_memories[m],
            "params": param_counts[m],
        }

    # Configuration summary
    config_record = {
        "dim": dim,
        "num_classes": num_classes,
        "seeds": list(seeds),
        "steps_per_regime": 120,
        "severities": list(SEVERITY_CONFIGS.keys()),
        "shortcut_audit": shortcut_audit,
        "controls": controls,
    }

    # -------------------------------------------------------------
    # Serialize All 6 Required JSON Artifacts
    # -------------------------------------------------------------
    with open(out_path / "phase_17_results.json", "w") as f:
        json.dump(results_agg, f, indent=2)

    with open(out_path / "phase_17_per_seed.json", "w") as f:
        json.dump(per_seed_results, f, indent=2)

    with open(out_path / "phase_17_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    with open(out_path / "phase_17_shift_results.json", "w") as f:
        json.dump(shift_results, f, indent=2)

    with open(out_path / "phase_17_retention.json", "w") as f:
        json.dump(retention_results, f, indent=2)

    with open(out_path / "phase_17_scaling.json", "w") as f:
        json.dump(scaling_results, f, indent=2)

    print(
        "\n[Serialization] Successfully wrote all 6 Phase 17 JSON artifacts to:",
        out_path,
    )
    return results_agg
