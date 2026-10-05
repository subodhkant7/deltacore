"""Phase 18: Non-Stationary Classification Benchmark Pipeline & Artifact Serialization.

Executes:
    1. Multi-family transfer evaluation:
       - Family A: Linear boundary rotation (A -> C -> A)
       - Family B: Boundary translation / intercept shift (A -> B_trans -> A)
       - Family C: Nonlinear decision boundary deformation (A -> C_nonlin -> A)
    2. Comprehensive shift dimensions:
       - Covariate shift (anisotropic covariance, invariant decision boundary)
       - Boundary shift (Family A & Family B)
       - Class-prior shift (imbalanced class frequencies, fixed conditional geometry)
       - Gradual shift (continuous drift over time)
       - Abrupt shift (discrete regime transitions)
       - Strong mismatch (adversarial falsification case: persistent stale state penalty)
    3. Causal ablations:
       - State ON (SafeAdaptiveDelta) vs State OFF (AdaptiveStateOFF)
       - Continuous persistent state vs Explicit reset at regime boundaries
       - FixedDelta vs SafeAdaptiveDelta
    4. Negative controls & shortcut audit:
       - Predefined shortcut audit (energy matching, variance balance, 1D leak, regime leak)
       - Label-shuffle negative control (verifying collapse to ~ 1/K chance)
       - Coordinate permutation control (measuring performance invariance)
    5. Dimensional scaling:
       - D in {32, 64, 128, 256}
       - Programmatic verification of exact 4*D^2 bytes persistent state footprint
    6. Multi-seed execution:
       - Seeds = [42, 43, 44, 45, 46]
    7. Cryptographic parameter hash immutability:
       - Bit-for-bit SHA-256 pre/post verification (Delta theta = 0)
    8. Serializes all 10 required Phase 18 JSON artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from deltacore.benchmarks.phase_18.controls import (
    audit_shortcut_statistics_18,
    permute_features_18,
    shuffle_labels_18,
)
from deltacore.benchmarks.phase_18.generators import (
    NonStationaryStream18,
    generate_covariate_shift_stream,
    generate_family_a_rotation,
    generate_family_b_translation,
    generate_family_c_nonlinear,
    generate_gradual_shift_stream,
    generate_prior_shift_stream,
    generate_stationary_dataset_18,
    generate_strong_mismatch_stream,
)
from deltacore.benchmarks.phase_18.models import (
    FrozenLinearClassifier18,
    SafeAdaptiveDeltaClassifier18,
    build_phase_18_model_suite,
    fit_offline_linear_head_18,
)
from deltacore.benchmarks.phase_18.protocol import (
    CausalStreamingProtocol,
    StreamEvaluationResult18,
)


def _serialize_result_metrics(res: StreamEvaluationResult18) -> dict[str, Any]:
    """Extract scalar summary metrics for serialization."""
    d = asdict(res)
    # Remove large step-by-step arrays from summary
    d.pop("rolling_accuracy_history", None)
    d.pop("state_norm_history", None)
    d.pop("step_latencies_us", None)
    return d


def run_phase_18_benchmark(
    output_dir: str | Path = "docs/benchmarks/artifacts/phase_18",
    seeds: list[int] | None = None,
    dimensions: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute complete Phase 18 multi-family classification transfer benchmark.

    Args:
        output_dir: Directory where the 10 Phase 18 JSON artifacts will be saved.
        seeds: List of random seeds for run-to-run reproducibility (default: [42, 43, 44, 45, 46]).
        dimensions: Feature dimensions for scaling evaluation (default: [32, 64, 128, 256]).
        verbose: Print progress to stdout.

    Returns:
        Dictionary containing aggregate benchmark results and artifact paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if dimensions is None:
        dimensions = [32, 64, 128, 256]

    default_dim = 32
    default_classes = 6

    # Frozen evaluation hyperparameter configuration (Section 3)
    frozen_config = {
        "eta0": 0.015,
        "rho": 1.50,
        "alpha_min": 0.95,
        "gamma": 0.10,
        "epsilon": 1e-6,
        "step_size_fixed": 0.015,
        "alpha_fixed": 0.95,
        "default_dim": default_dim,
        "num_classes": default_classes,
        "seeds": seeds,
        "scaling_dims": dimensions,
        "steps_per_regime": 120,
    }

    # Containers for results across seeds
    all_results: dict[str, Any] = {}
    per_seed_results: list[dict[str, Any]] = []
    task_metadata: dict[str, Any] = {}
    shift_results: dict[str, Any] = {}
    retention_results: dict[str, Any] = {}
    scaling_results: list[dict[str, Any]] = []
    hashes_manifest: list[dict[str, Any]] = []
    controls_results: dict[str, Any] = {}
    failures_log: list[dict[str, Any]] = []

    start_time = time.perf_counter()
    if verbose:
        print("=" * 80)
        print("DeltaCore Phase 18: Unseen Classification Transfer & Falsification")
        print(f"Seeds: {seeds} | Scaling D: {dimensions} | Output: {out_path}")
        print("=" * 80)

    # ==========================================================================
    # 1. Multi-Family & Shift Dimensions Evaluation across Seeds
    # ==========================================================================
    task_keys = [
        "family_a_rotation",
        "family_b_translation",
        "family_c_nonlinear",
        "shift_covariate",
        "shift_gradual",
        "shift_prior",
        "shift_strong_mismatch",
    ]

    # Pre-populate aggregated structures
    for t_key in task_keys:
        all_results[t_key] = {}

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n--- Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

        # 1. Pre-training on canonical stationary split
        stat_ds = generate_stationary_dataset_18(
            dim=default_dim,
            num_classes=default_classes,
            seed=seed,
        )

        # 2. Instantiate tasks
        tasks: dict[str, NonStationaryStream18] = {
            "family_a_rotation": generate_family_a_rotation(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
            "family_b_translation": generate_family_b_translation(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
            "family_c_nonlinear": generate_family_c_nonlinear(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
            "shift_covariate": generate_covariate_shift_stream(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
            "shift_gradual": generate_gradual_shift_stream(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
            "shift_prior": generate_prior_shift_stream(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
            "shift_strong_mismatch": generate_strong_mismatch_stream(
                dim=default_dim, num_classes=default_classes, seed=seed
            ),
        }

        # Save metadata on first seed
        if seed_idx == 0:
            for t_key, stream in tasks.items():
                task_metadata[t_key] = {
                    "family_name": stream.family_name,
                    "shift_type": stream.shift_type,
                    "sequence_length": len(stream.inputs),
                    "regime_bounds": stream.regime_bounds,
                    "regime_names": stream.regime_names,
                    "dim": stream.dim,
                    "num_classes": stream.num_classes,
                    "custom_metadata": stream.metadata,
                }

        seed_record: dict[str, Any] = {"seed": seed, "tasks": {}}

        for t_key, stream in tasks.items():
            # Build fresh models for each task
            models = build_phase_18_model_suite(stat_ds, seed=seed)
            task_model_results: dict[str, Any] = {}

            for m_name, model in models.items():
                h_pre = model.get_parameter_hash()

                # Run online causal evaluation
                res = CausalStreamingProtocol.evaluate_stream(
                    model=model,
                    inputs=stream.inputs,
                    targets=stream.targets,
                    regime_bounds=stream.regime_bounds,
                    reset_at_bounds=False,
                )

                h_post = model.get_parameter_hash()
                hashes_manifest.append(
                    {
                        "seed": seed,
                        "task": t_key,
                        "model": m_name,
                        "hash_before": h_pre,
                        "hash_after": h_post,
                        "immutable": h_pre == h_post,
                    }
                )

                res_dict = _serialize_result_metrics(res)
                task_model_results[m_name] = res_dict

                # Aggregate across seeds
                if m_name not in all_results[t_key]:
                    all_results[t_key][m_name] = {
                        "accuracy": [],
                        "balanced_accuracy": [],
                        "log_loss": [],
                        "latency_us": [],
                        "pre_shift_accuracy": [],
                        "post_shift_accuracy": [],
                        "first_passage_recovery": [],
                        "sustained_recovery": [],
                        "cumulative_excess_loss": [],
                        "forgetting": [],
                        "return_regime_accuracy": [],
                        "max_state_norm": [],
                        "adaptation_energy": [],
                        "min_safety_margin": [],
                        "diverged": [],
                    }
                all_results[t_key][m_name]["accuracy"].append(res.accuracy)
                all_results[t_key][m_name]["balanced_accuracy"].append(
                    res.balanced_accuracy
                )
                all_results[t_key][m_name]["log_loss"].append(res.log_loss)
                all_results[t_key][m_name]["latency_us"].append(res.mean_latency_us)
                all_results[t_key][m_name]["pre_shift_accuracy"].append(
                    res.pre_shift_accuracy
                )
                all_results[t_key][m_name]["post_shift_accuracy"].append(
                    res.post_shift_accuracy
                )
                all_results[t_key][m_name]["first_passage_recovery"].append(
                    res.first_passage_recovery
                )
                all_results[t_key][m_name]["sustained_recovery"].append(
                    res.sustained_recovery
                )
                all_results[t_key][m_name]["cumulative_excess_loss"].append(
                    res.cumulative_excess_loss
                )
                all_results[t_key][m_name]["forgetting"].append(res.forgetting)
                all_results[t_key][m_name]["return_regime_accuracy"].append(
                    res.return_regime_accuracy
                )
                all_results[t_key][m_name]["max_state_norm"].append(res.max_state_norm)
                all_results[t_key][m_name]["adaptation_energy"].append(
                    res.adaptation_energy
                )
                all_results[t_key][m_name]["min_safety_margin"].append(
                    res.min_safety_margin
                )
                all_results[t_key][m_name]["diverged"].append(res.diverged)

            seed_record["tasks"][t_key] = task_model_results

        per_seed_results.append(seed_record)

    # Compute mean and std for all_results
    processed_results: dict[str, Any] = {}
    for t_key, models_dict in all_results.items():
        processed_results[t_key] = {}
        for m_name, metric_lists in models_dict.items():
            processed_results[t_key][m_name] = {}
            for met_name, vals in metric_lists.items():
                arr = np.array(vals, dtype=np.float64)
                processed_results[t_key][m_name][f"{met_name}_mean"] = float(
                    np.mean(arr)
                )
                processed_results[t_key][m_name][f"{met_name}_std"] = float(np.std(arr))
                processed_results[t_key][m_name][f"{met_name}_median"] = float(
                    np.median(arr)
                )

    # Shift-specific summary
    shift_results = {
        "shift_dimensions": {
            "covariate": processed_results["shift_covariate"],
            "boundary_rotation": processed_results["family_a_rotation"],
            "boundary_translation": processed_results["family_b_translation"],
            "nonlinear_deformation": processed_results["family_c_nonlinear"],
            "gradual_drift": processed_results["shift_gradual"],
            "prior_imbalance": processed_results["shift_prior"],
            "strong_mismatch": processed_results["shift_strong_mismatch"],
        }
    }

    # ==========================================================================
    # 2. Causal Retention & Reset Ablation: Continuous vs Oracle Reset
    # ==========================================================================
    if verbose:
        print("\n--- Running Retention & Boundary Reset Ablation ---")

    ablation_tasks = [
        "family_a_rotation",
        "family_b_translation",
        "shift_strong_mismatch",
    ]
    retention_results = {}

    for t_key in ablation_tasks:
        continuous_accs: list[float] = []
        reset_accs: list[float] = []
        state_off_accs: list[float] = []

        for seed in seeds:
            stat_ds = generate_stationary_dataset_18(
                dim=default_dim, num_classes=default_classes, seed=seed
            )
            w_h, b_h = fit_offline_linear_head_18(
                stat_ds.train_inputs, stat_ds.train_targets, default_classes
            )

            if t_key == "family_a_rotation":
                stream = generate_family_a_rotation(
                    dim=default_dim, num_classes=default_classes, seed=seed
                )
            elif t_key == "family_b_translation":
                stream = generate_family_b_translation(
                    dim=default_dim, num_classes=default_classes, seed=seed
                )
            else:
                stream = generate_strong_mismatch_stream(
                    dim=default_dim, num_classes=default_classes, seed=seed
                )

            # Continuous state
            m_cont = SafeAdaptiveDeltaClassifier18(
                default_dim, default_classes, w_h, b_h
            )
            res_cont = CausalStreamingProtocol.evaluate_stream(
                m_cont,
                stream.inputs,
                stream.targets,
                stream.regime_bounds,
                reset_at_bounds=False,
            )
            continuous_accs.append(res_cont.accuracy)

            # Explicit reset at regime bounds
            m_reset = SafeAdaptiveDeltaClassifier18(
                default_dim, default_classes, w_h, b_h
            )
            res_reset = CausalStreamingProtocol.evaluate_stream(
                m_reset,
                stream.inputs,
                stream.targets,
                stream.regime_bounds,
                reset_at_bounds=True,
            )
            reset_accs.append(res_reset.accuracy)

            # State OFF
            m_off = FrozenLinearClassifier18(default_dim, default_classes, w_h, b_h)
            res_off = CausalStreamingProtocol.evaluate_stream(
                m_off,
                stream.inputs,
                stream.targets,
                stream.regime_bounds,
                reset_at_bounds=False,
            )
            state_off_accs.append(res_off.accuracy)

        c_arr = np.array(continuous_accs)
        r_arr = np.array(reset_accs)
        o_arr = np.array(state_off_accs)
        delta_persist_vs_reset = float(np.mean(c_arr - r_arr))

        retention_results[t_key] = {
            "continuous_acc_mean": float(np.mean(c_arr)),
            "continuous_acc_std": float(np.std(c_arr)),
            "reset_acc_mean": float(np.mean(r_arr)),
            "reset_acc_std": float(np.std(r_arr)),
            "state_off_acc_mean": float(np.mean(o_arr)),
            "delta_persistent_vs_reset": delta_persist_vs_reset,
            "persistence_benefit": delta_persist_vs_reset > 0.005,
            "persistence_penalty": delta_persist_vs_reset < -0.005,
        }

    # ==========================================================================
    # 3. Negative Controls & Shortcut Audit Suite
    # ==========================================================================
    if verbose:
        print("\n--- Running Negative Controls & Shortcut Audit ---")

    # 1. Audit on standard Family A, B, and C streams
    audit_reports: dict[str, Any] = {}
    for fam_key, gen_fn in [
        ("family_a", generate_family_a_rotation),
        ("family_b", generate_family_b_translation),
        ("family_c", generate_family_c_nonlinear),
    ]:
        st = gen_fn(dim=default_dim, num_classes=default_classes, seed=42)
        audit_reports[fam_key] = audit_shortcut_statistics_18(
            st.inputs, st.targets, st.regime_bounds, default_classes
        )

    # 2. Label-shuffle negative control across all 5 seeds on Family A
    label_shuffle_accs: list[float] = []
    for seed in seeds:
        stat_ds = generate_stationary_dataset_18(
            dim=default_dim, num_classes=default_classes, seed=seed
        )
        w_h, b_h = fit_offline_linear_head_18(
            stat_ds.train_inputs, stat_ds.train_targets, default_classes
        )
        stream = generate_family_a_rotation(
            dim=default_dim, num_classes=default_classes, seed=seed
        )
        shuffled_targets = shuffle_labels_18(stream.targets, seed=seed + 100)

        m = SafeAdaptiveDeltaClassifier18(default_dim, default_classes, w_h, b_h)
        res = CausalStreamingProtocol.evaluate_stream(
            m, stream.inputs, shuffled_targets, stream.regime_bounds
        )
        label_shuffle_accs.append(res.accuracy)

    # 3. Coordinate permutation control
    perm_diffs: list[float] = []
    for seed in seeds:
        stat_ds = generate_stationary_dataset_18(
            dim=default_dim, num_classes=default_classes, seed=seed
        )
        w_h, b_h = fit_offline_linear_head_18(
            stat_ds.train_inputs, stat_ds.train_targets, default_classes
        )
        stream = generate_family_a_rotation(
            dim=default_dim, num_classes=default_classes, seed=seed
        )

        # Baseline evaluation
        m_base = SafeAdaptiveDeltaClassifier18(default_dim, default_classes, w_h, b_h)
        res_base = CausalStreamingProtocol.evaluate_stream(
            m_base, stream.inputs, stream.targets, stream.regime_bounds
        )

        # Permuted evaluation: permute inputs and matching columns of w_head
        perm_inputs, perm_idx = permute_features_18(stream.inputs, seed=seed + 200)
        w_h_perm = w_h[:, perm_idx]

        m_perm = SafeAdaptiveDeltaClassifier18(
            default_dim, default_classes, w_h_perm, b_h
        )
        res_perm = CausalStreamingProtocol.evaluate_stream(
            m_perm, perm_inputs, stream.targets, stream.regime_bounds
        )

        perm_diffs.append(abs(res_base.accuracy - res_perm.accuracy))

    controls_results = {
        "shortcut_audits": audit_reports,
        "all_shortcut_audits_passed": all(
            r["passed_shortcut_audit"] for r in audit_reports.values()
        ),
        "label_shuffle": {
            "per_seed_accuracies": label_shuffle_accs,
            "mean_accuracy": float(np.mean(label_shuffle_accs)),
            "std_accuracy": float(np.std(label_shuffle_accs)),
            "chance_theoretical": 1.0 / default_classes,
            "collapsed_to_chance": float(np.mean(label_shuffle_accs))
            < (1.0 / default_classes + 0.05),
        },
        "feature_permutation_invariance": {
            "per_seed_absolute_differences": perm_diffs,
            "max_difference": float(np.max(perm_diffs)),
            "mean_difference": float(np.mean(perm_diffs)),
            "permutation_invariant": float(np.max(perm_diffs)) < 1e-4,
        },
    }

    # ==========================================================================
    # 4. Dimensional Scaling Evaluation: D in {32, 64, 128, 256}
    # ==========================================================================
    if verbose:
        print("\n--- Running Dimensional Scaling Evaluation ---")

    scaling_results = []
    for d_val in dimensions:
        d_seed_records: list[dict[str, Any]] = []

        for seed in seeds:
            stat_ds_d = generate_stationary_dataset_18(
                dim=d_val, num_classes=default_classes, seed=seed
            )
            w_h, b_h = fit_offline_linear_head_18(
                stat_ds_d.train_inputs, stat_ds_d.train_targets, default_classes
            )
            stream_d = generate_family_a_rotation(
                dim=d_val, num_classes=default_classes, seed=seed
            )

            model_d = SafeAdaptiveDeltaClassifier18(d_val, default_classes, w_h, b_h)
            res_d = CausalStreamingProtocol.evaluate_stream(
                model_d, stream_d.inputs, stream_d.targets, stream_d.regime_bounds
            )

            # Programmatic verification of exact 4*D^2 bytes
            expected_bytes = 4 * d_val * d_val
            actual_bytes = model_d.get_state_memory_bytes()
            assert actual_bytes == expected_bytes, (
                f"Memory mismatch at D={d_val}: actual {actual_bytes} != expected {expected_bytes}"
            )

            d_seed_records.append(
                {
                    "accuracy": res_d.accuracy,
                    "balanced_accuracy": res_d.balanced_accuracy,
                    "log_loss": res_d.log_loss,
                    "latency_us": res_d.mean_latency_us,
                    "max_state_norm": res_d.max_state_norm,
                    "min_safety_margin": res_d.min_safety_margin,
                    "diverged": res_d.diverged,
                }
            )

        accs = [r["accuracy"] for r in d_seed_records]
        lats = [r["latency_us"] for r in d_seed_records]
        norms = [r["max_state_norm"] for r in d_seed_records]
        safeties = [r["min_safety_margin"] for r in d_seed_records]
        divs = [r["diverged"] for r in d_seed_records]

        scaling_results.append(
            {
                "dim": d_val,
                "parameter_count": d_val * default_classes + default_classes,
                "persistent_state_bytes": 4 * d_val * d_val,
                "bytes_formula": "4 * D^2",
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "latency_us_mean": float(np.mean(lats)),
                "latency_us_std": float(np.std(lats)),
                "max_state_norm_mean": float(np.mean(norms)),
                "min_safety_margin_min": float(np.min(safeties)),
                "diverged_count": int(np.sum(divs)),
            }
        )

    # ==========================================================================
    # 5. Scientific Failure & Falsification Log
    # ==========================================================================
    # Document every configuration where persistent state or SafeAdaptiveDelta
    # underperformed an alternative baseline or reset.
    mismatch_pen = retention_results.get("shift_strong_mismatch", {}).get(
        "delta_persistent_vs_reset", 0.0
    )
    if mismatch_pen < 0:
        failures_log.append(
            {
                "task": "shift_strong_mismatch",
                "failure_mode": "Stale State Penalty under Incompatible Regime Transition",
                "severity": "Moderate",
                "continuous_accuracy": retention_results["shift_strong_mismatch"][
                    "continuous_acc_mean"
                ],
                "reset_accuracy": retention_results["shift_strong_mismatch"][
                    "reset_acc_mean"
                ],
                "delta": mismatch_pen,
                "reproducible": True,
                "scientific_interpretation": (
                    "Persistent associative state M_t formed during Regime A acts as an adversarial prior "
                    "when abruptly shifted to an orthogonal cyclic coordinate permutation. An oracle reset "
                    "eliminates this negative transfer immediately."
                ),
            }
        )

    # Check Family C nonlinear representation ceiling
    nonlin_res = processed_results["family_c_nonlinear"]
    safe_nonlin_acc = nonlin_res["SafeAdaptiveDelta"]["accuracy_mean"]
    froz_nonlin_acc = nonlin_res["FrozenLinear"]["accuracy_mean"]
    if safe_nonlin_acc < 0.90:
        failures_log.append(
            {
                "task": "family_c_nonlinear",
                "failure_mode": "Representation Ceiling on Non-Planar Boundary Deformation",
                "severity": "Architectural Limit",
                "safe_adaptive_accuracy": safe_nonlin_acc,
                "frozen_linear_accuracy": froz_nonlin_acc,
                "delta": safe_nonlin_acc - froz_nonlin_acc,
                "reproducible": True,
                "scientific_interpretation": (
                    "Nonlinear coordinate warp phi(x) bends decision boundaries nonlinearly. "
                    "Because SafeAdaptiveDelta operates via linear coordinate transformation z = (I + M)x "
                    "coupled to a frozen linear head, it cannot represent complex non-planar boundaries."
                ),
            }
        )

    # Check Prior shift: compare with classical online models
    prior_res = processed_results["shift_prior"]
    online_ridge_acc = prior_res["OnlineRidge"]["accuracy_mean"]
    online_logreg_acc = prior_res["OnlineLogisticRegression"]["accuracy_mean"]
    best_classical_prior = max(online_ridge_acc, online_logreg_acc)
    safe_prior_acc = prior_res["SafeAdaptiveDelta"]["accuracy_mean"]
    if best_classical_prior > safe_prior_acc:
        failures_log.append(
            {
                "task": "shift_prior",
                "failure_mode": "Sub-optimal Class-Prior Adaptation vs Classical Online Learning",
                "severity": "Minor",
                "safe_adaptive_accuracy": safe_prior_acc,
                "online_ridge_accuracy": best_classical_prior,
                "delta": safe_prior_acc - best_classical_prior,
                "reproducible": True,
                "scientific_interpretation": (
                    "When only class frequencies shift (P(y) imbalanced) while conditional feature geometry "
                    "P(x|y) is invariant, direct parameter adaptation (e.g. OnlineLogisticRegression adjusting intercepts) "
                    "adapts class priors more directly than associative input transformation."
                ),
            }
        )

    # ==========================================================================
    # 6. Serialize All 10 Required JSON Artifacts
    # ==========================================================================
    artifacts_map = {
        "phase_18_config.json": frozen_config,
        "phase_18_results.json": processed_results,
        "phase_18_per_seed.json": per_seed_results,
        "phase_18_task_metadata.json": task_metadata,
        "phase_18_shift_results.json": shift_results,
        "phase_18_retention.json": retention_results,
        "phase_18_scaling.json": scaling_results,
        "phase_18_hashes.json": hashes_manifest,
        "phase_18_controls.json": controls_results,
        "phase_18_failures.json": failures_log,
    }

    for filename, content in artifacts_map.items():
        file_path = out_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    total_time = time.perf_counter() - start_time
    if verbose:
        print("\n" + "=" * 80)
        print(f"Phase 18 benchmark completed in {total_time:.2f} seconds.")
        print(f"Serialized 10 JSON artifacts to {out_path}:")
        for fn in artifacts_map:
            print(f"  - {fn}")
        print("=" * 80)

    return {
        "config": frozen_config,
        "results": processed_results,
        "retention": retention_results,
        "controls": controls_results,
        "scaling": scaling_results,
        "failures": failures_log,
        "artifacts_dir": str(out_path),
        "execution_time_sec": total_time,
    }
