"""DeltaCore Phase 12.1 Forensic Audit & Scientific Rerun.

Performs:
1. SelectiveStateAdaptive parameter and training audit (recording named_parameters, requires_grad, optimizer, gradient norm, delta theta).
2. Training of the 5-parameter controller on spatio-temporal streams via unrolled backprop.
3. Rerun of the principal retention experiment on D=64 across seeds [0, 1, 2, 3, 4] for:
   - Selective_fixed_high
   - Selective_fixed_low
   - Selective_adaptive
   - Selective_state_adaptive (trained 5-param controller)
   - Selective_oracle
   - SafeAdaptiveDelta (useful baseline)
   Both for Phase 12 as-run configuration (feat_dim=8, untrained) and Phase 12.1 corrected configuration (feat_dim=64, trained).
4. Evaluation of USEFUL_STATE vs NO_USEFUL_STATE gate.
5. Measurement of continuous vs reset retention delta: Delta_retention = E_continuous - E_reset.
6. Recording of state evolution diagnostics (state norm, update norm, retention alpha, parameter trajectories).
7. Emission of machine-readable artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deltacore.streaming.metrics import (
    compute_relative_step_error,
)
from deltacore.streaming.models import (
    SafeAdaptiveDeltaPredictor,
    SelectiveRetentionPredictor,
    SelectiveStateAdaptivePredictor,
)
from deltacore.streaming.spatiotemporal_regimes import (
    generate_spatiotemporal_stale_stream,
    generate_spatiotemporal_stream,
)


def run_phase_12_untrained_audit(seed: int = 0) -> dict[str, Any]:
    """Audit the exact Phase 12 model configuration before training."""
    dim = 64
    # Phase 12 instantiation had feat_dim defaulting to 8
    model = SelectiveStateAdaptivePredictor(
        dim=dim,
        use_nonlinear_features=True,
        shuffled_control=False,
        seed=seed,
    )

    named_params = []
    for name, p in model.named_parameters():
        named_params.append(
            {
                "name": name,
                "requires_grad": p.requires_grad,
                "numel": p.numel(),
                "shape": list(p.shape),
                "values": p.detach().cpu().tolist(),
            }
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer inspection in Phase 12
    optimizer_param_groups = []
    # In Phase 12 benchmark, no optimizer was created for SelectiveStateAdaptivePredictor!
    # If one had tried:
    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_param_groups = opt.param_groups
        opt_param_count = sum(len(g["params"]) for g in optimizer_param_groups)
    except ValueError:
        # torch.optim.Adam raises ValueError if param list is empty of trainable parameters in some versions
        opt_param_count = 0

    return {
        "model_name": "SelectiveStateAdaptivePredictor (Phase 12 As-Run)",
        "dim": dim,
        "feat_dim": model.feat_dim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "optimizer_parameter_count": opt_param_count,
        "number_of_optimization_steps": 0,
        "gradient_norm": 0.0,
        "parameter_delta": 0.0,
        "named_parameters": named_params,
        "answers": {
            "why_5_total_but_0_trainable": (
                "In deltacore/streaming/models.py lines 888-895, w_controller and b_controller were explicitly "
                "instantiated as nn.Parameter(..., requires_grad=False) with hardcoded heuristic values "
                "[-1.20, -0.05, -0.60, -0.60] and [2.00]. In examples/phase_12_spatiotemporal_benchmark.py, "
                "no optimizer was created and no loss.backward() was ever invoked. The streaming evaluation loop "
                "solely updated the buffer M via adapt_step() without any parameter optimization."
            ),
            "was_model_trained": False,
            "did_controller_learn": False,
        },
    }


def train_5param_controller(
    dim: int = 64,
    feat_dim: int = 64,
    seed: int = 42,
    num_steps: int = 30,
    lr: float = 0.05,
) -> dict[str, Any]:
    """Train the exact 5-parameter controller on an independent spatio-temporal stream."""
    train_stream = generate_spatiotemporal_stream(
        seq_len=128, height=8, width=8, channels=1, seed=seed + 1000
    )
    inputs = train_stream.inputs
    targets = train_stream.targets
    T, D = inputs.shape

    # Exact 5 parameters
    w = nn.Parameter(
        torch.tensor(
            [-0.50, 0.0, -0.50, -0.50], dtype=torch.float32, requires_grad=True
        )
    )
    b = nn.Parameter(torch.tensor([1.00], dtype=torch.float32, requires_grad=True))
    optimizer = torch.optim.Adam([w, b], lr=lr)

    w_init = w.detach().clone()
    b_init = b.detach().clone()

    def forward_rollout(
        w_c: torch.Tensor, b_c: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        M = torch.zeros(D, D, dtype=torch.float32)
        total_loss = torch.tensor(0.0)
        alpha_min = 0.10
        ema_decay = 0.90
        res_ema = torch.tensor(0.0)
        last_d_norm = torch.tensor(0.0)

        alphas = []
        state_norms = []
        update_norms = []

        for t in range(T):
            x = inputs[t]
            y = targets[t]
            phi = torch.tanh(x)
            pred = M @ phi
            err = y - pred
            err_norm = torch.linalg.norm(err)
            total_loss = total_loss + torch.mean(err**2)

            m_norm = torch.linalg.norm(M)
            if t == 0:
                res_ema = err_norm
            else:
                res_ema = ema_decay * res_ema + (1.0 - ema_decay) * err_norm

            z = torch.stack([err_norm, m_norm, last_d_norm, res_ema])
            v = torch.dot(w_c, z) + b_c[0]
            alpha = alpha_min + (1.0 - alpha_min) * torch.sigmoid(v)
            alphas.append(alpha.item())
            state_norms.append(m_norm.item())

            phi_norm_sq = torch.dot(phi, phi)
            raw_eta = torch.clamp(1.9 / torch.clamp(phi_norm_sq, min=1e-5), max=0.40)

            delta_m = (alpha - 1.0) * M + raw_eta * torch.outer(err, phi)
            M = M + delta_m
            d_norm = torch.linalg.norm(delta_m).detach()
            update_norms.append(d_norm.item())
            last_d_norm = d_norm

        return total_loss / T, {
            "mean_alpha": sum(alphas) / len(alphas),
            "mean_state_norm": sum(state_norms) / len(state_norms),
            "mean_update_norm": sum(update_norms) / len(update_norms),
        }

    loss_init, _ = forward_rollout(w, b)
    loss_init_val = float(loss_init.item())

    trajectories = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "w": [],
        "b": [],
    }

    for step in range(num_steps):
        optimizer.zero_grad()
        loss, _ = forward_rollout(w, b)
        loss.backward()
        assert w.grad is not None and b.grad is not None
        g_norm = float(torch.sqrt(torch.sum(w.grad**2) + torch.sum(b.grad**2)).item())
        trajectories["step"].append(step)
        trajectories["loss"].append(float(loss.item()))
        trajectories["grad_norm"].append(g_norm)
        trajectories["w"].append(w.detach().cpu().tolist())
        trajectories["b"].append(float(b.detach().item()))
        optimizer.step()

    loss_final, _ = forward_rollout(w, b)
    loss_final_val = float(loss_final.item())

    delta_w = float(torch.linalg.norm(w.detach() - w_init).item())
    delta_b = float(torch.abs(b.detach() - b_init).item())
    total_delta = delta_w + delta_b

    return {
        "num_optimization_steps": num_steps,
        "w_init": w_init.tolist(),
        "b_init": b_init.tolist(),
        "w_trained": w.detach().tolist(),
        "b_trained": b.detach().tolist(),
        "training_loss_before": loss_init_val,
        "training_loss_after": loss_final_val,
        "gradient_norm_final": trajectories["grad_norm"][-1],
        "mean_gradient_norm": sum(trajectories["grad_norm"])
        / len(trajectories["grad_norm"]),
        "parameter_delta": total_delta,
        "trajectories": trajectories,
        "w_param": w.detach(),
        "b_param": b.detach(),
    }


def evaluate_principal_retention(
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Rerun only the principal retention experiment on D=64 across seeds [0, 1, 2, 3, 4]."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    dim = 64
    height, width, channels = 8, 8, 1

    # First, train controller for each seed (or shared training stream)
    # We train controller with seed-specific training data to test robust convergence
    trained_controllers = {}
    for s in seeds:
        trained_controllers[s] = train_5param_controller(dim=dim, feat_dim=dim, seed=s)

    # Useful Prediction Gate Definition:
    # E_baseline (Zero prediction) = 1.0
    # delta_useful = 0.015 (3 * empirical seed std of 0.005)
    # Threshold = 0.985
    useful_threshold = 0.985
    delta_floor = 1e-3  # 0.1% minimum effect size to rise above measurement floor

    per_seed_results: list[dict[str, Any]] = []

    for s in seeds:
        eval_stream = generate_spatiotemporal_stream(
            seq_len=256,
            height=height,
            width=width,
            channels=channels,
            shift_interval=64,
            seed=s,
        )
        change_points = eval_stream.change_points

        # Also stale memory stream for negative transfer evaluation (Task C)
        stale_stream = generate_spatiotemporal_stale_stream(
            seq_len=384,
            height=height,
            width=width,
            channels=channels,
            seed=s,
        )
        cp_stale = stale_stream.change_points

        train_info = trained_controllers[s]

        # Model configs to test
        # 1. Useful baseline (SafeAdaptiveDelta)
        # 2. Phase 12.1 corrected (full rank feat_dim=64, trained controller)
        # 3. Phase 12 original as-run (bottleneck feat_dim=8, untrained controller)
        models_to_test = {
            "UsefulBaseline_SafeAdaptiveDelta": SafeAdaptiveDeltaPredictor(
                dim=dim, eta_max=0.50
            ),
            # Corrected full-rank models (feat_dim=64)
            "Selective_fixed_high": SelectiveRetentionPredictor(
                dim=dim,
                feat_dim=dim,
                retention_mode="fixed_high",
                use_nonlinear_features=True,
                seed=s,
            ),
            "Selective_fixed_low": SelectiveRetentionPredictor(
                dim=dim,
                feat_dim=dim,
                retention_mode="fixed_low",
                use_nonlinear_features=True,
                seed=s,
            ),
            "Selective_adaptive": SelectiveRetentionPredictor(
                dim=dim,
                feat_dim=dim,
                retention_mode="adaptive",
                use_nonlinear_features=True,
                seed=s,
            ),
            "Selective_oracle": SelectiveRetentionPredictor(
                dim=dim,
                feat_dim=dim,
                retention_mode="oracle",
                use_nonlinear_features=True,
                oracle_change_points=change_points,
                seed=s,
            ),
            "Selective_state_adaptive_trained": SelectiveStateAdaptivePredictor(
                dim=dim,
                feat_dim=dim,
                use_nonlinear_features=True,
                shuffled_control=False,
                seed=s,
            ),
            # Phase 12 original as-run configurations (for forensic verification)
            "P12_Selective_state_adaptive_untrained_dim8": SelectiveStateAdaptivePredictor(
                dim=dim,
                feat_dim=8,  # Default in Phase 12
                use_nonlinear_features=True,
                shuffled_control=False,
                seed=s,
            ),
            "P12_Selective_fixed_high_dim8": SelectiveRetentionPredictor(
                dim=dim,
                feat_dim=8,
                retention_mode="fixed_high",
                use_nonlinear_features=True,
                seed=s,
            ),
        }

        # Apply trained parameters to Selective_state_adaptive_trained
        w_trained = train_info["w_param"]
        b_trained = train_info["b_param"]
        models_to_test["Selective_state_adaptive_trained"].w_controller.data.copy_(
            w_trained
        )
        models_to_test["Selective_state_adaptive_trained"].b_controller.data.copy_(
            b_trained
        )

        seed_entry: dict[str, Any] = {
            "seed": s,
            "controller_training": {
                "steps": train_info["num_optimization_steps"],
                "w_init": train_info["w_init"],
                "b_init": train_info["b_init"],
                "w_trained": train_info["w_trained"],
                "b_trained": train_info["b_trained"],
                "param_delta": train_info["parameter_delta"],
                "training_loss_before": train_info["training_loss_before"],
                "training_loss_after": train_info["training_loss_after"],
                "mean_grad_norm": train_info["mean_gradient_norm"],
            },
            "models": {},
        }

        for m_name, model in models_to_test.items():
            # Evaluation on primary spatio-temporal task (continuous state)
            model.reset_state()
            preds_cont = []
            state_norms = []
            update_norms = []
            retentions = []

            T_eval = eval_stream.inputs.shape[0]
            for t in range(T_eval):
                x_t = eval_stream.inputs[t]
                y_t = eval_stream.targets[t]
                p_t = model.predict_step(x_t)
                preds_cont.append(p_t)
                model.adapt_step(x_t, y_t)
                state_norms.append(model.get_state_norm())
                update_norms.append(model.last_update_norm)
                if hasattr(model, "last_retention"):
                    retentions.append(model.last_retention)
                else:
                    retentions.append(1.0)

            preds_cont_t = torch.stack(preds_cont)
            err_cont = compute_relative_step_error(preds_cont_t, eval_stream.targets)

            # Evaluation on primary spatio-temporal task (reset state at regime boundaries)
            model.reset_state()
            preds_reset = []
            for t in range(T_eval):
                if t in change_points:
                    model.reset_state()
                x_t = eval_stream.inputs[t]
                y_t = eval_stream.targets[t]
                p_t = model.predict_step(x_t)
                preds_reset.append(p_t)
                model.adapt_step(x_t, y_t)

            preds_reset_t = torch.stack(preds_reset)
            err_reset = compute_relative_step_error(preds_reset_t, eval_stream.targets)

            # Retention Delta: E_continuous - E_reset
            delta_retention = err_cont - err_reset

            # Useful prediction gate determination
            gate_status = (
                "USEFUL_STATE" if err_cont <= useful_threshold else "NO_USEFUL_STATE"
            )
            retention_interpretable = (gate_status == "USEFUL_STATE") and (
                abs(delta_retention) >= delta_floor
            )

            # Stale memory evaluation (A1 -> B -> A2)
            # Continuous:
            T_stale = stale_stream.inputs.shape[0]
            model.reset_state()
            for t in range(T_stale):
                x_t = stale_stream.inputs[t]
                y_t = stale_stream.targets[t]
                if t >= cp_stale[1]:  # Phase A2
                    break
                model.adapt_step(x_t, y_t)
            # Now evaluate in Phase A2 with continuous stale state
            preds_stale_cont = []
            for t in range(cp_stale[1], T_stale):
                x_t = stale_stream.inputs[t]
                y_t = stale_stream.targets[t]
                p_t = model.predict_step(x_t)
                preds_stale_cont.append(p_t)
                model.adapt_step(x_t, y_t)
            err_stale_cont = compute_relative_step_error(
                torch.stack(preds_stale_cont), stale_stream.targets[cp_stale[1] :]
            )

            # Reset: reset state right before entering Phase A2
            model.reset_state()
            for t in range(cp_stale[1]):
                model.adapt_step(stale_stream.inputs[t], stale_stream.targets[t])
            model.reset_state()  # Reset obsolete memory!
            preds_stale_reset = []
            for t in range(cp_stale[1], T_stale):
                x_t = stale_stream.inputs[t]
                y_t = stale_stream.targets[t]
                p_t = model.predict_step(x_t)
                preds_stale_reset.append(p_t)
                model.adapt_step(x_t, y_t)
            err_stale_reset = compute_relative_step_error(
                torch.stack(preds_stale_reset), stale_stream.targets[cp_stale[1] :]
            )
            delta_stale_retention = err_stale_cont - err_stale_reset

            seed_entry["models"][m_name] = {
                "relative_error_continuous": err_cont,
                "relative_error_reset": err_reset,
                "delta_retention": delta_retention,
                "gate_status": gate_status,
                "retention_interpretable": retention_interpretable,
                "stale_a2_err_continuous": err_stale_cont,
                "stale_a2_err_reset": err_stale_reset,
                "delta_stale_retention": delta_stale_retention,
                "mean_state_norm": float(sum(state_norms) / len(state_norms)),
                "mean_update_norm": float(sum(update_norms) / len(update_norms)),
                "mean_retention_alpha": float(sum(retentions) / len(retentions)),
            }

        per_seed_results.append(seed_entry)

    # Compute aggregate metrics across seeds
    model_names = list(per_seed_results[0]["models"].keys())
    aggregated: dict[str, Any] = {}

    for m_name in model_names:
        rel_errors_cont = [
            s["models"][m_name]["relative_error_continuous"] for s in per_seed_results
        ]
        rel_errors_reset = [
            s["models"][m_name]["relative_error_reset"] for s in per_seed_results
        ]
        deltas = [s["models"][m_name]["delta_retention"] for s in per_seed_results]
        stale_deltas = [
            s["models"][m_name]["delta_stale_retention"] for s in per_seed_results
        ]
        gate_statuses = [s["models"][m_name]["gate_status"] for s in per_seed_results]
        interpretable_count = sum(
            1
            for s in per_seed_results
            if s["models"][m_name]["retention_interpretable"]
        )

        mean_err_cont = sum(rel_errors_cont) / len(rel_errors_cont)
        std_err_cont = math.sqrt(
            sum((e - mean_err_cont) ** 2 for e in rel_errors_cont)
            / len(rel_errors_cont)
        )

        mean_err_reset = sum(rel_errors_reset) / len(rel_errors_reset)
        std_err_reset = math.sqrt(
            sum((e - mean_err_reset) ** 2 for e in rel_errors_reset)
            / len(rel_errors_reset)
        )

        mean_delta = sum(deltas) / len(deltas)
        std_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / len(deltas))

        mean_stale_delta = sum(stale_deltas) / len(stale_deltas)
        std_stale_delta = math.sqrt(
            sum((d - mean_stale_delta) ** 2 for d in stale_deltas) / len(stale_deltas)
        )

        majority_gate = (
            "USEFUL_STATE"
            if (sum(1 for g in gate_statuses if g == "USEFUL_STATE") >= 3)
            else "NO_USEFUL_STATE"
        )

        aggregated[m_name] = {
            "mean_relative_error_continuous": mean_err_cont,
            "std_relative_error_continuous": std_err_cont,
            "mean_relative_error_reset": mean_err_reset,
            "std_relative_error_reset": std_err_reset,
            "mean_delta_retention": mean_delta,
            "std_delta_retention": std_delta,
            "mean_delta_stale_retention": mean_stale_delta,
            "std_delta_stale_retention": std_stale_delta,
            "gate_status": majority_gate,
            "interpretable_seeds": interpretable_count,
            "per_seed_errors_continuous": rel_errors_cont,
            "per_seed_errors_reset": rel_errors_reset,
            "per_seed_deltas_retention": deltas,
            "per_seed_deltas_stale": stale_deltas,
        }

    return {
        "useful_threshold": useful_threshold,
        "delta_floor": delta_floor,
        "seeds": seeds,
        "aggregated": aggregated,
        "per_seed": per_seed_results,
        "controller_trajectories_sample": trained_controllers[0]["trajectories"],
    }


def main() -> None:
    output_dir = Path("docs/benchmarks/artifacts/phase_12_1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Executing Section 2: Selective Parameter Audit...")
    audit_data = run_phase_12_untrained_audit(seed=0)
    with open(output_dir / "selective_parameter_audit.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    print("Selective Parameter Audit saved.")

    print("Executing Section 5 & 6: Controller Training & Principal Retention Rerun...")
    rerun_data = evaluate_principal_retention(seeds=[0, 1, 2, 3, 4])

    # Save phase_12_1_results.json
    results_json = {
        "benchmark_metadata": {
            "phase": "12.1",
            "title": "Forensic Scientific & Code Integrity Correction",
            "task": "Spatio-Temporal Regime Switch D=64 (8x8x1)",
            "useful_prediction_threshold": rerun_data["useful_threshold"],
            "retention_noise_floor": rerun_data["delta_floor"],
            "seeds": rerun_data["seeds"],
        },
        "summary": rerun_data["aggregated"],
        "controller_training_sample_seed0": {
            "w_init": rerun_data["per_seed"][0]["controller_training"]["w_init"],
            "b_init": rerun_data["per_seed"][0]["controller_training"]["b_init"],
            "w_trained": rerun_data["per_seed"][0]["controller_training"]["w_trained"],
            "b_trained": rerun_data["per_seed"][0]["controller_training"]["b_trained"],
            "param_delta": rerun_data["per_seed"][0]["controller_training"][
                "param_delta"
            ],
            "mean_grad_norm": rerun_data["per_seed"][0]["controller_training"][
                "mean_grad_norm"
            ],
            "loss_before": rerun_data["per_seed"][0]["controller_training"][
                "training_loss_before"
            ],
            "loss_after": rerun_data["per_seed"][0]["controller_training"][
                "training_loss_after"
            ],
        },
    }
    with open(output_dir / "phase_12_1_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Save phase_12_1_per_seed.json
    with open(output_dir / "phase_12_1_per_seed.json", "w") as f:
        json.dump(rerun_data["per_seed"], f, indent=2)
    print("Results and Per-Seed data saved.")


if __name__ == "__main__":
    main()
