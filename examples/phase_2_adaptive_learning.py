r"""Phase 2 Experiment: Adaptive Step-Size Dynamics Under Distribution Shift.

Investigates how different step-size conditioning policies behave when an associative
memory encounters an abrupt distribution shift in key-value associations:
    - Regime A (t = 0..19): Keys mapped to Target Set A.
    - Distribution Shift at t = 20.
    - Regime B (t = 20..39): Same keys mapped to Target Set B (altered associations).

Compares 4 controllers under identical seeds, sequences, and initializations:
    1. Fixed Delta (ConstantStepSize)
    2. Input-Conditioned Delta (InputConditionedStepSize)
    3. Error-Conditioned Delta (ErrorConditionedStepSize)
    4. State-Conditioned Delta (StateConditionedStepSize)

Measures and reports real empirical telemetry:
    - Pre-shift error (end of Regime A)
    - Post-shift error (immediately after shift)
    - Recovery steps (steps to reduce post-shift error by 50%)
    - Final error (end of Regime B)
    - Mean and max step size
    - Mean and max update norm
    - Adaptation gain G_adapt = E_fixed - E_adaptive
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for standalone script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch  # noqa: E402

from deltacore.diagnostics.adaptation import (  # noqa: E402
    compute_adaptation_gain,
    compute_recovery_steps,
    compute_step_size_stats,
)
from deltacore.scans.adaptive import adaptive_scan  # noqa: E402
from deltacore.updates.adaptive_delta import AdaptiveDeltaRule  # noqa: E402
from deltacore.updates.controllers import (  # noqa: E402
    ConstantStepSize,
    ErrorConditionedStepSize,
    InputConditionedStepSize,
    StateConditionedStepSize,
)


def run_experiment():
    torch.manual_seed(123)

    # Dimensionality
    k_dim = 4
    v_dim = 3
    num_items = 4
    steps_per_regime = 20
    total_steps = steps_per_regime * 2

    # 1. Generate Non-Orthogonal Key Manifold
    base_keys = torch.randn(num_items, k_dim, dtype=torch.float64)
    base_keys = base_keys / torch.linalg.norm(base_keys, dim=-1, keepdim=True)

    # 2. Generate Regime A and Regime B Targets (Altered associations)
    targets_a = torch.randn(num_items, v_dim, dtype=torch.float64)
    targets_b = -1.5 * targets_a + 0.5 * torch.randn(
        num_items, v_dim, dtype=torch.float64
    )

    # 3. Assemble Sequential Stream [T, K] and [T, V]
    seq_keys = []
    seq_targets = []

    # Regime A: repeat items sequentially
    for step in range(steps_per_regime):
        idx = step % num_items
        seq_keys.append(base_keys[idx])
        seq_targets.append(targets_a[idx])

    # Regime B: same keys presented with altered targets
    for step in range(steps_per_regime):
        idx = step % num_items
        seq_keys.append(base_keys[idx])
        seq_targets.append(targets_b[idx])

    keys_tensor = torch.stack(seq_keys)  # [T, K]
    targets_tensor = torch.stack(seq_targets)  # [T, V]

    # 4. Define Candidate Controllers
    eta_max = 0.4
    controllers = {
        "Fixed Delta": ConstantStepSize(0.2),
        "Input-Conditioned": InputConditionedStepSize(
            dim=k_dim, eta_max=eta_max, bias=0.0
        ),
        "Error-Conditioned": ErrorConditionedStepSize(
            eta_max=eta_max, scale=0.8, bias=-0.2
        ),
        "State-Conditioned": StateConditionedStepSize(
            eta_max=eta_max, scale=-0.2, bias=0.5
        ),
    }

    results = {}

    print("=" * 85)
    print("DeltaCore Phase 2: Adaptive Update Dynamics Under Distribution Shift")
    print("=" * 85)
    print(
        f"Sequence Length T = {total_steps} (Regime A: steps 0-19, Regime B: steps 20-39)"
    )
    print(
        f"Input Dim K = {k_dim}, Value Dim V = {v_dim}, Target Shift at t = {steps_per_regime}"
    )
    print("=" * 85)

    for name, ctrl in controllers.items():
        ctrl = ctrl.to(dtype=torch.float64)
        rule = AdaptiveDeltaRule(controller=ctrl)

        scan_res = adaptive_scan(keys_tensor, targets_tensor, rule=rule)

        # Compute step-wise squared error: [T]
        step_errors = torch.sum(scan_res.errors**2, dim=-1)  # [T]
        err_norms = torch.linalg.norm(scan_res.errors, dim=-1)  # [T]

        # Metrics
        pre_shift_err = torch.mean(
            step_errors[steps_per_regime - 4 : steps_per_regime]
        ).item()
        post_shift_err = step_errors[steps_per_regime].item()
        final_err = torch.mean(step_errors[total_steps - 4 : total_steps]).item()

        # Recovery steps: steps after t=20 to reduce error to <= 50% of post-shift peak
        regime_b_errs = err_norms[steps_per_regime:]
        recovery = compute_recovery_steps(
            regime_b_errs, threshold_ratio=0.5, baseline_error=regime_b_errs[0].item()
        )

        eta_stats = compute_step_size_stats(scan_res.step_sizes)
        norm_stats = {
            "mean": float(torch.mean(scan_res.update_norms).item()),
            "max": float(torch.max(scan_res.update_norms).item()),
        }

        results[name] = {
            "pre_shift": pre_shift_err,
            "post_shift": post_shift_err,
            "final_err": final_err,
            "recovery": recovery,
            "eta_mean": eta_stats["mean"],
            "eta_max": eta_stats["max"],
            "norm_mean": norm_stats["mean"],
            "norm_max": norm_stats["max"],
        }

    # Fixed baseline for adaptation gain
    fixed_final = results["Fixed Delta"]["final_err"]

    print(
        f"{'Controller':<20} | {'Pre-Shift':<9} | {'Post-Shift':<10} | {'Final Err':<9} | {'Recov':<5} | {'Mean η':<6} | {'Max η':<6} | {'Gain G':<8}"
    )
    print("-" * 85)
    for name, m in results.items():
        rec_str = str(m["recovery"]) if m["recovery"] is not None else "N/A"
        gain = compute_adaptation_gain(fixed_final, m["final_err"])
        gain_str = f"{gain:+.4f}" if name != "Fixed Delta" else "baseline"
        print(
            f"{name:<20} | {m['pre_shift']:<9.4f} | {m['post_shift']:<10.4f} | {m['final_err']:<9.4f} | {rec_str:<5} | {m['eta_mean']:<6.3f} | {m['eta_max']:<6.3f} | {gain_str:<8}"
        )
    print("=" * 85)
    print("Scientific Observation:")
    print("  - Pre-shift error demonstrates convergence rate on initial mapping.")
    print("  - Post-shift error spikes as the state encounters conflicting targets.")
    print(
        "  - Adaptation gain G_adapt = E_fixed - E_adaptive quantifies whether a controller"
    )
    print(
        "    achieved faster/better adaptation (G > 0) or underperformed fixed Delta (G < 0)."
    )
    print("=" * 85)


if __name__ == "__main__":
    run_experiment()
