r"""Phase 3 Experiment: Minimal Self-Referential Adaptive Memory Dynamics.

Compares three memory systems under identical seeds, dimensions, and sequences:
    1. Phase 1: Fixed Delta (Constant learning rate \eta_0)
    2. Phase 2: Error-Conditioned Adaptive Delta (\eta_t = f(e_t), external controller)
    3. Phase 3: Minimal Self-Referential Delta (Coupled evolving M_t and C_t)

Evaluates two experimental regimes:
    - Mode A (Stationary): Stationary key-value distribution over T=60 steps to evaluate
      whether self-referential controller memory introduces unnecessary state drift.
    - Mode B (Distribution Shift): Abrupt regime shift at t=40 over T=80 steps to evaluate
      shock response, recovery latency, step-size velocity (\Delta\eta_t), and state growth.

Recovery Latency Specification:
    RecoverySteps = min { j >= 0 : ||e_{t_shift + j}||_2 <= \tau * ||e_{t_shift}||_2 }
    with \tau = 0.5 (50% reduction of immediate post-shift disturbance shock).
    If recovery is not achieved before the end of the horizon, returns None (reported as 'N/A').
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for standalone script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Any  # noqa: E402

import torch  # noqa: E402

from deltacore.diagnostics.self_reference import (  # noqa: E402
    compute_coupling_correlation,
    compute_step_size_changes,
)
from deltacore.memory.associative import AssociativeMemory  # noqa: E402
from deltacore.scans.adaptive import adaptive_scan  # noqa: E402
from deltacore.scans.sequential import sequential_scan  # noqa: E402
from deltacore.updates.adaptive_delta import AdaptiveDeltaRule  # noqa: E402
from deltacore.updates.controllers import ErrorConditionedStepSize  # noqa: E402
from deltacore.updates.delta import DeltaRule  # noqa: E402
from deltacore.updates.dynamics_memory import DynamicsMemory  # noqa: E402
from deltacore.updates.self_referential import (  # noqa: E402
    CoupledState,
    SelfReferentialSystem,
)


def compute_first_passage_recovery_steps(
    errors: torch.Tensor,
    shift_index: int,
    tau: float = 0.5,
) -> int | None:
    r"""Compute first-passage recovery latency following distribution shift.

    Mathematical definition:
        $$\text{FirstPassageRecoverySteps} = \min \{ j \ge 0 : \|e_{t_{\text{shift}} + j}\|_2 \le \tau \cdot \|e_{t_{\text{shift}}}\|_2 \}$$

    Args:
        errors: Error tensor of shape `[T, V]`.
        shift_index: Time index $t_{\text{shift}}$ of the distribution shift.
        tau: Contraction ratio threshold (\tau = 0.5 defines 50% shock attenuation).

    Returns:
        Integer steps to first cross recovery threshold, or None if not met within horizon.
    """
    post_shift_errs = errors[shift_index:]
    if len(post_shift_errs) == 0:
        return None

    err_norms = torch.linalg.norm(post_shift_errs, dim=-1)
    e_ref = err_norms[0].item()

    if e_ref <= 1e-12:
        return 0

    threshold = tau * e_ref
    for j in range(len(err_norms)):
        if err_norms[j].item() <= threshold:
            return j

    return None


def compute_sustained_recovery_steps(
    errors: torch.Tensor,
    shift_index: int,
    window: int = 4,
    tau: float = 0.5,
) -> int | None:
    r"""Compute sustained recovery latency following distribution shift.

    Mathematical definition:
        $$\text{SustainedRecoverySteps}(W) = \min \{ j \ge 0 : \|e_{t_{\text{shift}} + j + w}\|_2 \le \tau \cdot \|e_{t_{\text{shift}}}\|_2, \; \forall w \in \{0, \dots, W-1\} \}$$

    Requires the error to remain below the threshold for the next $W$ consecutive steps.

    Args:
        errors: Error tensor of shape `[T, V]`.
        shift_index: Time index $t_{\text{shift}}$ of the distribution shift.
        window: Number of consecutive steps $W$ error must remain contracted.
        tau: Contraction ratio threshold.

    Returns:
        Integer steps to enter sustained recovery, or None if not met within horizon.
    """
    post_shift_errs = errors[shift_index:]
    if len(post_shift_errs) < window:
        return None

    err_norms = torch.linalg.norm(post_shift_errs, dim=-1)
    e_ref = err_norms[0].item()

    if e_ref <= 1e-12:
        return 0

    threshold = tau * e_ref
    for j in range(len(err_norms) - window + 1):
        if (err_norms[j : j + window] <= threshold).all():
            return j

    return None


# Alias for backward compatibility
compute_operational_recovery_steps = compute_first_passage_recovery_steps


def run_mode_a_stationarity():
    """Mode A: Stationary distribution to test for controller-induced state drift."""
    torch.manual_seed(42)

    k_dim = 4
    v_dim = 3
    num_items = 4
    t_steps = 60

    # Fixed items repeatedly queried
    base_keys = torch.randn(num_items, k_dim, dtype=torch.float64)
    base_keys = base_keys / torch.linalg.norm(base_keys, dim=-1, keepdim=True)
    base_targets = torch.randn(num_items, v_dim, dtype=torch.float64)

    seq_keys = [base_keys[t % num_items] for t in range(t_steps)]
    seq_targets = [base_targets[t % num_items] for t in range(t_steps)]

    keys = torch.stack(seq_keys)  # [T, K]
    targets = torch.stack(seq_targets)  # [T, V]

    eta_base = 0.2
    eta_max = 0.4
    rho = 0.05

    print("\n" + "=" * 95)
    print("MODE A: STATIONARY DISTRIBUTION (No Distribution Shift, T=60)")
    print("=" * 95)
    print(
        "Goal: Measure whether evolving dynamics memory introduces unnecessary state drift."
    )
    print(f"Parameters: K={k_dim}, V={v_dim}, Items={num_items}, Steps={t_steps}")
    print("-" * 95)

    # 1. Fixed Delta
    res_fixed = sequential_scan(keys, targets, rule=DeltaRule(step_size=eta_base))
    fixed_errs = targets - res_fixed.predictions
    fixed_err_norms = torch.linalg.norm(fixed_errs, dim=-1)
    m_fixed_final = (
        res_fixed.final_memory.data
        if isinstance(res_fixed.final_memory, AssociativeMemory)
        else res_fixed.final_memory
    )
    m_fixed_norm = torch.linalg.norm(m_fixed_final).item()

    # 2. Phase 2 Error-Conditioned
    ctrl_p2 = ErrorConditionedStepSize(eta_max=eta_max, scale=0.8, bias=-0.2).to(
        dtype=torch.float64
    )
    res_p2 = adaptive_scan(keys, targets, rule=AdaptiveDeltaRule(controller=ctrl_p2))
    p2_err_norms = torch.linalg.norm(res_p2.errors, dim=-1)
    m_p2_final = (
        res_p2.final_memory.data
        if isinstance(res_p2.final_memory, AssociativeMemory)
        else res_p2.final_memory
    )
    m_p2_norm = torch.linalg.norm(m_p2_final).item()

    # 3. Phase 3 Self-Referential
    sys_p3 = SelfReferentialSystem(eta_max=eta_max, rho=rho).to(dtype=torch.float64)
    init_state = CoupledState(
        content_memory=AssociativeMemory.zeros(v_dim, k_dim, dtype=torch.float64),
        dynamics_memory=DynamicsMemory.zeros(
            1, sys_p3.feature_extractor.d_c, dtype=torch.float64
        ),
    )
    res_p3 = sys_p3.scan(keys, targets, initial_state=init_state)
    p3_err_norms = torch.linalg.norm(res_p3.errors, dim=-1)
    m_p3_norm = torch.linalg.norm(res_p3.final_state.content_memory.data).item()
    c_p3_norm = torch.linalg.norm(res_p3.final_state.dynamics_memory.data).item()

    systems = [
        (
            "Phase 1: Fixed Delta",
            fixed_err_norms,
            torch.full((t_steps,), eta_base),
            m_fixed_norm,
            0.0,
        ),
        ("Phase 2: Error-Conditioned", p2_err_norms, res_p2.step_sizes, m_p2_norm, 0.0),
        (
            "Phase 3: Self-Referential",
            p3_err_norms,
            res_p3.step_sizes,
            m_p3_norm,
            c_p3_norm,
        ),
    ]

    print(
        f"{'System':<28} | {'Init Err':<9} | {'Mid Err (t=30)':<14} | {'Final Err':<10} | {'Mean η':<7} | {'Var(η)':<8} | {'||M_T||':<8} | {'||C_T||':<8}"
    )
    print("-" * 95)
    for name, err_n, etas, m_norm, c_norm in systems:
        init_e = err_n[0].item()
        mid_e = torch.mean(err_n[26:30]).item()
        final_e = torch.mean(err_n[-4:]).item()
        mean_eta = torch.mean(etas).item()
        var_eta = torch.var(etas).item()
        c_str = f"{c_norm:.4f}" if c_norm > 0.0 else "N/A"
        print(
            f"{name:<28} | {init_e:<9.4f} | {mid_e:<14.4f} | {final_e:<10.4f} | {mean_eta:<7.4f} | {var_eta:<8.2e} | {m_norm:<8.4f} | {c_str:<8}"
        )
    print("=" * 95)


def run_mode_b_distribution_shift() -> dict[str, Any]:
    """Mode B: Controlled distribution shift to measure shock adaptation and recovery latency."""
    torch.manual_seed(123)

    k_dim = 4
    v_dim = 3
    num_items = 4
    steps_per_regime = 40
    total_steps = steps_per_regime * 2
    shift_index = steps_per_regime

    # Generate keys
    base_keys = torch.randn(num_items, k_dim, dtype=torch.float64)
    base_keys = base_keys / torch.linalg.norm(base_keys, dim=-1, keepdim=True)

    # Regime A targets and Regime B targets (orthogonal conflicting shift)
    targets_a = torch.randn(num_items, v_dim, dtype=torch.float64)
    targets_b = -1.5 * targets_a + 0.5 * torch.randn(
        num_items, v_dim, dtype=torch.float64
    )

    seq_keys = []
    seq_targets = []
    for t in range(steps_per_regime):
        seq_keys.append(base_keys[t % num_items])
        seq_targets.append(targets_a[t % num_items])
    for t in range(steps_per_regime):
        seq_keys.append(base_keys[t % num_items])
        seq_targets.append(targets_b[t % num_items])

    keys = torch.stack(seq_keys)  # [T, K]
    targets = torch.stack(seq_targets)  # [T, V]

    eta_base = 0.2
    eta_max = 0.4
    rho = 0.05

    print("\n" + "=" * 95)
    print(
        f"MODE B: ABRUPT DISTRIBUTION SHIFT (T={total_steps}, Shift at t={shift_index})"
    )
    print("=" * 95)
    print("Regime A: Steps 0-39 | Regime B: Steps 40-79 (Altered Target Associations)")
    print(
        "Recovery Metric: Steps to reduce post-shift disturbance shock by 50% (tau = 0.5)"
    )
    print("-" * 95)

    # 1. Fixed Delta Baseline
    res_fixed = sequential_scan(keys, targets, rule=DeltaRule(step_size=eta_base))
    fixed_errs = targets - res_fixed.predictions
    fixed_err_norms = torch.linalg.norm(fixed_errs, dim=-1)
    m_fixed_final = (
        res_fixed.final_memory.data
        if isinstance(res_fixed.final_memory, AssociativeMemory)
        else res_fixed.final_memory
    )
    m_fixed_norm = torch.linalg.norm(m_fixed_final).item()
    rec_fixed_fp = compute_first_passage_recovery_steps(
        fixed_errs, shift_index=shift_index, tau=0.5
    )
    rec_fixed_sus = compute_sustained_recovery_steps(
        fixed_errs, shift_index=shift_index, window=4, tau=0.5
    )

    # 2. Phase 2 Error-Conditioned
    ctrl_p2 = ErrorConditionedStepSize(eta_max=eta_max, scale=0.8, bias=-0.2).to(
        dtype=torch.float64
    )
    res_p2 = adaptive_scan(keys, targets, rule=AdaptiveDeltaRule(controller=ctrl_p2))
    p2_err_norms = torch.linalg.norm(res_p2.errors, dim=-1)
    m_p2_final = (
        res_p2.final_memory.data
        if isinstance(res_p2.final_memory, AssociativeMemory)
        else res_p2.final_memory
    )
    m_p2_norm = torch.linalg.norm(m_p2_final).item()
    rec_p2_fp = compute_first_passage_recovery_steps(
        res_p2.errors, shift_index=shift_index, tau=0.5
    )
    rec_p2_sus = compute_sustained_recovery_steps(
        res_p2.errors, shift_index=shift_index, window=4, tau=0.5
    )
    delta_eta_p2 = compute_step_size_changes(res_p2.step_sizes)

    # 3. Phase 3 Self-Referential
    sys_p3 = SelfReferentialSystem(eta_max=eta_max, rho=rho).to(dtype=torch.float64)
    init_state = CoupledState(
        content_memory=AssociativeMemory.zeros(v_dim, k_dim, dtype=torch.float64),
        dynamics_memory=DynamicsMemory.zeros(
            1, sys_p3.feature_extractor.d_c, dtype=torch.float64
        ),
    )
    res_p3 = sys_p3.scan(keys, targets, initial_state=init_state)
    p3_err_norms = torch.linalg.norm(res_p3.errors, dim=-1)
    m_p3_norm = torch.linalg.norm(res_p3.final_state.content_memory.data).item()
    c_p3_norm = torch.linalg.norm(res_p3.final_state.dynamics_memory.data).item()
    rec_p3_fp = compute_first_passage_recovery_steps(
        res_p3.errors, shift_index=shift_index, tau=0.5
    )
    rec_p3_sus = compute_sustained_recovery_steps(
        res_p3.errors, shift_index=shift_index, window=4, tau=0.5
    )
    delta_eta_p3 = compute_step_size_changes(res_p3.step_sizes)

    # Coupling correlation between dynamics updates and step-size velocity
    coupling_corr = compute_coupling_correlation(
        res_p3.dynamics_update_norms, delta_eta_p3
    )

    data = [
        {
            "name": "Fixed Delta",
            "err_norms": fixed_err_norms,
            "etas": torch.full((total_steps,), eta_base),
            "delta_eta": torch.zeros(total_steps),
            "rec_fp": rec_fixed_fp,
            "rec_sus": rec_fixed_sus,
            "m_norm": m_fixed_norm,
            "c_norm": None,
            "has_nan": torch.isnan(m_fixed_final).any().item(),
        },
        {
            "name": "Phase 2 Adaptive",
            "err_norms": p2_err_norms,
            "etas": res_p2.step_sizes,
            "delta_eta": delta_eta_p2,
            "rec_fp": rec_p2_fp,
            "rec_sus": rec_p2_sus,
            "m_norm": m_p2_norm,
            "c_norm": None,
            "has_nan": torch.isnan(m_p2_final).any().item(),
        },
        {
            "name": "Phase 3 Self-Referential",
            "err_norms": p3_err_norms,
            "etas": res_p3.step_sizes,
            "delta_eta": delta_eta_p3,
            "rec_fp": rec_p3_fp,
            "rec_sus": rec_p3_sus,
            "m_norm": m_p3_norm,
            "c_norm": c_p3_norm,
            "has_nan": (
                torch.isnan(res_p3.final_state.content_memory.data).any().item()
                or torch.isnan(res_p3.final_state.dynamics_memory.data).any().item()
            ),
        },
    ]

    # Required Comparative Table:
    # | System | Shift Error | 1st-Pass | Sustained(W=4) | Final Error | Mean η | Δη | Final ||M|| | NaN |
    print(
        f"{'System':<25} | {'Shift Error':<11} | {'1st-Pass':<8} | {'Sustained(W=4)':<14} | {'Final Error':<11} | {'Mean η':<7} | {'Δη':<6} | {'Final ||M||':<11} | {'NaN':<5}"
    )
    print("-" * 115)
    for d in data:
        shift_e = d["err_norms"][shift_index].item()
        final_e = torch.mean(d["err_norms"][-4:]).item()
        rec_fp_str = str(d["rec_fp"]) if d["rec_fp"] is not None else "N/A"
        rec_sus_str = str(d["rec_sus"]) if d["rec_sus"] is not None else "N/A"
        mean_eta = torch.mean(d["etas"]).item()
        mean_d_eta = torch.mean(d["delta_eta"]).item()
        m_norm_str = f"{d['m_norm']:.2f}"
        nan_str = "True" if d["has_nan"] else "False"

        print(
            f"{d['name']:<25} | {shift_e:<11.4f} | {rec_fp_str:<8} | {rec_sus_str:<14} | {final_e:<11.4f} | {mean_eta:<7.4f} | {mean_d_eta:<6.4f} | {m_norm_str:<11} | {nan_str:<5}"
        )
    print("=" * 115)

    print("\nSelf-Reference Coupled Diagnostics:")
    print(f"  - Terminal Dynamics Memory Norm ||C_T||_F: {c_p3_norm:.4f}")
    print(
        f"  - Peak Content Memory Update Norm: {torch.max(res_p3.content_update_norms).item():.4f}"
    )
    print(
        f"  - Peak Dynamics Memory Update Norm: {torch.max(res_p3.dynamics_update_norms).item():.4f}"
    )
    print(f"  - Pearson Correlation (||ΔC_t|| vs Δη_t): {coupling_corr:+.4f}")
    print(
        "    (Note: Pearson correlation is an empirical diagnostic only; it does NOT establish causal direction)"
    )

    return {
        "fixed": data[0],
        "p2": data[1],
        "p3": data[2],
        "coupling_corr": coupling_corr,
    }


def main():
    print(
        "==================================================================================="
    )
    print("DeltaCore Phase 3: Minimal Self-Referential Adaptive Memory Experiment")
    print(
        "==================================================================================="
    )
    run_mode_a_stationarity()
    run_mode_b_distribution_shift()


if __name__ == "__main__":
    main()
