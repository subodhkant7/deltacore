# ==============================================================================
# DeltaCore: examples/phase_8_five_memory.py
# Phase 8 Experiment: Five-Memory Self-Modifying Reference Core Comparison.
# ==============================================================================

import math
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch  # noqa: E402

from deltacore.memory.associative import AssociativeMemory  # noqa: E402
from deltacore.memory.five_memory import FiveMemoryState  # noqa: E402
from deltacore.observatory.fingerprint import (  # noqa: E402
    compare_fingerprints,
    generate_fingerprint,
)
from deltacore.observatory.plots import generate_all_plots  # noqa: E402
from deltacore.observatory.report import generate_observatory_report  # noqa: E402
from deltacore.observatory.schema import StateTrajectory, TrajectoryStep  # noqa: E402
from deltacore.stability.controllers import (  # noqa: E402
    SafeDynamicsRateController,
    SafeStepSizeController,
    UnconstrainedController,
)
from deltacore.updates.adaptive_delta import AdaptiveDeltaRule  # noqa: E402
from deltacore.updates.controllers import InputConditionedStepSize  # noqa: E402
from deltacore.updates.delta import DeltaRule  # noqa: E402
from deltacore.updates.dynamics_memory import DynamicsMemory  # noqa: E402
from deltacore.updates.five_memory import (  # noqa: E402
    FiveMemoryConfig,
    FiveMemoryScanResult,
    FiveMemorySystem,
)
from deltacore.updates.self_referential import (  # noqa: E402
    CoupledState,
    SelfReferentialSystem,
)


def generate_synthetic_data(
    t_steps: int = 60,
    d_in: int = 8,
    k_dim: int = 8,
    v_dim: int = 6,
    shift_step: int | None = 30,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Generate synthetic sequence data for stationary or distribution-shift conditions."""
    gen = torch.Generator().manual_seed(seed)

    # Static key projection
    w_key = torch.randn(k_dim, d_in, generator=gen) / math.sqrt(d_in)

    # Value association mapping regimes
    w_val_1 = torch.randn(v_dim, k_dim, generator=gen) / math.sqrt(k_dim)
    w_val_2 = -w_val_1  # Exact inverted distribution shift

    # Input tokens
    x_seq = torch.randn(t_steps, d_in, generator=gen)

    # Compute keys and targets
    keys = torch.matmul(x_seq, w_key.t())  # [T, K]
    targets = torch.zeros(t_steps, v_dim)

    for t in range(t_steps):
        noise = torch.randn(v_dim, generator=gen) * 0.05
        if shift_step is not None and t >= shift_step:
            targets[t] = torch.matmul(w_val_2, keys[t]) + noise
        else:
            targets[t] = torch.matmul(w_val_1, keys[t]) + noise

    w_val_init = w_val_1.clone()
    return x_seq, keys, targets, w_key, w_val_init


def run_fixed_delta(
    keys: torch.Tensor,
    targets: torch.Tensor,
    step_size: float = 0.15,
) -> StateTrajectory:
    t_steps, k_dim = keys.shape
    v_dim = targets.shape[-1]
    rule = DeltaRule(step_size=step_size)
    memory = AssociativeMemory.zeros(v_dim, k_dim)

    steps: list[TrajectoryStep] = []
    first_fail: int | None = None
    all_finite = True

    for t in range(t_steps):
        k_t = keys[t]
        y_t = targets[t]
        res = rule.step(key=k_t, target=y_t, memory=memory)
        memory = res.new_memory

        err_norm = float(torch.linalg.norm(res.error).item())
        mem_norm = float(torch.linalg.norm(memory.data).item())
        upd_norm = float(torch.linalg.norm(res.update).item())

        is_finite = math.isfinite(err_norm) and math.isfinite(mem_norm)
        if not is_finite and first_fail is None:
            first_fail = t
            all_finite = False

        steps.append(
            TrajectoryStep(
                step=t,
                observed=True,
                error_norm=err_norm if is_finite else None,
                step_size=step_size,
                step_size_change=0.0 if t > 0 else None,
                memory_norm=mem_norm if is_finite else None,
                update_norm=upd_norm if is_finite else None,
                normalized_step=float(step_size * torch.sum(k_t**2).item()),
                stability_margin=float(2.0 - step_size * torch.sum(k_t**2).item()),
                finite_state=is_finite,
            )
        )

    return StateTrajectory(
        model="FixedDelta",
        task="token_association",
        seed=42,
        steps=steps,
        all_states_finite=all_finite,
        first_nonfinite_step=first_fail,
        terminal_state_finite=all_finite,
    )


def run_phase_2_adaptive_delta(
    keys: torch.Tensor,
    targets: torch.Tensor,
) -> StateTrajectory:
    t_steps, k_dim = keys.shape
    v_dim = targets.shape[-1]
    controller = InputConditionedStepSize(dim=k_dim, eta_max=0.4)
    rule = AdaptiveDeltaRule(controller=controller)
    memory = AssociativeMemory.zeros(v_dim, k_dim)

    steps: list[TrajectoryStep] = []
    first_fail: int | None = None
    all_finite = True

    prev_eta = None
    for t in range(t_steps):
        k_t = keys[t]
        y_t = targets[t]
        res = rule.step(key=k_t, target=y_t, memory=memory)
        memory = res.new_memory

        err_norm = float(torch.linalg.norm(res.error).item())
        mem_norm = float(torch.linalg.norm(memory.data).item())
        upd_norm = float(torch.linalg.norm(res.update).item())
        eta_val = float(res.step_size.item())
        eta_change = eta_val - prev_eta if prev_eta is not None else None
        prev_eta = eta_val

        is_finite = math.isfinite(err_norm) and math.isfinite(mem_norm)
        if not is_finite and first_fail is None:
            first_fail = t
            all_finite = False

        steps.append(
            TrajectoryStep(
                step=t,
                observed=True,
                error_norm=err_norm if is_finite else None,
                step_size=eta_val,
                step_size_change=eta_change,
                memory_norm=mem_norm if is_finite else None,
                update_norm=upd_norm if is_finite else None,
                normalized_step=float(eta_val * torch.sum(k_t**2).item()),
                stability_margin=float(2.0 - eta_val * torch.sum(k_t**2).item()),
                finite_state=is_finite,
            )
        )

    return StateTrajectory(
        model="AdaptiveDelta",
        task="token_association",
        seed=42,
        steps=steps,
        all_states_finite=all_finite,
        first_nonfinite_step=first_fail,
        terminal_state_finite=all_finite,
    )


def run_phase_3_self_referential(
    keys: torch.Tensor,
    targets: torch.Tensor,
    safe: bool = False,
) -> StateTrajectory:
    t_steps, k_dim = keys.shape
    v_dim = targets.shape[-1]
    d_c = 3
    name = "SafeSelfReferential" if safe else "SelfReferential"

    if safe:
        stab_ctrl = SafeStepSizeController(beta=1.9)
        dyn_ctrl = SafeDynamicsRateController(beta_c=1.9)
    else:
        stab_ctrl = UnconstrainedController()
        dyn_ctrl = UnconstrainedController()

    system = SelfReferentialSystem(
        content_stability_controller=stab_ctrl,
        dynamics_stability_controller=dyn_ctrl,
    )
    init_state = CoupledState(
        content_memory=AssociativeMemory.zeros(v_dim, k_dim),
        dynamics_memory=DynamicsMemory.zeros(1, d_c),
    )

    res = system.scan(keys, targets, initial_state=init_state)

    steps: list[TrajectoryStep] = []
    first_fail = None
    all_finite = True

    res_mem = res.memory_states or []
    res_dyn = res.dynamics_states or []
    res_margins = res.stability_margins
    res_norm_steps = res.normalized_steps

    for t in range(t_steps):
        e_norm = float(torch.linalg.norm(res.errors[t]).item())
        m_norm = (
            float(torch.linalg.norm(res_mem[t + 1]).item())
            if t + 1 < len(res_mem)
            else 0.0
        )
        c_norm = (
            float(torch.linalg.norm(res_dyn[t + 1]).item())
            if t + 1 < len(res_dyn)
            else 0.0
        )
        u_norm = float(res.content_update_norms[t].item())
        cu_norm = float(res.dynamics_update_norms[t].item())
        s_size = float(res.step_sizes[t].item())
        marg = float(res_margins[t].item()) if res_margins is not None else 2.0
        norm_s = float(res_norm_steps[t].item()) if res_norm_steps is not None else 0.0
        d_marg = (
            float(res.dynamics_stability_margins[t].item())
            if res.dynamics_stability_margins is not None
            else None
        )
        clip = (
            bool(res.step_size_clips[t].item())
            if res.step_size_clips is not None
            else False
        )

        is_finite = math.isfinite(e_norm) and math.isfinite(m_norm)
        if not is_finite and first_fail is None:
            first_fail = t
            all_finite = False

        steps.append(
            TrajectoryStep(
                step=t,
                observed=True,
                error_norm=e_norm if is_finite else None,
                step_size=s_size,
                memory_norm=m_norm if is_finite else None,
                dynamics_memory_norm=c_norm if is_finite else None,
                update_norm=u_norm if is_finite else None,
                dynamics_update_norm=cu_norm if is_finite else None,
                normalized_step=norm_s,
                stability_margin=marg,
                dynamics_stability_margin=d_marg,
                clip_event=clip,
                finite_state=is_finite,
            )
        )

    return StateTrajectory(
        model=name,
        task="token_association",
        seed=42,
        steps=steps,
        all_states_finite=all_finite,
        first_nonfinite_step=first_fail,
        terminal_state_finite=all_finite,
    )


def run_phase_8_five_memory(
    x_seq: torch.Tensor,
    targets: torch.Tensor,
    w_key_init: torch.Tensor,
    w_val_init: torch.Tensor,
) -> tuple[StateTrajectory, FiveMemoryScanResult]:
    t_steps, d_in = x_seq.shape
    v_dim = targets.shape[-1]
    k_dim = w_key_init.shape[0]

    config = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=d_in,
        feat_dim=d_in,
        lr_dim=2,
        ret_dim=2,
        eta_max=0.4,
        eta_bias=0.0,
        ret_min=0.1,
        ret_bias=2.0,
        eta_key=0.01,
        lambda_key=0.99,
        eta_val=0.01,
        lambda_val=0.99,
        rho_eta=0.05,
        lambda_eta=0.98,
        rho_ret=0.05,
        lambda_ret=0.98,
        beta=1.9,
    )
    system = FiveMemorySystem(config)

    # Initial state with warm key and value projection matrices
    init_state = FiveMemoryState(
        content=torch.zeros(v_dim, k_dim),
        key=w_key_init.clone(),
        value=torch.matmul(w_val_init, w_key_init),
        learning_rate=torch.zeros(config.lr_dim, d_in),
        retention=torch.zeros(config.ret_dim, d_in),
    )

    res: FiveMemoryScanResult = system.scan(x_seq, init_state, targets=targets)

    steps: list[TrajectoryStep] = []
    first_fail = None
    all_finite = True

    prev_eta = None
    for t in range(t_steps):
        e_norm = float(torch.linalg.norm(res.errors[t]).item())
        m_norm = float(res.content_memory_norms[t + 1].item())
        k_norm = float(res.key_memory_norms[t + 1].item())
        v_norm = float(res.value_memory_norms[t + 1].item())
        lr_norm = float(res.learning_rate_memory_norms[t + 1].item())
        ret_norm = float(res.retention_memory_norms[t + 1].item())

        u_norm = float(res.content_update_norms[t].item())
        raw_lr = float(res.raw_learning_rates[t].item())
        safe_lr = float(res.safe_learning_rates[t].item())
        raw_ret = float(res.raw_retentions[t].item())
        safe_ret = float(res.safe_retentions[t].item())

        s_marg = (
            float(res.stability_margins[t].item())
            if res.stability_margins.numel() > 0
            else None
        )
        n_step = (
            float(res.normalized_steps[t].item())
            if res.normalized_steps.numel() > 0
            else None
        )
        clip = bool(res.clip_events[t].item()) if res.clip_events.numel() > 0 else False

        s_change = safe_lr - prev_eta if prev_eta is not None else None
        prev_eta = safe_lr

        is_finite = math.isfinite(e_norm) and math.isfinite(m_norm)
        if not is_finite and first_fail is None:
            first_fail = t
            all_finite = False

        steps.append(
            TrajectoryStep(
                step=t,
                observed=True,
                error_norm=e_norm if is_finite else None,
                step_size=safe_lr,
                step_size_change=s_change,
                memory_norm=m_norm if is_finite else None,
                update_norm=u_norm if is_finite else None,
                normalized_step=n_step,
                stability_margin=s_marg,
                clip_event=clip,
                finite_state=is_finite,
                key_memory_norm=k_norm,
                value_memory_norm=v_norm,
                learning_rate_memory_norm=lr_norm,
                retention_memory_norm=ret_norm,
                raw_learning_rate=raw_lr,
                safe_learning_rate=safe_lr,
                raw_retention=raw_ret,
                safe_retention=safe_ret,
            )
        )

    traj = StateTrajectory(
        model="FiveMemory",
        task="token_association",
        seed=42,
        steps=steps,
        all_states_finite=all_finite,
        first_nonfinite_step=first_fail,
        terminal_state_finite=all_finite,
    )
    return traj, res


def run_experiment_condition(condition_name: str, shift_step: int | None) -> None:
    print(f"\n{'=' * 80}")
    print(f"Condition: {condition_name} (shift_step={shift_step})")
    print(f"{'=' * 80}")

    x_seq, keys, targets, w_key, w_val = generate_synthetic_data(
        t_steps=60, d_in=8, k_dim=8, v_dim=6, shift_step=shift_step, seed=42
    )

    t_fixed = run_fixed_delta(keys, targets, step_size=0.15)
    t_adap = run_phase_2_adaptive_delta(keys, targets)
    t_sr = run_phase_3_self_referential(keys, targets, safe=False)
    t_ssr = run_phase_3_self_referential(keys, targets, safe=True)
    t_5m, scan_5m = run_phase_8_five_memory(x_seq, targets, w_key, w_val)

    trajectories = [t_fixed, t_adap, t_sr, t_ssr, t_5m]
    fingerprints = [generate_fingerprint(t) for t in trajectories]

    comparison = compare_fingerprints(fingerprints)
    table = comparison["comparison_table"]

    header = f"{'Model':<20} | {'Final Err':<10} | {'1st Pass':<9} | {'Sustained':<10} | {'Upd Energy':<11} | {'Max Norm':<9} | {'Min Margin':<10} | {'5M Activity':<11}"
    print(header)
    print("-" * len(header))
    for row, fp in zip(table, fingerprints, strict=False):
        act_str = (
            f"{fp.five_memory_state_activity:.4f}"
            if fp.five_memory_state_activity is not None
            else "—"
        )
        print(
            f"{row['model']:<20} | {str(row['final_error']):<10} | {str(row['recovery_latency']):<9} | {str(row['sustained_recovery']):<10} | {str(row['update_energy']):<11} | {str(row['max_state_norm']):<9} | {str(row['min_margin']):<10} | {act_str:<11}"
        )

    # Save artifact and generate report for Phase 8 Five-Memory run
    if shift_step is not None:
        out_dir = repo_root / "docs" / "benchmarks" / "artifacts" / "phase_8"
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / "five_memory_shift_trajectory.json"
        t_5m.save(traj_path)

        plots = generate_all_plots(t_5m, output_dir=out_dir / "plots")
        report_path = out_dir / "five_memory_observatory_report.md"
        generate_observatory_report(
            trajectory=t_5m,
            output_path=report_path,
            plot_paths={k: str(v) for k, v in plots.items()},
        )
        print(f"\nObservatory report generated at: {report_path}")


def main() -> None:
    print("=" * 80)
    print("DeltaCore Phase 8: Five-Memory Self-Modifying Reference Core Comparison")
    print("=" * 80)

    # Condition 1: Stationary data
    run_experiment_condition("Stationary", shift_step=None)

    # Condition 2: Distribution shift at step 30
    run_experiment_condition("Distribution Shift", shift_step=30)


if __name__ == "__main__":
    main()
