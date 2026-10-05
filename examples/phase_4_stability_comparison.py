# ==============================================================================
# DeltaCore: examples/phase_4_stability_comparison.py
# Phase 4 Experiment: Stability Controllers & Mathematical Guarantees Comparison.
# ==============================================================================

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Any  # noqa: E402

import torch  # noqa: E402

from deltacore.diagnostics.stability import (  # noqa: E402
    compute_first_passage_recovery_steps,
    compute_sustained_recovery_steps,
)
from deltacore.memory.associative import AssociativeMemory  # noqa: E402
from deltacore.stability.controllers import (  # noqa: E402
    SafeDynamicsRateController,
    SafeStepSizeController,
    UnconstrainedController,
)
from deltacore.updates.dynamics_memory import DynamicsMemory  # noqa: E402
from deltacore.updates.self_referential import (  # noqa: E402
    CoupledState,
    SelfReferentialScanResult,
    SelfReferentialSystem,
)


def run_configuration(
    name: str,
    system: SelfReferentialSystem,
    keys: torch.Tensor,
    targets: torch.Tensor,
    shift_index: int | None = None,
    tau: float = 0.5,
    window: int = 4,
) -> dict[str, Any]:
    r"""Run a single self-referential configuration and extract all comparative stability metrics."""
    t_steps = keys.shape[0]
    v_dim = targets.shape[-1]
    k_dim = keys.shape[-1]
    d_c = 3

    # Identical zero initialization for both memories
    init_state = CoupledState(
        content_memory=AssociativeMemory.zeros(v_dim, k_dim, dtype=keys.dtype),
        dynamics_memory=DynamicsMemory.zeros(1, d_c, dtype=keys.dtype),
    )

    scan_res: SelfReferentialScanResult = system.scan(
        keys, targets, initial_state=init_state
    )

    # 1. Error metrics
    errors = targets - scan_res.predictions
    final_error = torch.linalg.norm(errors[-1]).item()

    # Recovery metrics if shift_index specified
    fp_recovery = None
    sustained_recovery = None
    if shift_index is not None and shift_index < t_steps:
        fp_recovery = compute_first_passage_recovery_steps(
            errors, shift_index=shift_index, tau=tau
        )
        sustained_recovery = compute_sustained_recovery_steps(
            errors, shift_index=shift_index, window=window, tau=tau
        )

    # 2. State & update norms
    mem_states = scan_res.memory_states or []
    dyn_states = scan_res.dynamics_states or []
    mem_norms = [torch.linalg.norm(m).item() for m in mem_states]
    max_mem_norm = max(mem_norms) if mem_norms else 0.0

    dyn_norms = [torch.linalg.norm(c).item() for c in dyn_states]
    max_dyn_norm = max(dyn_norms) if dyn_norms else 0.0

    max_update_norm = (
        torch.max(scan_res.content_update_norms).item()
        if len(scan_res.content_update_norms) > 0
        else 0.0
    )

    # 3. Step sizes and clipping
    safe_steps = scan_res.step_sizes
    max_step_size = torch.max(safe_steps).item() if len(safe_steps) > 0 else 0.0
    content_clips = (
        int(scan_res.step_size_clips.sum().item())
        if scan_res.step_size_clips is not None
        else 0
    )
    dynamics_clips = (
        int(scan_res.dynamics_rate_clips.sum().item())
        if scan_res.dynamics_rate_clips is not None
        else 0
    )

    # 4. Normalized steps and stability margins
    norm_steps = scan_res.normalized_steps
    max_norm_step = (
        torch.max(norm_steps).item()
        if norm_steps is not None and len(norm_steps) > 0
        else 0.0
    )
    min_stab_margin = (
        torch.min(scan_res.stability_margins).item()
        if scan_res.stability_margins is not None
        and len(scan_res.stability_margins) > 0
        else 2.0
    )

    norm_dyn_rates = scan_res.normalized_dynamics_rates
    max_norm_dyn_rate = (
        torch.max(norm_dyn_rates).item()
        if norm_dyn_rates is not None and len(norm_dyn_rates) > 0
        else 0.0
    )
    min_dyn_margin = (
        torch.min(scan_res.dynamics_stability_margins).item()
        if scan_res.dynamics_stability_margins is not None
        and len(scan_res.dynamics_stability_margins) > 0
        else 2.0
    )

    # 5. Detailed Finiteness Tracking (NaN / Inf)
    all_states_finite = True
    first_nonfinite_step: int | None = None

    t_count = len(mem_states)
    for idx in range(t_count):
        m_tensor = mem_states[idx]
        m_finite = bool(
            torch.isfinite(m_tensor).all().item()
            and torch.isfinite(torch.linalg.norm(m_tensor.float())).item()
        )
        c_tensor = dyn_states[idx] if idx < len(dyn_states) else None
        c_finite = bool(
            (
                torch.isfinite(c_tensor).all().item()
                and torch.isfinite(torch.linalg.norm(c_tensor.float())).item()
            )
            if c_tensor is not None
            else True
        )
        p_finite = (
            bool(torch.isfinite(scan_res.predictions[idx]).all().item())
            if idx < len(scan_res.predictions)
            else True
        )
        if not (m_finite and c_finite and p_finite):
            all_states_finite = False
            if first_nonfinite_step is None:
                first_nonfinite_step = idx

    terminal_state_finite = (
        (
            bool(
                torch.isfinite(mem_states[-1]).all().item()
                and torch.isfinite(torch.linalg.norm(mem_states[-1].float())).item()
            )
            if len(mem_states) > 0
            else True
        )
        and (
            bool(
                torch.isfinite(dyn_states[-1]).all().item()
                and torch.isfinite(torch.linalg.norm(dyn_states[-1].float())).item()
            )
            if len(dyn_states) > 0
            else True
        )
        and (
            bool(torch.isfinite(scan_res.predictions[-1]).all().item())
            if len(scan_res.predictions) > 0
            else True
        )
        if t_count > 0
        else True
    )

    return {
        "name": name,
        "final_error": final_error,
        "first_passage_recovery": fp_recovery,
        "sustained_recovery": sustained_recovery,
        "max_mem_norm": max_mem_norm,
        "max_dyn_norm": max_dyn_norm,
        "max_update_norm": max_update_norm,
        "max_step_size": max_step_size,
        "content_clips": content_clips,
        "dynamics_clips": dynamics_clips,
        "max_norm_step": max_norm_step,
        "min_stab_margin": min_stab_margin,
        "max_norm_dyn_rate": max_norm_dyn_rate,
        "min_dyn_margin": min_dyn_margin,
        "all_states_finite": all_states_finite,
        "terminal_state_finite": terminal_state_finite,
        "first_nonfinite_step": first_nonfinite_step,
        "finite_state": all_states_finite,
    }


def format_table(headers: list[str], rows: list[list[str]], title: str) -> str:
    """Format an ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )

    lines = [f"\n=== {title} ===", sep, hdr, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    print("=" * 80)
    print(
        "DeltaCore Phase 4: Stability Controllers & Mathematical Guarantees Comparison"
    )
    print("=" * 80)

    # Common architecture parameters
    k_dim = 4
    v_dim = 4
    seed = 42

    # Configurations to compare
    def get_configs(
        eta_max: float, rho: float
    ) -> list[tuple[str, SelfReferentialSystem]]:
        return [
            (
                "A. Unconstrained Phase 3",
                SelfReferentialSystem(
                    eta_max=eta_max,
                    rho=rho,
                    content_stability_controller=UnconstrainedController(),
                    dynamics_stability_controller=UnconstrainedController(),
                ),
            ),
            (
                "B. Safe Content Step Only",
                SelfReferentialSystem(
                    eta_max=eta_max,
                    rho=rho,
                    content_stability_controller=SafeStepSizeController(beta=1.0),
                    dynamics_stability_controller=UnconstrainedController(),
                ),
            ),
            (
                "C. Safe Dynamics Rate Only",
                SelfReferentialSystem(
                    eta_max=eta_max,
                    rho=rho,
                    content_stability_controller=UnconstrainedController(),
                    dynamics_stability_controller=SafeDynamicsRateController(
                        beta_c=1.0
                    ),
                ),
            ),
            (
                "D. Both Safety Controls",
                SelfReferentialSystem(
                    eta_max=eta_max,
                    rho=rho,
                    content_stability_controller=SafeStepSizeController(beta=1.0),
                    dynamics_stability_controller=SafeDynamicsRateController(
                        beta_c=1.0
                    ),
                ),
            ),
        ]

    # -------------------------------------------------------------------------
    # Experiment 1: Standard Regime Shift (Adaptive Tracking & Recovery)
    # -------------------------------------------------------------------------
    print("\n[Running Experiment 1: Standard Regime Shift Benchmark]")
    gen1 = torch.Generator().manual_seed(seed)
    n_patterns = 4
    base_keys = torch.randn(n_patterns, k_dim, generator=gen1)
    base_keys = base_keys / torch.linalg.norm(base_keys, dim=-1, keepdim=True)
    base_vals_a = torch.randn(n_patterns, v_dim, generator=gen1)
    base_vals_b = -base_vals_a + 0.2 * torch.randn(n_patterns, v_dim, generator=gen1)

    t1 = 80
    shift_idx = 40
    keys_exp1 = torch.zeros(t1, k_dim)
    targets_exp1 = torch.zeros(t1, v_dim)
    for t in range(t1):
        idx = t % n_patterns
        keys_exp1[t] = base_keys[idx]
        targets_exp1[t] = base_vals_a[idx] if t < shift_idx else base_vals_b[idx]

    configs_exp1 = get_configs(eta_max=0.8, rho=0.2)
    results_exp1 = [
        run_configuration(
            name, sys, keys_exp1, targets_exp1, shift_index=shift_idx, tau=0.5, window=4
        )
        for name, sys in configs_exp1
    ]

    headers_exp1 = [
        "Configuration",
        "Final Err",
        "1st Pass Rec",
        "Sustained(4)",
        "Max ||M||_F",
        "Max ||C||_F",
        "Max ||ΔM||_F",
        "Max η_t",
        "Clips (M/C)",
        "All Fin",
        "Term Fin",
        "1st NonFin",
    ]
    rows_exp1 = []
    for r in results_exp1:
        fp_str = (
            str(r["first_passage_recovery"])
            if r["first_passage_recovery"] is not None
            else "N/A"
        )
        sust_str = (
            str(r["sustained_recovery"])
            if r["sustained_recovery"] is not None
            else "N/A"
        )
        clips_str = f"{r['content_clips']} / {r['dynamics_clips']}"
        nonfin_str = (
            str(r["first_nonfinite_step"])
            if r["first_nonfinite_step"] is not None
            else "None"
        )
        rows_exp1.append(
            [
                r["name"],
                f"{r['final_error']:.4f}",
                fp_str,
                sust_str,
                f"{r['max_mem_norm']:.4f}",
                f"{r['max_dyn_norm']:.4f}",
                f"{r['max_update_norm']:.4f}",
                f"{r['max_step_size']:.4f}",
                clips_str,
                str(r["all_states_finite"]),
                str(r["terminal_state_finite"]),
                nonfin_str,
            ]
        )
    print(
        format_table(
            headers_exp1,
            rows_exp1,
            "Experiment 1: Standard Regime Shift (T=80, Shift at t=40)",
        )
    )

    # -------------------------------------------------------------------------
    # Experiment 2: Adversarial Stress Test (Runaway Regime & Instability)
    # -------------------------------------------------------------------------
    print("\n[Running Experiment 2: Adversarial Numerical Stress Test]")
    # Stress conditions:
    # 1. Large key norms (||k|| = 3.0 => ||k||^2 = 9.0)
    # 2. Large target magnitude (||v|| = 10.0)
    # 3. High raw learning rate and dynamics rate (eta_max = 2.5, rho = 2.0)
    # 4. Repeated identical keys with conflicting alternating targets
    # Under these conditions:
    # Unconstrained normalized step: eta ||k||^2 ~ 1.25 * 9.0 = 11.25 >> 2.0 (EXPANSIVE)
    # Safe controller normalized step: eta_safe ||k||^2 <= beta = 1.0 (CONTRACTIVE)
    gen2 = torch.Generator().manual_seed(101)
    t2 = 50
    # 2 conflicting patterns
    k_stress = torch.randn(2, k_dim, generator=gen2)
    k_stress = (
        3.0 * k_stress / torch.linalg.norm(k_stress, dim=-1, keepdim=True)
    )  # ||k|| = 3.0
    v_stress_pos = 10.0 * torch.randn(2, v_dim, generator=gen2)
    v_stress_neg = -v_stress_pos  # Conflicting targets

    keys_exp2 = torch.zeros(t2, k_dim)
    targets_exp2 = torch.zeros(t2, v_dim)
    for t in range(t2):
        pattern_id = t % 2
        # Switch targets every 5 steps to create aggressive feedback oscillation
        phase = (t // 5) % 2
        keys_exp2[t] = k_stress[pattern_id]
        targets_exp2[t] = (
            v_stress_pos[pattern_id] if phase == 0 else v_stress_neg[pattern_id]
        )

    configs_exp2 = get_configs(eta_max=2.5, rho=2.0)
    results_exp2 = [
        run_configuration(
            name, sys, keys_exp2, targets_exp2, shift_index=5, tau=0.5, window=4
        )
        for name, sys in configs_exp2
    ]

    headers_exp2 = [
        "Configuration",
        "Final Err",
        "Max ||M||_F",
        "Max ||C||_F",
        "Max ||ΔM||_F",
        "Max η_t",
        "Max η||k||²",
        "Clips (M/C)",
        "All Fin",
        "Term Fin",
        "1st NonFin",
    ]
    rows_exp2 = []
    for r in results_exp2:
        clips_str = f"{r['content_clips']} / {r['dynamics_clips']}"
        nonfin_str = (
            str(r["first_nonfinite_step"])
            if r["first_nonfinite_step"] is not None
            else "None"
        )
        rows_exp2.append(
            [
                r["name"],
                f"{r['final_error']:.4f}",
                f"{r['max_mem_norm']:.4f}",
                f"{r['max_dyn_norm']:.4f}",
                f"{r['max_update_norm']:.4f}",
                f"{r['max_step_size']:.4f}",
                f"{r['max_norm_step']:.4f}",
                clips_str,
                str(r["all_states_finite"]),
                str(r["terminal_state_finite"]),
                nonfin_str,
            ]
        )
    print(
        format_table(
            headers_exp2,
            rows_exp2,
            "Experiment 2: Adversarial Runaway Stress Test (T=50, ||k||=3.0, η_max=2.5)",
        )
    )

    # -------------------------------------------------------------------------
    # Stability Telemetry Sample from Experiment 2 (Both Controls)
    # -------------------------------------------------------------------------
    r_both = results_exp2[3]
    r_uncon = results_exp2[0]
    print("\n=== Theoretical Contract Verification ===")
    print(
        f"Unconstrained System Max Normalized Step: {r_uncon['max_norm_step']:.4f} (Boundary: 2.0)"
    )
    print(
        f"Unconstrained System Minimum Margin:      {r_uncon['min_stab_margin']:.4f} (< 0 implies EXPANSION)"
    )
    print(
        f"Safe System (Both) Max Normalized Step:   {r_both['max_norm_step']:.4f} (Guaranteed <= beta = 1.0)"
    )
    print(
        f"Safe System (Both) Minimum Margin:        {r_both['min_stab_margin']:.4f} (Guaranteed >= 2 - beta = 1.0)"
    )
    print(
        f"Safety Enforced Clips: Content={r_both['content_clips']}, Dynamics={r_both['dynamics_clips']}"
    )
    print(f"Unconstrained State Norm Max:             {r_uncon['max_mem_norm']:.4f}")
    print(f"Safe System (Both) State Norm Max:        {r_both['max_mem_norm']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
