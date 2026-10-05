# ==============================================================================
# DeltaCore: tests/test_five_memory_equivalence.py
# Verification of special-case reductions to simpler DeltaCore primitives.
# ==============================================================================

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.five_memory import FiveMemoryState
from deltacore.stability.controllers import SafeStepSizeController
from deltacore.updates.delta import DeltaRule
from deltacore.updates.five_memory import (
    FiveMemoryConfig,
    FiveMemorySystem,
)


def test_case_1_reduction_to_fixed_delta_rule():
    r"""Case 1: Freeze learning-rate and retention memories and key/val adaptation.
    The five-memory system must reduce exactly to DeltaCore's Phase 1 DeltaRule.
    """
    v_dim, k_dim, in_dim = 4, 3, 5
    dtype = torch.float64
    gen = torch.Generator().manual_seed(1234)

    # Configure five-memory system for frozen, constant dynamics
    fixed_eta = 0.15
    config = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        feat_dim=in_dim,
        lr_dim=1,
        ret_dim=1,
        eta_max=fixed_eta,
        eta_bias=10.0,  # sigmoid(10) -> 1.0, so raw_eta = fixed_eta
        ret_min=1.0,  # fixed retention = 1.0
        ret_bias=10.0,
        eta_key=0.0,  # static key generator
        lambda_key=1.0,
        eta_val=0.0,  # static value generator
        lambda_val=1.0,
        rho_eta=0.0,  # frozen LR memory
        lambda_eta=1.0,
        rho_ret=0.0,  # frozen retention memory
        lambda_ret=1.0,
        apply_stability_control=False,  # direct fixed step size
    )
    five_system = FiveMemorySystem(config)

    # Initial states
    c0 = torch.randn(v_dim, k_dim, generator=gen, dtype=dtype)
    k0 = torch.randn(k_dim, in_dim, generator=gen, dtype=dtype)
    v0 = torch.randn(v_dim, in_dim, generator=gen, dtype=dtype)
    lr0 = torch.zeros(1, in_dim, dtype=dtype)
    ret0 = torch.zeros(1, in_dim, dtype=dtype)

    state = FiveMemoryState(
        content=c0.clone(),
        key=k0.clone(),
        value=v0.clone(),
        learning_rate=lr0,
        retention=ret0,
    )
    delta_rule = DeltaRule(step_size=fixed_eta)
    assoc_mem = AssociativeMemory(c0.clone())

    # Run 5 sequential steps
    for _ in range(5):
        x = torch.randn(in_dim, generator=gen, dtype=dtype)
        target = torch.randn(v_dim, generator=gen, dtype=dtype)

        # 1. FiveMemorySystem step
        res_five = five_system.step(x, state, target=target)
        state = res_five.new_state

        # 2. Phase 1 DeltaRule step using corresponding static key, target, and effective step size
        k_static = torch.matmul(k0, x)
        delta_rule = DeltaRule(step_size=res_five.safe_learning_rate.item())
        res_delta = delta_rule.step(key=k_static, target=target, memory=assoc_mem)
        assoc_mem = res_delta.new_memory

        # Verification of exact mathematical equivalence
        assert torch.allclose(res_five.key_representation, k_static, atol=1e-12)
        assert torch.allclose(
            res_five.content_prediction, res_delta.prediction, atol=1e-12
        )
        assert torch.allclose(res_five.prediction_error, res_delta.error, atol=1e-12)
        assert torch.allclose(res_five.content_update, res_delta.update, atol=1e-12)
        assert torch.allclose(state.content, assoc_mem.data, atol=1e-12)


def test_case_2_disabled_key_adaptation_equivalence():
    r"""Case 2: Disable key-memory adaptation.
    Keys generated must remain exactly equivalent to static linear projection k_t = M_key,0 * x_t.
    """
    v_dim, k_dim, in_dim = 3, 4, 6
    gen = torch.Generator().manual_seed(5678)

    config_static_key = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        eta_key=0.0,
        lambda_key=1.0,  # Frozen key memory
    )
    system = FiveMemorySystem(config_static_key)

    state = FiveMemoryState.initialize(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        generator=gen,
    )
    initial_k_matrix = state.key.clone()

    t_steps = 10
    inputs = torch.randn(t_steps, in_dim, generator=gen)
    targets = torch.randn(t_steps, v_dim, generator=gen)

    scan_res = system.scan(inputs, state, targets=targets)

    # For every step, generated key must match static matrix multiplication
    for t in range(t_steps):
        expected_k = torch.matmul(initial_k_matrix, inputs[t])
        assert torch.equal(scan_res.keys[t], expected_k)

    # Final key matrix is completely unchanged
    assert torch.equal(scan_res.final_state.key, initial_k_matrix)


def test_case_3_disabled_value_adaptation_equivalence():
    r"""Case 3: Disable value-memory adaptation.
    Values generated must remain exactly equivalent to static linear projection v_t = M_val,0 * x_t.
    """
    v_dim, k_dim, in_dim = 4, 3, 5
    gen = torch.Generator().manual_seed(9012)

    config_static_val = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        eta_val=0.0,
        lambda_val=1.0,  # Frozen value memory
    )
    system = FiveMemorySystem(config_static_val)

    state = FiveMemoryState.initialize(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        generator=gen,
    )
    initial_v_matrix = state.value.clone()

    t_steps = 10
    inputs = torch.randn(t_steps, in_dim, generator=gen)

    # Autonomous scan without external targets
    scan_res = system.scan(inputs, state, targets=None)

    for t in range(t_steps):
        expected_v = torch.matmul(initial_v_matrix, inputs[t])
        assert torch.equal(scan_res.values[t], expected_v)

    assert torch.equal(scan_res.final_state.value, initial_v_matrix)


def test_case_4_phase_4_stability_clamp_reduction():
    r"""Case 4: Verify that FiveMemorySystem's step-size clamping reproduces Phase 4 SafeStepSizeController."""
    v_dim, k_dim, in_dim = 2, 2, 2
    beta_val = 1.8
    config = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        eta_max=10.0,  # high raw step size to trigger clipping
        eta_bias=5.0,
        beta=beta_val,
        apply_stability_control=True,
    )
    system = FiveMemorySystem(config)
    safe_controller = SafeStepSizeController(beta=beta_val)

    state = FiveMemoryState.initialize(v_dim=v_dim, k_dim=k_dim, in_dim=in_dim)
    # Large input token to produce large key norm
    x = torch.ones(in_dim) * 5.0
    res = system.step(x, state)

    k_t = res.key_representation
    # Compare with Phase 4 controller
    ctrl_res = safe_controller.safe_step(res.raw_learning_rate, k_t)

    assert torch.allclose(res.safe_learning_rate, ctrl_res.safe_value, atol=1e-7)
    assert bool(res.clip_event) == bool(ctrl_res.clipped)
