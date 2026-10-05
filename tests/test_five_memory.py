# ==============================================================================
# DeltaCore: tests/test_five_memory.py
# Verification of Properties A through J for the Five-Memory Reference Core.
# ==============================================================================

import math

import pytest
import torch

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.updates.five_memory import (
    FiveMemoryConfig,
    FiveMemorySystem,
)


@pytest.fixture
def base_config() -> FiveMemoryConfig:
    return FiveMemoryConfig(
        v_dim=4,
        k_dim=3,
        in_dim=5,
        feat_dim=5,
        lr_dim=2,
        ret_dim=2,
        eta_max=0.5,
        eta_bias=0.0,
        ret_min=0.0,
        ret_bias=2.0,
        eta_key=0.02,
        lambda_key=0.99,
        eta_val=0.02,
        lambda_val=0.99,
        rho_eta=0.05,
        lambda_eta=0.98,
        tau_eta=0.5,
        rho_ret=0.05,
        lambda_ret=0.98,
        tau_ret=0.5,
        beta=1.9,
    )


def test_property_a_dtype_device_preservation(base_config):
    """Property A: All five states preserve dtype and device under transitions."""
    for dtype in (torch.float32, torch.float64):
        gen = torch.Generator().manual_seed(42)
        state = FiveMemoryState.initialize(
            v_dim=base_config.v_dim,
            k_dim=base_config.k_dim,
            in_dim=base_config.in_dim,
            feat_dim=base_config.feat_dim,
            lr_dim=base_config.lr_dim,
            ret_dim=base_config.ret_dim,
            generator=gen,
            dtype=dtype,
        )
        system = FiveMemorySystem(base_config)
        x = torch.randn(base_config.in_dim, generator=gen, dtype=dtype)
        target = torch.randn(base_config.v_dim, generator=gen, dtype=dtype)

        res = system.step(x, state, target=target)

        assert res.new_state.content.dtype == dtype
        assert res.new_state.key.dtype == dtype
        assert res.new_state.value.dtype == dtype
        assert res.new_state.learning_rate.dtype == dtype
        assert res.new_state.retention.dtype == dtype

        assert res.new_state.content.device == state.content.device
        assert res.new_state.key.device == state.key.device
        assert res.new_state.value.device == state.value.device
        assert res.new_state.learning_rate.device == state.learning_rate.device
        assert res.new_state.retention.device == state.retention.device


def test_property_b_no_unexpected_mutation(base_config):
    """Property B: No state mutates in-place; input state remains unmodified."""
    gen = torch.Generator().manual_seed(101)
    state = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    c_orig = state.content.clone()
    k_orig = state.key.clone()
    v_orig = state.value.clone()
    lr_orig = state.learning_rate.clone()
    ret_orig = state.retention.clone()

    system = FiveMemorySystem(base_config)
    x = torch.randn(base_config.in_dim, generator=gen)
    target = torch.randn(base_config.v_dim, generator=gen)

    # Execute step
    _ = system.step(x, state, target=target)

    # Verify input state is bit-for-bit unchanged
    assert torch.equal(state.content, c_orig)
    assert torch.equal(state.key, k_orig)
    assert torch.equal(state.value, v_orig)
    assert torch.equal(state.learning_rate, lr_orig)
    assert torch.equal(state.retention, ret_orig)


def test_property_c_all_five_states_evolve(base_config):
    """Property C: All five states evolve when update is non-degenerate."""
    gen = torch.Generator().manual_seed(202)
    state = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    system = FiveMemorySystem(base_config)
    x = torch.randn(base_config.in_dim, generator=gen)
    target = torch.randn(base_config.v_dim, generator=gen)

    res = system.step(x, state, target=target)

    # Every state must have evolved (norm of update > 0)
    assert torch.linalg.norm(res.content_update) > 1e-6
    assert torch.linalg.norm(res.key_memory_update) > 1e-6
    assert torch.linalg.norm(res.value_memory_update) > 1e-6
    assert torch.linalg.norm(res.learning_rate_update) > 1e-6
    assert torch.linalg.norm(res.retention_update) > 1e-6


def test_property_d_freezing_learning_rate_memory(base_config):
    """Property D: Freezing learning-rate memory reproduces static/unadapted LR controller."""
    frozen_config = FiveMemoryConfig(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        rho_eta=0.0,
        lambda_eta=1.0,  # Frozen learning-rate memory
    )
    gen = torch.Generator().manual_seed(303)
    state = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    system = FiveMemorySystem(frozen_config)

    x = torch.randn(base_config.in_dim, generator=gen)
    target = torch.randn(base_config.v_dim, generator=gen)
    res = system.step(x, state, target=target)

    # Learning rate memory must remain identical
    assert torch.equal(res.new_state.learning_rate, state.learning_rate)
    assert torch.equal(res.learning_rate_update, torch.zeros_like(state.learning_rate))


def test_property_e_freezing_retention_memory(base_config):
    """Property E: Freezing retention memory reproduces static retention case."""
    frozen_config = FiveMemoryConfig(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        rho_ret=0.0,
        lambda_ret=1.0,  # Frozen retention memory
    )
    gen = torch.Generator().manual_seed(404)
    state = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    system = FiveMemorySystem(frozen_config)

    x = torch.randn(base_config.in_dim, generator=gen)
    target = torch.randn(base_config.v_dim, generator=gen)
    res = system.step(x, state, target=target)

    # Retention memory must remain identical
    assert torch.equal(res.new_state.retention, state.retention)
    assert torch.equal(res.retention_update, torch.zeros_like(state.retention))


def test_property_f_determinism(base_config):
    """Property F: Identical initial state + identical sequence produces identical trajectory."""
    gen = torch.Generator().manual_seed(505)
    inputs = torch.randn(8, base_config.in_dim, generator=gen)
    targets = torch.randn(8, base_config.v_dim, generator=gen)

    state1 = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=torch.Generator().manual_seed(777),
    )
    state2 = state1.clone()

    system = FiveMemorySystem(base_config)
    res1 = system.scan(inputs, state1, targets=targets)
    res2 = system.scan(inputs, state2, targets=targets)

    assert torch.equal(res1.predictions, res2.predictions)
    assert torch.equal(res1.errors, res2.errors)
    assert torch.equal(res1.safe_learning_rates, res2.safe_learning_rates)
    assert torch.equal(res1.safe_retentions, res2.safe_retentions)
    assert torch.equal(res1.final_state.content, res2.final_state.content)
    assert torch.equal(res1.final_state.key, res2.final_state.key)
    assert torch.equal(res1.final_state.value, res2.final_state.value)
    assert torch.equal(res1.final_state.learning_rate, res2.final_state.learning_rate)
    assert torch.equal(res1.final_state.retention, res2.final_state.retention)


def test_property_g_learning_rate_memory_alters_future_content_updates(base_config):
    """Property G: Changing learning-rate memory alters future content updates."""
    gen = torch.Generator().manual_seed(606)
    state_a = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    # State B has different learning-rate memory
    state_b = FiveMemoryState(
        content=state_a.content.clone(),
        key=state_a.key.clone(),
        value=state_a.value.clone(),
        learning_rate=state_a.learning_rate + 2.0,  # perturb LR memory
        retention=state_a.retention.clone(),
    )

    system = FiveMemorySystem(base_config)
    x = torch.randn(base_config.in_dim, generator=gen)
    target = torch.randn(base_config.v_dim, generator=gen)

    res_a = system.step(x, state_a, target=target)
    res_b = system.step(x, state_b, target=target)

    # Different effective step sizes
    assert not torch.allclose(res_a.safe_learning_rate, res_b.safe_learning_rate)
    # Different content updates
    assert not torch.allclose(res_a.content_update, res_b.content_update)


def test_property_h_retention_memory_alters_future_retention(base_config):
    """Property H: Changing retention memory alters future memory retention."""
    gen = torch.Generator().manual_seed(707)
    state_a = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    # Give non-zero initial content memory so retention decay difference is measurable
    content_init = torch.ones_like(state_a.content)
    state_a = FiveMemoryState(
        content=content_init.clone(),
        key=state_a.key.clone(),
        value=state_a.value.clone(),
        learning_rate=state_a.learning_rate.clone(),
        retention=state_a.retention.clone() - 5.0,  # low retention
    )
    state_b = FiveMemoryState(
        content=content_init.clone(),
        key=state_a.key.clone(),
        value=state_a.value.clone(),
        learning_rate=state_a.learning_rate.clone(),
        retention=state_a.retention.clone() + 5.0,  # high retention
    )

    system = FiveMemorySystem(base_config)
    x = torch.randn(base_config.in_dim, generator=gen)
    target = torch.randn(base_config.v_dim, generator=gen)

    res_a = system.step(x, state_a, target=target)
    res_b = system.step(x, state_b, target=target)

    assert not torch.allclose(res_a.safe_retention, res_b.safe_retention)
    assert not torch.allclose(res_a.new_state.content, res_b.new_state.content)


def test_property_i_key_value_independent_replaceability(base_config):
    """Property I: Key and value memories are independently replaceable."""
    gen = torch.Generator().manual_seed(808)
    state = FiveMemoryState.initialize(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
        generator=gen,
    )
    # Replace key memory with new tensor
    new_key_tensor = torch.randn_like(state.key) * 0.5
    state_with_new_key = FiveMemoryState(
        content=state.content.clone(),
        key=new_key_tensor,
        value=state.value.clone(),
        learning_rate=state.learning_rate.clone(),
        retention=state.retention.clone(),
    )

    system = FiveMemorySystem(base_config)
    x = torch.randn(base_config.in_dim, generator=gen)

    res_orig = system.step(x, state)
    res_mod = system.step(x, state_with_new_key)

    # Keys must differ
    assert not torch.allclose(res_orig.key_representation, res_mod.key_representation)
    # Value representation must be identical since value memory was not modified
    assert torch.allclose(res_orig.value_representation, res_mod.value_representation)


def test_property_j_degenerate_zero_input(base_config):
    """Property J: Zero/degenerate input behavior is well-defined, finite, and documented."""
    state = FiveMemoryState.zeros(
        v_dim=base_config.v_dim,
        k_dim=base_config.k_dim,
        in_dim=base_config.in_dim,
        feat_dim=base_config.feat_dim,
        lr_dim=base_config.lr_dim,
        ret_dim=base_config.ret_dim,
    )
    system = FiveMemorySystem(base_config)
    x_zero = torch.zeros(base_config.in_dim)

    res = system.step(x_zero, state)

    # Output vectors must be zero and strictly finite
    assert torch.equal(res.key_representation, torch.zeros(base_config.k_dim))
    assert torch.equal(res.value_representation, torch.zeros(base_config.v_dim))
    assert torch.equal(res.content_prediction, torch.zeros(base_config.v_dim))
    assert torch.equal(res.prediction_error, torch.zeros(base_config.v_dim))

    assert math.isfinite(res.safe_learning_rate.item())
    assert math.isfinite(res.safe_retention.item())

    # All resulting state tensors must be finite
    assert torch.all(torch.isfinite(res.new_state.content))
    assert torch.all(torch.isfinite(res.new_state.key))
    assert torch.all(torch.isfinite(res.new_state.value))
    assert torch.all(torch.isfinite(res.new_state.learning_rate))
    assert torch.all(torch.isfinite(res.new_state.retention))
