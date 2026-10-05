# ==============================================================================
# DeltaCore: tests/test_five_memory_autograd.py
# Autograd differentiability and gradcheck verification for the five-memory system.
# ==============================================================================

import torch

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem


def test_five_memory_gradcheck():
    r"""Verify gradients through the complete coupled five-memory transition using FP64 gradcheck:
    input -> key/value memories -> content read -> error -> learning-rate memory -> retention memory -> new state
    """
    v_dim, k_dim, in_dim, feat_dim = 2, 2, 2, 2
    lr_dim, ret_dim = 1, 1

    config = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        feat_dim=feat_dim,
        lr_dim=lr_dim,
        ret_dim=ret_dim,
        eta_max=0.5,
        eta_bias=0.0,
        ret_min=0.1,
        ret_bias=1.0,
        eta_key=0.05,
        lambda_key=0.95,
        eta_val=0.05,
        lambda_val=0.95,
        rho_eta=0.05,
        lambda_eta=0.95,
        rho_ret=0.05,
        lambda_ret=0.95,
        apply_stability_control=True,
    )
    system = FiveMemorySystem(config)

    # Use FP64 for high numerical precision in finite differences
    dtype = torch.float64
    gen = torch.Generator().manual_seed(999)

    x = torch.randn(in_dim, generator=gen, dtype=dtype, requires_grad=True)
    c = torch.randn(v_dim, k_dim, generator=gen, dtype=dtype, requires_grad=True)
    k = torch.randn(k_dim, in_dim, generator=gen, dtype=dtype, requires_grad=True)
    v = torch.randn(v_dim, in_dim, generator=gen, dtype=dtype, requires_grad=True)
    lr = torch.randn(lr_dim, feat_dim, generator=gen, dtype=dtype, requires_grad=True)
    ret = torch.randn(ret_dim, feat_dim, generator=gen, dtype=dtype, requires_grad=True)

    def func(
        x_in: torch.Tensor,
        c_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        lr_in: torch.Tensor,
        ret_in: torch.Tensor,
    ) -> torch.Tensor:
        st = FiveMemoryState(
            content=c_in,
            key=k_in,
            value=v_in,
            learning_rate=lr_in,
            retention=ret_in,
        )
        res = system.step(x_in, st)
        # Sum of output prediction and all evolved memory states
        return (
            res.content_prediction.sum()
            + res.new_state.content.sum()
            + res.new_state.key.sum()
            + res.new_state.value.sum()
            + res.new_state.learning_rate.sum()
            + res.new_state.retention.sum()
        )

    # PyTorch gradcheck
    assert torch.autograd.gradcheck(
        func,
        (x, c, k, v, lr, ret),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


def test_sequential_recurrence_backprop():
    r"""Verify backpropagation through multiple recurrent sequential steps without detaching."""
    v_dim, k_dim, in_dim = 3, 3, 4
    config = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        feat_dim=in_dim,
        lr_dim=1,
        ret_dim=1,
    )
    system = FiveMemorySystem(config)
    dtype = torch.float64
    gen = torch.Generator().manual_seed(888)

    t_steps = 3
    inputs = torch.randn(
        t_steps, in_dim, generator=gen, dtype=dtype, requires_grad=True
    )

    c0 = (
        (torch.randn(v_dim, k_dim, generator=gen, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    k0 = (
        (torch.randn(k_dim, in_dim, generator=gen, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    v0 = (
        (torch.randn(v_dim, in_dim, generator=gen, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    lr0 = (
        (torch.randn(1, in_dim, generator=gen, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    ret0 = (
        (torch.randn(1, in_dim, generator=gen, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )

    state = FiveMemoryState(
        content=c0, key=k0, value=v0, learning_rate=lr0, retention=ret0
    )

    # Unroll sequentially without detaching
    total_loss = torch.tensor(0.0, dtype=dtype)
    for t in range(t_steps):
        # When target is None, value_representation directly drives prediction_error
        res = system.step(inputs[t], state, target=None)
        total_loss = (
            total_loss
            + torch.sum(res.prediction_error**2)
            + torch.sum(res.content_prediction**2)
        )
        state = res.new_state

    total_loss.backward()

    # Verify non-zero gradients on all inputs and initial state tensors
    assert inputs.grad is not None and torch.linalg.norm(inputs.grad) > 0
    assert c0.grad is not None and torch.linalg.norm(c0.grad) > 0
    assert k0.grad is not None and torch.linalg.norm(k0.grad) > 0
    assert v0.grad is not None and torch.linalg.norm(v0.grad) > 0
    assert lr0.grad is not None and torch.linalg.norm(lr0.grad) > 0
    assert ret0.grad is not None and torch.linalg.norm(ret0.grad) > 0
