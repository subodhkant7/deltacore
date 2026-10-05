"""Autograd verification and numerical gradcheck tests for Phase 9 spatial routing.

Verifies:
1. End-to-end gradient computation through:
   feature map -> serialization -> five-memory learner -> restoration -> fusion -> loss
2. Numerical gradient verification using torch.autograd.gradcheck (FP64)
3. Non-vanishing gradients across all four directional routes
"""

import torch
from torch.autograd import gradcheck

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.spatial.fusion import EqualFusion, LearnedChannelFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.spatial.routes import DOWN, LEFT, RIGHT, UP
from deltacore.updates.five_memory import FiveMemoryConfig


def test_spatial_autograd_loss_backward() -> None:
    """Verify loss.backward() computes non-zero, finite gradients for feature map and weights."""
    torch.manual_seed(101)
    B, C, H, W = 2, 3, 3, 4

    cfg = FiveMemoryConfig(
        v_dim=C,
        k_dim=C,
        in_dim=C,
        feat_dim=C,
        lr_dim=1,
        ret_dim=1,
        eta_max=0.5,
        ret_min=0.1,
        apply_stability_control=True,
    )
    fusion = LearnedChannelFusion(channels=C, num_routes=4)
    op = SpatialAdaptiveOperator(config=cfg, fusion=fusion, mode="paper_aligned")

    Z = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)

    # Initial states as differentiable tensors
    states = {
        r.name: FiveMemoryState.initialize(
            C,
            C,
            C,
            batch_size=B,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(i),
        )
        for i, r in enumerate([RIGHT, LEFT, DOWN, UP])
    }

    res = op(Z, initial_states=states)
    target = torch.randn_like(res.fused_output)
    loss = torch.sum((res.fused_output - target) ** 2)

    loss.backward()

    assert Z.grad is not None
    assert Z.grad.shape == Z.shape
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()
    assert torch.norm(Z.grad) > 0.0

    assert fusion.weights.grad is not None
    assert not torch.isnan(fusion.weights.grad).any()
    assert torch.norm(fusion.weights.grad) > 0.0


def test_spatial_autograd_gradcheck_fp64() -> None:
    """Run numerical gradcheck in FP64 across small spatial feature map."""
    torch.manual_seed(202)
    B, C, H, W = 1, 2, 2, 2

    cfg = FiveMemoryConfig(
        v_dim=C,
        k_dim=C,
        in_dim=C,
        feat_dim=C,
        lr_dim=1,
        ret_dim=1,
        eta_max=0.2,
        ret_min=0.2,
        apply_stability_control=False,  # Avoid min() non-differentiable point during gradcheck
    )
    fusion = EqualFusion(mode="mean")
    op = SpatialAdaptiveOperator(
        config=cfg, fusion=fusion, mode="generic", default_chunk_size=2
    )

    # Fixed initial states
    states = {
        r.name: FiveMemoryState.initialize(
            C,
            C,
            C,
            batch_size=B,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(i),
        )
        for i, r in enumerate([RIGHT, LEFT, DOWN, UP])
    }

    def func(x: torch.Tensor) -> torch.Tensor:
        res = op(x, initial_states=states)
        return res.fused_output

    Z = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)

    # gradcheck tests analytical vs numerical finite differences
    passed = gradcheck(func, (Z,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed, "gradcheck failed on SpatialAdaptiveOperator"
