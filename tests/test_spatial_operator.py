"""Property tests and invariants for SpatialAdaptiveOperator.

Verifies:
- Property A: Output shape equals input shape.
- Property B: No cross-direction state contamination.
- Property C: Route permutation changes outputs only through documented route assignment.
- Property D: Disabling three directions reduces exactly to the enabled route.
- Property E: Batch, device, and dtype preservation.
- Property F: Deterministic repeated execution.
- Property G: Autograd works end-to-end through all four routes.
"""

import pytest
import torch

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.spatial.fusion import LearnedChannelFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.spatial.routes import DOWN, LEFT, RIGHT, UP
from deltacore.updates.five_memory import FiveMemoryConfig


def _create_operator(
    channels: int = 4,
    mode: str = "generic",
    chunk_size: int = 2,
    seed: int = 42,
    fusion=None,
    routes=(RIGHT, LEFT, DOWN, UP),
) -> SpatialAdaptiveOperator:
    torch.manual_seed(seed)
    cfg = FiveMemoryConfig(
        v_dim=channels,
        k_dim=channels,
        in_dim=channels,
        feat_dim=channels,
        lr_dim=2,
        ret_dim=2,
        eta_max=0.5,
        ret_min=0.1,
        apply_stability_control=True,
    )
    return SpatialAdaptiveOperator(
        config=cfg,
        fusion=fusion,
        mode=mode,
        default_chunk_size=chunk_size,
        routes=routes,
    )


def test_property_a_shape_preservation() -> None:
    """Property A: Output shape equals input shape [B, C, H, W]."""
    B, C, H, W = 2, 4, 3, 5
    op = _create_operator(channels=C)
    Z = torch.randn(B, C, H, W)
    res = op(Z)
    assert res.fused_output.shape == (B, C, H, W)
    for r_name, r_out in res.directional_outputs.items():
        assert r_out.shape == (B, C, H, W), (
            f"Route {r_name} produced shape {r_out.shape}"
        )


def test_property_b_no_cross_direction_state_contamination() -> None:
    """Property B: No cross-direction state contamination.

    Perturbing or changing LEFT's initial state must have zero influence on
    RIGHT's final state or output.
    """
    B, C, H, W = 2, 4, 3, 3
    op = _create_operator(channels=C, seed=10)
    Z = torch.randn(B, C, H, W, dtype=torch.float64)

    # Initial states
    state_right = FiveMemoryState.initialize(
        C,
        C,
        C,
        batch_size=B,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(1),
    )
    state_left_1 = FiveMemoryState.initialize(
        C,
        C,
        C,
        batch_size=B,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(2),
    )
    state_left_2 = FiveMemoryState.initialize(
        C,
        C,
        C,
        batch_size=B,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(999),
    )

    init_states_1 = {"RIGHT": state_right, "LEFT": state_left_1}
    init_states_2 = {"RIGHT": state_right, "LEFT": state_left_2}

    res1 = op(Z, initial_states=init_states_1)
    res2 = op(Z, initial_states=init_states_2)

    # RIGHT output and final state must be identical bitwise between run 1 and run 2
    assert torch.equal(
        res1.directional_outputs["RIGHT"], res2.directional_outputs["RIGHT"]
    )
    assert torch.equal(
        res1.final_states["RIGHT"].content, res2.final_states["RIGHT"].content
    )

    # LEFT output must differ because initial state changed
    assert not torch.equal(
        res1.directional_outputs["LEFT"], res2.directional_outputs["LEFT"]
    )


def test_property_c_route_permutation_consistency() -> None:
    """Property C: Route permutation changes outputs only through documented assignment."""
    B, C, H, W = 1, 3, 2, 2
    Z = torch.randn(B, C, H, W, dtype=torch.float64)

    op_standard = _create_operator(channels=C, routes=(RIGHT, LEFT, DOWN, UP), seed=77)
    op_permuted = _create_operator(channels=C, routes=(UP, DOWN, LEFT, RIGHT), seed=77)

    # Copy branch parameters so systems are identical
    op_permuted.directional_branches["RIGHT"].load_state_dict(
        op_standard.directional_branches["RIGHT"].state_dict()
    )
    op_permuted.directional_branches["LEFT"].load_state_dict(
        op_standard.directional_branches["LEFT"].state_dict()
    )
    op_permuted.directional_branches["DOWN"].load_state_dict(
        op_standard.directional_branches["DOWN"].state_dict()
    )
    op_permuted.directional_branches["UP"].load_state_dict(
        op_standard.directional_branches["UP"].state_dict()
    )

    # Provide explicit initial states so both operators start with identical states per route
    init_states = {
        r.name: FiveMemoryState.initialize(
            C,
            C,
            C,
            batch_size=B,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(idx * 100),
        )
        for idx, r in enumerate([RIGHT, LEFT, DOWN, UP])
    }

    res_std = op_standard(Z, initial_states=init_states)
    res_perm = op_permuted(Z, initial_states=init_states)

    # Under EqualFusion(mode="mean"), permutation of terms in sum/mean does not alter fused output
    assert torch.allclose(res_std.fused_output, res_perm.fused_output, atol=1e-12)
    # Individual route restored predictions are identical
    assert torch.equal(
        res_std.directional_outputs["RIGHT"], res_perm.directional_outputs["RIGHT"]
    )
    assert torch.equal(
        res_std.directional_outputs["UP"], res_perm.directional_outputs["UP"]
    )


def test_property_d_single_route_reduction() -> None:
    """Property D: Disabling three directions reduces exactly to the enabled route."""
    B, C, H, W = 2, 4, 3, 4
    Z = torch.randn(B, C, H, W, dtype=torch.float64)

    # Operator with only RIGHT enabled
    op_single = _create_operator(channels=C, routes=(RIGHT,), seed=88)
    res_single = op_single(Z)

    # Fused output must equal RIGHT restored prediction exactly
    assert torch.equal(res_single.fused_output, res_single.directional_outputs["RIGHT"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_property_e_batch_dtype_preservation(
    dtype: torch.dtype, batch_size: int
) -> None:
    """Property E: Batch, device, and dtype preservation across rectangular feature maps."""
    H, W, C = 4, 6, 3
    op = _create_operator(channels=C, mode="paper_aligned")
    Z = torch.randn(batch_size, C, H, W, dtype=dtype)
    res = op(Z)

    assert res.fused_output.shape == (batch_size, C, H, W)
    assert res.fused_output.dtype == dtype
    assert res.fused_output.device == Z.device


def test_property_f_deterministic_repeated_execution() -> None:
    """Property F: Repeated execution produces identical results."""
    B, C, H, W = 2, 3, 3, 3
    op = _create_operator(channels=C, seed=99)
    Z = torch.randn(B, C, H, W)

    # Fixed initial states
    states = {
        r.name: FiveMemoryState.initialize(
            C, C, C, batch_size=B, generator=torch.Generator().manual_seed(123)
        )
        for r in (RIGHT, LEFT, DOWN, UP)
    }

    res1 = op(Z, initial_states=states)
    res2 = op(Z, initial_states=states)

    assert torch.equal(res1.fused_output, res2.fused_output)


def test_property_g_autograd_end_to_end() -> None:
    """Property G: Autograd works end-to-end through feature map -> 4 routes -> fusion -> loss."""
    B, C, H, W = 1, 2, 2, 3
    fusion = LearnedChannelFusion(channels=C, num_routes=4)
    op = _create_operator(channels=C, fusion=fusion, mode="paper_aligned")

    Z = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
    res = op(Z)
    loss = (res.fused_output**2).sum()
    loss.backward()

    assert Z.grad is not None
    assert Z.grad.shape == Z.shape
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()
    assert torch.norm(Z.grad) > 0.0

    # Verify fusion weights also received gradients
    assert fusion.weights.grad is not None
    assert not torch.isnan(fusion.weights.grad).any()


def test_learnable_initial_states_optimization() -> None:
    """Verify that learnable_initial_states=True creates trainable parameters receiving gradients."""
    C = 3
    cfg = FiveMemoryConfig(
        v_dim=C,
        k_dim=C,
        in_dim=C,
        feat_dim=C,
        lr_dim=2,
        ret_dim=2,
    )
    op = SpatialAdaptiveOperator(
        config=cfg,
        routes=(RIGHT, LEFT),
        learnable_initial_states=True,
    )
    # Check that initial state parameters exist
    trainable_params = [p for p in op.parameters() if p.requires_grad]
    assert len(trainable_params) > 0

    x = torch.randn(2, C, 4, 4, requires_grad=True)
    res = op(x)
    loss = (res.fused_output**2).sum()
    loss.backward()

    for p in trainable_params:
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
    assert x.grad is not None
