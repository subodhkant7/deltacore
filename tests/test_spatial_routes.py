"""Unit tests for DeltaCore 2D spatial routes.

Verifies:
1. Exact mathematical round-trip identity: restore(serialize(Z)) == Z
2. Arbitrary rectangular grids: 1x1, 1xN, Nx1, 2x3, 3x2, 4x5, 7x11
3. FP32 and FP64 precision
4. Multi-batch (B > 1) and multi-channel (C > 1) preservation
5. Explicit coordinate encoding and ordering verification for RIGHT, LEFT, DOWN, UP
6. Device, dtype, and non-mutation invariants
"""

import pytest
import torch

from deltacore.spatial.routes import (
    ALL_ROUTES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    RouteDirection,
    get_spatial_route,
)


@pytest.mark.parametrize(
    "height,width",
    [
        (1, 1),
        (1, 7),
        (5, 1),
        (2, 3),
        (3, 2),
        (4, 5),
        (7, 11),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("channels", [1, 4])
def test_spatial_route_roundtrip_identity(
    height: int,
    width: int,
    dtype: torch.dtype,
    batch_size: int,
    channels: int,
) -> None:
    """Verify restore(serialize(Z)) == Z exactly across all routes and shapes."""
    torch.manual_seed(42 + height * 13 + width * 7)
    Z = torch.randn(batch_size, channels, height, width, dtype=dtype)
    Z_clone = Z.clone()

    for route in ALL_ROUTES:
        seq = route.serialize(Z)
        assert seq.shape == (batch_size, height * width, channels)
        assert seq.dtype == dtype
        assert seq.device == Z.device

        restored = route.restore(seq, height=height, width=width)
        assert restored.shape == Z.shape
        assert restored.dtype == dtype
        assert restored.device == Z.device

        # Exact numerical equality
        assert torch.equal(restored, Z), (
            f"Route {route.name} failed round-trip identity for shape {(height, width)}"
        )

    # Invariant: input feature map Z was not mutated
    assert torch.equal(Z, Z_clone)


def test_spatial_route_explicit_coordinate_ordering() -> None:
    """Verify serialize produces exact token sequence corresponding to documented coordinate order.

    Grid: H=3, W=4
    RIGHT: row-major (0,0), (0,1), ..., (2,3)
    LEFT: reverse row-major (2,3), (2,2), ..., (0,0)
    DOWN: column-major (0,0), (1,0), (2,0), (0,1), ..., (2,3)
    UP: reverse column-major (2,3), (1,3), (0,3), ..., (0,0)
    """
    H, W = 3, 4
    B, C = 1, 1

    # Coordinate-encoded feature map: value at (h, w) is 100 * h + w
    Z = torch.zeros(B, C, H, W, dtype=torch.float64)
    for h in range(H):
        for w in range(W):
            Z[0, 0, h, w] = 100.0 * h + float(w)

    # 1. RIGHT: row-major
    seq_right = RIGHT.serialize(Z).squeeze(0).squeeze(-1)  # shape [N]
    expected_right = [
        100.0 * h + float(w) for (h, w) in RIGHT.coordinate_sequence(H, W)
    ]
    assert torch.equal(seq_right, torch.tensor(expected_right, dtype=torch.float64))

    # 2. LEFT: reverse row-major
    seq_left = LEFT.serialize(Z).squeeze(0).squeeze(-1)
    expected_left = [100.0 * h + float(w) for (h, w) in LEFT.coordinate_sequence(H, W)]
    assert torch.equal(seq_left, torch.tensor(expected_left, dtype=torch.float64))
    # Reverse of RIGHT
    assert torch.equal(seq_left, torch.flip(seq_right, dims=[0]))

    # 3. DOWN: column-major
    seq_down = DOWN.serialize(Z).squeeze(0).squeeze(-1)
    expected_down = [100.0 * h + float(w) for (h, w) in DOWN.coordinate_sequence(H, W)]
    assert torch.equal(seq_down, torch.tensor(expected_down, dtype=torch.float64))
    # Explicitly check column 0: (0,0)=0, (1,0)=100, (2,0)=200
    assert seq_down[0].item() == 0.0
    assert seq_down[1].item() == 100.0
    assert seq_down[2].item() == 200.0
    # Column 1: (0,1)=1, (1,1)=101, (2,1)=201
    assert seq_down[3].item() == 1.0
    assert seq_down[4].item() == 101.0
    assert seq_down[5].item() == 201.0

    # 4. UP: reverse column-major
    seq_up = UP.serialize(Z).squeeze(0).squeeze(-1)
    expected_up = [100.0 * h + float(w) for (h, w) in UP.coordinate_sequence(H, W)]
    assert torch.equal(seq_up, torch.tensor(expected_up, dtype=torch.float64))
    # Reverse of DOWN
    assert torch.equal(seq_up, torch.flip(seq_down, dims=[0]))

    # Restore all and verify they match original Z
    for route in [RIGHT, LEFT, DOWN, UP]:
        seq = route.serialize(Z)
        rec = route.restore(seq, H, W)
        assert torch.equal(rec, Z)


def test_spatial_route_invalid_shapes() -> None:
    """Test boundary checks and clear error messages for malformed inputs."""
    Z_3d = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="Expected 4D feature map"):
        RIGHT.serialize(Z_3d)

    seq_4d = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="Expected 3D sequence"):
        RIGHT.restore(seq_4d, height=3, width=4)

    seq_mismatch = torch.randn(2, 10, 4)
    with pytest.raises(ValueError, match="Sequence length N=10 does not match"):
        RIGHT.restore(seq_mismatch, height=3, width=4)  # 3*4 = 12 != 10


def test_get_spatial_route_factory() -> None:
    """Test factory retrieval by string, enum, or route instance."""
    assert get_spatial_route("right") == RIGHT
    assert get_spatial_route("Left") == LEFT
    assert get_spatial_route(RouteDirection.DOWN) == DOWN
    assert get_spatial_route(UP) == UP
    with pytest.raises(ValueError, match="Unknown route direction"):
        get_spatial_route("DIAGONAL")
