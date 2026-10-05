"""DeltaCore Spatial Routing Specifications and Operators.

Provides exact serialization and restoration mappings for 2D feature maps:
    RIGHT (row-major: (0,0) -> (0,W-1) -> ...)
    LEFT  (reverse row-major)
    DOWN  (column-major: (0,0) -> (H-1,0) -> ...)
    UP    (reverse column-major)

Guarantees:
- Exact round-trip mathematical identity: restore(serialize(Z)) == Z
- Shape, batch, channel, dtype, and device preservation
- Zero in-place tensor mutation
"""

from __future__ import annotations

from enum import Enum

import torch


class RouteDirection(str, Enum):
    """Enumeration of standard 2D spatial traversal directions."""

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    DOWN = "DOWN"
    UP = "UP"


class SpatialRoute:
    """Base class and factory for 2D spatial routing.

    A spatial route defines a deterministic traversal order over a 2D spatial
    lattice of shape (H, W), mapping a 4D feature map of shape (B, C, H, W)
    into a serialized 3D sequence of shape (B, N, C), and its exact inverse
    restoring (B, N, C) back to (B, C, H, W).
    """

    def __init__(self, direction: RouteDirection | str) -> None:
        if isinstance(direction, str):
            direction = RouteDirection(direction.upper())
        self.direction = direction

    @property
    def name(self) -> str:
        return self.direction.value

    def serialize(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Serialize a 4D feature map into a 3D token sequence.

        Args:
            feature_map: Tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, N, C) where N = H * W.
        """
        if feature_map.ndim != 4:
            raise ValueError(
                f"Expected 4D feature map [B, C, H, W], got tensor of shape {tuple(feature_map.shape)}"
            )

        b, c, h, w = feature_map.shape
        n = h * w

        if self.direction == RouteDirection.RIGHT:
            # Row-major: (0,0) -> (0,1) -> ... -> (0, W-1) -> (1,0) -> ...
            # Permute to (B, H, W, C) then flatten spatial dimensions to N
            return feature_map.permute(0, 2, 3, 1).reshape(b, n, c)

        elif self.direction == RouteDirection.LEFT:
            # Reverse row-major: (H-1, W-1) -> ... -> (0,0)
            right_seq = feature_map.permute(0, 2, 3, 1).reshape(b, n, c)
            return torch.flip(right_seq, dims=[1])

        elif self.direction == RouteDirection.DOWN:
            # Column-major: (0,0) -> (1,0) -> ... -> (H-1,0) -> (0,1) -> ...
            # Permute to (B, W, H, C) then flatten spatial dimensions to N
            return feature_map.permute(0, 3, 2, 1).reshape(b, n, c)

        elif self.direction == RouteDirection.UP:
            # Reverse column-major: (H-1, W-1) -> ... -> (0,0)
            down_seq = feature_map.permute(0, 3, 2, 1).reshape(b, n, c)
            return torch.flip(down_seq, dims=[1])

        else:
            raise NotImplementedError(f"Unsupported direction: {self.direction}")

    def restore(self, sequence: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Restore a 3D token sequence back into a 4D spatial feature map.

        Args:
            sequence: Tensor of shape (B, N, C).
            height: Original spatial height H.
            width: Original spatial width W.

        Returns:
            Tensor of shape (B, C, H, W).
        """
        if sequence.ndim != 3:
            raise ValueError(
                f"Expected 3D sequence [B, N, C], got tensor of shape {tuple(sequence.shape)}"
            )

        b, n, c = sequence.shape
        expected_n = height * width
        if n != expected_n:
            raise ValueError(
                f"Sequence length N={n} does not match H*W={height}*{width}={expected_n}"
            )

        if self.direction == RouteDirection.RIGHT:
            # seq shape: (B, H*W, C) -> (B, H, W, C) -> permute to (B, C, H, W)
            return sequence.reshape(b, height, width, c).permute(0, 3, 1, 2)

        elif self.direction == RouteDirection.LEFT:
            # Reverse row-major: unflip along sequence dim then invert row-major
            unflipped = torch.flip(sequence, dims=[1])
            return unflipped.reshape(b, height, width, c).permute(0, 3, 1, 2)

        elif self.direction == RouteDirection.DOWN:
            # Column-major seq shape: (B, W*H, C) -> (B, W, H, C) -> permute to (B, C, H, W)
            return sequence.reshape(b, width, height, c).permute(0, 3, 2, 1)

        elif self.direction == RouteDirection.UP:
            # Reverse column-major: unflip along sequence dim then invert column-major
            unflipped = torch.flip(sequence, dims=[1])
            return unflipped.reshape(b, width, height, c).permute(0, 3, 2, 1)

        else:
            raise NotImplementedError(f"Unsupported direction: {self.direction}")

    def coordinate_sequence(self, height: int, width: int) -> list[tuple[int, int]]:
        """Return the explicit list of (h, w) coordinates visited at each token step."""
        coords: list[tuple[int, int]] = []
        if self.direction == RouteDirection.RIGHT:
            for h in range(height):
                for w in range(width):
                    coords.append((h, w))
        elif self.direction == RouteDirection.LEFT:
            for h in reversed(range(height)):
                for w in reversed(range(width)):
                    coords.append((h, w))
        elif self.direction == RouteDirection.DOWN:
            for w in range(width):
                for h in range(height):
                    coords.append((h, w))
        elif self.direction == RouteDirection.UP:
            for w in reversed(range(width)):
                for h in reversed(range(height)):
                    coords.append((h, w))
        return coords

    def __repr__(self) -> str:
        return f"SpatialRoute({self.direction.value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SpatialRoute):
            return self.direction == other.direction
        return False


# Canonical instances
RIGHT = SpatialRoute(RouteDirection.RIGHT)
LEFT = SpatialRoute(RouteDirection.LEFT)
DOWN = SpatialRoute(RouteDirection.DOWN)
UP = SpatialRoute(RouteDirection.UP)

ALL_ROUTES = (RIGHT, LEFT, DOWN, UP)


def get_spatial_route(direction: RouteDirection | str | SpatialRoute) -> SpatialRoute:
    """Retrieve canonical SpatialRoute instance."""
    if isinstance(direction, SpatialRoute):
        return direction
    if isinstance(direction, RouteDirection):
        val = direction.value
    else:
        val = str(direction).upper()

    mapping = {
        "RIGHT": RIGHT,
        "LEFT": LEFT,
        "DOWN": DOWN,
        "UP": UP,
    }
    if val not in mapping:
        raise ValueError(
            f"Unknown route direction '{direction}'. Expected one of {list(mapping.keys())}"
        )
    return mapping[val]
