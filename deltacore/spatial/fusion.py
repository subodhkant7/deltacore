r"""DeltaCore Spatial Directional Fusion Operators.

Provides modular fusion layers for combining directional 2D outputs:
    EqualFusion: Unweighted mean or sum across directional outputs.
    LearnedChannelFusion: Direction-wise per-channel weighting:
        Y = \sum_{r \in {R, L, D, U}} \lambda_r \odot Y_r
        where \lambda_r \in \mathbb{R}^C.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from deltacore.spatial.routes import SpatialRoute


class SpatialFusion(nn.Module, ABC):
    """Abstract base class for combining multi-directional spatial outputs."""

    @abstractmethod
    def forward(
        self,
        directional_outputs: Mapping[SpatialRoute | str, torch.Tensor]
        | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Combine directional tensors of shape [B, C, H, W] into a fused tensor [B, C, H, W]."""
        pass


class EqualFusion(SpatialFusion):
    r"""Combines directional outputs via equal unweighted aggregation.

    Semantics:
        If mode="mean":
            Y = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} Y_r
        If mode="sum":
            Y = \sum_{r \in \mathcal{R}} Y_r

    Args:
        mode: Aggregation mode ("mean" or "sum", default: "mean").
    """

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        mode_clean = mode.lower()
        if mode_clean not in ("mean", "sum"):
            raise ValueError(f"EqualFusion mode must be 'mean' or 'sum', got '{mode}'")
        self.mode = mode_clean

    def forward(
        self,
        directional_outputs: Mapping[SpatialRoute | str, torch.Tensor]
        | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(directional_outputs, Mapping):
            tensors = list(directional_outputs.values())
        else:
            tensors = list(directional_outputs)

        if not tensors:
            raise ValueError("directional_outputs must not be empty.")

        stacked = torch.stack(tensors, dim=0)  # [R, B, C, H, W]
        if self.mode == "mean":
            return torch.mean(stacked, dim=0)
        return torch.sum(stacked, dim=0)


class LearnedChannelFusion(SpatialFusion):
    r"""Combines directional outputs via learnable per-route, per-channel weights.

    Semantics:
        Y = \sum_{r=0}^{R-1} \lambda_r \odot Y_r

    where \lambda_r \in \mathbb{R}^C is broadcast across batch and spatial dimensions:
        [1, C, 1, 1].

    Normalization Options:
        - normalize=True (default):
          Weights are normalized via softmax across routes for each channel:
              \lambda_{r, c} = \frac{\exp(w_{r, c})}{\sum_{r'=0}^{R-1} \exp(w_{r', c})}
          guaranteeing \sum_{r} \lambda_{r, c} = 1 and \lambda_{r, c} > 0.
        - normalize=False:
          Direct unconstrained channel scaling: \lambda_{r, c} = w_{r, c}.

    Args:
        channels: Channel dimension C of feature maps.
        num_routes: Number of directional routes R (default: 4 for RIGHT, LEFT, DOWN, UP).
        normalize: Whether to apply softmax normalization across routes per channel.
        init_equal: If True, initializes weights equally across routes.
    """

    def __init__(
        self,
        channels: int,
        num_routes: int = 4,
        normalize: bool = True,
        init_equal: bool = True,
    ) -> None:
        super().__init__()
        if num_routes < 1:
            raise ValueError(f"num_routes must be >= 1, got {num_routes}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self.channels = channels
        self.num_routes = num_routes
        self.normalize = normalize

        # Parameters of shape [R, C]
        if init_equal:
            # Equal initialization
            raw_weights = torch.zeros(num_routes, channels)
        else:
            raw_weights = torch.randn(num_routes, channels) * 0.02

        self.weights = nn.Parameter(raw_weights)

    def get_effective_weights(self) -> torch.Tensor:
        r"""Compute effective directional weights \lambda of shape [R, 1, C, 1, 1]."""
        if self.normalize:
            # Softmax across route dimension (dim 0)
            norm_weights = torch.softmax(self.weights, dim=0)
        else:
            norm_weights = self.weights
        # Reshape to [R, 1, C, 1, 1] for broadcasting against [B, C, H, W]
        return norm_weights.view(self.num_routes, 1, self.channels, 1, 1)

    def forward(
        self,
        directional_outputs: Mapping[SpatialRoute | str, torch.Tensor]
        | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(directional_outputs, Mapping):
            tensors = list(directional_outputs.values())
        else:
            tensors = list(directional_outputs)

        if len(tensors) != self.num_routes:
            raise ValueError(
                f"Expected {self.num_routes} directional outputs, got {len(tensors)}."
            )

        stacked = torch.stack(tensors, dim=0)  # [R, B, C, H, W]
        lambdas = self.get_effective_weights()  # [R, 1, C, 1, 1]
        fused = torch.sum(lambdas * stacked, dim=0)  # [B, C, H, W]
        return fused
