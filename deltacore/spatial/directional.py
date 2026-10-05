"""DeltaCore Directional Five-Memory Execution.

Wraps a route-agnostic FiveMemory learner with a SpatialRoute:
    feature map [B, C, H, W]
          │
          ▼
    serialization P_r -> [B, N, C]
          │
          ▼
    FiveMemory boundary-refresh scan
          │
          ▼
    trajectory & sequence output [B, N, V]
          │
          ▼
    spatial restoration P_r^{-1} -> [B, V, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.scans.boundary_chunked import (
    BoundaryChunkResult,
    BoundaryRefreshChunkScan,
)
from deltacore.spatial.routes import SpatialRoute, get_spatial_route
from deltacore.updates.five_memory import FiveMemorySystem


@dataclass(frozen=True)
class DirectionalExecutionResult:
    r"""Output and diagnostic trajectory for a single directional traversal.

    Attributes:
        route: SpatialRoute applied (RIGHT, LEFT, DOWN, UP).
        restored_predictions: Predictions restored to original 2D lattice [B, V, H, W].
        sequence_predictions: Raw 1D serialized sequence predictions [B, N, V].
        scan_result: Underlying BoundaryChunkResult with all diagnostics.
        final_state: Final FiveMemoryState after consuming entire sequence.
    """

    route: SpatialRoute
    restored_predictions: torch.Tensor
    sequence_predictions: torch.Tensor
    scan_result: BoundaryChunkResult
    final_state: FiveMemoryState


class DirectionalFiveMemory(nn.Module):
    r"""Executes the FiveMemory learner along a specified 2D spatial route.

    The learner remains strictly route-agnostic; all spatial serialization and
    restoration are handled externally by the specified SpatialRoute.

    Args:
        route: SpatialRoute instance (e.g. RIGHT, LEFT, DOWN, UP).
        system: FiveMemorySystem instance.
        chunk_scan: Optional BoundaryRefreshChunkScan instance. If None,
            constructed with system and default_chunk_size=1.
    """

    def __init__(
        self,
        route: SpatialRoute | str,
        system: FiveMemorySystem,
        chunk_scan: BoundaryRefreshChunkScan | None = None,
    ) -> None:
        super().__init__()
        self.route = get_spatial_route(route)
        self.system = system
        self.chunk_scan = (
            chunk_scan
            if chunk_scan is not None
            else BoundaryRefreshChunkScan(system, default_chunk_size=1)
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        initial_state: FiveMemoryState,
        targets: torch.Tensor | None = None,
        chunk_size: int | None = None,
        record_trajectory: bool = False,
    ) -> DirectionalExecutionResult:
        r"""Execute directional traversal over 2D spatial feature map.

        Args:
            feature_map: Spatial input tensor [B, C, H, W].
            initial_state: Starting FiveMemoryState at step 0.
            targets: Optional spatial target tensor [B, V, H, W].
            chunk_size: Optional chunk length override.
            record_trajectory: If True, records all intermediate states.

        Returns:
            DirectionalExecutionResult containing restored spatial predictions
            and full diagnostic telemetry.
        """
        if feature_map.ndim != 4:
            raise ValueError(
                f"DirectionalFiveMemory expects 4D feature map [B, C, H, W], got {list(feature_map.shape)}"
            )

        b, c, h, w = feature_map.shape

        # 1. Serialize input feature map
        input_seq = self.route.serialize(feature_map)  # [B, N, C]

        # 2. Serialize targets if provided
        target_seq: torch.Tensor | None = None
        if targets is not None:
            if (
                targets.ndim != 4
                or targets.shape[0] != b
                or targets.shape[2:] != (h, w)
            ):
                raise ValueError(
                    f"Targets shape {list(targets.shape)} must match feature map [B={b}, V, H={h}, W={w}]"
                )
            target_seq = self.route.serialize(targets)  # [B, N, V]

        # 3. Execute boundary-refresh chunk scan
        scan_res = self.chunk_scan.scan(
            inputs=input_seq,
            initial_state=initial_state,
            targets=target_seq,
            chunk_size=chunk_size,
            record_trajectory=record_trajectory,
        )

        # 4. Restore predictions back to 2D spatial lattice
        restored = self.route.restore(
            scan_res.predictions,
            height=h,
            width=w,
        )  # [B, V, H, W]

        return DirectionalExecutionResult(
            route=self.route,
            restored_predictions=restored,
            sequence_predictions=scan_res.predictions,
            scan_result=scan_res,
            final_state=scan_res.final_state,
        )
