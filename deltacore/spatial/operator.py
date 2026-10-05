"""DeltaCore 2D Spatial Adaptive Operator.

Composes four independent directional FiveMemory learners over a 2D feature map:
    RIGHT (row-major)
    LEFT  (reverse row-major)
    DOWN  (column-major)
    UP    (reverse column-major)

Features:
- Independent FiveMemory instances per direction (no cross-direction state contamination)
- Flexible chunking modes:
    * "paper_aligned": C_right = W, C_left = W, C_down = H, C_up = H
    * "generic": user-specified uniform or per-route chunk size
- Pluggable spatial fusion (EqualFusion, LearnedChannelFusion)
- Batch, device, and dtype preservation
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.spatial.directional import (
    DirectionalExecutionResult,
    DirectionalFiveMemory,
)
from deltacore.spatial.fusion import EqualFusion, SpatialFusion
from deltacore.spatial.routes import (
    ALL_ROUTES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    SpatialRoute,
    get_spatial_route,
)
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem


@dataclass(frozen=True)
class SpatialOperatorResult:
    r"""Comprehensive output and trajectory telemetry from SpatialAdaptiveOperator.

    Attributes:
        fused_output: Spatial output after multi-directional fusion [B, C, H, W].
        directional_results: Dict mapping route name to DirectionalExecutionResult.
        directional_outputs: Dict mapping route name to restored tensor [B, C, H, W].
        final_states: Dict mapping route name to final FiveMemoryState.
    """

    fused_output: torch.Tensor
    directional_results: dict[str, DirectionalExecutionResult]
    directional_outputs: dict[str, torch.Tensor]
    final_states: dict[str, FiveMemoryState]


class SpatialAdaptiveOperator(nn.Module):
    r"""2D Spatial Adaptive Operator executing four independent directional learners.

    Applies DeltaCore's five-memory reference learner along multiple spatial
    traversals, restoring and fusing their outputs.

    Modes:
        - mode="paper_aligned":
            Automatically configures chunk size matching VisionHOPE row/col dimensions:
                C_right = W, C_left = W, C_down = H, C_up = H
        - mode="generic":
            Uses uniform chunk_size for all directions (or default_chunk_size).

    Args:
        config: FiveMemoryConfig governing all directional learners.
        fusion: SpatialFusion operator (default: EqualFusion(mode="mean")).
        mode: Chunking mode ("generic" or "paper_aligned", default: "generic").
        default_chunk_size: Default chunk size for mode="generic" (default: 1).
        routes: Sequence of SpatialRoute instances to execute (default: RIGHT, LEFT, DOWN, UP).
    """

    def __init__(
        self,
        config: FiveMemoryConfig,
        fusion: SpatialFusion | None = None,
        mode: str = "generic",
        default_chunk_size: int = 1,
        routes: Sequence[SpatialRoute] = ALL_ROUTES,
        learnable_initial_states: bool = False,
        trainable_dynamics: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode.lower()
        if self.mode not in ("generic", "paper_aligned"):
            raise ValueError(f"mode must be 'generic' or 'paper_aligned', got '{mode}'")

        self.default_chunk_size = default_chunk_size
        self.routes = [get_spatial_route(r) for r in routes]
        self.learnable_initial_states = learnable_initial_states
        self.trainable_dynamics = trainable_dynamics

        # Four independently initialized five-memory systems
        branches: dict[str, DirectionalFiveMemory] = {}
        for r in self.routes:
            sys = FiveMemorySystem(config, trainable_dynamics=trainable_dynamics)
            branches[r.name] = DirectionalFiveMemory(
                route=r,
                system=sys,
            )
        self.directional_branches = nn.ModuleDict(branches)

        # Learnable initial states if requested
        if self.learnable_initial_states:
            self.init_state_params = nn.ParameterDict()
            for r in self.routes:
                self.init_state_params[f"{r.name}_content"] = nn.Parameter(
                    torch.zeros(config.v_dim, config.k_dim)
                )
                self.init_state_params[f"{r.name}_key"] = nn.Parameter(
                    torch.randn(config.k_dim, config.in_dim) * 0.02
                )
                self.init_state_params[f"{r.name}_val"] = nn.Parameter(
                    torch.randn(config.v_dim, config.in_dim) * 0.02
                )
                self.init_state_params[f"{r.name}_lr"] = nn.Parameter(
                    torch.zeros(config.lr_dim, config.get_feat_dim())
                )
                self.init_state_params[f"{r.name}_ret"] = nn.Parameter(
                    torch.zeros(config.ret_dim, config.get_feat_dim())
                )

        # Spatial fusion operator
        self.fusion = fusion if fusion is not None else EqualFusion(mode="mean")

    def get_chunk_size_for_route(
        self, route: SpatialRoute, height: int, width: int, override_chunk: int | None
    ) -> int:
        """Resolve chunk size according to configured execution mode."""
        if self.mode == "paper_aligned":
            # Paper semantics: RIGHT/LEFT chunk along rows (W), DOWN/UP chunk along cols (H)
            if route in (RIGHT, LEFT):
                return width
            elif route in (DOWN, UP):
                return height
            else:
                return width
        else:
            return (
                override_chunk
                if override_chunk is not None
                else self.default_chunk_size
            )

    def forward(
        self,
        feature_map: torch.Tensor,
        initial_states: Mapping[str, FiveMemoryState] | None = None,
        chunk_size: int | None = None,
        record_trajectories: bool = False,
    ) -> SpatialOperatorResult:
        r"""Execute spatial multi-directional routing and fusion.

        Args:
            feature_map: 4D input feature map [B, C, H, W].
            initial_states: Optional dict mapping route name to FiveMemoryState.
                If None, freshly initialized for each direction.
            chunk_size: Optional chunk size override (used when mode="generic").
            record_trajectories: If True, records detailed state trajectories.

        Returns:
            SpatialOperatorResult with fused output and directional diagnostics.
        """
        if feature_map.ndim != 4:
            raise ValueError(
                f"SpatialAdaptiveOperator expects 4D input [B, C, H, W], got {list(feature_map.shape)}"
            )

        b, c, h, w = feature_map.shape
        directional_results: dict[str, DirectionalExecutionResult] = {}
        directional_outputs: dict[str, torch.Tensor] = {}
        final_states: dict[str, FiveMemoryState] = {}

        for route in self.routes:
            branch = self.directional_branches[route.name]

            # Route-specific chunk size
            c_size = self.get_chunk_size_for_route(
                route=route, height=h, width=w, override_chunk=chunk_size
            )

            # Resolve initial state
            if initial_states is not None and route.name in initial_states:
                state_r = initial_states[route.name]
            elif self.learnable_initial_states:
                p_c = self.init_state_params[f"{route.name}_content"].to(
                    device=feature_map.device, dtype=feature_map.dtype
                )
                p_k = self.init_state_params[f"{route.name}_key"].to(
                    device=feature_map.device, dtype=feature_map.dtype
                )
                p_v = self.init_state_params[f"{route.name}_val"].to(
                    device=feature_map.device, dtype=feature_map.dtype
                )
                p_lr = self.init_state_params[f"{route.name}_lr"].to(
                    device=feature_map.device, dtype=feature_map.dtype
                )
                p_ret = self.init_state_params[f"{route.name}_ret"].to(
                    device=feature_map.device, dtype=feature_map.dtype
                )
                state_r = FiveMemoryState(
                    content=p_c.unsqueeze(0).repeat(b, 1, 1),
                    key=p_k.unsqueeze(0).repeat(b, 1, 1),
                    value=p_v.unsqueeze(0).repeat(b, 1, 1),
                    learning_rate=p_lr.unsqueeze(0).repeat(b, 1, 1),
                    retention=p_ret.unsqueeze(0).repeat(b, 1, 1),
                )
            else:
                state_r = FiveMemoryState.initialize(
                    v_dim=self.config.v_dim,
                    k_dim=self.config.k_dim,
                    in_dim=self.config.in_dim,
                    feat_dim=self.config.get_feat_dim(),
                    lr_dim=self.config.lr_dim,
                    ret_dim=self.config.ret_dim,
                    batch_size=b,
                    dtype=feature_map.dtype,
                    device=feature_map.device,
                )

            # Traversal along route
            res_r = branch(
                feature_map=feature_map,
                initial_state=state_r,
                chunk_size=c_size,
                record_trajectory=record_trajectories,
            )

            directional_results[route.name] = res_r
            directional_outputs[route.name] = res_r.restored_predictions
            final_states[route.name] = res_r.final_state

        # Fuse directional outputs
        fused = self.fusion(directional_outputs)

        return SpatialOperatorResult(
            fused_output=fused,
            directional_results=directional_results,
            directional_outputs=directional_outputs,
            final_states=final_states,
        )
