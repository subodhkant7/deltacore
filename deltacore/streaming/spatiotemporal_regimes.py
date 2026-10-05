r"""DeltaCore Phase 12: Controlled Spatio-Temporal Regimes and Distribution Shifts.

Constructs reproducible, deterministic spatial fields X_t \in \mathbb{R}^{H \times W \times C}
evolving under structured temporal dynamics across multiple regimes:
    x_t = flatten(X_t) \in \mathbb{R}^D, \quad D = H \cdot W \cdot C

Regimes:
    - Regime A: Horizontal transport / wave motion (velocity v_x)
    - Regime B: Vertical transport / wave motion (velocity v_y)
    - Regime C: Localized 2D diffusion + rotating vortex / nonlinear perturbation
    - Regime A2: Related horizontal transport with modified wave speed / frequency

Tasks:
    1. Task A: Moving Spatial Field Prediction
    2. Task B: Spatio-Temporal Regime Switching (A -> B -> C -> A)
    3. Task C: Stale-Memory Challenge (A1 -> B -> A2)
    4. Spatial Structure Ablation: Shuffled spatial pixel permutation P_{spatial}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from deltacore.streaming.regimes import StreamData


@dataclass(frozen=True)
class SpatioTemporalDimensions:
    """Dimensions of the spatial field."""

    height: int
    width: int
    channels: int

    @property
    def dim(self) -> int:
        return self.height * self.width * self.channels


def create_initial_spatial_field(
    dims: SpatioTemporalDimensions,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a smooth initial spatial field using a 2D Gaussian blob."""
    H, W, C = dims.height, dims.width, dims.channels
    y = torch.linspace(-2.0, 2.0, H)
    x = torch.linspace(-2.0, 2.0, W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    field = torch.zeros(H, W, C, dtype=torch.float32)
    for c in range(C):
        cx = float(torch.randn(1, generator=generator).item() * 0.5)
        cy = float(torch.randn(1, generator=generator).item() * 0.5)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.5)
        field[:, :, c] = blob

    # Normalize to unit Frobenius norm
    norm = float(torch.linalg.norm(field).item())
    return field / max(norm, 1e-6)


def simulate_spatiotemporal_step(
    X: torch.Tensor,
    regime: str,
    dt: float = 0.1,
    diff_coeff: float = 0.05,
) -> torch.Tensor:
    r"""Simulate a single timestep of spatial field evolution X \in [H, W, C].

    Guaranteed to remain bounded and dissipative.
    """
    H, W, C = X.shape
    X_next = torch.zeros_like(X)

    if regime == "horizontal_transport":
        # Wave motion along horizontal axis (roll along W)
        for c in range(C):
            shifted = torch.roll(X[:, :, c], shifts=1, dims=1)
            # Add slight dissipative smoothing
            smooth = (
                torch.roll(X[:, :, c], shifts=-1, dims=1)
                + torch.roll(X[:, :, c], shifts=1, dims=1)
            ) * 0.5
            X_next[:, :, c] = 0.85 * shifted + 0.10 * smooth

    elif regime == "vertical_transport":
        # Wave motion along vertical axis (roll along H)
        for c in range(C):
            shifted = torch.roll(X[:, :, c], shifts=1, dims=0)
            smooth = (
                torch.roll(X[:, :, c], shifts=-1, dims=0)
                + torch.roll(X[:, :, c], shifts=1, dims=0)
            ) * 0.5
            X_next[:, :, c] = 0.85 * shifted + 0.10 * smooth

    elif regime == "diffusion_vortex":
        # 2D Laplacian diffusion + localized nonlinear rotation
        for c in range(C):
            laplacian = (
                torch.roll(X[:, :, c], shifts=1, dims=0)
                + torch.roll(X[:, :, c], shifts=-1, dims=0)
                + torch.roll(X[:, :, c], shifts=1, dims=1)
                + torch.roll(X[:, :, c], shifts=-1, dims=1)
                - 4.0 * X[:, :, c]
            )
            diffused = X[:, :, c] + diff_coeff * laplacian
            # Nonlinear vortex perturbation
            vortex = torch.tanh(diffused) * 0.90
            X_next[:, :, c] = vortex

    elif regime == "horizontal_transport_a2":
        # Related horizontal wave with modified velocity and phase
        for c in range(C):
            shifted = torch.roll(X[:, :, c], shifts=2, dims=1)
            X_next[:, :, c] = 0.88 * shifted + 0.08 * torch.sin(X[:, :, c])

    else:
        # Default dissipative step
        X_next = 0.90 * X

    # Ensure dissipative boundedness: max norm bounded
    val_max = float(torch.max(torch.abs(X_next)).item())
    if val_max > 5.0:
        X_next = torch.clamp(X_next, -5.0, 5.0)

    return X_next


def generate_spatiotemporal_stream(
    seq_len: int = 512,
    height: int = 8,
    width: int = 8,
    channels: int = 1,
    noise_std: float = 0.05,
    shift_interval: int | None = None,
    seed: int = 42,
) -> StreamData:
    r"""Generate Spatio-Temporal Benchmark Stream (A -> B -> C -> A).

    Spatial field: [H, W, C] -> flattened to D = H * W * C.
    Default shift points: [T/4, T/2, 3T/4], or based on shift_interval.
    """
    dims = SpatioTemporalDimensions(height, width, channels)
    D = dims.dim
    gen = torch.Generator().manual_seed(seed)

    if shift_interval is not None and shift_interval > 0:
        change_points = list(range(shift_interval, seq_len, shift_interval))
    else:
        change_points = [seq_len // 4, seq_len // 2, (3 * seq_len) // 4]

    regime_cycle = [
        "horizontal_transport",
        "vertical_transport",
        "diffusion_vortex",
        "horizontal_transport",
    ]

    inputs = torch.zeros(seq_len, D, dtype=torch.float32)
    targets = torch.zeros(seq_len, D, dtype=torch.float32)
    regime_ids: list[int] = []

    X_curr = create_initial_spatial_field(dims, generator=gen)

    for t in range(seq_len):
        # Determine regime
        r_idx = 0
        for i, cp in enumerate(change_points):
            if t >= cp:
                r_idx = (i + 1) % len(regime_cycle)

        regime_name = regime_cycle[r_idx]
        regime_ids.append(r_idx)

        # Compute next spatial state
        X_next = simulate_spatiotemporal_step(X_curr, regime_name)
        noise = (
            torch.randn(height, width, channels, generator=gen, dtype=torch.float32)
            * noise_std
        )
        X_target = X_next + noise

        # Flatten spatial dimensions
        inputs[t] = X_curr.flatten()
        targets[t] = X_target.flatten()
        X_curr = X_next

    metadata: dict[str, Any] = {
        "task": "Spatio-Temporal Regime Switching",
        "height": height,
        "width": width,
        "channels": channels,
        "dim": D,
        "seq_len": seq_len,
        "seed": seed,
        "shift_interval": shift_interval,
    }

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata=metadata,
    )


def generate_spatiotemporal_stale_stream(
    seq_len: int = 512,
    height: int = 8,
    width: int = 8,
    channels: int = 1,
    noise_std: float = 0.05,
    seed: int = 42,
) -> StreamData:
    r"""Generate Task C: Stale-Memory Challenge (A1 -> B -> A2).

    A1: Horizontal wave (shifts=1)
    B:  Vertical wave (shifts=1)
    A2: Related horizontal wave (shifts=2, modified frequency)

    Phase 1: [0, T/3) Regime A1
    Phase 2: [T/3, 2T/3) Regime B
    Phase 3: [2T/3, T) Regime A2
    """
    dims = SpatioTemporalDimensions(height, width, channels)
    D = dims.dim
    gen = torch.Generator().manual_seed(seed)
    change_points = [seq_len // 3, (2 * seq_len) // 3]

    inputs = torch.zeros(seq_len, D, dtype=torch.float32)
    targets = torch.zeros(seq_len, D, dtype=torch.float32)
    regime_ids: list[int] = []

    X_curr = create_initial_spatial_field(dims, generator=gen)

    for t in range(seq_len):
        if t < change_points[0]:
            r_idx = 0
            regime_name = "horizontal_transport"
        elif t < change_points[1]:
            r_idx = 1
            regime_name = "vertical_transport"
        else:
            r_idx = 2
            regime_name = "horizontal_transport_a2"

        regime_ids.append(r_idx)
        X_next = simulate_spatiotemporal_step(X_curr, regime_name)
        noise = (
            torch.randn(height, width, channels, generator=gen, dtype=torch.float32)
            * noise_std
        )
        X_target = X_next + noise

        inputs[t] = X_curr.flatten()
        targets[t] = X_target.flatten()
        X_curr = X_next

    metadata: dict[str, Any] = {
        "task": "Stale-Memory Challenge (A1 -> B -> A2)",
        "height": height,
        "width": width,
        "channels": channels,
        "dim": D,
        "seq_len": seq_len,
        "seed": seed,
    }

    return StreamData(
        inputs=inputs,
        targets=targets,
        regime_ids=regime_ids,
        change_points=change_points,
        metadata=metadata,
    )


def apply_spatial_permutation(
    stream: StreamData,
    permutation_seed: int = 999,
) -> StreamData:
    r"""Ablation 12: Destroy spatial arrangement via fixed permutation P_{spatial}.

    Permutes the spatial coordinates identically across all timesteps,
    preserving exact marginal values while destroying 2D spatial locality.
    """
    D = stream.inputs.shape[1]
    gen = torch.Generator().manual_seed(permutation_seed)
    perm = torch.randperm(D, generator=gen)

    perm_inputs = stream.inputs[:, perm]
    perm_targets = stream.targets[:, perm]

    meta = dict(stream.metadata)
    meta["spatial_ablation"] = "shuffled_permutation"
    meta["permutation_seed"] = permutation_seed

    return StreamData(
        inputs=perm_inputs,
        targets=perm_targets,
        regime_ids=list(stream.regime_ids),
        change_points=list(stream.change_points),
        metadata=meta,
    )
