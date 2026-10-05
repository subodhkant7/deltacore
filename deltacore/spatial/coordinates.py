"""DeltaCore Spatial Coordinate Injection & Transformation Primitives.

Provides:
- Normalized 2D coordinate grid generation: (x, y) in [-1, 1]^2
- Explicit coordinate channel injection: [B, C, H, W] -> [B, C+2, H, W]
- Shuffled coordinate control: preserves coordinate marginal distribution while destroying true spatial location
- Translation stress transformation: random cyclic shifts preserving pattern semantics
- Pixel shuffle control: permuting spatial token indices to isolate spatial structure from pixel distribution
"""

from __future__ import annotations

import math

import torch


def generate_2d_coordinates(
    height: int,
    width: int,
    normalized: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Generate a 2D coordinate grid [2, H, W].

    Args:
        height: Spatial grid height H.
        width: Spatial grid width W.
        normalized: If True, coordinates are linearly scaled to [-1.0, 1.0].
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor of shape [2, H, W], where channel 0 is x (width) and channel 1 is y (height).
    """
    if normalized:
        x_coords = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        y_coords = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    else:
        x_coords = torch.arange(width, device=device, dtype=dtype)
        y_coords = torch.arange(height, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    # Stack into [2, H, W] where channel 0 is x and channel 1 is y
    coords = torch.stack([grid_x, grid_y], dim=0)
    return coords


def inject_2d_coordinates(
    feature_map: torch.Tensor,
    normalized: bool = True,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Inject 2D spatial coordinates into the channel dimension of a feature map.

    Args:
        feature_map: Input tensor [B, C, H, W].
        normalized: If True, uses coordinates in [-1.0, 1.0].
        shuffle: If True, randomly permutes coordinate assignments across (H, W)
            independently for each sample in the batch. Preserves the marginal
            distribution of coordinates while destroying spatial address correspondence.
        generator: Optional torch.Generator for deterministic shuffling.

    Returns:
        Augmented tensor [B, C+2, H, W].
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must be 4D [B, C, H, W], got shape {list(feature_map.shape)}"
        )

    b, _c, h, w = feature_map.shape
    coords = generate_2d_coordinates(
        height=h,
        width=w,
        normalized=normalized,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )  # [2, H, W]

    coords_batch = coords.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, 2, H, W]

    if shuffle:
        # Permute (H*W) spatial locations independently per sample
        flat_coords = coords_batch.view(b, 2, h * w)
        shuffled_flat = torch.empty_like(flat_coords)
        for i in range(b):
            perm = (
                torch.randperm(h * w, generator=generator)
                if generator is not None
                else torch.randperm(h * w)
            )
            shuffled_flat[i] = flat_coords[i, :, perm]
        coords_batch = shuffled_flat.view(b, 2, h, w)

    return torch.cat([feature_map, coords_batch], dim=1)


def translate_spatial_patterns(
    feature_map: torch.Tensor,
    max_shift: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Apply random spatial cyclic translations independently to each sample.

    Preserves pattern class identity while testing translation invariance.

    Args:
        feature_map: Input tensor [B, C, H, W].
        max_shift: Maximum integer shift along height and width.
        generator: Optional torch.Generator for deterministic shifts.

    Returns:
        Translated tensor [B, C, H, W].
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must be 4D [B, C, H, W], got shape {list(feature_map.shape)}"
        )

    b = feature_map.shape[0]
    translated = torch.empty_like(feature_map)

    for i in range(b):
        if generator is not None:
            shift_h = int(
                torch.randint(
                    -max_shift, max_shift + 1, (1,), generator=generator
                ).item()
            )
            shift_w = int(
                torch.randint(
                    -max_shift, max_shift + 1, (1,), generator=generator
                ).item()
            )
        else:
            shift_h = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            shift_w = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())

        translated[i] = torch.roll(
            feature_map[i], shifts=(shift_h, shift_w), dims=(-2, -1)
        )

    return translated


def pixel_shuffle_control(
    feature_map: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Randomly permute spatial pixel positions independently for each sample.

    Preserves exact pixel value distribution and ||X_i||_F while destroying all 2D structure.

    Args:
        feature_map: Input tensor [B, C, H, W].
        generator: Optional torch.Generator for deterministic permutation.

    Returns:
        Pixel-permuted tensor [B, C, H, W].
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must be 4D [B, C, H, W], got shape {list(feature_map.shape)}"
        )

    b, c, h, w = feature_map.shape
    flat = feature_map.view(b, c, h * w)
    shuffled_flat = torch.empty_like(flat)

    for i in range(b):
        perm = (
            torch.randperm(h * w, generator=generator)
            if generator is not None
            else torch.randperm(h * w)
        )
        shuffled_flat[i] = flat[i, :, perm]

    return shuffled_flat.view(b, c, h, w)


def compute_coordinate_statistics(
    feature_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    r"""Compute basic spatial coordinate and energy statistics for 4D feature maps.

    Computes per-sample:
    - total_energy: \sum_{c,h,w} X_{b,c,h,w}^2, shape [B]
    - row_energy: \sum_{c,w} X_{b,c,h,w}^2, shape [B, H]
    - col_energy: \sum_{c,h} X_{b,c,h,w}^2, shape [B, W]
    - center_of_mass_y: \sum_h h \cdot row_energy[b, h] / total_energy[b], shape [B]
    - center_of_mass_x: \sum_w w \cdot col_energy[b, w] / total_energy[b], shape [B]

    Args:
        feature_map: Input tensor [B, C, H, W].

    Returns:
        Dictionary mapping statistic names to tensors:
        - "total_energy": [B]
        - "row_energy": [B, H]
        - "col_energy": [B, W]
        - "center_of_mass_y": [B]
        - "center_of_mass_x": [B]
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must be 4D [B, C, H, W], got shape {list(feature_map.shape)}"
        )

    b, _c, h, w = feature_map.shape
    squared = feature_map**2

    total_energy = squared.sum(dim=(1, 2, 3))  # [B]
    row_energy = squared.sum(dim=(1, 3))  # [B, H]
    col_energy = squared.sum(dim=(1, 2))  # [B, W]

    eps = 1e-8
    safe_total = torch.clamp(total_energy, min=eps)

    h_indices = torch.arange(
        h, device=feature_map.device, dtype=feature_map.dtype
    )  # [H]
    w_indices = torch.arange(
        w, device=feature_map.device, dtype=feature_map.dtype
    )  # [W]

    com_y = (row_energy * h_indices.unsqueeze(0)).sum(dim=1) / safe_total  # [B]
    com_x = (col_energy * w_indices.unsqueeze(0)).sum(dim=1) / safe_total  # [B]

    return {
        "total_energy": total_energy,
        "row_energy": row_energy,
        "col_energy": col_energy,
        "center_of_mass_y": com_y,
        "center_of_mass_x": com_x,
    }


def coordinate_statistic_control(
    feature_map: torch.Tensor,
    target_total_energy: float = 1.0,
    match_center_of_mass: bool = True,
) -> torch.Tensor:
    r"""Construct or transform examples to control coordinate statistics (B2 Control).

    Controls basic statistics across classes where practical:
    1. Total Energy: Normalized strictly so that \sum_{c,h,w} X_{b,c,h,w}^2 = target_total_energy
       (||X_b||_F = \sqrt{target_total_energy}).
    2. Center of Mass: If match_center_of_mass=True, calculates empirical center of mass
       (CoM_y, CoM_x) and applies cyclic roll shifts so that the center of mass aligns at
       the grid center ((H-1)/2, (W-1)/2).
    3. Row and Column Energy: By centering mass and strictly matching total energy,
       first- and second-order spatial moment discrepancies across classes are mitigated.

    Args:
        feature_map: Input tensor [B, C, H, W].
        target_total_energy: Desired total squared Frobenius norm per sample.
        match_center_of_mass: If True, centers the mass of each sample.

    Returns:
        Controlled feature map [B, C, H, W].
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must be 4D [B, C, H, W], got shape {list(feature_map.shape)}"
        )

    b, _c, h, w = feature_map.shape
    stats = compute_coordinate_statistics(feature_map)

    controlled = feature_map.clone()

    if match_center_of_mass:
        target_y = (h - 1) / 2.0
        target_x = (w - 1) / 2.0

        for i in range(b):
            curr_y = stats["center_of_mass_y"][i].item()
            curr_x = stats["center_of_mass_x"][i].item()

            shift_y = int(round(target_y - curr_y))
            shift_x = int(round(target_x - curr_x))

            if shift_y != 0 or shift_x != 0:
                controlled[i] = torch.roll(
                    controlled[i], shifts=(shift_y, shift_x), dims=(-2, -1)
                )

    # Re-normalize total energy strictly to target_total_energy (Frobenius norm sqrt(target))
    target_norm = math.sqrt(max(target_total_energy, 1e-8))
    for i in range(b):
        cur_norm = torch.norm(controlled[i]).item()
        if cur_norm > 1e-6:
            controlled[i] = controlled[i] * (target_norm / cur_norm)

    return controlled
