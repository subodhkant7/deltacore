# ==============================================================================
# DeltaCore: deltacore/scans/delta_affine.py
# Conversion of error-correcting delta updates into affine scan operators.
# ==============================================================================

import torch

from deltacore.scans.affine import AffineScanMetadata, AffineScanOperator
from deltacore.stability.controllers import BaseStabilityController


def delta_to_affine(
    key: torch.Tensor,
    target: torch.Tensor,
    step_size: float | torch.Tensor = 1.0,
    stability_controller: BaseStabilityController | None = None,
) -> AffineScanOperator:
    r"""Convert a delta-rule token transition into an exact AffineScanOperator $(A_t, B_t)$.

    Mathematical derivation:
        $$M_{t+1} = M_t + \eta_t (v_t - M_t k_t) k_t^\top$$
        $$M_{t+1} = M_t (I_K - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$$
        $$A_t = I_K - \eta_t k_t k_t^\top \in \mathbb{R}^{K \times K}$$
        $$B_t = \eta_t v_t k_t^\top \in \mathbb{R}^{V \times K}$$

    Args:
        key: Input key vector $k_t \in \mathbb{R}^K$ or batched `[B, K]`.
        target: Target value vector $v_t \in \mathbb{R}^V$ or batched `[B, V]`.
        step_size: Non-negative scalar learning rate $\eta_t \ge 0$ (float or tensor).
        stability_controller: Optional BaseStabilityController to constrain step size.

    Returns:
        AffineScanOperator with exact $A_t$ and $B_t$ preserving autograd gradients.

    Raises:
        TypeError: If types, dtypes, or devices mismatch.
        ValueError: If dimensions or values are invalid.
    """
    if not isinstance(key, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("key and target must be torch.Tensor instances.")
    if key.dtype != target.dtype:
        raise TypeError(
            f"Dtype mismatch: key is {key.dtype}, target is {target.dtype}."
        )
    if key.device != target.device:
        raise ValueError(
            f"Device mismatch: key is on {key.device}, target is on {target.device}."
        )
    if key.ndim not in (1, 2) or target.ndim not in (1, 2):
        raise ValueError(
            f"Expected 1D or 2D key/target, got key.ndim={key.ndim}, target.ndim={target.ndim}."
        )
    if key.ndim != target.ndim:
        raise ValueError(
            f"key and target must have matching ndim, got key.ndim={key.ndim}, target.ndim={target.ndim}."
        )

    # Process step size and optional stability constraint
    if isinstance(step_size, torch.Tensor):
        eta_raw = step_size
    else:
        eta_raw = torch.tensor(step_size, dtype=key.dtype, device=key.device)

    if (eta_raw < 0.0).any():
        raise ValueError(f"step_size must be non-negative, got {step_size}.")

    clipped_count = 0
    if stability_controller is not None:
        stab_res = stability_controller.safe_step(eta_raw, key)
        eta = stab_res.safe_value
        norm_step = float(torch.max(stab_res.normalized_value).item())
        stab_margin = float(torch.min(stab_res.stability_margin).item())
        clipped_count = int(stab_res.clipped.sum().item())
    else:
        eta = eta_raw
        if key.ndim == 1:
            k_sq = torch.sum(key**2)
            norm_step = float((eta * k_sq).item())
        else:
            k_sq = torch.sum(key**2, dim=-1)
            norm_step = float(torch.max(eta * k_sq).item())
        stab_margin = float(2.0 - norm_step)

    metadata = AffineScanMetadata(
        min_stability_margin=stab_margin,
        max_normalized_step=norm_step,
        clipped_count=clipped_count,
    )

    # Unbatched transition: key is [K], target is [V]
    if key.ndim == 1:
        k_dim = key.shape[0]
        v_dim = target.shape[0]

        I_k = torch.eye(k_dim, dtype=key.dtype, device=key.device)
        kk_T = torch.outer(key, key)
        vk_T = torch.outer(target, key)

        A_t = I_k - eta * kk_T
        B_t = eta * vk_T

    # Batched transition: key is [B, K], target is [B, V]
    else:
        b, k_dim = key.shape
        b_v, v_dim = target.shape
        if b != b_v:
            raise ValueError(
                f"Batch dimension mismatch: key has {b}, target has {b_v}."
            )

        I_k = (
            torch.eye(k_dim, dtype=key.dtype, device=key.device)
            .unsqueeze(0)
            .expand(b, -1, -1)
        )
        kk_T = torch.bmm(
            key.unsqueeze(-1), key.unsqueeze(-2)
        )  # [B, K, 1] @ [B, 1, K] -> [B, K, K]
        vk_T = torch.bmm(
            target.unsqueeze(-1), key.unsqueeze(-2)
        )  # [B, V, 1] @ [B, 1, K] -> [B, V, K]

        eta_view_a = eta.view(b, 1, 1) if eta.ndim == 1 else eta
        eta_view_b = eta.view(b, 1, 1) if eta.ndim == 1 else eta

        A_t = I_k - eta_view_a * kk_T
        B_t = eta_view_b * vk_T

    return AffineScanOperator(A_t, B_t, metadata=metadata)
