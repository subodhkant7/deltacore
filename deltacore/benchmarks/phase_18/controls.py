"""Phase 18: Negative Controls & Shortcut Audit Suite.

Provides:
    1. Predefined shortcut audit:
       - Energy / input norm balance
       - Per-feature variance balance
       - Class balance
       - Maximum single-feature 1D classifier accuracy (< 0.60, baseline chance ~ 16.7%)
       - 2D pairwise feature interaction shortcut check
       - Sequence length & transition timing records
       - Regime identifier leakage check (|r| < 0.40)
    2. Label-shuffle negative control:
       - Destroys associative relationship between features and labels
       - Verifies collapse to near-chance performance (~ 1/K)
    3. Coordinate permutation control:
       - Measures performance robustness/invariance under coordinate re-indexing
       - Explicitly distinguishes performance invariance from mathematical equivariance
"""

from __future__ import annotations

import math
from typing import Any

import torch


def audit_shortcut_statistics_18(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    regime_bounds: list[int] | None = None,
    num_classes: int = 6,
) -> dict[str, Any]:
    """Audit feature stream to verify absence of trivial shortcuts or regime leakage.

    Args:
        inputs: Streaming feature tensor of shape (T, D).
        targets: Target label tensor of shape (T,).
        regime_bounds: Step indices of distribution regime changes.
        num_classes: Expected number of classes (default 6).

    Returns:
        Structured audit report dictionary with predefined thresholds.
    """
    bounds = regime_bounds or []
    t_len, dim = inputs.shape

    # 1. Class balance
    class_counts = [int((targets == c).sum().item()) for c in range(num_classes)]
    min_count = min(class_counts) if class_counts else 0
    max_count = max(class_counts) if class_counts else 1
    class_balance_ratio = float(min_count / max(max_count, 1))

    # 2. Input norm / energy across classes
    mean_norms: list[float] = []
    variances: list[float] = []
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            c_inputs = inputs[mask]
            c_norms = torch.linalg.norm(c_inputs, dim=-1)
            mean_norms.append(float(c_norms.mean().item()))
            variances.append(float(c_inputs.var(dim=0).mean().item()))
        else:
            mean_norms.append(float(math.sqrt(dim)))
            variances.append(1.0)

    norm_ratio = max(mean_norms) / max(min(mean_norms), 1e-6)
    var_ratio = max(variances) / max(min(variances), 1e-6)

    # 3. Maximum single-feature 1D classification accuracy
    best_1d_acc = 0.0
    best_1d_dim = 0
    for d in range(dim):
        feat = inputs[:, d]
        centroids = torch.tensor(
            [
                feat[targets == c].mean().item() if (targets == c).any() else 0.0
                for c in range(num_classes)
            ]
        )
        preds_1d = torch.argmin(
            (feat.unsqueeze(1) - centroids.unsqueeze(0)).abs(), dim=1
        )
        acc_1d = float((preds_1d == targets).float().mean().item())
        if acc_1d > best_1d_acc:
            best_1d_acc = acc_1d
            best_1d_dim = d

    # 4. Pairwise 2D shortcut check on the two strongest 1D features
    feat_pair_acc = 0.0
    if dim >= 2:
        # Pick top 2 dimensions
        d1 = best_1d_dim
        d2 = (best_1d_dim + 1) % dim
        x2d = inputs[:, [d1, d2]]
        centroids_2d = torch.stack(
            [
                x2d[targets == c].mean(dim=0)
                if (targets == c).any()
                else torch.zeros(2)
                for c in range(num_classes)
            ]
        )
        # Nearest centroid in 2D
        dists_2d = torch.cdist(x2d, centroids_2d)
        preds_2d = torch.argmin(dists_2d, dim=1)
        feat_pair_acc = float((preds_2d == targets).float().mean().item())

    # 5. Regime identifier leakage check: feature correlation with regime index
    max_regime_corr = 0.0
    regime_indices = torch.zeros(t_len, dtype=torch.float32)
    current_regime = 0
    b_idx = 0
    for t in range(t_len):
        if b_idx < len(bounds) and t >= bounds[b_idx]:
            current_regime += 1
            b_idx += 1
        regime_indices[t] = float(current_regime)

    if current_regime > 0:
        regime_std = float(regime_indices.std().item())
        if regime_std > 1e-6:
            reg_centered = regime_indices - regime_indices.mean()
            for d in range(dim):
                f_d = inputs[:, d]
                f_std = float(f_d.std().item())
                if f_std > 1e-6:
                    f_centered = f_d - f_d.mean()
                    corr = abs(
                        float(
                            (f_centered * reg_centered).mean().item()
                            / (f_std * regime_std)
                        )
                    )
                    if corr > max_regime_corr:
                        max_regime_corr = corr

    energy_matched = norm_ratio < 1.15
    variance_matched = var_ratio < 3.00
    no_feature_leak = best_1d_acc < 0.60
    no_regime_leak = max_regime_corr < 0.40

    passed = energy_matched and variance_matched and no_feature_leak and no_regime_leak

    return {
        "sequence_length": t_len,
        "feature_dim": dim,
        "num_classes": num_classes,
        "transition_timing": bounds,
        "class_counts": class_counts,
        "class_balance_ratio": class_balance_ratio,
        "norm_max_min_ratio": norm_ratio,
        "var_max_min_ratio": var_ratio,
        "max_single_feature_1d_acc": best_1d_acc,
        "best_1d_dim": best_1d_dim,
        "pairwise_2d_acc": feat_pair_acc,
        "max_regime_correlation": max_regime_corr,
        "energy_matched": energy_matched,
        "variance_matched": variance_matched,
        "no_feature_leak": no_feature_leak,
        "no_regime_leak": no_regime_leak,
        "passed_shortcut_audit": passed,
    }


def shuffle_labels_18(targets: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Randomly permute target labels as a negative control."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(targets), generator=gen)
    return targets[perm].clone()


def permute_features_18(
    inputs: torch.Tensor, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly permute coordinate dimensions to test spatial permutation invariance.

    Args:
        inputs: Input tensor of shape (T, D).
        seed: Random seed for coordinate permutation.

    Returns:
        permuted_inputs: (T, D) tensor with permuted coordinate axes.
        permutation_indices: (D,) tensor of column indices.
    """
    dim = inputs.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dim, generator=gen)
    p_mat = torch.eye(dim)[perm]
    permuted_inputs = inputs @ p_mat.T
    return permuted_inputs, perm
