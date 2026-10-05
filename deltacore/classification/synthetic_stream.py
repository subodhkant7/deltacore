"""Phase 17: Synthetic Non-Stationary Classification Stream Generator.

Implements online multiclass classification streams (K=6, D=32 initially),
with structured distribution shifts (Regimes A -> B -> C -> A), shortcut audits,
matched class energy, label shuffling negative controls, and feature permutation controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StationaryDataset:
    """Stationary classification dataset partition for Task A and offline pre-training."""

    train_inputs: torch.Tensor  # (N_train, D)
    train_targets: torch.Tensor  # (N_train,)
    val_inputs: torch.Tensor  # (N_val, D)
    val_targets: torch.Tensor  # (N_val,)
    test_inputs: torch.Tensor  # (N_test, D)
    test_targets: torch.Tensor  # (N_test,)
    class_prototypes: torch.Tensor  # (K, D)
    num_classes: int
    dim: int


@dataclass(frozen=True)
class NonStationaryStream:
    """Sequential non-stationary streaming classification dataset."""

    inputs: torch.Tensor  # (T, D)
    targets: torch.Tensor  # (T,)
    regime_bounds: list[int]  # step boundaries e.g. [150, 300, 450]
    regime_names: list[
        str
    ]  # e.g. ["Regime A", "Regime B", "Regime C", "Regime A (Return)"]
    class_prototypes: torch.Tensor  # (K, D)
    num_classes: int
    dim: int
    severity: str


SEVERITY_CONFIGS: dict[str, dict[str, float]] = {
    "mild": {
        "cov_condition": 2.5,
        "rotation_angle": math.pi / 4.0,
        "noise_scale": 0.35,
    },
    "moderate": {
        "cov_condition": 5.0,
        "rotation_angle": math.pi / 2.5,
        "noise_scale": 0.40,
    },
    "severe": {
        "cov_condition": 10.0,
        "rotation_angle": math.pi / 2.0,
        "noise_scale": 0.48,
    },
}


def _create_orthogonal_prototypes(
    dim: int, num_classes: int, seed: int
) -> torch.Tensor:
    """Generate normalized distributed class prototypes on hypersphere using dense Rademacher codes.

    Dense binary codes ensure that the discriminant signal is distributed across all
    coordinates, preventing any single feature from leaking the class label.
    """
    gen = torch.Generator().manual_seed(seed)
    code = torch.randint(0, 2, (num_classes, dim), generator=gen).float() * 2.0 - 1.0
    prototypes = (code / torch.linalg.norm(code, dim=-1, keepdim=True)) * math.sqrt(dim)
    return prototypes


def generate_stationary_dataset(
    dim: int = 32,
    num_classes: int = 6,
    n_train: int = 480,
    n_val: int = 180,
    n_test: int = 240,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> StationaryDataset:
    """Generate balanced stationary dataset from Regime A for baseline pre-training."""
    prototypes = _create_orthogonal_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 1)

    def _sample_split(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_class = n_samples // num_classes
        x_list = []
        y_list = []
        for c in range(num_classes):
            proto = prototypes[c]
            noise = torch.randn(per_class, dim, generator=gen) * noise_scale
            samples = proto.unsqueeze(0) + noise
            # Normalize to constant energy sqrt(D) to avoid energy shortcuts
            norms = torch.linalg.norm(samples, dim=-1, keepdim=True).clamp_min(1e-6)
            samples = (samples / norms) * math.sqrt(dim)
            x_list.append(samples)
            y_list.append(torch.full((per_class,), c, dtype=torch.long))

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        # Random shuffle within split
        perm = torch.randperm(len(x), generator=gen)
        return x[perm], y[perm]

    train_x, train_y = _sample_split(n_train)
    val_x, val_y = _sample_split(n_val)
    test_x, test_y = _sample_split(n_test)

    return StationaryDataset(
        train_inputs=train_x,
        train_targets=train_y,
        val_inputs=val_x,
        val_targets=val_y,
        test_inputs=test_x,
        test_targets=test_y,
        class_prototypes=prototypes,
        num_classes=num_classes,
        dim=dim,
    )


def _generate_rotation_matrix(
    dim: int, prototypes: torch.Tensor, angle: float, seed: int = 42
) -> torch.Tensor:
    """Create orthogonal rotation matrix that rotates the class prototype subspace."""
    del seed
    q_proto, _ = torch.linalg.qr(prototypes.T)  # (D, K)

    num_classes = prototypes.size(0)
    r_k = torch.eye(num_classes)
    for i in range(0, num_classes - 1, 2):
        r_k[i, i] = math.cos(angle)
        r_k[i, i + 1] = -math.sin(angle)
        r_k[i + 1, i] = math.sin(angle)
        r_k[i + 1, i + 1] = math.cos(angle)
    # Rotate within the prototype subspace while preserving the orthogonal complement
    rot_mat = q_proto @ r_k @ q_proto.T + (torch.eye(dim) - q_proto @ q_proto.T)
    return rot_mat


def _generate_covariance_transform(
    dim: int, condition_number: float, seed: int
) -> torch.Tensor:
    """Create symmetric positive-definite covariance square root with specific condition number."""
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
    evals = torch.linspace(
        math.sqrt(condition_number), 1.0 / math.sqrt(condition_number), dim
    )
    diag_l = torch.diag(evals)
    transform = q @ diag_l @ q.T
    return transform


def generate_nonstationary_stream(
    dim: int = 32,
    num_classes: int = 6,
    steps_per_regime: int = 150,
    severity: str = "moderate",
    shift_mode: str = "all_regimes",  # "all_regimes", "covariate", "decision_boundary"
    seed: int = 42,
) -> NonStationaryStream:
    """Construct multi-regime non-stationary streaming classification sequence.

    Regimes:
        - Regime A: Directional hyperspherical prototypes with isotropic noise.
        - Regime B: Covariance structure change (anisotropic correlation deformation).
        - Regime C: Decision boundary rotation (feature-to-label hyperplane rotation).
        - Regime A (Return): Exact return to Regime A distribution.
    """
    if severity not in SEVERITY_CONFIGS:
        raise ValueError(
            f"Unknown severity: {severity}. Expected one of {list(SEVERITY_CONFIGS.keys())}"
        )

    cfg = SEVERITY_CONFIGS[severity]
    prototypes = _create_orthogonal_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 10)

    cov_trans = _generate_covariance_transform(
        dim, cfg["cov_condition"], seed=seed + 20
    )
    rot_mat = _generate_rotation_matrix(
        dim, prototypes, cfg["rotation_angle"], seed=seed + 30
    )

    regimes = ["Regime A", "Regime B", "Regime C", "Regime A (Return)"]
    if shift_mode == "covariate":
        # A -> B -> A
        regimes = ["Regime A", "Regime B (Covariance)", "Regime A (Return)"]
    elif shift_mode == "decision_boundary":
        # A -> C -> A
        regimes = ["Regime A", "Regime C (Boundary Rotation)", "Regime A (Return)"]

    x_list = []
    y_list = []
    bounds = []
    current_step = 0

    for reg_idx, rname in enumerate(regimes):
        for _ in range(steps_per_regime):
            # Balanced class sampling
            c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
            proto = prototypes[c]
            noise = torch.randn(dim, generator=gen) * cfg["noise_scale"]
            raw_sample = proto + noise

            # Apply regime-specific non-stationary transformation
            if "Regime B" in rname:
                sample = cov_trans @ raw_sample
            elif "Regime C" in rname:
                # Rotate sample relative to prototype boundaries
                sample = rot_mat @ raw_sample
            else:
                # Regime A (original or return)
                sample = raw_sample

            # Energy normalization: ensure matched energy across classes and regimes
            norm = torch.linalg.norm(sample).clamp_min(1e-6)
            norm_sample = (sample / norm) * math.sqrt(dim)

            x_list.append(norm_sample)
            y_list.append(c)

        current_step += steps_per_regime
        if reg_idx < len(regimes) - 1:
            bounds.append(current_step)

    inputs = torch.stack(x_list)
    targets = torch.tensor(y_list, dtype=torch.long)

    return NonStationaryStream(
        inputs=inputs,
        targets=targets,
        regime_bounds=bounds,
        regime_names=regimes,
        class_prototypes=prototypes,
        num_classes=num_classes,
        dim=dim,
        severity=severity,
    )


def audit_shortcut_statistics(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 6,
) -> dict[str, Any]:
    """Audit feature stream to verify absence of trivial class shortcuts."""
    classes = torch.unique(targets).tolist()
    mean_norms: list[float] = []
    variances: list[float] = []

    for c in classes:
        mask = targets == c
        c_inputs = inputs[mask]
        c_norms = torch.linalg.norm(c_inputs, dim=-1)
        mean_norms.append(float(c_norms.mean().item()))
        variances.append(float(c_inputs.var(dim=0).mean().item()))

    norm_ratio = max(mean_norms) / max(min(mean_norms), 1e-6)
    var_ratio = max(variances) / max(min(variances), 1e-6)

    # Check maximum 1D classification accuracy achievable by any single feature alone
    best_single_feat_acc = 0.0
    for d in range(inputs.shape[-1]):
        feat = inputs[:, d]
        centroids = torch.tensor(
            [feat[targets == c].mean().item() for c in range(num_classes)]
        )
        preds_1d = torch.argmin(
            (feat.unsqueeze(1) - centroids.unsqueeze(0)).abs(), dim=1
        )
        acc_1d = float((preds_1d == targets).float().mean().item())
        if acc_1d > best_single_feat_acc:
            best_single_feat_acc = acc_1d

    energy_matched = norm_ratio < 1.10
    variance_matched = var_ratio < 1.30
    no_feature_leak = best_single_feat_acc < 0.65  # chance is 1/K ~ 16.7%

    return {
        "num_samples": len(inputs),
        "num_classes": len(classes),
        "class_mean_norms": mean_norms,
        "class_mean_variances": variances,
        "norm_max_min_ratio": norm_ratio,
        "var_max_min_ratio": var_ratio,
        "max_single_feature_1d_acc": best_single_feat_acc,
        "energy_matched": energy_matched,
        "variance_matched": variance_matched,
        "no_feature_leak": no_feature_leak,
        "passed_shortcut_audit": energy_matched
        and variance_matched
        and no_feature_leak,
    }


def shuffle_labels(targets: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Randomly permute labels as a negative control."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(targets), generator=gen)
    return targets[perm].clone()


def permute_features(
    inputs: torch.Tensor, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly permute coordinate dimensions to test spatial permutation equivariance."""
    dim = inputs.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dim, generator=gen)
    p_mat = torch.eye(dim)[perm]
    permuted_inputs = inputs @ p_mat.T
    return permuted_inputs, perm
