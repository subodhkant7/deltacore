"""Phase 18: Non-Stationary Classification Stream Generators.

Constructs 3 genuinely distinct task families and 6 distribution shift dimensions:
    - Family A: Linear boundary rotation (A -> C -> A)
    - Family B: Boundary translation / intercept shift (A -> B_trans -> A)
    - Family C: Nonlinear decision boundary deformation (A -> C_nonlin -> A)
    - Shift dimensions:
        1. Covariate shift (sheared covariance, invariant labels)
        2. Boundary shift (rotated/translated hyperplanes)
        3. Class-prior shift (imbalanced class frequencies)
        4. Gradual shift (continuous rotation drift over time)
        5. Abrupt shift (discrete regime transitions)
        6. Strong mismatch (adversarial negative-transfer falsification case)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StationaryDataset18:
    """Stationary dataset container for offline development and head training."""

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
class NonStationaryStream18:
    """Non-stationary classification stream container."""

    inputs: torch.Tensor  # (T, D)
    targets: torch.Tensor  # (T,)
    regime_bounds: list[int]  # Step indices where regime changes occur
    regime_names: list[str]  # Human-readable names of regimes
    family_name: str  # "Family A", "Family B", "Family C"
    shift_type: (
        str  # "rotation", "translation", "nonlinear", "prior", "gradual", "mismatch"
    )
    num_classes: int
    dim: int
    metadata: dict[str, Any]


def _create_dense_prototypes(dim: int, num_classes: int, seed: int) -> torch.Tensor:
    """Generate normalized distributed class prototypes on hypersphere using dense Rademacher codes.

    Dense binary codes ensure that the discriminant signal is distributed across all
    coordinates, preventing any single feature from leaking the class label.
    """
    gen = torch.Generator().manual_seed(seed)
    code = torch.randint(0, 2, (num_classes, dim), generator=gen).float() * 2.0 - 1.0
    prototypes = (code / torch.linalg.norm(code, dim=-1, keepdim=True)) * math.sqrt(dim)
    return prototypes


def generate_stationary_dataset_18(
    dim: int = 32,
    num_classes: int = 6,
    n_train: int = 480,
    n_val: int = 180,
    n_test: int = 240,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> StationaryDataset18:
    """Generate balanced stationary dataset from canonical Regime A for baseline pre-training."""
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 1)

    def _sample_split(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_class = n_samples // num_classes
        x_list = []
        y_list = []
        for c in range(num_classes):
            proto = prototypes[c]
            noise = torch.randn(per_class, dim, generator=gen) * noise_scale
            samples = proto.unsqueeze(0) + noise
            norms = torch.linalg.norm(samples, dim=-1, keepdim=True).clamp_min(1e-6)
            samples = (samples / norms) * math.sqrt(dim)
            x_list.append(samples)
            y_list.append(torch.full((per_class,), c, dtype=torch.long))

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        perm = torch.randperm(len(x), generator=gen)
        return x[perm], y[perm]

    train_x, train_y = _sample_split(n_train)
    val_x, val_y = _sample_split(n_val)
    test_x, test_y = _sample_split(n_test)

    return StationaryDataset18(
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


# ==============================================================================
# Task Family A: Linear Boundary Rotation (Phase 17 Direct Transfer)
# ==============================================================================


def generate_family_a_rotation(
    dim: int = 32,
    num_classes: int = 6,
    steps_per_regime: int = 120,
    angle: float = math.pi / 2.5,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Family A: Linear decision boundary rotation across prototype subspace.

    Regime A (120) -> Regime C_rot (120) -> Regime A_return (120).
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 10)

    # Subspace rotation matrix
    q_proto, _ = torch.linalg.qr(prototypes.T)  # (D, K)
    r_k = torch.eye(num_classes)
    for i in range(0, num_classes - 1, 2):
        r_k[i, i] = math.cos(angle)
        r_k[i, i + 1] = -math.sin(angle)
        r_k[i + 1, i] = math.sin(angle)
        r_k[i + 1, i + 1] = math.cos(angle)
    rot_mat = q_proto @ r_k @ q_proto.T + (torch.eye(dim) - q_proto @ q_proto.T)

    regimes = ["Regime A (Base)", "Regime C (Rotated)", "Regime A (Return)"]
    x_list = []
    y_list = []
    bounds = []
    current_step = 0

    for reg_idx, rname in enumerate(regimes):
        for _ in range(steps_per_regime):
            c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
            proto = prototypes[c]
            if "Rotated" in rname:
                proto = rot_mat @ proto
            noise = torch.randn(dim, generator=gen) * noise_scale
            raw = proto + noise
            norm = torch.linalg.norm(raw).clamp_min(1e-6)
            norm_sample = (raw / norm) * math.sqrt(dim)
            x_list.append(norm_sample)
            y_list.append(c)

        current_step += steps_per_regime
        if reg_idx < len(regimes) - 1:
            bounds.append(current_step)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=bounds,
        regime_names=regimes,
        family_name="Family A — Boundary Rotation",
        shift_type="rotation",
        num_classes=num_classes,
        dim=dim,
        metadata={"angle": angle, "noise_scale": noise_scale},
    )


# ==============================================================================
# Task Family B: Boundary Translation / Intercept Shift
# ==============================================================================


def generate_family_b_translation(
    dim: int = 32,
    num_classes: int = 6,
    steps_per_regime: int = 120,
    translation_scale: float = 0.50,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Family B: Decision boundary translation and intercept offset shift.

    Instead of rotating feature axes, class centroids undergo distinct translation
    vectors in prototype subspace: P_B^(k) = P_A^(k) + tau * (P_A^(k+1) - P_A^(k)).
    Tests whether associative state can adapt to intercept geometry without rotation.
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 20)

    # Shift class centroids towards neighboring classes in prototype subspace
    trans = (
        prototypes[(torch.arange(num_classes) + 1) % num_classes] - prototypes
    ) * translation_scale
    shifted_prototypes = prototypes + trans

    regimes = ["Regime A (Base)", "Regime B (Translated)", "Regime A (Return)"]
    x_list = []
    y_list = []
    bounds = []
    current_step = 0

    for reg_idx, rname in enumerate(regimes):
        p_curr = shifted_prototypes if "Translated" in rname else prototypes
        for _ in range(steps_per_regime):
            c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
            proto = p_curr[c]
            noise = torch.randn(dim, generator=gen) * noise_scale
            raw = proto + noise
            norm = torch.linalg.norm(raw).clamp_min(1e-6)
            norm_sample = (raw / norm) * math.sqrt(dim)
            x_list.append(norm_sample)
            y_list.append(c)

        current_step += steps_per_regime
        if reg_idx < len(regimes) - 1:
            bounds.append(current_step)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=bounds,
        regime_names=regimes,
        family_name="Family B — Boundary Translation",
        shift_type="translation",
        num_classes=num_classes,
        dim=dim,
        metadata={"translation_scale": translation_scale, "noise_scale": noise_scale},
    )


# ==============================================================================
# Task Family C: Nonlinear Decision Boundary Deformation
# ==============================================================================


def generate_family_c_nonlinear(
    dim: int = 32,
    num_classes: int = 6,
    steps_per_regime: int = 120,
    angle_0: float = math.pi / 2.5,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Family C: Nonlinear decision boundary deformation.

    Applies a radial/quadratic nonlinear boundary warp in the prototype subspace
    that deforms the decision boundaries nonlinearly across coordinate pairs.
    Tests whether the minimal linear-head + associative-state mechanism can adapt
    when boundaries are non-planar.
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    q_proto, _ = torch.linalg.qr(prototypes.T)  # (D, K)
    gen = torch.Generator().manual_seed(seed + 30)

    regimes = ["Regime A (Base)", "Regime C (Nonlinear Warp)", "Regime A (Return)"]
    x_list = []
    y_list = []
    bounds = []
    current_step = 0

    for reg_idx, rname in enumerate(regimes):
        for _ in range(steps_per_regime):
            c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
            proto = prototypes[c]
            noise = torch.randn(dim, generator=gen) * noise_scale
            raw = proto + noise
            if "Nonlinear" in rname:
                u = q_proto.T @ raw  # (K,)
                u_warped = u.clone()
                for i in range(0, num_classes - 1, 2):
                    u_warped[i] = (
                        math.cos(angle_0) * u[i]
                        - math.sin(angle_0) * u[i + 1]
                        + 0.5 * (u[i] ** 2 - u[i + 1] ** 2) / math.sqrt(dim)
                    )
                    u_warped[i + 1] = (
                        math.sin(angle_0) * u[i] + math.cos(angle_0) * u[i + 1]
                    )
                raw = q_proto @ u_warped + (raw - q_proto @ u)
            norm = torch.linalg.norm(raw).clamp_min(1e-6)
            norm_sample = (raw / norm) * math.sqrt(dim)
            x_list.append(norm_sample)
            y_list.append(c)

        current_step += steps_per_regime
        if reg_idx < len(regimes) - 1:
            bounds.append(current_step)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=bounds,
        regime_names=regimes,
        family_name="Family C — Nonlinear Deformation",
        shift_type="nonlinear",
        num_classes=num_classes,
        dim=dim,
        metadata={"angle_0": angle_0, "noise_scale": noise_scale},
    )


# ==============================================================================
# Additional Shift Dimensions: Gradual, Prior Imbalance, and Strong Mismatch
# ==============================================================================


def generate_gradual_shift_stream(
    dim: int = 32,
    num_classes: int = 6,
    total_steps: int = 360,
    max_angle: float = math.pi / 2.0,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Gradual shift: Continuous smooth rotation over time without discrete transitions.

    theta(t) = (t / T) * max_angle.
    Tests adaptation tracking under continuous drift.
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 40)
    q_proto, _ = torch.linalg.qr(prototypes.T)

    x_list = []
    y_list = []

    for t in range(total_steps):
        current_angle = (t / float(total_steps)) * max_angle
        # Construct time-varying rotation
        r_k = torch.eye(num_classes)
        for i in range(0, num_classes - 1, 2):
            r_k[i, i] = math.cos(current_angle)
            r_k[i, i + 1] = -math.sin(current_angle)
            r_k[i + 1, i] = math.sin(current_angle)
            r_k[i + 1, i + 1] = math.cos(current_angle)
        rot_t = q_proto @ r_k @ q_proto.T + (torch.eye(dim) - q_proto @ q_proto.T)

        c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
        proto = rot_t @ prototypes[c]
        noise = torch.randn(dim, generator=gen) * noise_scale
        raw = proto + noise
        norm = torch.linalg.norm(raw).clamp_min(1e-6)
        norm_sample = (raw / norm) * math.sqrt(dim)
        x_list.append(norm_sample)
        y_list.append(c)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=[total_steps // 3, 2 * (total_steps // 3)],
        regime_names=["Early Drift", "Mid Drift", "Late Drift"],
        family_name="Continuous Drift",
        shift_type="gradual",
        num_classes=num_classes,
        dim=dim,
        metadata={"max_angle": max_angle},
    )


def generate_prior_shift_stream(
    dim: int = 32,
    num_classes: int = 6,
    steps_per_regime: int = 120,
    imbalance_ratio: float = 0.80,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Class-prior shift: Conditional feature geometry P(x|y) is fixed, but P(y) changes.

    Regime A: Balanced P(y=k) = 1/K.
    Regime B: Classes 0 and 1 occur with probability 0.40 each (imbalance_ratio total);
              remaining classes occur with 0.05 each.
    Regime A: Return to balanced.
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 50)

    probs_balanced = torch.full((num_classes,), 1.0 / num_classes)
    probs_imbalanced = torch.full(
        (num_classes,), (1.0 - imbalance_ratio) / (num_classes - 2)
    )
    probs_imbalanced[0] = imbalance_ratio / 2.0
    probs_imbalanced[1] = imbalance_ratio / 2.0

    regimes = ["Regime A (Balanced)", "Regime B (Imbalanced)", "Regime A (Return)"]
    x_list = []
    y_list = []
    bounds = []
    current_step = 0

    for reg_idx, rname in enumerate(regimes):
        p_dist = probs_imbalanced if "Imbalanced" in rname else probs_balanced
        for _ in range(steps_per_regime):
            c = int(torch.multinomial(p_dist, 1, generator=gen).item())
            proto = prototypes[c]
            noise = torch.randn(dim, generator=gen) * noise_scale
            raw = proto + noise
            norm = torch.linalg.norm(raw).clamp_min(1e-6)
            norm_sample = (raw / norm) * math.sqrt(dim)
            x_list.append(norm_sample)
            y_list.append(c)

        current_step += steps_per_regime
        if reg_idx < len(regimes) - 1:
            bounds.append(current_step)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=bounds,
        regime_names=regimes,
        family_name="Class Prior Imbalance",
        shift_type="prior",
        num_classes=num_classes,
        dim=dim,
        metadata={"imbalance_ratio": imbalance_ratio},
    )


def generate_strong_mismatch_stream(
    dim: int = 32,
    num_classes: int = 6,
    steps_a: int = 120,
    steps_mismatch: int = 40,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Required Adversarial / Falsification Case: Strong Regime Mismatch.

    Constructs A -> B_mismatch where Regime A requires adaptation (accumulating non-zero
    associative state M_A), and Regime B is abruptly incompatible (opposite subspace rotation)
    with a short window (40 steps), deliberately penalizing continuous persistent state
    relative to an oracle boundary reset.
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 60)
    q_proto, _ = torch.linalg.qr(prototypes.T)

    # Regime A rotation (builds up state M_A)
    angle_a = math.pi / 2.5
    r_a = torch.eye(num_classes)
    for i in range(0, num_classes - 1, 2):
        r_a[i, i] = math.cos(angle_a)
        r_a[i, i + 1] = -math.sin(angle_a)
        r_a[i + 1, i] = math.sin(angle_a)
        r_a[i + 1, i + 1] = math.cos(angle_a)
    rot_a = q_proto @ r_a @ q_proto.T + (torch.eye(dim) - q_proto @ q_proto.T)

    # Regime B: Incompatible opposite rotation
    angle_b = -math.pi / 2.5
    r_b = torch.eye(num_classes)
    for i in range(0, num_classes - 1, 2):
        r_b[i, i] = math.cos(angle_b)
        r_b[i, i + 1] = -math.sin(angle_b)
        r_b[i + 1, i] = math.sin(angle_b)
        r_b[i + 1, i + 1] = math.cos(angle_b)
    rot_b = q_proto @ r_b @ q_proto.T + (torch.eye(dim) - q_proto @ q_proto.T)

    regimes = ["Regime A (Rotated)", "Regime B (Incompatible Mismatch)"]
    x_list = []
    y_list = []
    bounds = [steps_a]

    # Regime A (accumulates M_A)
    for _ in range(steps_a):
        c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
        proto = rot_a @ prototypes[c]
        noise = torch.randn(dim, generator=gen) * noise_scale
        raw = proto + noise
        norm = torch.linalg.norm(raw).clamp_min(1e-6)
        x_list.append((raw / norm) * math.sqrt(dim))
        y_list.append(c)

    # Regime B (Strong Mismatch)
    for _ in range(steps_mismatch):
        c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
        proto = rot_b @ prototypes[c]
        noise = torch.randn(dim, generator=gen) * noise_scale
        raw = proto + noise
        norm = torch.linalg.norm(raw).clamp_min(1e-6)
        x_list.append((raw / norm) * math.sqrt(dim))
        y_list.append(c)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=bounds,
        regime_names=regimes,
        family_name="Adversarial Strong Mismatch",
        shift_type="mismatch",
        num_classes=num_classes,
        dim=dim,
        metadata={"steps_a": steps_a, "steps_mismatch": steps_mismatch},
    )


def generate_covariate_shift_stream(
    dim: int = 32,
    num_classes: int = 6,
    steps_per_regime: int = 120,
    condition_number: float = 6.0,
    noise_scale: float = 0.35,
    seed: int = 42,
) -> NonStationaryStream18:
    """Covariate shift: Input distribution P(x) changes via anisotropic covariance distortion.

    Class prototypes and Bayes decision boundaries P(y|x) remain strictly invariant.
    Regime A (Base) -> Regime B (Sheared Covariance) -> Regime A (Return).
    """
    prototypes = _create_dense_prototypes(dim, num_classes, seed)
    gen = torch.Generator().manual_seed(seed + 70)

    # Generate symmetric positive-definite covariance square root
    q_cov, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
    evals = torch.linspace(
        math.sqrt(condition_number), 1.0 / math.sqrt(condition_number), dim
    )
    cov_trans = q_cov @ torch.diag(evals) @ q_cov.T

    regimes = ["Regime A (Base)", "Regime B (Covariance)", "Regime A (Return)"]
    x_list = []
    y_list = []
    bounds = []
    current_step = 0

    for reg_idx, rname in enumerate(regimes):
        for _ in range(steps_per_regime):
            c = int(torch.randint(0, num_classes, (1,), generator=gen).item())
            proto = prototypes[c]
            noise = torch.randn(dim, generator=gen) * noise_scale
            if "Covariance" in rname:
                noise = cov_trans @ noise
            raw = proto + noise
            norm = torch.linalg.norm(raw).clamp_min(1e-6)
            x_list.append((raw / norm) * math.sqrt(dim))
            y_list.append(c)

        current_step += steps_per_regime
        if reg_idx < len(regimes) - 1:
            bounds.append(current_step)

    return NonStationaryStream18(
        inputs=torch.stack(x_list),
        targets=torch.tensor(y_list, dtype=torch.long),
        regime_bounds=bounds,
        regime_names=regimes,
        family_name="Covariate Anisotropic Shift",
        shift_type="covariate",
        num_classes=num_classes,
        dim=dim,
        metadata={"condition_number": condition_number, "noise_scale": noise_scale},
    )
