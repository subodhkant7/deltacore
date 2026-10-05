"""Comprehensive unit and property tests for Phase 9.2 Spatial Capacity & Optimization Ablation.

Verifies:
- Trainable parameter registration across DeltaCore variants
- Frozen vs trainable parameter separation
- Coordinate injection, normalization, and shuffled controls
- Translation stress transformation invariants
- Pixel-shuffle control histogram preservation
- Train / Validation generation independence
- Exact runtime parameter counting (conv3x3, mlp_matched, gap_linear, deltacore)
- Optimizer and autograd gradient flow through trainable dynamics
- Confusion matrix and balanced accuracy computation
- Observatory JSON serialization and Plots U through AC generation
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from deltacore.observatory.plots import (
    plot_coordinate_effect,
    plot_directional_accuracy_points,
    plot_performance_vs_parameters,
    plot_pixel_shuffle_control,
    plot_state_and_gradient_norms,
    plot_task_a_rel_error_by_model,
    plot_task_a_train_val_curves,
    plot_task_b_acc_vs_params,
    plot_translation_stress,
)
from deltacore.spatial.coordinates import (
    compute_coordinate_statistics,
    coordinate_statistic_control,
    generate_2d_coordinates,
    inject_2d_coordinates,
    pixel_shuffle_control,
    translate_spatial_patterns,
)
from deltacore.spatial.fusion import LearnedChannelFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.spatial.routes import DOWN, LEFT, RIGHT, UP
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem


class TestCoordinatePrimitives:
    """Test suite for 2D spatial coordinate generation and transformations."""

    def test_generate_2d_coordinates_normalized(self) -> None:
        """Verify normalized 2D coordinate grid bounds in [-1.0, 1.0]."""
        coords = generate_2d_coordinates(height=8, width=12, normalized=True)
        assert coords.shape == (2, 8, 12)
        x_coords, y_coords = coords[0], coords[1]
        assert abs(x_coords[0, 0].item() - (-1.0)) < 1e-5
        assert abs(x_coords[0, -1].item() - 1.0) < 1e-5
        assert abs(y_coords[0, 0].item() - (-1.0)) < 1e-5
        assert abs(y_coords[-1, 0].item() - 1.0) < 1e-5

    def test_generate_2d_coordinates_unnormalized(self) -> None:
        """Verify unnormalized 2D coordinate grid bounds."""
        coords = generate_2d_coordinates(height=5, width=6, normalized=False)
        assert coords.shape == (2, 5, 6)
        assert coords[0, 0, 5].item() == 5.0
        assert coords[1, 4, 0].item() == 4.0

    def test_inject_2d_coordinates_shape(self) -> None:
        """Verify coordinate injection increases channel dimension by exactly 2."""
        x = torch.randn(3, 4, 10, 10)
        x_aug = inject_2d_coordinates(x, normalized=True, shuffle=False)
        assert x_aug.shape == (3, 6, 10, 10)
        # Verify first 4 channels are unchanged
        assert torch.equal(x_aug[:, :4], x)

    def test_inject_2d_coordinates_invalid_dim_rejected(self) -> None:
        """Non-4D input tensors must raise ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            inject_2d_coordinates(torch.randn(3, 10, 10))

    def test_shuffled_coordinate_control_preserves_histogram(self) -> None:
        """Shuffled coordinates must preserve coordinate values while scrambling spatial positions."""
        g = torch.Generator().manual_seed(42)
        x = torch.zeros(2, 1, 6, 6)
        x_reg = inject_2d_coordinates(x, normalized=True, shuffle=False)
        x_shuff = inject_2d_coordinates(x, normalized=True, shuffle=True, generator=g)

        coords_reg = x_reg[:, 1:]
        coords_shuff = x_shuff[:, 1:]

        # Must not be identically equal due to scrambling
        assert not torch.equal(coords_reg, coords_shuff)
        # Sorted values along spatial dimension must match exactly (histogram preservation)
        flat_reg, _ = torch.sort(coords_reg.view(2, 2, -1), dim=-1)
        flat_shuff, _ = torch.sort(coords_shuff.view(2, 2, -1), dim=-1)
        assert torch.allclose(flat_reg, flat_shuff, atol=1e-6)

    def test_translate_spatial_patterns_invariants(self) -> None:
        """Translation stress must preserve tensor shape and Frobenius norm."""
        g = torch.Generator().manual_seed(99)
        x = torch.randn(4, 3, 8, 8)
        x_trans = translate_spatial_patterns(x, max_shift=2, generator=g)
        assert x_trans.shape == x.shape
        # Frobenius norm is strictly invariant under cyclic translation
        for i in range(4):
            assert abs(torch.norm(x[i]).item() - torch.norm(x_trans[i]).item()) < 1e-4

    def test_translate_invalid_dim_rejected(self) -> None:
        """Non-4D input to translate_spatial_patterns must raise ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            translate_spatial_patterns(torch.randn(8, 8))

    def test_pixel_shuffle_control_invariants(self) -> None:
        """Pixel shuffle must preserve pixel value distribution while scrambling 2D layout."""
        g = torch.Generator().manual_seed(123)
        x = torch.randn(3, 4, 7, 7)
        x_shuff = pixel_shuffle_control(x, generator=g)
        assert x_shuff.shape == x.shape
        assert not torch.equal(x, x_shuff)
        # Sorted pixel values per channel must match exactly
        flat_orig, _ = torch.sort(x.view(3, 4, -1), dim=-1)
        flat_shuff, _ = torch.sort(x_shuff.view(3, 4, -1), dim=-1)
        assert torch.allclose(flat_orig, flat_shuff, atol=1e-6)

    def test_pixel_shuffle_invalid_dim_rejected(self) -> None:
        """Non-4D input to pixel_shuffle_control must raise ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            pixel_shuffle_control(torch.randn(7, 7))

    def test_compute_coordinate_statistics_shapes_and_values(self) -> None:
        """Verify computed energy and center of mass statistics."""
        x = torch.zeros(2, 3, 5, 5)
        # Put energy at (y=2, x=2) - center pixel
        x[:, :, 2, 2] = 1.0
        stats = compute_coordinate_statistics(x)
        assert stats["total_energy"].shape == (2,)
        assert stats["row_energy"].shape == (2, 5)
        assert stats["col_energy"].shape == (2, 5)
        assert stats["center_of_mass_y"].shape == (2,)
        assert stats["center_of_mass_x"].shape == (2,)
        # For center pixel, center of mass should be exactly 2.0
        assert torch.allclose(stats["center_of_mass_y"], torch.tensor([2.0, 2.0]))
        assert torch.allclose(stats["center_of_mass_x"], torch.tensor([2.0, 2.0]))

    def test_coordinate_statistic_control_unit_norm(self) -> None:
        """coordinate_statistic_control enforces strict target Frobenius norm."""
        x = torch.randn(4, 2, 6, 6) * 5.0
        controlled = coordinate_statistic_control(x, target_total_energy=1.0)
        assert controlled.shape == x.shape
        for i in range(4):
            norm_val = torch.norm(controlled[i]).item()
            assert abs(norm_val - 1.0) < 1e-4

    def test_coordinate_statistic_control_centers_mass(self) -> None:
        """coordinate_statistic_control shifts off-center mass towards grid center."""
        x = torch.zeros(1, 1, 7, 7)
        # Place localized spot at top-left (0, 0)
        x[0, 0, 0, 0] = 5.0
        controlled = coordinate_statistic_control(x, match_center_of_mass=True)
        stats = compute_coordinate_statistics(controlled)
        # Target center for 7x7 grid is (3.0, 3.0)
        assert abs(stats["center_of_mass_y"][0].item() - 3.0) < 1e-4
        assert abs(stats["center_of_mass_x"][0].item() - 3.0) < 1e-4

    def test_coordinate_statistics_invalid_dim_rejected(self) -> None:
        """Non-4D tensors raise ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            compute_coordinate_statistics(torch.randn(5, 5))
        with pytest.raises(ValueError, match="must be 4D"):
            coordinate_statistic_control(torch.randn(5, 5))


class TestParameterAccounting:
    """Test suite for runtime parameter verification across Phase 9.2 models."""

    def test_five_memory_system_static_has_zero_parameters(self) -> None:
        """When trainable_dynamics=False, FiveMemorySystem has exactly 0 parameters."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        sys = FiveMemorySystem(cfg, trainable_dynamics=False)
        assert sum(p.numel() for p in sys.parameters()) == 0

    def test_five_memory_system_trainable_dynamics_parameters(self) -> None:
        """When trainable_dynamics=True, FiveMemorySystem registers 28 transition parameters."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        sys = FiveMemorySystem(cfg, trainable_dynamics=True)
        # eta_key(4), lambda_key(4), eta_val(4), lambda_val(4), rho_eta(2), lambda_eta(2),
        # rho_ret(2), lambda_ret(2), tau_eta(1), tau_ret(1), eta_bias(1), ret_bias(1)
        expected = 4 + 4 + 4 + 4 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1
        assert sum(p.numel() for p in sys.parameters() if p.requires_grad) == expected

    def test_spatial_operator_reference_parameter_count(self) -> None:
        """delta_reference: 64 params per route + 16 fusion params for 4 routes."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        # 1-Dir
        op1 = SpatialAdaptiveOperator(
            cfg, routes=(RIGHT,), learnable_initial_states=True
        )
        assert sum(p.numel() for p in op1.parameters() if p.requires_grad) == 64

        # 4-Dir with LearnedChannelFusion
        fusion = LearnedChannelFusion(channels=4, num_routes=4)
        op4 = SpatialAdaptiveOperator(
            cfg,
            fusion=fusion,
            routes=(RIGHT, LEFT, DOWN, UP),
            learnable_initial_states=True,
        )
        assert (
            sum(p.numel() for p in op4.parameters() if p.requires_grad)
            == 4 * 64 + 16
            == 272
        )

    def test_spatial_operator_trainable_dynamics_parameter_count(self) -> None:
        """delta_trainable_dynamics: (64 + 28) = 92 params per route."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        op1 = SpatialAdaptiveOperator(
            cfg, routes=(RIGHT,), learnable_initial_states=True, trainable_dynamics=True
        )
        assert sum(p.numel() for p in op1.parameters() if p.requires_grad) == 92

        fusion = LearnedChannelFusion(channels=4, num_routes=4)
        op4 = SpatialAdaptiveOperator(
            cfg,
            fusion=fusion,
            routes=(RIGHT, LEFT, DOWN, UP),
            learnable_initial_states=True,
            trainable_dynamics=True,
        )
        assert (
            sum(p.numel() for p in op4.parameters() if p.requires_grad)
            == 4 * 92 + 16
            == 384
        )

    def test_spatial_operator_coordinates_parameter_count(self) -> None:
        """delta_reference_xy: in_dim=6 -> 88 params per route."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=6, feat_dim=6, lr_dim=2, ret_dim=2
        )
        fusion = LearnedChannelFusion(channels=4, num_routes=4)
        op = SpatialAdaptiveOperator(
            cfg,
            fusion=fusion,
            routes=(RIGHT, LEFT, DOWN, UP),
            learnable_initial_states=True,
        )
        # 88 per route * 4 + 16 = 368
        assert sum(p.numel() for p in op.parameters() if p.requires_grad) == 368

    def test_conv3x3_and_gap_linear_parameter_counts(self) -> None:
        """Verify baseline model parameter counts match Phase 9.2 specifications."""
        conv_a = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
        assert sum(p.numel() for p in conv_a.parameters()) == 148

        gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 6)
        )
        assert sum(p.numel() for p in gap_linear.parameters()) == 30

        conv_b = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 6),
        )
        assert sum(p.numel() for p in conv_b.parameters()) == 148 + 30 == 178

    def test_parameter_matched_mlp_count(self) -> None:
        """Pointwise MLP with hidden_dim=30 has exactly 274 parameters."""
        mlp = nn.Sequential(
            nn.Conv2d(4, 30, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(30, 4, kernel_size=1, bias=True),
        )
        # (4*30 + 30) + (30*4 + 4) = 150 + 124 = 274
        assert sum(p.numel() for p in mlp.parameters()) == 274


class TestOptimizationAndDiagnostics:
    """Test suite for gradient flow, trainability, and metrics."""

    def test_trainable_dynamics_gradient_flow(self) -> None:
        """All trainable dynamics parameters receive gradients through BPTT."""
        cfg = FiveMemoryConfig(
            v_dim=3, k_dim=3, in_dim=3, feat_dim=3, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(
            cfg, routes=(RIGHT,), learnable_initial_states=True, trainable_dynamics=True
        )
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        res = op(x).fused_output
        loss = res.sum()
        loss.backward()

        trainable_params = [p for p in op.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
        for p in trainable_params:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
        assert x.grad is not None

    def test_confusion_matrix_and_balanced_accuracy_calculation(self) -> None:
        """Verify confusion matrix and balanced accuracy calculations."""
        y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        y_pred = torch.tensor([0, 1, 1, 1, 2, 0])

        confusion = torch.zeros(3, 3, dtype=torch.int32)
        for t, p in zip(y_true, y_pred, strict=False):
            confusion[t, p] += 1

        assert confusion[0, 0].item() == 1
        assert confusion[0, 1].item() == 1
        assert confusion[1, 1].item() == 2
        assert confusion[2, 2].item() == 1

        per_class_acc = []
        for c in range(3):
            tot = int((y_true == c).sum().item())
            cor = int(confusion[c, c].item())
            per_class_acc.append((cor / tot) * 100.0)

        assert per_class_acc == [50.0, 100.0, 50.0]
        balanced_acc = float(np.mean(per_class_acc))
        assert abs(balanced_acc - 66.6667) < 1e-3


class TestObservatoryPlotsUThroughAC:
    """Smoke tests for Plots U through AC in deltacore.observatory.plots."""

    def test_plots_u_through_ac_smoke(self, tmp_path: Path) -> None:
        """Verify that all Phase 9.2 plots generate valid image files headlessly."""
        # Plot U
        p_u = plot_task_a_rel_error_by_model(
            ["conv", "delta"],
            [0.35, 0.87],
            [0.01, 0.02],
            output_path=tmp_path / "plot_u.png",
        )
        assert p_u.exists() and p_u.stat().st_size > 0

        # Plot V
        curves = {"conv": ([0.8, 0.4], [0.8, 0.35]), "delta": ([1.0, 0.9], [1.0, 0.88])}
        p_v = plot_task_a_train_val_curves(curves, output_path=tmp_path / "plot_v.png")
        assert p_v.exists() and p_v.stat().st_size > 0

        # Plot W
        p_w = plot_coordinate_effect(
            ["No Coords", "True", "Shuff"],
            [1.0, 0.87, 0.95],
            output_path=tmp_path / "plot_w.png",
        )
        assert p_w.exists() and p_w.stat().st_size > 0

        # Plot X
        p_x = plot_task_b_acc_vs_params(
            ["gap", "conv"],
            [30.0, 40.0],
            [30, 178],
            output_path=tmp_path / "plot_x.png",
        )
        assert p_x.exists() and p_x.stat().st_size > 0

        # Plot Y
        p_y = plot_translation_stress(
            ["gap", "conv"],
            [30.0, 40.0],
            [30.0, 41.7],
            output_path=tmp_path / "plot_y.png",
        )
        assert p_y.exists() and p_y.stat().st_size > 0

        # Plot Z
        p_z = plot_pixel_shuffle_control(
            ["gap", "conv"],
            [30.0, 40.0],
            [30.0, 23.0],
            output_path=tmp_path / "plot_z.png",
        )
        assert p_z.exists() and p_z.stat().st_size > 0

        # Plot AA
        seed_points = {"RIGHT": [33.3, 35.0, 38.3], "LEFT": [50.0, 33.3, 33.3]}
        p_aa = plot_directional_accuracy_points(
            ["RIGHT", "LEFT"], seed_points, output_path=tmp_path / "plot_aa.png"
        )
        assert p_aa.exists() and p_aa.stat().st_size > 0

        # Plot AB
        state_norms = {"content": [0.1, 0.2], "key": [0.05, 0.08]}
        p_ab = plot_state_and_gradient_norms(
            [0.5, 0.3], state_norms, output_path=tmp_path / "plot_ab.png"
        )
        assert p_ab.exists() and p_ab.stat().st_size > 0

        # Plot AC
        p_ac = plot_performance_vs_parameters(
            [30, 178, 302],
            [30.0, 40.0, 35.0],
            ["gap", "conv", "delta"],
            output_path=tmp_path / "plot_ac.png",
        )
        assert p_ac.exists() and p_ac.stat().st_size > 0


class TestDatasetAndStressInvariants:
    """Test suite for dataset generation, train/val independence, and stress transformations."""

    def test_task_a_exact_shift_formula(self) -> None:
        """Verify Task A target is exactly 0.5 * X_right + 0.5 * X_down."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_a_dataset,
        )

        X, Y = generate_task_a_dataset(
            num_samples=4, height=6, width=6, channels=2, seed=12
        )
        X_r = torch.roll(X, shifts=-1, dims=3)
        X_d = torch.roll(X, shifts=-1, dims=2)
        expected = 0.5 * X_r + 0.5 * X_d
        assert torch.allclose(Y, expected, atol=1e-6)

    def test_task_a_seed_determinism(self) -> None:
        """Fixed seed must produce bitwise identical Task A datasets."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_a_dataset,
        )

        x1, y1 = generate_task_a_dataset(
            num_samples=8, height=4, width=4, channels=2, seed=77
        )
        x2, y2 = generate_task_a_dataset(
            num_samples=8, height=4, width=4, channels=2, seed=77
        )
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    def test_task_b_strict_unit_norm_invariant(self) -> None:
        """Every single sample in Task B must satisfy ||X_i||_F = 1.0000 +- 1e-4."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_b_dataset,
        )

        X, y = generate_task_b_dataset(
            samples_per_class=10, height=8, width=8, channels=3, seed=88
        )
        assert X.shape[0] == 60
        for i in range(60):
            norm_val = torch.norm(X[i]).item()
            assert abs(norm_val - 1.0) < 1e-4, (
                f"Sample {i} norm is {norm_val}, expected 1.0"
            )

    def test_task_b_class_balance_and_labels(self) -> None:
        """Task B must have balanced labels across all 6 classes."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_b_dataset,
        )

        _, y = generate_task_b_dataset(
            samples_per_class=12, height=6, width=6, channels=2, seed=42
        )
        assert y.shape[0] == 72
        for c in range(6):
            assert (y == c).sum().item() == 12

    def test_task_b_train_val_seed_independence(self) -> None:
        """Distinct seeds must produce statistically distinct datasets."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_b_dataset,
        )

        x_train, _ = generate_task_b_dataset(samples_per_class=5, seed=303)
        x_val, _ = generate_task_b_dataset(samples_per_class=5, seed=404)
        assert not torch.equal(x_train, x_val)

    def test_translation_stress_zero_shift_identity(self) -> None:
        """max_shift=0 must return bitwise identical tensor."""
        x = torch.randn(3, 2, 5, 5)
        x_trans = translate_spatial_patterns(x, max_shift=0)
        assert torch.equal(x, x_trans)

    def test_translation_stress_seed_repeatability(self) -> None:
        """Generator with fixed seed produces deterministic translation shifts."""
        x = torch.randn(4, 2, 6, 6)
        g1 = torch.Generator().manual_seed(101)
        g2 = torch.Generator().manual_seed(101)
        t1 = translate_spatial_patterns(x, max_shift=2, generator=g1)
        t2 = translate_spatial_patterns(x, max_shift=2, generator=g2)
        assert torch.equal(t1, t2)

    def test_pixel_shuffle_seed_repeatability(self) -> None:
        """Generator with fixed seed produces deterministic pixel shuffle."""
        x = torch.randn(4, 2, 6, 6)
        g1 = torch.Generator().manual_seed(202)
        g2 = torch.Generator().manual_seed(202)
        s1 = pixel_shuffle_control(x, generator=g1)
        s2 = pixel_shuffle_control(x, generator=g2)
        assert torch.equal(s1, s2)

    def test_coordinate_injection_rectangular_grid(self) -> None:
        """Coordinate injection works correctly on non-square grids (H != W)."""
        x = torch.randn(2, 3, 7, 14)
        x_aug = inject_2d_coordinates(x, normalized=True)
        assert x_aug.shape == (2, 5, 7, 14)
        assert x_aug[0, 3, 0, 0].item() == -1.0
        assert x_aug[0, 3, 0, -1].item() == 1.0
        assert x_aug[0, 4, 0, 0].item() == -1.0
        assert x_aug[0, 4, -1, 0].item() == 1.0

    def test_coordinate_injection_dtype_preservation(self) -> None:
        """Coordinate injection preserves float64 dtype."""
        x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        x_aug = inject_2d_coordinates(x, normalized=True)
        assert x_aug.dtype == torch.float64


class TestModelArchitectureAndStateInvariants:
    """Test suite for state dictionaries, serialization, and architectural properties."""

    def test_spatial_operator_state_dict_roundtrip(self) -> None:
        """SpatialAdaptiveOperator state_dict can be saved and restored identically."""
        cfg = FiveMemoryConfig(
            v_dim=3, k_dim=3, in_dim=3, feat_dim=3, lr_dim=2, ret_dim=2
        )
        op1 = SpatialAdaptiveOperator(
            cfg,
            routes=(RIGHT, LEFT),
            learnable_initial_states=True,
            trainable_dynamics=True,
        )
        sd = op1.state_dict()

        op2 = SpatialAdaptiveOperator(
            cfg,
            routes=(RIGHT, LEFT),
            learnable_initial_states=True,
            trainable_dynamics=True,
        )
        op2.load_state_dict(sd)

        x = torch.randn(2, 3, 5, 5)
        out1 = op1(x).fused_output
        out2 = op2(x).fused_output
        assert torch.equal(out1, out2)

    def test_trainable_dynamics_2dir_parameter_count(self) -> None:
        """2-direction operator with trainable dynamics has 2 * 92 = 184 parameters."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(
            cfg,
            routes=(RIGHT, LEFT),
            learnable_initial_states=True,
            trainable_dynamics=True,
        )
        assert sum(p.numel() for p in op.parameters() if p.requires_grad) == 184

    def test_trainable_dynamics_chunk_sweep_forward(self) -> None:
        """trainable_dynamics operates across chunk sizes C in {1, 2, 4, 16}."""
        cfg = FiveMemoryConfig(
            v_dim=3, k_dim=3, in_dim=3, feat_dim=3, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(
            cfg, routes=(RIGHT,), learnable_initial_states=True, trainable_dynamics=True
        )
        x = torch.randn(2, 3, 4, 4)
        for c in [1, 2, 4, 16]:
            res = op(x, chunk_size=c)
            assert res.fused_output.shape == (2, 3, 4, 4)

    def test_mlp_matched_forward_backward_pass(self) -> None:
        """ParameterMatchedMLP handles forward and backward pass cleanly."""
        from examples.phase_9_2_capacity_ablation_benchmark import ParameterMatchedMLP

        mlp = ParameterMatchedMLP(in_channels=4, out_channels=4, hidden_dim=30)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        out = mlp(x)
        assert out.shape == (2, 4, 8, 8)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_spatial_classifier_wrapper_with_coords(self) -> None:
        """SpatialClassifier with use_coords=True accepts 4-channel input and outputs 6 logits."""
        from examples.phase_9_2_capacity_ablation_benchmark import SpatialClassifier

        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=6, feat_dim=6, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(
            cfg, routes=(RIGHT,), learnable_initial_states=True
        )
        clf = SpatialClassifier(op, channels=4, num_classes=6, use_coords=True)
        x = torch.randn(3, 4, 6, 6)
        logits = clf(x)
        assert logits.shape == (3, 6)

    def test_trainable_dynamics_with_coordinates_parameter_count(self) -> None:
        """delta_trainable_dynamics_xy has 480 parameters."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=6, feat_dim=6, lr_dim=2, ret_dim=2
        )
        fusion = LearnedChannelFusion(channels=4, num_routes=4)
        op = SpatialAdaptiveOperator(
            cfg,
            fusion=fusion,
            routes=(RIGHT, LEFT, DOWN, UP),
            learnable_initial_states=True,
            trainable_dynamics=True,
        )
        assert sum(p.numel() for p in op.parameters() if p.requires_grad) == 480

    def test_single_direction_trainable_dynamics_with_coordinates_parameter_count(
        self,
    ) -> None:
        """1-Dir delta_trainable_dynamics_xy has 88 (init) + 28 (dyn) = 116 parameters."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=6, feat_dim=6, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(
            cfg, routes=(RIGHT,), learnable_initial_states=True, trainable_dynamics=True
        )
        assert sum(p.numel() for p in op.parameters() if p.requires_grad) == 116

    def test_equal_fusion_mean_vs_sum(self) -> None:
        """EqualFusion supports mean and sum convex combinations."""
        from deltacore.spatial.fusion import EqualFusion

        f_mean = EqualFusion(mode="mean")
        f_sum = EqualFusion(mode="sum")
        t = torch.ones(2, 3, 4, 4)
        out_mean = f_mean([t, t, t, t])
        out_sum = f_sum([t, t, t, t])
        assert torch.allclose(out_mean, t)
        assert torch.allclose(out_sum, 4.0 * t)

    def test_equal_fusion_invalid_mode_rejected(self) -> None:
        """Invalid mode in EqualFusion raises ValueError."""
        from deltacore.spatial.fusion import EqualFusion

        with pytest.raises(ValueError, match="mode must be"):
            EqualFusion(mode="invalid")

    def test_equal_fusion_empty_routes_rejected(self) -> None:
        """EqualFusion requires non-empty directional output list."""
        from deltacore.spatial.fusion import EqualFusion

        f = EqualFusion()
        with pytest.raises(ValueError, match="directional_outputs must not be empty"):
            f([])

    def test_learned_channel_fusion_invalid_num_routes(self) -> None:
        """LearnedChannelFusion requires num_routes >= 1."""
        with pytest.raises(ValueError, match="num_routes must be >= 1"):
            LearnedChannelFusion(channels=4, num_routes=0)

    def test_translation_stress_single_sample_batch(self) -> None:
        """Translation stress functions on batch size 1."""
        x = torch.randn(1, 4, 8, 8)
        xt = translate_spatial_patterns(x, max_shift=2)
        assert xt.shape == (1, 4, 8, 8)
        assert abs(torch.norm(xt).item() - torch.norm(x).item()) < 1e-4

    def test_translation_stress_large_batch(self) -> None:
        """Translation stress functions on batch size 16."""
        x = torch.randn(16, 2, 6, 6)
        xt = translate_spatial_patterns(x, max_shift=3)
        assert xt.shape == (16, 2, 6, 6)

    def test_pixel_shuffle_single_sample_batch(self) -> None:
        """Pixel shuffle functions on batch size 1."""
        x = torch.randn(1, 3, 5, 5)
        xs = pixel_shuffle_control(x)
        assert xs.shape == (1, 3, 5, 5)
        assert abs(torch.norm(xs).item() - torch.norm(x).item()) < 1e-4

    def test_pixel_shuffle_preserves_tensor_sum(self) -> None:
        """Pixel shuffle preserves sum of elements across grid."""
        x = torch.randn(4, 2, 6, 6)
        xs = pixel_shuffle_control(x)
        for i in range(4):
            assert abs(x[i].sum().item() - xs[i].sum().item()) < 1e-4

    def test_task_a_dataset_sample_count(self) -> None:
        """Task A dataset returns exact requested sample count."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_a_dataset,
        )

        X, Y = generate_task_a_dataset(num_samples=17, height=5, width=5, channels=3)
        assert X.shape == (17, 3, 5, 5)
        assert Y.shape == (17, 3, 5, 5)

    def test_task_b_dataset_sample_count(self) -> None:
        """Task B dataset returns 6 * samples_per_class samples."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_b_dataset,
        )

        X, y = generate_task_b_dataset(
            samples_per_class=7, height=6, width=6, channels=2
        )
        assert X.shape == (42, 2, 6, 6)
        assert y.shape == (42,)

    def test_task_a_dataset_non_trivial_variance(self) -> None:
        """Task A targets have non-zero variance across space."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            generate_task_a_dataset,
        )

        _, Y = generate_task_a_dataset(num_samples=5, height=8, width=8, channels=2)
        assert Y.var().item() > 0.1

    def test_spatial_classifier_mlp_backbone(self) -> None:
        """SpatialClassifier wraps ParameterMatchedMLP correctly."""
        from examples.phase_9_2_capacity_ablation_benchmark import (
            ParameterMatchedMLP,
            SpatialClassifier,
        )

        mlp = ParameterMatchedMLP(in_channels=4, out_channels=4, hidden_dim=20)
        clf = SpatialClassifier(mlp, channels=4, num_classes=6)
        x = torch.randn(2, 4, 8, 8)
        logits = clf(x)
        assert logits.shape == (2, 6)

    def test_spatial_classifier_conv_backbone(self) -> None:
        """SpatialClassifier wraps Conv2d backbone correctly."""
        from examples.phase_9_2_capacity_ablation_benchmark import SpatialClassifier

        conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        clf = SpatialClassifier(conv, channels=4, num_classes=6)
        x = torch.randn(2, 4, 8, 8)
        logits = clf(x)
        assert logits.shape == (2, 6)

    def test_saturated_sigmoid_diagnostic_check(self) -> None:
        """Verify calculation of saturated sigmoid fraction."""
        t = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        sig = torch.sigmoid(t)
        sat = ((sig > 0.99) | (sig < 0.01)).float().mean().item()
        assert abs(sat - 0.4) < 1e-4

    def test_near_zero_gradient_fraction_check(self) -> None:
        """Verify near zero gradient fraction calculation."""
        g = torch.tensor([0.0, 1e-8, 0.05, 0.1])
        near_zero = (torch.abs(g) < 1e-6).float().mean().item()
        assert abs(near_zero - 0.5) < 1e-4

    def test_spatial_operator_directional_outputs_extraction(self) -> None:
        """SpatialAdaptiveOperator exposes individual directional outputs."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(cfg, routes=(RIGHT, DOWN))
        x = torch.randn(2, 4, 6, 6)
        res = op(x)
        assert "RIGHT" in res.directional_outputs
        assert "DOWN" in res.directional_outputs
        assert res.directional_outputs["RIGHT"].shape == (2, 4, 6, 6)
        assert res.directional_outputs["DOWN"].shape == (2, 4, 6, 6)

    def test_spatial_operator_final_states_extraction(self) -> None:
        """SpatialAdaptiveOperator exposes final FiveMemoryState for each route."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(cfg, routes=(RIGHT, UP))
        x = torch.randn(2, 4, 5, 5)
        res = op(x)
        assert "RIGHT" in res.final_states
        assert "UP" in res.final_states
        assert res.final_states["RIGHT"].content.shape == (2, 4, 4)

    def test_spatial_operator_unbatched_rejection(self) -> None:
        """SpatialAdaptiveOperator rejects non-4D input."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        op = SpatialAdaptiveOperator(cfg)
        with pytest.raises(ValueError, match="expects 4D input"):
            op(torch.randn(4, 8, 8))

    def test_spatial_operator_invalid_mode_rejection(self) -> None:
        """SpatialAdaptiveOperator rejects invalid chunking mode."""
        cfg = FiveMemoryConfig(
            v_dim=4, k_dim=4, in_dim=4, feat_dim=4, lr_dim=2, ret_dim=2
        )
        with pytest.raises(ValueError, match="mode must be"):
            SpatialAdaptiveOperator(cfg, mode="unsupported_mode")

    def test_true_shuffled_coordinate_control_c1(self) -> None:
        """C1: delta_reference_xy and delta_reference_xy_shuffled share identical parameters (368p)."""
        from deltacore.spatial.coordinates import inject_2d_coordinates
        from examples.phase_9_2_capacity_ablation_benchmark import (
            build_deltacore_operator,
        )

        m_clean = build_deltacore_operator(
            in_channels=6,
            out_channels=4,
            trainable_initial_states=True,
            trainable_dynamics=False,
            use_learned_fusion=True,
        )
        m_shuff = build_deltacore_operator(
            in_channels=6,
            out_channels=4,
            trainable_initial_states=True,
            trainable_dynamics=False,
            use_learned_fusion=True,
        )
        p_clean = sum(p.numel() for p in m_clean.parameters() if p.requires_grad)
        p_shuff = sum(p.numel() for p in m_shuff.parameters() if p.requires_grad)
        assert p_clean == 368
        assert p_shuff == 368
        assert p_clean == p_shuff

        x = torch.randn(2, 4, 10, 10)
        x_clean = inject_2d_coordinates(x, normalized=True, shuffle=False)
        x_shuff = inject_2d_coordinates(x, normalized=True, shuffle=True)
        assert x_clean.shape == (2, 6, 10, 10)
        assert x_shuff.shape == (2, 6, 10, 10)

        # Coordinate channels marginal histogram preservation
        clean_coords = x_clean[:, 4:, :, :].reshape(-1)
        shuff_coords = x_shuff[:, 4:, :, :].reshape(-1)
        assert torch.allclose(
            torch.sort(clean_coords).values, torch.sort(shuff_coords).values
        )
