"""DeltaCore Phase 9.1: Spatial Learning Validation Benchmark.

Audits spatial learning capability through genuine optimization on two tasks:
- Task A: Spatial Neighborhood Reconstruction / Shift Transformation
- Task B: Normalized Spatial Pattern Classification (Constant Energy ||X||_F = 1.0)

Evaluates:
- Baseline A: Zero-output predictor
- Baseline B: Static linear spatial predictor
- Non-spatial baseline: 1D sequence unrolling without directional routing
- Baseline C: Single-direction DeltaCore (RIGHT, LEFT, DOWN, UP)
- Baseline D: Two-direction DeltaCore (RIGHT+LEFT, DOWN+UP)
- Baseline E: Four-direction DeltaCore (All routes with fusion)

Protocol:
- PyTorch + AdamW optimizer
- Fixed seeds with multi-seed statistical evaluation (seeds 0, 1, 2, 3, 4)
- Train / Validation splits
- Chunk-size sensitivity (C = 1, 2, 4, 8, full)
- Absolute Frobenius error ||Y - Y_hat||_F and Normalized Relative Error E_rel
- Generates publication plots O, P, Q, R, S, T into docs/benchmarks/artifacts/phase_9_1/

SCIENTIFIC PRINCIPLE:
Do NOT claim directional specialization or vision understanding.
Report objective empirical results across baselines, including negative findings.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from deltacore.benchmarks.metrics.accuracy import compute_relative_error
from deltacore.observatory.plots import (
    plot_absolute_vs_relative_error,
    plot_accuracy_vs_directions,
    plot_metric_vs_directional_route,
    plot_runtime_vs_chunk_size,
    plot_train_val_loss,
    plot_validation_error_vs_chunk_size,
)
from deltacore.spatial.fusion import EqualFusion, LearnedChannelFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.spatial.routes import DOWN, LEFT, RIGHT, UP, SpatialRoute
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem

# ==============================================================================
# 1. Dataset Generation: Task A & Task B
# ==============================================================================


def generate_task_a_dataset(
    num_samples: int = 80,
    height: int = 10,
    width: int = 10,
    channels: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Task A: Spatial Neighborhood Aggregation / Shift Transform.

    Y(c, h, w) = 0.5 * X(c, h, (w + 1) % W) + 0.5 * X(c, (h + 1) % H, w)
    Couples orthogonal spatial axes; cannot be solved by identity pass-through.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(
        num_samples, channels, height, width, generator=g, dtype=torch.float32
    )

    # Shifted neighbors
    X_right = torch.roll(X, shifts=-1, dims=3)  # w + 1
    X_down = torch.roll(X, shifts=-1, dims=2)  # h + 1
    Y = 0.5 * X_right + 0.5 * X_down
    return X, Y


def generate_task_b_dataset(
    samples_per_class: int = 20,
    height: int = 10,
    width: int = 10,
    channels: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Task B: Spatial Pattern Classification with Unit-Norm Normalization.

    6 Classes:
        0: horizontal
        1: vertical
        2: diagonal
        3: checkerboard
        4: localized
        5: asymmetric

    CRITICAL INVARIANT:
        Every sample is strictly normalized so ||X_i||_F = 1.000.
        Classification CANNOT exploit total tensor magnitude.
    """
    torch.manual_seed(seed)
    X_list, y_list = [], []

    for c_idx in range(6):
        for _ in range(samples_per_class):
            noise = torch.randn(1, channels, height, width, dtype=torch.float32) * 0.1
            pattern = torch.zeros(1, channels, height, width, dtype=torch.float32)

            if c_idx == 0:  # horizontal stripe
                h_ramp = torch.sin(
                    2.0 * math.pi * torch.arange(height).view(1, 1, height, 1) / height
                )
                pattern = h_ramp.expand(1, channels, height, width)
            elif c_idx == 1:  # vertical stripe
                w_ramp = torch.sin(
                    2.0 * math.pi * torch.arange(width).view(1, 1, 1, width) / width
                )
                pattern = w_ramp.expand(1, channels, height, width)
            elif c_idx == 2:  # diagonal
                diag = torch.sin(
                    2.0
                    * math.pi
                    * (
                        torch.arange(height).view(1, 1, height, 1)
                        + torch.arange(width).view(1, 1, 1, width)
                    )
                    / max(height, width)
                )
                pattern = diag.expand(1, channels, height, width)
            elif c_idx == 3:  # checkerboard
                gh = torch.arange(height).view(-1, 1)
                gw = torch.arange(width).view(1, -1)
                cb = ((gh + gw) % 2).to(torch.float32).view(1, 1, height, width)
                pattern = (cb * 2.0 - 1.0).expand(1, channels, height, width)
            elif c_idx == 4:  # localized square
                pattern[
                    :,
                    :,
                    height // 2 - 1 : height // 2 + 2,
                    width // 2 - 1 : width // 2 + 2,
                ] = 2.0
            elif c_idx == 5:  # asymmetric quadrant
                pattern[:, :, : height // 2, : width // 2] = 1.0
                pattern[:, :, height // 2 :, width // 2 :] = -1.0

            sample = pattern + noise
            # Strict unit-norm normalization
            sample = sample / (torch.norm(sample) + 1e-7)
            X_list.append(sample)
            y_list.append(c_idx)

    X_all = torch.cat(X_list, dim=0)  # [num_samples, channels, height, width]
    y_all = torch.tensor(y_list, dtype=torch.long)
    return X_all, y_all


# ==============================================================================
# 2. Baseline Model Definitions
# ==============================================================================


class ZeroPredictor(nn.Module):
    """Baseline A: Zero-output predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class StaticLinearSpatialPredictor(nn.Module):
    """Baseline B: Static linear 2D spatial convolution filter without recurrent state."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NonSpatialSequencePredictor(nn.Module):
    """Non-Spatial Baseline: Flattened 1D sequence unrolling without 2D route multiplicity."""

    def __init__(self, config: FiveMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.system = FiveMemorySystem(config)
        self.init_c = nn.Parameter(torch.zeros(config.v_dim, config.k_dim))
        self.init_k = nn.Parameter(torch.randn(config.k_dim, config.in_dim) * 0.02)
        self.init_v = nn.Parameter(torch.randn(config.v_dim, config.in_dim) * 0.02)
        self.init_lr = nn.Parameter(torch.zeros(config.lr_dim, config.get_feat_dim()))
        self.init_ret = nn.Parameter(torch.zeros(config.ret_dim, config.get_feat_dim()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Flatten spatial dimensions into 1D sequence of length N = H*W
        seq = x.view(b, c, h * w).permute(0, 2, 1)  # [B, N, C]
        from deltacore.memory.five_memory import FiveMemoryState

        st = FiveMemoryState(
            content=self.init_c.unsqueeze(0).repeat(b, 1, 1),
            key=self.init_k.unsqueeze(0).repeat(b, 1, 1),
            value=self.init_v.unsqueeze(0).repeat(b, 1, 1),
            learning_rate=self.init_lr.unsqueeze(0).repeat(b, 1, 1),
            retention=self.init_ret.unsqueeze(0).repeat(b, 1, 1),
        )
        res = self.system.scan(seq, st)
        # Restore sequence back to [B, C, H, W]
        restored = res.predictions.permute(0, 2, 1).view(b, c, h, w)
        return restored


class SpatialClassifier(nn.Module):
    """Classifier head wrapping a spatial backbone (DeltaCore or baseline) for Task B."""

    def __init__(
        self, backbone: nn.Module, channels: int, num_classes: int = 6
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.backbone, SpatialAdaptiveOperator):
            out = self.backbone(x).fused_output
        else:
            out = self.backbone(x)
        # Global spatial average pooling over H, W
        pooled = torch.mean(out, dim=(-2, -1))  # [B, C]
        logits = self.head(pooled)
        return logits


# ==============================================================================
# 3. Training & Evaluation Engine
# ==============================================================================


def train_reconstruction_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    lr: float = 0.01,
    batch_size: int = 16,
) -> tuple[list[float], list[float], float, float, float]:
    """Train a spatial reconstruction model using AdamW and evaluate validation metrics."""
    params = [p for p in model.parameters() if p.requires_grad]
    train_losses = []
    val_losses = []

    start_time = time.perf_counter()

    if params:
        optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
        num_samples = x_train.shape[0]

        for _epoch in range(epochs):
            model.train()
            perm = torch.randperm(num_samples)
            epoch_loss = 0.0
            batches = 0

            for i in range(0, num_samples, batch_size):
                idx = perm[i : i + batch_size]
                bx, by = x_train[idx], y_train[idx]

                optimizer.zero_grad()
                if isinstance(model, SpatialAdaptiveOperator):
                    pred = model(bx).fused_output
                else:
                    pred = model(bx)

                loss = torch.mean((pred - by) ** 2)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            train_losses.append(epoch_loss / max(1, batches))

            # Validation loss
            model.eval()
            with torch.no_grad():
                if isinstance(model, SpatialAdaptiveOperator):
                    v_pred = model(x_val).fused_output
                else:
                    v_pred = model(x_val)
                v_loss = torch.mean((v_pred - y_val) ** 2).item()
                val_losses.append(v_loss)
    else:
        # Zero predictor or parameter-free model
        train_losses = [1.0]
        val_losses = [1.0]

    runtime = (time.perf_counter() - start_time) * 1000.0

    # Final validation evaluation
    model.eval()
    with torch.no_grad():
        if isinstance(model, SpatialAdaptiveOperator):
            y_hat = model(x_val).fused_output
        else:
            y_hat = model(x_val)

        abs_err = torch.norm(y_val - y_hat).item()
        rel_err = compute_relative_error(y_hat, y_val)

    return train_losses, val_losses, abs_err, rel_err, runtime


def train_classification_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    lr: float = 0.01,
    batch_size: int = 16,
) -> tuple[float, float]:
    """Train a spatial classification model and evaluate validation accuracy."""
    params = [p for p in model.parameters() if p.requires_grad]
    start_time = time.perf_counter()

    if params:
        optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        num_samples = x_train.shape[0]

        for _epoch in range(epochs):
            model.train()
            perm = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                idx = perm[i : i + batch_size]
                bx, by = x_train[idx], y_train[idx]
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_val)
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == y_val).sum().item()
        accuracy = (correct / y_val.shape[0]) * 100.0

    runtime = (time.perf_counter() - start_time) * 1000.0
    return accuracy, runtime


# ==============================================================================
# 4. Helper Factory
# ==============================================================================


def create_operator(
    channels: int,
    routes: tuple[SpatialRoute, ...],
    chunk_size: int = 1,
    use_learned_fusion: bool = False,
    mode: str = "generic",
) -> SpatialAdaptiveOperator:
    cfg = FiveMemoryConfig(
        v_dim=channels,
        k_dim=channels,
        in_dim=channels,
        feat_dim=channels,
        lr_dim=2,
        ret_dim=2,
        eta_max=0.5,
        ret_min=0.1,
        apply_stability_control=True,
    )
    fusion = (
        LearnedChannelFusion(channels=channels, num_routes=len(routes))
        if use_learned_fusion
        else EqualFusion(mode="mean")
    )
    return SpatialAdaptiveOperator(
        config=cfg,
        fusion=fusion,
        mode=mode,
        default_chunk_size=chunk_size,
        routes=routes,
        learnable_initial_states=True,
    )


# ==============================================================================
# 5. Main Execution Protocol
# ==============================================================================


def run_spatial_learning_benchmark() -> None:
    print("=" * 88)
    print("DeltaCore Phase 9.1: Spatial Learning Validation Audit")
    print("=" * 88)

    H, W, C = 10, 10, 4
    epochs = 15
    lr = 0.01
    seeds = [0, 1, 2, 3, 4]
    out_dir = Path("docs/benchmarks/artifacts/phase_9_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Task A: Spatial Neighborhood Reconstruction Transform ---")
    print("Generating train (N=64) and validation (N=32) spatial datasets...")
    x_train, y_train = generate_task_a_dataset(
        num_samples=64, height=H, width=W, channels=C, seed=101
    )
    x_val, y_val = generate_task_a_dataset(
        num_samples=32, height=H, width=W, channels=C, seed=202
    )

    # Multi-model evaluation on Task A
    model_factories = {
        "Baseline A (Zero)": lambda: ZeroPredictor(),
        "Baseline B (Static Linear)": lambda: StaticLinearSpatialPredictor(channels=C),
        "Non-Spatial (1D Seq)": lambda: NonSpatialSequencePredictor(
            FiveMemoryConfig(
                v_dim=C, k_dim=C, in_dim=C, feat_dim=C, lr_dim=2, ret_dim=2
            )
        ),
        "1-Dir (RIGHT)": lambda: create_operator(C, (RIGHT,)),
        "1-Dir (LEFT)": lambda: create_operator(C, (LEFT,)),
        "1-Dir (DOWN)": lambda: create_operator(C, (DOWN,)),
        "1-Dir (UP)": lambda: create_operator(C, (UP,)),
        "2-Dir (Horizontal R+L)": lambda: create_operator(C, (RIGHT, LEFT)),
        "2-Dir (Vertical D+U)": lambda: create_operator(C, (DOWN, UP)),
        "4-Dir (All Equal)": lambda: create_operator(
            C, (RIGHT, LEFT, DOWN, UP), use_learned_fusion=False
        ),
        "4-Dir (All Learned)": lambda: create_operator(
            C, (RIGHT, LEFT, DOWN, UP), use_learned_fusion=True
        ),
    }

    results_task_a: dict[str, dict[str, list[float]]] = {
        k: {"abs_err": [], "rel_err": [], "runtime": []} for k in model_factories
    }
    sample_curves: dict[str, tuple[list[float], list[float]]] = {}

    print(f"\nRunning Task A across seeds {seeds}...")
    for seed in seeds:
        torch.manual_seed(seed)
        for name, factory in model_factories.items():
            m = factory()
            tr_losses, v_losses, abs_err, rel_err, rtime = train_reconstruction_model(
                m, x_train, y_train, x_val, y_val, epochs=epochs, lr=lr
            )
            results_task_a[name]["abs_err"].append(abs_err)
            results_task_a[name]["rel_err"].append(rel_err)
            results_task_a[name]["runtime"].append(rtime)
            if seed == 0 and name in (
                "Baseline B (Static Linear)",
                "4-Dir (All Learned)",
            ):
                sample_curves[name] = (tr_losses, v_losses)

    print("\n" + "=" * 88)
    print(
        f"{'Model / Architecture':<28} | {'Params':<8} | {'Abs Error (Mean±Std)':<24} | {'Rel Error E_rel':<16} | {'Runtime (ms)':<10}"
    )
    print("-" * 88)

    model_names_plot = []
    mean_abs_plot = []
    mean_rel_plot = []

    for name, factory in model_factories.items():
        m_temp = factory()
        param_count = sum(p.numel() for p in m_temp.parameters() if p.requires_grad)
        abs_vals = results_task_a[name]["abs_err"]
        rel_vals = results_task_a[name]["rel_err"]
        r_vals = results_task_a[name]["runtime"]

        m_abs, s_abs = np.mean(abs_vals), np.std(abs_vals)
        m_rel, s_rel = np.mean(rel_vals), np.std(rel_vals)
        m_rt = np.mean(r_vals)

        print(
            f"{name:<28} | {param_count:<8} | {m_abs:>8.4f} ± {s_abs:<12.4f} | {m_rel:>6.4f} ± {s_rel:<6.4f} | {m_rt:<10.1f}"
        )
        model_names_plot.append(
            name.replace("Baseline ", "B-").replace("DeltaCore ", "")
        )
        mean_abs_plot.append(float(m_abs))
        mean_rel_plot.append(float(m_rel))

    print("-" * 88)

    # --- Chunk Size Evaluation on Task A ---
    print(
        "\n--- Evaluating Boundary Chunk Size Sensitivity (C in {1, 2, 4, 8, full}) ---"
    )
    chunk_sizes = [1, 2, 4, 8, 100]  # 100 exceeds T=100 (full sequence)
    chunk_metrics = {}

    for c in chunk_sizes:
        op_c = create_operator(
            C, (RIGHT, LEFT, DOWN, UP), chunk_size=c, use_learned_fusion=True
        )
        _, _, abs_err, rel_err, rtime = train_reconstruction_model(
            op_c, x_train, y_train, x_val, y_val, epochs=epochs, lr=lr
        )
        c_label = str(c) if c <= 8 else "full"
        chunk_metrics[c_label] = {"val_err": rel_err, "runtime": rtime}
        print(
            f"Chunk C={c_label:<5} -> Relative Error: {rel_err:.4f}, Runtime: {rtime:.1f} ms"
        )

    # --- Task B: Spatial Classification Benchmark ---
    print(
        "\n--- Task B: Spatial Pattern Classification (Constant Energy ||X||_F = 1.0) ---"
    )
    xb_train, yb_train = generate_task_b_dataset(
        samples_per_class=20, height=H, width=W, channels=C, seed=303
    )
    xb_val, yb_val = generate_task_b_dataset(
        samples_per_class=10, height=H, width=W, channels=C, seed=404
    )

    classification_models = {
        "Baseline A (Random)": lambda: None,
        "Baseline B (Linear)": lambda: nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 6)
        ),
        "1-Dir (RIGHT)": lambda: SpatialClassifier(
            create_operator(C, (RIGHT,)), channels=C
        ),
        "2-Dir (Horizontal)": lambda: SpatialClassifier(
            create_operator(C, (RIGHT, LEFT)), channels=C
        ),
        "2-Dir (Vertical)": lambda: SpatialClassifier(
            create_operator(C, (DOWN, UP)), channels=C
        ),
        "4-Dir (All Learned)": lambda: SpatialClassifier(
            create_operator(C, (RIGHT, LEFT, DOWN, UP), use_learned_fusion=True),
            channels=C,
        ),
    }

    acc_results = {}
    print(f"\nEvaluating Task B Classification Accuracy across seeds {seeds}...")
    for name, factory in classification_models.items():
        if name == "Baseline A (Random)":
            acc_results[name] = 100.0 / 6.0
            print(
                f"{name:<28} | Accuracy: {acc_results[name]:.2f}% (theoretical random)"
            )
            continue

        acc_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            m = factory()
            acc, _ = train_classification_model(
                m, xb_train, yb_train, xb_val, yb_val, epochs=epochs, lr=lr
            )
            acc_runs.append(acc)

        m_acc, s_acc = np.mean(acc_runs), np.std(acc_runs)
        acc_results[name] = float(m_acc)
        print(f"{name:<28} | Accuracy: {m_acc:>6.2f}% ± {s_acc:.2f}%")

    # ==============================================================================
    # 6. Generate Publication Plots O through T
    # ==============================================================================
    print(
        "\nGenerating Phase 9.1 Publication Plots O through T in docs/benchmarks/artifacts/phase_9_1/ ..."
    )

    # Plot O: Training vs validation loss curve
    if "4-Dir (All Learned)" in sample_curves:
        tr_curve, val_curve = sample_curves["4-Dir (All Learned)"]
        plot_train_val_loss(
            tr_curve, val_curve, output_path=out_dir / "plot_o_train_val_loss.png"
        )

    # Plot P: Accuracy vs direction count
    dir_acc = {
        "1-Dir": acc_results.get("1-Dir (RIGHT)", 0.0),
        "2-Dir (H)": acc_results.get("2-Dir (Horizontal)", 0.0),
        "2-Dir (V)": acc_results.get("2-Dir (Vertical)", 0.0),
        "4-Dir": acc_results.get("4-Dir (All Learned)", 0.0),
    }
    plot_accuracy_vs_directions(
        dir_acc,
        metric_label="Accuracy (%)",
        output_path=out_dir / "plot_p_accuracy_vs_directions.png",
    )

    # Plot Q: Validation error vs chunk size
    q_errors = {k: v["val_err"] for k, v in chunk_metrics.items()}
    plot_validation_error_vs_chunk_size(
        q_errors, output_path=out_dir / "plot_q_val_error_vs_chunk_size.png"
    )

    # Plot R: Runtime vs chunk size
    r_runtimes = {k: v["runtime"] for k, v in chunk_metrics.items()}
    plot_runtime_vs_chunk_size(
        r_runtimes, output_path=out_dir / "plot_r_runtime_vs_chunk_size.png"
    )

    # Plot S: Error vs directional route
    route_errs = {
        "RIGHT": float(np.mean(results_task_a["1-Dir (RIGHT)"]["rel_err"])),
        "LEFT": float(np.mean(results_task_a["1-Dir (LEFT)"]["rel_err"])),
        "DOWN": float(np.mean(results_task_a["1-Dir (DOWN)"]["rel_err"])),
        "UP": float(np.mean(results_task_a["1-Dir (UP)"]["rel_err"])),
        "R+L": float(np.mean(results_task_a["2-Dir (Horizontal R+L)"]["rel_err"])),
        "D+U": float(np.mean(results_task_a["2-Dir (Vertical D+U)"]["rel_err"])),
        "All-4": float(np.mean(results_task_a["4-Dir (All Learned)"]["rel_err"])),
    }
    plot_metric_vs_directional_route(
        route_errs,
        metric_name="Relative Error E_rel",
        output_path=out_dir / "plot_s_route_comparison.png",
    )

    # Plot T: Absolute vs Relative Error across models
    plot_absolute_vs_relative_error(
        model_names_plot[:6],
        mean_abs_plot[:6],
        mean_rel_plot[:6],
        output_path=out_dir / "plot_t_absolute_vs_relative_error.png",
    )

    print("Phase 9.1 Plots O through T generated successfully.")
    print("=" * 88)


if __name__ == "__main__":
    run_spatial_learning_benchmark()
