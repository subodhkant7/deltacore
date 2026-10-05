"""Phase 17: Classification Telemetry & Evaluation Metrics.

Computes accuracy, balanced accuracy, log-loss, first-passage recovery, sustained recovery,
cumulative excess classification loss, forgetting, adaptation energy, and runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    """Full telemetry and benchmark evaluation metrics for an online classification run."""

    model_name: str
    accuracy: float
    balanced_accuracy: float
    log_loss: float
    pre_shift_accuracy: float
    post_shift_accuracy: float
    first_passage_recovery: int
    sustained_recovery: int
    cumulative_excess_loss: float
    forgetting: float
    adaptation_energy: float
    max_state_norm: float
    min_safety_margin: float
    mean_latency_us: float
    persistent_state_bytes: int
    parameter_count: int
    rolling_accuracy_history: list[float]
    log_loss_history: list[float]
    state_norm_history: list[float]
    eta_history: list[float]
    diverged: int


def compute_log_loss(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7
) -> float:
    """Compute mean cross-entropy log loss."""
    clamped = probs.clamp(min=eps, max=1.0)
    # Gather true class probabilities
    true_probs = clamped[torch.arange(len(targets)), targets]
    loss = -torch.log(true_probs).mean()
    return float(loss.item())


def compute_balanced_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    """Compute macro-averaged recall across all classes."""
    recalls: list[float] = []
    for c in range(num_classes):
        mask = targets == c
        n_c = mask.sum().item()
        if n_c > 0:
            rec = (preds[mask] == c).float().sum().item() / n_c
            recalls.append(float(rec))
        else:
            recalls.append(1.0)
    return float(np.mean(recalls))


def compute_recovery_metrics(
    step_correct: list[int],
    shift_step: int,
    end_step: int,
    pre_window: int = 30,
    recovery_window: int = 10,
    sustained_window: int = 10,
) -> tuple[float, float, int, int]:
    """Calculate pre-shift accuracy, post-shift accuracy, first-passage and sustained recovery.

    Args:
        step_correct: List of 0 or 1 for each streaming step.
        shift_step: Step index where shift occurs.
        end_step: End of shift evaluation window.
        pre_window: Number of steps before shift to calculate pre-shift baseline.
        recovery_window: Window size for rolling accuracy.
        sustained_window: Number of consecutive steps for sustained recovery (K=10).

    Returns:
        Tuple of (pre_acc, post_acc, first_passage_steps, sustained_steps).
    """
    total_steps = len(step_correct)
    pre_start = max(0, shift_step - pre_window)
    pre_slice = step_correct[pre_start:shift_step]
    pre_acc = float(np.mean(pre_slice)) if pre_slice else 0.5

    # Immediate post-shift accuracy (e.g. first 20 steps)
    post_slice = step_correct[shift_step : min(shift_step + 20, total_steps)]
    post_acc = float(np.mean(post_slice)) if post_slice else 0.0

    threshold = max(0.20, pre_acc - 0.05)

    # First-passage recovery: rolling window accuracy >= threshold
    first_passage = max(0, end_step - shift_step)
    for t in range(shift_step, min(end_step, total_steps - recovery_window)):
        window = step_correct[t : t + recovery_window]
        if np.mean(window) >= threshold:
            first_passage = t - shift_step
            break

    # Sustained recovery: sustained_window consecutive correct classifications
    # or consecutive rolling accuracy >= threshold
    sustained = max(0, end_step - shift_step)
    for t in range(shift_step, min(end_step, total_steps - sustained_window)):
        window = step_correct[t : t + sustained_window]
        if np.mean(window) >= threshold:
            sustained = t - shift_step
            break

    return pre_acc, post_acc, first_passage, sustained


def compute_classification_telemetry(
    predictions: list[int],
    probabilities: list[torch.Tensor],
    targets: list[int],
    regime_bounds: list[int],
    state_norms: list[float],
    etas: list[float],
    safety_margins: list[float],
    update_norms: list[float],
    latencies: list[float],
    persistent_bytes: int,
    parameter_count: int,
    model_name: str,
    num_classes: int = 6,
) -> ClassificationMetrics:
    """Compute comprehensive classification performance and stability telemetry."""
    preds_t = torch.tensor(predictions, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)
    probs_t = torch.stack(probabilities)

    step_correct = (preds_t == targets_t).int().tolist()
    overall_acc = float((preds_t == targets_t).float().mean().item())
    balanced_acc = compute_balanced_accuracy(preds_t, targets_t, num_classes)
    loss = compute_log_loss(probs_t, targets_t)

    # Stepwise log loss
    eps = 1e-7
    clamped_probs = probs_t.clamp(min=eps, max=1.0)
    step_losses = [
        -float(torch.log(clamped_probs[i, targets[i]]).item())
        for i in range(len(targets))
    ]

    # Rolling accuracy (window W=15)
    w_size = 15
    rolling_acc = []
    for i in range(len(step_correct)):
        start = max(0, i - w_size + 1)
        rolling_acc.append(float(np.mean(step_correct[start : i + 1])))

    # Shift recovery on first regime boundary
    shift_0 = regime_bounds[0] if regime_bounds else len(targets) // 2
    shift_end = regime_bounds[1] if len(regime_bounds) > 1 else len(targets)
    pre_acc, post_acc, fp_rec, sust_rec = compute_recovery_metrics(
        step_correct=step_correct,
        shift_step=shift_0,
        end_step=shift_end,
    )

    # Cumulative excess loss during first shift interval
    pre_loss_slice = step_losses[max(0, shift_0 - 30) : shift_0]
    pre_loss_mean = float(np.mean(pre_loss_slice)) if pre_loss_slice else 0.5
    excess_loss = float(
        sum(
            max(0.0, loss_val - pre_loss_mean)
            for loss_val in step_losses[shift_0:shift_end]
        )
    )

    # Forgetting: accuracy in initial Regime A vs return to Regime A
    if len(regime_bounds) >= 3:
        b0 = regime_bounds[0]
        b_last = regime_bounds[-1]
        acc_initial_a = float(np.mean(step_correct[:b0]))
        acc_return_a = float(np.mean(step_correct[b_last:]))
        forgetting = max(0.0, acc_initial_a - acc_return_a)
    else:
        forgetting = 0.0

    adapt_energy = float(sum(u**2 for u in update_norms)) if update_norms else 0.0
    max_norm = max(state_norms) if state_norms else 0.0
    min_margin = min(safety_margins) if safety_margins else 2.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    diverged = 1 if (np.isnan(overall_acc) or np.isnan(loss) or max_norm > 1e6) else 0

    return ClassificationMetrics(
        model_name=model_name,
        accuracy=overall_acc,
        balanced_accuracy=balanced_acc,
        log_loss=loss,
        pre_shift_accuracy=pre_acc,
        post_shift_accuracy=post_acc,
        first_passage_recovery=fp_rec,
        sustained_recovery=sust_rec,
        cumulative_excess_loss=excess_loss,
        forgetting=forgetting,
        adaptation_energy=adapt_energy,
        max_state_norm=max_norm,
        min_safety_margin=min_margin,
        mean_latency_us=mean_lat,
        persistent_state_bytes=persistent_bytes,
        parameter_count=parameter_count,
        rolling_accuracy_history=rolling_acc,
        log_loss_history=step_losses,
        state_norm_history=state_norms,
        eta_history=etas,
        diverged=diverged,
    )
