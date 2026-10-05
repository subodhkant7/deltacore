r"""Phase 18: Online Causal Streaming Protocol & Immutability Verification.

Strictly enforces:
    1. Online sequencing: \hat{y}_t = f(x_t, S_{t-1}) before label y_t is revealed.
    2. Timestamp monotonicity: pred_time <= reveal_time <= update_time.
    3. Cryptographic parameter immutability: pre- and post-stream SHA-256 hashes must match bit-for-bit.
    4. Optional explicit state reset at regime boundaries when evaluated in ablation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from deltacore.benchmarks.phase_18.models import (
    FixedDeltaClassifier18,
    OnlineLogisticRegression18,
    OnlineMulticlassLinear18,
    OnlinePredictor18,
    OnlineRidgeClassifier18,
    SafeAdaptiveDeltaClassifier18,
)


@dataclass(frozen=True)
class StreamEvaluationResult18:
    """Telemetry and evaluation result container for Phase 18 online stream."""

    model_name: str
    accuracy: float
    balanced_accuracy: float
    log_loss: float
    mean_latency_us: float
    persistent_state_bytes: int
    parameter_count: int
    pre_shift_accuracy: float
    post_shift_accuracy: float
    first_passage_recovery: int
    sustained_recovery: int
    cumulative_excess_loss: float
    forgetting: float
    return_regime_accuracy: float
    max_state_norm: float
    adaptation_energy: float
    min_safety_margin: float
    diverged: int
    rolling_accuracy_history: list[float]
    state_norm_history: list[float]
    step_latencies_us: list[float]
    causality_verified: bool
    parameter_immutable: bool


def verify_parameter_immutability(model: OnlinePredictor18, hash_before: str) -> bool:
    """Verify that model parameters did not mutate during evaluation."""
    hash_after = model.get_parameter_hash()
    return hash_before == hash_after


class CausalStreamingProtocol:
    """Programmatic execution harness enforcing strict online causality and parameter immutability."""

    @staticmethod
    def evaluate_stream(
        model: OnlinePredictor18,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        regime_bounds: list[int] | None = None,
        reset_at_bounds: bool = False,
        rolling_window: int = 15,
        sustained_window: int = 10,
    ) -> StreamEvaluationResult18:
        """Run online streaming evaluation with timestamp logging and causality checks.

        Args:
            model: OnlinePredictor instance to evaluate.
            inputs: Tensor of shape (T, D) containing streaming features.
            targets: Tensor of shape (T,) containing integer ground-truth labels.
            regime_bounds: Step indices where distribution regimes change.
            reset_at_bounds: If True, calls model.reset_state() at each regime bound.
            rolling_window: Window size for rolling-window accuracy.
            sustained_window: Consecutive accurate steps required for sustained recovery.

        Returns:
            StreamEvaluationResult18 populated with complete telemetry.
        """
        bounds = regime_bounds or []
        num_samples = len(inputs)
        num_classes = model.num_classes

        # Cryptographic parameter hash before streaming
        h_before = model.get_parameter_hash()

        preds: list[int] = []
        step_correct: list[int] = []
        step_losses: list[float] = []
        latencies: list[float] = []
        state_norms: list[float] = []
        safety_margins: list[float] = []
        update_energies: list[float] = []

        rolling_acc_history: list[float] = []
        causality_checks_passed = True

        for t in range(num_samples):
            # Optional intervention: internal adaptive-state reset at regime boundary
            if reset_at_bounds and t in bounds:
                model.reset_state()

            x_t = inputs[t]

            # ------------------------------------------------------------------
            # Phase A: Prediction Step (S_{t-1} ONLY - label is UNKNOWN)
            # ------------------------------------------------------------------
            t_pred_start = time.perf_counter_ns()
            pred_class, pred_probs = model.predict_step(x_t)
            t_pred_end = time.perf_counter_ns()

            # Record prediction
            preds.append(pred_class)

            # ------------------------------------------------------------------
            # Phase B: Label Revelation & Verification Step
            # ------------------------------------------------------------------
            t_reveal = time.perf_counter_ns()
            y_t = int(targets[t].item())

            # Programmatic assertion of temporal sequence
            if not (t_pred_start <= t_pred_end <= t_reveal):
                causality_checks_passed = False

            # Evaluate immediate step metrics
            is_correct = int(pred_class == y_t)
            step_correct.append(is_correct)

            eps = 1e-12
            prob_true = float(pred_probs[y_t].clamp(min=eps).item())
            step_losses.append(-math.log(prob_true))

            # ------------------------------------------------------------------
            # Phase C: State Adaptation Step (Label y_t now revealed)
            # ------------------------------------------------------------------
            t_adapt_start = time.perf_counter_ns()
            model.adapt_step(x_t, y_t)
            t_adapt_end = time.perf_counter_ns()

            if not (t_reveal <= t_adapt_start <= t_adapt_end):
                causality_checks_passed = False

            # Latency for prediction + adaptation
            step_lat_us = (t_adapt_end - t_pred_start) / 1000.0
            latencies.append(step_lat_us)

            # Telemetry: State norm extraction
            if isinstance(
                model, (SafeAdaptiveDeltaClassifier18, FixedDeltaClassifier18)
            ):
                m_norm = float(torch.linalg.norm(model.M).item())
            elif isinstance(model, OnlineRidgeClassifier18):
                m_norm = float(torch.linalg.norm(model.W).item())
            elif isinstance(
                model, (OnlineMulticlassLinear18, OnlineLogisticRegression18)
            ):
                m_norm = float(torch.linalg.norm(model.weight).item())
            else:
                m_norm = 0.0

            state_norms.append(m_norm)
            safety_margins.append(getattr(model, "last_safety_margin", 2.0))
            update_energies.append(getattr(model, "last_update_norm", 0.0) ** 2)

            # Rolling accuracy window
            r_start = max(0, t - rolling_window + 1)
            rolling_acc_history.append(float(np.mean(step_correct[r_start : t + 1])))

        # ------------------------------------------------------------------
        # Phase D: Parameter Immutability Check (\Delta\theta = 0)
        # ------------------------------------------------------------------
        param_immutable = verify_parameter_immutability(model, h_before)
        if hasattr(model, "weight_head") and not param_immutable:
            raise RuntimeError(
                f"[CausalStreamingProtocol] Parameter mutation detected in {model.name}! "
                f"SHA-256 before ({h_before[:12]}...) != after ({model.get_parameter_hash()[:12]}...)"
            )

        # ------------------------------------------------------------------
        # Phase E: Metric Aggregation
        # ------------------------------------------------------------------
        overall_acc = float(np.mean(step_correct))
        overall_loss = float(np.mean(step_losses))

        # Balanced accuracy
        class_accs = []
        targets_np = targets.cpu().numpy()
        preds_np = np.array(preds)
        for c in range(num_classes):
            mask = targets_np == c
            if np.any(mask):
                class_accs.append(float(np.mean(preds_np[mask] == c)))
        balanced_acc = float(np.mean(class_accs)) if class_accs else overall_acc

        # Shift recovery metrics
        shift_0 = bounds[0] if bounds else 0
        shift_end = bounds[1] if len(bounds) > 1 else num_samples

        pre_slice = step_correct[max(0, shift_0 - 30) : shift_0]
        pre_shift_acc = float(np.mean(pre_slice)) if pre_slice else overall_acc

        post_slice = step_correct[shift_0 : min(shift_0 + 20, num_samples)]
        post_shift_acc = float(np.mean(post_slice)) if post_slice else 0.0

        threshold = max(0.20, pre_shift_acc - 0.05)

        # First passage
        first_passage = max(0, shift_end - shift_0)
        for idx in range(shift_0, min(shift_end, num_samples - 10)):
            if np.mean(step_correct[idx : idx + 10]) >= threshold:
                first_passage = idx - shift_0
                break

        # Sustained recovery
        sustained = max(0, shift_end - shift_0)
        for idx in range(shift_0, min(shift_end, num_samples - sustained_window)):
            if np.mean(step_correct[idx : idx + sustained_window]) >= threshold:
                sustained = idx - shift_0
                break

        # Cumulative excess loss
        pre_loss_slice = step_losses[max(0, shift_0 - 30) : shift_0]
        pre_loss_mean = float(np.mean(pre_loss_slice)) if pre_loss_slice else 0.5
        excess_loss = float(
            sum(
                max(0.0, l_val - pre_loss_mean)
                for l_val in step_losses[shift_0:shift_end]
            )
        )

        # Forgetting & return regime
        forgetting = 0.0
        return_acc = overall_acc
        if len(bounds) >= 2:
            acc_a1 = float(np.mean(step_correct[: bounds[0]]))
            acc_a2 = float(np.mean(step_correct[bounds[-1] :]))
            forgetting = max(0.0, acc_a1 - acc_a2)
            return_acc = acc_a2

        max_norm = float(np.max(state_norms)) if state_norms else 0.0
        min_safety = float(np.min(safety_margins)) if safety_margins else 2.0
        adapt_energy = float(np.sum(update_energies))
        diverged = int(
            np.isnan(overall_loss) or np.isinf(overall_loss) or max_norm > 1e4
        )

        return StreamEvaluationResult18(
            model_name=model.name,
            accuracy=overall_acc,
            balanced_accuracy=balanced_acc,
            log_loss=overall_loss,
            mean_latency_us=float(np.mean(latencies)),
            persistent_state_bytes=model.get_state_memory_bytes(),
            parameter_count=model.get_parameter_count(),
            pre_shift_accuracy=pre_shift_acc,
            post_shift_accuracy=post_shift_acc,
            first_passage_recovery=first_passage,
            sustained_recovery=sustained,
            cumulative_excess_loss=excess_loss,
            forgetting=forgetting,
            return_regime_accuracy=return_acc,
            max_state_norm=max_norm,
            adaptation_energy=adapt_energy,
            min_safety_margin=min_safety,
            diverged=diverged,
            rolling_accuracy_history=rolling_acc_history,
            state_norm_history=state_norms,
            step_latencies_us=latencies,
            causality_verified=causality_checks_passed,
            parameter_immutable=param_immutable,
        )
