"""DeltaCore Classification Module: Online Non-Stationary Classification & Adaptive State.

Implements online classification tasks, non-stationary streams (A -> B -> C -> A),
adaptive classification models (SafeAdaptiveDelta, FixedDelta, classical baselines),
recovery and retention diagnostics, and full Phase 17 evaluation pipelines.
"""

from __future__ import annotations

from deltacore.classification.metrics import (
    ClassificationMetrics,
    compute_balanced_accuracy,
    compute_classification_telemetry,
    compute_log_loss,
    compute_recovery_metrics,
)
from deltacore.classification.models import (
    FixedDeltaClassifier,
    FrozenLinearClassifier,
    GRUClassifier,
    LSTMClassifier,
    OnlineClassificationPredictor,
    OnlineLogisticRegression,
    OnlineMulticlassLinear,
    OnlineRidgeClassifier,
    SafeAdaptiveDeltaClassifier,
    SmallMLPClassifier,
    compute_model_parameter_hash,
)
from deltacore.classification.phase_17_runner import (
    evaluate_online_stream,
    run_phase_17_benchmark,
)
from deltacore.classification.synthetic_stream import (
    NonStationaryStream,
    StationaryDataset,
    audit_shortcut_statistics,
    generate_nonstationary_stream,
    generate_stationary_dataset,
    permute_features,
    shuffle_labels,
)

__all__ = [
    # Data & Streams
    "NonStationaryStream",
    "StationaryDataset",
    "generate_nonstationary_stream",
    "generate_stationary_dataset",
    "audit_shortcut_statistics",
    "shuffle_labels",
    "permute_features",
    # Models
    "OnlineClassificationPredictor",
    "OnlineLogisticRegression",
    "OnlineRidgeClassifier",
    "OnlineMulticlassLinear",
    "FrozenLinearClassifier",
    "SmallMLPClassifier",
    "GRUClassifier",
    "LSTMClassifier",
    "FixedDeltaClassifier",
    "SafeAdaptiveDeltaClassifier",
    "compute_model_parameter_hash",
    # Metrics
    "ClassificationMetrics",
    "compute_log_loss",
    "compute_balanced_accuracy",
    "compute_recovery_metrics",
    "compute_classification_telemetry",
    # Runner
    "evaluate_online_stream",
    "run_phase_17_benchmark",
]
