"""DeltaCore Phase 18: Unseen Classification Regime Transfer & Falsification.

Core modules:
    - generators: Multi-family non-stationary classification stream generators
    - models: Frozen, classical, recurrent, and DeltaCore online predictors
    - protocol: Strict online sequencing, causality assertions, and hash verification
    - controls: Shortcut audit, label-shuffle negative control, and feature permutation
    - run: Multi-seed benchmark pipeline and artifact serialization
    - analysis: Hypothesis testing and failure mode characterization
"""

from __future__ import annotations

from deltacore.benchmarks.phase_18.analysis import (
    evaluate_hypotheses_18,
    generate_failure_matrix_markdown,
    generate_resource_matrix_markdown,
    generate_transfer_matrix_markdown,
)
from deltacore.benchmarks.phase_18.controls import (
    audit_shortcut_statistics_18,
    permute_features_18,
    shuffle_labels_18,
)
from deltacore.benchmarks.phase_18.generators import (
    NonStationaryStream18,
    StationaryDataset18,
    generate_covariate_shift_stream,
    generate_family_a_rotation,
    generate_family_b_translation,
    generate_family_c_nonlinear,
    generate_gradual_shift_stream,
    generate_prior_shift_stream,
    generate_stationary_dataset_18,
    generate_strong_mismatch_stream,
)
from deltacore.benchmarks.phase_18.models import (
    FixedDeltaClassifier18,
    FrozenLinearClassifier18,
    OnlineLogisticRegression18,
    OnlineMulticlassLinear18,
    OnlineRidgeClassifier18,
    SafeAdaptiveDeltaClassifier18,
    StateOffAblationClassifier18,
    build_phase_18_model_suite,
    compute_model_parameter_hash_18,
    fit_offline_linear_head_18,
)
from deltacore.benchmarks.phase_18.protocol import (
    CausalStreamingProtocol,
    StreamEvaluationResult18,
    verify_parameter_immutability,
)
from deltacore.benchmarks.phase_18.run import run_phase_18_benchmark

__all__ = [
    "CausalStreamingProtocol",
    "FixedDeltaClassifier18",
    "FrozenLinearClassifier18",
    "NonStationaryStream18",
    "OnlineLogisticRegression18",
    "OnlineMulticlassLinear18",
    "OnlineRidgeClassifier18",
    "SafeAdaptiveDeltaClassifier18",
    "StateOffAblationClassifier18",
    "StationaryDataset18",
    "StreamEvaluationResult18",
    "audit_shortcut_statistics_18",
    "build_phase_18_model_suite",
    "compute_model_parameter_hash_18",
    "evaluate_hypotheses_18",
    "fit_offline_linear_head_18",
    "generate_covariate_shift_stream",
    "generate_failure_matrix_markdown",
    "generate_family_a_rotation",
    "generate_family_b_translation",
    "generate_family_c_nonlinear",
    "generate_gradual_shift_stream",
    "generate_prior_shift_stream",
    "generate_resource_matrix_markdown",
    "generate_stationary_dataset_18",
    "generate_strong_mismatch_stream",
    "generate_transfer_matrix_markdown",
    "permute_features_18",
    "run_phase_18_benchmark",
    "shuffle_labels_18",
    "verify_parameter_immutability",
]
