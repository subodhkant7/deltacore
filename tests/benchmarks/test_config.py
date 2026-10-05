# ==============================================================================
# DeltaCore: tests/benchmarks/test_config.py
# Unit tests for BenchmarkConfig validation and serialization.
# ==============================================================================

import pytest

from deltacore.benchmarks.schemas.config import BenchmarkConfig


def test_valid_config_creation() -> None:
    """Verify valid BenchmarkConfig creates without error."""
    cfg = BenchmarkConfig(
        experiment_name="test_task",
        seed=42,
        sequence_length=100,
        key_dim=16,
        value_dim=32,
        batch_size=8,
        shift_position=50,
        chunk_size=10,
        controller="fixed",
        stability_mode="dual_safe",
        dtype="float32",
        device="cpu",
    )
    assert cfg.experiment_name == "test_task"
    assert cfg.seed == 42
    assert cfg.sequence_length == 100
    assert cfg.key_dim == 16
    assert cfg.value_dim == 32
    assert cfg.shift_position == 50


def test_invalid_experiment_name_rejected() -> None:
    """Empty or non-string experiment_name must raise ValueError."""
    with pytest.raises(ValueError, match="experiment_name"):
        BenchmarkConfig(experiment_name="")


def test_negative_seed_rejected() -> None:
    """Negative seed must raise ValueError."""
    with pytest.raises(ValueError, match="seed"):
        BenchmarkConfig(experiment_name="task", seed=-1)


def test_non_positive_dimensions_rejected() -> None:
    """Non-positive dimensions must raise ValueError."""
    with pytest.raises(ValueError, match="sequence_length"):
        BenchmarkConfig(experiment_name="task", sequence_length=0)
    with pytest.raises(ValueError, match="key_dim"):
        BenchmarkConfig(experiment_name="task", key_dim=0)
    with pytest.raises(ValueError, match="value_dim"):
        BenchmarkConfig(experiment_name="task", value_dim=-5)


def test_invalid_shift_position_rejected() -> None:
    """shift_position outside [0, T] must raise ValueError."""
    with pytest.raises(ValueError, match="shift_position"):
        BenchmarkConfig(experiment_name="task", sequence_length=100, shift_position=105)
    with pytest.raises(ValueError, match="shift_position"):
        BenchmarkConfig(experiment_name="task", sequence_length=100, shift_position=-1)


def test_invalid_controller_rejected() -> None:
    """Unknown controller name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown controller"):
        BenchmarkConfig(experiment_name="task", controller="unregistered_model")


def test_invalid_stability_mode_rejected() -> None:
    """Unknown stability mode must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown stability_mode"):
        BenchmarkConfig(experiment_name="task", stability_mode="hyper_safe")


def test_invalid_dtype_rejected() -> None:
    """Unsupported dtype must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown dtype"):
        BenchmarkConfig(experiment_name="task", dtype="int32")


def test_config_dict_roundtrip() -> None:
    """BenchmarkConfig should serialize and deserialize identically via dict."""
    cfg = BenchmarkConfig(
        experiment_name="shift_experiment",
        seed=123,
        sequence_length=64,
        key_dim=8,
        value_dim=8,
        shift_position=32,
        task_params={"tau": 0.5, "window": 4},
    )
    d = cfg.to_dict()
    assert isinstance(d, dict)
    reconstructed = BenchmarkConfig.from_dict(d)
    assert reconstructed == cfg
