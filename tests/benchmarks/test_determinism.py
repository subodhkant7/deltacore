# ==============================================================================
# DeltaCore: tests/benchmarks/test_determinism.py
# Unit tests verifying PRNG seeding and deterministic data generation across tasks.
# ==============================================================================

import torch

from deltacore.benchmarks.runners.determinism import get_environment_info, set_seed
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.distribution_shift import DistributionShiftTask
from deltacore.benchmarks.tasks.stationary_recall import StationaryRecallTask


def test_set_seed_reproducibility() -> None:
    """Repeated set_seed calls with the same seed must produce identical random tensors."""
    gen1 = set_seed(42)
    t1 = torch.randn(10, generator=gen1)

    gen2 = set_seed(42)
    t2 = torch.randn(10, generator=gen2)

    assert torch.equal(t1, t2)


def test_different_seeds_produce_different_tensors() -> None:
    """Different seeds must produce different pseudo-random sequences."""
    gen1 = set_seed(10)
    t1 = torch.randn(10, generator=gen1)

    gen2 = set_seed(20)
    t2 = torch.randn(10, generator=gen2)

    assert not torch.equal(t1, t2)


def test_task_data_generation_determinism() -> None:
    """Task data generation with identical config must produce identical keys and targets."""
    cfg = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=101,
        sequence_length=32,
        key_dim=8,
        value_dim=8,
    )
    task = StationaryRecallTask()
    k1, v1, _ = task.generate_data(cfg)
    k2, v2, _ = task.generate_data(cfg)

    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


def test_distribution_shift_task_determinism() -> None:
    """DistributionShiftTask data generation is deterministic for identical seeds."""
    cfg = BenchmarkConfig(
        experiment_name="distribution_shift",
        seed=77,
        sequence_length=40,
        key_dim=8,
        value_dim=8,
        shift_position=20,
    )
    task = DistributionShiftTask()
    k1, v1, _ = task.generate_data(cfg)
    k2, v2, _ = task.generate_data(cfg)

    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


def test_environment_info_metadata() -> None:
    """Environment info returns valid provenance metadata."""
    info = get_environment_info()
    assert "python_version" in info
    assert "torch_version" in info
    assert "cuda_available" in info
    assert "num_threads" in info
