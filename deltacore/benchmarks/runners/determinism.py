# ==============================================================================
# DeltaCore: deltacore/benchmarks/runners/determinism.py
# PRNG seeding and determinism management for reproducible benchmarks.
# ==============================================================================

import random
import sys

import torch


def set_seed(seed: int) -> torch.Generator:
    """Set PRNG seeds for Python random, PyTorch CPU, and CUDA (if available).

    Important Scientific Note:
        Deterministic PRNG seeding guarantees repeatable execution within identical
        hardware and software runtime environments. Cross-platform bit-identical
        results are NOT guaranteed across differing CPU architectures (e.g. x86_64
        vs. arm64) or GPU compute capabilities due to differing fused multiply-add
        (FMA) reductions and compiler vectorization.

    Args:
        seed: Non-negative integer random seed.

    Returns:
        Configured torch.Generator instance for deterministic tensor creation.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"Seed must be a non-negative integer, got {seed} (type {type(seed)})."
        )

    # 1. Python standard library
    random.seed(seed)

    # 2. NumPy (optional/lazy import if available)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # 3. PyTorch PRNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 4. Explicit generator
    gen = torch.Generator().manual_seed(seed)
    return gen


def get_environment_info() -> dict[str, str]:
    """Capture environment metadata for benchmark provenance."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    info["num_threads"] = str(torch.get_num_threads())
    return info
