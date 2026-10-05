# ==============================================================================
# DeltaCore: deltacore/benchmarks/runners/timing.py
# High-precision timing protocol with CUDA synchronization and warmup separation.
# ==============================================================================

import time
from collections.abc import Callable
from typing import Any

import torch


def measure_execution_time(
    fn: Callable[[], Any],
    repetitions: int = 5,
    warmup_count: int = 2,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Measure execution runtime according to standard benchmark protocol.

    Separates initial untimed warmup iterations from timed measurements.
    Enforces torch.cuda.synchronize() on CUDA devices.

    Args:
        fn: Zero-argument callable to benchmark.
        repetitions: Number of timed trials (must be >= 1).
        warmup_count: Number of untimed warmup iterations.
        device: Execution device.

    Returns:
        Dictionary containing:
            'median_ms': Median runtime across repetitions.
            'mean_ms': Mean runtime.
            'std_ms': Standard deviation of runtime.
            'min_ms': Fastest trial.
            'max_ms': Slowest trial.
            'warmup_ms': Runtime of the first warmup trial (cold start).
    """
    dev = torch.device(device)
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()

    # 1. Warmup / cold start measurement
    warmup_time = 0.0
    for w in range(warmup_count):
        if is_cuda:
            torch.cuda.synchronize(dev)
        t_w0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize(dev)
        if w == 0:
            warmup_time = (time.perf_counter() - t_w0) * 1000.0

    # 2. Repeated measurements
    trial_times: list[float] = []
    for _ in range(max(1, repetitions)):
        if is_cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize(dev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        trial_times.append(elapsed_ms)

    sorted_times = sorted(trial_times)
    n = len(sorted_times)
    mid = n // 2
    median_val = (
        sorted_times[mid]
        if n % 2 == 1
        else 0.5 * (sorted_times[mid - 1] + sorted_times[mid])
    )
    mean_val = sum(trial_times) / n
    variance = sum((x - mean_val) ** 2 for x in trial_times) / max(1, n - 1)
    std_val = variance**0.5

    return {
        "median_ms": float(median_val),
        "mean_ms": float(mean_val),
        "std_ms": float(std_val),
        "min_ms": float(min(trial_times)),
        "max_ms": float(max(trial_times)),
        "warmup_ms": float(warmup_time),
    }
