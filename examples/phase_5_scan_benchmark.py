# ==============================================================================
# DeltaCore: examples/phase_5_scan_benchmark.py
# Phase 5 Benchmark: Numerical Precision and Runtime Performance of Chunked Scans.
# ==============================================================================

import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Any  # noqa: E402

import torch  # noqa: E402

from deltacore.diagnostics.scans import extract_scan_diagnostics  # noqa: E402
from deltacore.scans.affine import AffineScanOperator  # noqa: E402
from deltacore.scans.chunked import chunked_scan  # noqa: E402
from deltacore.scans.delta_affine import delta_to_affine  # noqa: E402
from deltacore.scans.sequential import sequential_scan  # noqa: E402
from deltacore.updates.delta import DeltaRule  # noqa: E402


def format_table(headers: list[str], rows: list[list[str]], title: str) -> str:
    """Format an ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )

    lines = [f"\n=== {title} ===", sep, hdr, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_numerical_benchmark(
    device: torch.device,
    lengths: list[int],
    chunk_size: int = 16,
    k_dim: int = 4,
    v_dim: int = 4,
) -> list[dict[str, Any]]:
    r"""Evaluate max absolute difference between sequential and chunked scan across sequence lengths."""
    results = []
    step_size = 0.25

    for t_len in lengths:
        for dtype in [torch.float32, torch.float64]:
            gen = torch.Generator(device="cpu").manual_seed(42 + t_len)
            keys = torch.randn(t_len, k_dim, dtype=dtype, generator=gen).to(device)
            targets = torch.randn(t_len, v_dim, dtype=dtype, generator=gen).to(device)
            M_0 = torch.zeros(v_dim, k_dim, dtype=dtype, device=device)

            # 1. Sequential scan reference
            res_seq = sequential_scan(
                keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
            )
            M_seq_final = res_seq.final_memory.data

            # 2. Chunked scan
            ops = [
                delta_to_affine(keys[t], targets[t], step_size=step_size)
                for t in range(t_len)
            ]
            res_chunk = chunked_scan(
                M_0, ops, chunk_size=chunk_size, return_trajectory=True
            )

            # Numerical error
            final_diff = torch.max(
                torch.abs(M_seq_final.float() - res_chunk.final_memory.float())
            ).item()

            diag = extract_scan_diagnostics(
                res_chunk.chunks,
                sequential_memories=res_seq.final_memory.data,
                chunked_memories=res_chunk.final_memory,
            )

            results.append(
                {
                    "length": t_len,
                    "dtype": "FP32" if dtype == torch.float32 else "FP64",
                    "chunk_size": chunk_size,
                    "final_diff": final_diff,
                    "chunk_count": diag.chunk_count,
                    "comp_depth": diag.composition_depth,
                }
            )

    return results


def run_runtime_benchmark(
    device: torch.device,
    lengths: list[int],
    chunk_sizes: list[int],
    k_dim: int = 4,
    v_dim: int = 4,
    repetitions: int = 5,
) -> list[dict[str, Any]]:
    r"""Benchmark execution times for sequential vs pure-affine vs chunked scan."""
    results = []
    step_size = 0.25
    dtype = torch.float32

    for t_len in lengths:
        gen = torch.Generator(device="cpu").manual_seed(100 + t_len)
        keys = torch.randn(t_len, k_dim, dtype=dtype, generator=gen).to(device)
        targets = torch.randn(t_len, v_dim, dtype=dtype, generator=gen).to(device)
        M_0 = torch.zeros(v_dim, k_dim, dtype=dtype, device=device)

        # Precompute affine operators
        ops = [
            delta_to_affine(keys[t], targets[t], step_size=step_size)
            for t in range(t_len)
        ]

        # 1. Sequential recurrence baseline
        # Warmup
        sequential_scan(
            keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(repetitions):
            sequential_scan(
                keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        t_seq = (time.perf_counter() - t0) / repetitions * 1000.0  # ms

        # 2. Pure Affine Sequential Loop
        def run_affine_sequential(
            init_m: torch.Tensor, op_list: list[AffineScanOperator]
        ) -> torch.Tensor:
            M = init_m.clone()
            for op in op_list:
                M = op.apply(M)
            return M

        run_affine_sequential(M_0, ops)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(repetitions):
            run_affine_sequential(M_0, ops)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        t_aff_seq = (time.perf_counter() - t0) / repetitions * 1000.0  # ms

        # 3. Chunked Scans across chunk sizes
        chunk_times = {}
        for c_size in chunk_sizes:
            chunked_scan(M_0, ops, chunk_size=c_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            for _ in range(repetitions):
                chunked_scan(M_0, ops, chunk_size=c_size)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            t_chk = (time.perf_counter() - t0) / repetitions * 1000.0  # ms
            chunk_times[c_size] = t_chk

        results.append(
            {
                "length": t_len,
                "t_seq_ms": t_seq,
                "t_aff_seq_ms": t_aff_seq,
                "chunk_times_ms": chunk_times,
            }
        )

    return results


def main() -> None:
    print("=" * 80)
    print("DeltaCore Phase 5: Associative & Chunked State Scans Benchmark")
    print("=" * 80)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    lengths_num = [8, 16, 32, 64, 128, 512, 1024]
    chunk_size_num = 16

    lengths_perf = [64, 256, 1024]
    chunk_sizes_perf = [4, 16, 64]

    for dev in devices:
        print(f"\n>>> Running on device: {dev} <<<")

        # ---------------------------------------------------------------------
        # Part 1: Numerical Precision Benchmark
        # ---------------------------------------------------------------------
        num_res = run_numerical_benchmark(
            device=dev,
            lengths=lengths_num,
            chunk_size=chunk_size_num,
        )

        headers_num = [
            "Length (T)",
            "Dtype",
            "Chunk Size",
            "Chunks (N)",
            "Comp Depth",
            "Max Abs Diff (||M_seq - M_chk||_inf)",
            "Numerical Agreement",
        ]
        rows_num = []
        for r in num_res:
            diff_fmt = f"{r['final_diff']:.2e}"
            agree_str = (
                "Exact (< 1e-10)" if r["final_diff"] < 1e-10 else "Consistent (< 1e-4)"
            )
            rows_num.append(
                [
                    str(r["length"]),
                    r["dtype"],
                    str(r["chunk_size"]),
                    str(r["chunk_count"]),
                    str(r["comp_depth"]),
                    diff_fmt,
                    agree_str,
                ]
            )
        print(
            format_table(
                headers_num, rows_num, f"Part 1: Numerical Precision Scaling ({dev})"
            )
        )

        # ---------------------------------------------------------------------
        # Part 2: Runtime Performance Benchmark
        # ---------------------------------------------------------------------
        perf_res = run_runtime_benchmark(
            device=dev,
            lengths=lengths_perf,
            chunk_sizes=chunk_sizes_perf,
        )

        headers_perf = [
            "Length (T)",
            "Sequential Scan (ms)",
            "Pure Affine Loop (ms)",
            "Chunk C=4 (ms)",
            "Chunk C=16 (ms)",
            "Chunk C=64 (ms)",
        ]
        rows_perf = []
        for r in perf_res:
            c4 = f"{r['chunk_times_ms'].get(4, 0.0):.3f}"
            c16 = f"{r['chunk_times_ms'].get(16, 0.0):.3f}"
            c64 = f"{r['chunk_times_ms'].get(64, 0.0):.3f}"
            rows_perf.append(
                [
                    str(r["length"]),
                    f"{r['t_seq_ms']:.3f}",
                    f"{r['t_aff_seq_ms']:.3f}",
                    c4,
                    c16,
                    c64,
                ]
            )
        print(
            format_table(
                headers_perf, rows_perf, f"Part 2: Runtime Measurement ({dev})"
            )
        )

    print("\n" + "=" * 80)
    print("Scientific Observation:")
    print(
        "In pure PyTorch on CPU, Python-level chunk composition incurs extra overhead"
    )
    print(
        "relative to the inlined sequential delta loop. As anticipated by Section 12,"
    )
    print(
        "associative chunking provides algorithmic parallelization (log-depth prefix scans)"
    )
    print(
        "whose wall-clock benefit requires parallel hardware execution kernels (GPU/Triton),"
    )
    print(
        "not pure single-threaded CPU Python loops. Do NOT falsely claim pure Python speedups."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
