# ==============================================================================
# DeltaCore: tests/test_chunked_equivalence.py
# Verification of exact numerical equivalence between Sequential and Chunked Affine scans.
# ==============================================================================

import pytest
import torch

from deltacore.scans.chunked import build_chunks, chunked_scan, run_chunk
from deltacore.scans.delta_affine import delta_to_affine
from deltacore.scans.sequential import sequential_scan
from deltacore.updates.delta import DeltaRule


class TestChunkedScanEquivalence:
    """Verifies that chunked affine execution matches sequential execution across all chunk configurations."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 16])
    def test_chunk_size_equivalence(self, dtype: torch.dtype, chunk_size: int) -> None:
        """Verify sequential final memory == chunked final memory for various chunk sizes."""
        gen = torch.Generator().manual_seed(42)
        seq_len = 16
        k_dim, v_dim = 4, 3
        step_size = 0.35

        keys = torch.randn(seq_len, k_dim, dtype=dtype, generator=gen)
        targets = torch.randn(seq_len, v_dim, dtype=dtype, generator=gen)
        M_0 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)

        # Sequential scan reference
        res_seq = sequential_scan(
            keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
        )
        M_seq_final = res_seq.final_memory.data

        # Construct affine operators
        ops = [
            delta_to_affine(keys[t], targets[t], step_size=step_size)
            for t in range(seq_len)
        ]

        # Chunked scan
        res_chunk = chunked_scan(
            M_0, ops, chunk_size=chunk_size, return_trajectory=True
        )

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(M_seq_final, res_chunk.final_memory, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_non_divisible_sequence_length(self, dtype: torch.dtype) -> None:
        """Verify T = 17, chunk_size = 5 works with exact equivalence."""
        gen = torch.Generator().manual_seed(123)
        seq_len = 17
        chunk_size = 5
        k_dim, v_dim = 3, 3
        step_size = 0.25

        keys = torch.randn(seq_len, k_dim, dtype=dtype, generator=gen)
        targets = torch.randn(seq_len, v_dim, dtype=dtype, generator=gen)
        M_0 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)

        res_seq = sequential_scan(
            keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
        )

        ops = [
            delta_to_affine(keys[t], targets[t], step_size=step_size)
            for t in range(seq_len)
        ]

        chunks = build_chunks(ops, chunk_size=chunk_size)
        assert len(chunks) == 4  # 5, 5, 5, 2
        assert chunks[-1].num_transitions == 2

        res_chunk = chunked_scan(
            M_0, ops, chunk_size=chunk_size, return_trajectory=True
        )

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(
            res_seq.final_memory.data, res_chunk.final_memory, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_full_trajectory_reconstruction(self, dtype: torch.dtype) -> None:
        """Verify sequential per-step memories == chunked reconstructed trajectory."""
        gen = torch.Generator().manual_seed(456)
        seq_len = 12
        chunk_size = 4
        k_dim, v_dim = 4, 3
        step_size = 0.3

        keys = torch.randn(seq_len, k_dim, dtype=dtype, generator=gen)
        targets = torch.randn(seq_len, v_dim, dtype=dtype, generator=gen)
        M_0 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)

        # Sequential step-by-step memory collection
        M_curr = M_0.clone()
        seq_trajectory = []
        for t in range(seq_len):
            rule_step = DeltaRule(step_size=step_size).step(M_curr, keys[t], targets[t])
            M_curr = rule_step.new_memory
            seq_trajectory.append(M_curr)

        # Chunked scan with full trajectory reconstruction
        ops = [
            delta_to_affine(keys[t], targets[t], step_size=step_size)
            for t in range(seq_len)
        ]
        res_chunk = chunked_scan(
            M_0, ops, chunk_size=chunk_size, return_trajectory=True
        )

        assert res_chunk.trajectory is not None
        assert len(res_chunk.trajectory) == seq_len

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        for t in range(seq_len):
            assert torch.allclose(
                seq_trajectory[t], res_chunk.trajectory[t], atol=tol, rtol=tol
            )

    def test_run_chunk_boundary_api(self) -> None:
        """Directly verify run_chunk advances state correctly across chunk boundary."""
        k_dim, v_dim = 3, 2
        ops = [
            delta_to_affine(torch.randn(k_dim), torch.randn(v_dim), step_size=0.2)
            for _ in range(5)
        ]
        chunk = build_chunks(ops, chunk_size=5)[0]

        M_0 = torch.randn(v_dim, k_dim)

        # Step by step manual advance
        M_step = M_0.clone()
        for op in ops:
            M_step = op.apply(M_step)

        # run_chunk advance
        M_chunk = run_chunk(M_0, chunk)

        assert torch.allclose(M_step, M_chunk, atol=1e-6)

    def test_adaptive_precomputed_coefficients_chunk_equivalence(self) -> None:
        """Verify Phase 2 precomputed adaptive step sizes eta_t match chunked scan."""
        gen = torch.Generator().manual_seed(789)
        seq_len = 15
        chunk_size = 4
        k_dim, v_dim = 3, 3

        keys = torch.randn(seq_len, k_dim, generator=gen)
        targets = torch.randn(seq_len, v_dim, generator=gen)
        # Precomputed variable step sizes (e.g. from input-conditioned controller)
        etas = 0.1 + 0.3 * torch.rand(seq_len, generator=gen)
        M_0 = torch.randn(v_dim, k_dim, generator=gen)

        # Sequential reference with variable eta_t
        M_seq = M_0.clone()
        for t in range(seq_len):
            step_res = DeltaRule().step(M_seq, keys[t], targets[t], step_size=etas[t])
            M_seq = step_res.new_memory

        # Chunked scan with precomputed variable eta_t
        ops = [
            delta_to_affine(keys[t], targets[t], step_size=etas[t])
            for t in range(seq_len)
        ]
        res_chunk = chunked_scan(M_0, ops, chunk_size=chunk_size)

        assert torch.allclose(M_seq, res_chunk.final_memory, atol=1e-5)
