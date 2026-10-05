# ==============================================================================
# DeltaCore: tests/test_affine_equivalence.py
# Verification of exact numerical equivalence between SequentialScan and AffineScan.
# ==============================================================================

import pytest
import torch

from deltacore.memory.read import read
from deltacore.scans.delta_affine import delta_to_affine
from deltacore.scans.sequential import sequential_scan
from deltacore.updates.delta import DeltaRule


class TestAffineSequentialEquivalence:
    """Verifies that affine operator representation matches DeltaRule sequential scan exactly."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("k_dim, v_dim", [(3, 2), (4, 4), (8, 6)])
    def test_random_sequence_equivalence(
        self, dtype: torch.dtype, k_dim: int, v_dim: int
    ) -> None:
        """Sequential DeltaRule matches sequential Affine application for random sequences."""
        gen = torch.Generator().manual_seed(42)
        seq_len = 15
        step_size = 0.4

        keys = torch.randn(seq_len, k_dim, dtype=dtype, generator=gen)
        targets = torch.randn(seq_len, v_dim, dtype=dtype, generator=gen)
        M_0 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)

        # Path A: sequential_scan
        res_seq = sequential_scan(
            keys,
            targets,
            initial_memory=M_0,
            rule=DeltaRule(step_size=step_size),
        )
        final_M_seq = res_seq.final_memory.data
        preds_seq = res_seq.predictions

        # Path B: sequentially apply delta_to_affine operators
        M_curr = M_0.clone()
        preds_affine = []
        for t in range(seq_len):
            # Prediction before update: M_t @ k_t
            preds_affine.append(read(M_curr, keys[t]))
            op_t = delta_to_affine(keys[t], targets[t], step_size=step_size)
            M_curr = op_t.apply(M_curr)

        preds_affine_tensor = torch.stack(preds_affine, dim=0)

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(final_M_seq, M_curr, atol=tol, rtol=tol)
        assert torch.allclose(preds_seq, preds_affine_tensor, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_zero_keys_equivalence(self, dtype: torch.dtype) -> None:
        """Zero keys produce identity transitions and identical predictions."""
        v_dim, k_dim = 3, 3
        seq_len = 5
        step_size = 0.5

        keys = torch.zeros(seq_len, k_dim, dtype=dtype)
        targets = torch.randn(seq_len, v_dim, dtype=dtype)
        M_0 = torch.randn(v_dim, k_dim, dtype=dtype)

        res_seq = sequential_scan(
            keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
        )

        M_curr = M_0.clone()
        for t in range(seq_len):
            op = delta_to_affine(keys[t], targets[t], step_size=step_size)
            M_curr = op.apply(M_curr)

        tol = 1e-12 if dtype == torch.float64 else 1e-6
        assert torch.allclose(res_seq.final_memory.data, M_curr, atol=tol)
        assert torch.allclose(M_curr, M_0, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_repeated_keys_conflicting_targets_equivalence(
        self, dtype: torch.dtype
    ) -> None:
        """Repeated identical keys with conflicting targets match across both paths."""
        v_dim, k_dim = 3, 3
        seq_len = 8
        step_size = 0.3

        key_single = torch.tensor([1.0, -1.0, 0.5], dtype=dtype)
        keys = key_single.unsqueeze(0).expand(seq_len, -1)
        targets = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [2.0, -2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=dtype,
        )
        M_0 = torch.zeros(v_dim, k_dim, dtype=dtype)

        res_seq = sequential_scan(
            keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
        )

        M_curr = M_0.clone()
        for t in range(seq_len):
            op = delta_to_affine(keys[t], targets[t], step_size=step_size)
            M_curr = op.apply(M_curr)

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(res_seq.final_memory.data, M_curr, atol=tol, rtol=tol)

    def test_correlated_keys_equivalence(self) -> None:
        """Highly correlated keys match across both paths."""
        dtype = torch.float64
        v_dim, k_dim = 4, 4
        seq_len = 10
        step_size = 0.2

        base = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        # Correlated keys with small perturbation
        keys = base.unsqueeze(0) + 0.01 * torch.randn(seq_len, k_dim, dtype=dtype)
        targets = torch.randn(seq_len, v_dim, dtype=dtype)
        M_0 = torch.randn(v_dim, k_dim, dtype=dtype)

        res_seq = sequential_scan(
            keys, targets, initial_memory=M_0, rule=DeltaRule(step_size=step_size)
        )

        M_curr = M_0.clone()
        for t in range(seq_len):
            op = delta_to_affine(keys[t], targets[t], step_size=step_size)
            M_curr = op.apply(M_curr)

        assert torch.allclose(res_seq.final_memory.data, M_curr, atol=1e-12)
