# ==============================================================================
# DeltaCore: tests/test_scan_autograd.py
# Autograd compatibility and gradient equivalence between sequential and chunked scans.
# ==============================================================================

import pytest
import torch
from torch.autograd import gradcheck

from deltacore.scans.affine import AffineScanOperator
from deltacore.scans.chunked import chunked_scan
from deltacore.scans.delta_affine import delta_to_affine
from deltacore.updates.delta import DeltaRule


class TestScanAutograd:
    """Verifies differentiability, gradchecks, and sequential/chunked gradient equivalence."""

    def test_affine_apply_autograd_and_gradcheck(self) -> None:
        """Verify autograd and gradcheck through AffineScanOperator.apply."""
        v_dim, k_dim = 2, 2
        dtype = torch.float64

        A = torch.randn(k_dim, k_dim, dtype=dtype, requires_grad=True)
        B = torch.randn(v_dim, k_dim, dtype=dtype, requires_grad=True)
        M = torch.randn(v_dim, k_dim, dtype=dtype, requires_grad=True)

        def apply_fn(
            m_in: torch.Tensor, a_in: torch.Tensor, b_in: torch.Tensor
        ) -> torch.Tensor:
            op = AffineScanOperator(a_in, b_in)
            return op.apply(m_in)

        assert gradcheck(apply_fn, (M, A, B), eps=1e-6, atol=1e-5)

    def test_affine_compose_autograd_and_gradcheck(self) -> None:
        """Verify gradcheck through AffineScanOperator.compose."""
        v_dim, k_dim = 2, 2
        dtype = torch.float64

        A1 = torch.randn(k_dim, k_dim, dtype=dtype, requires_grad=True)
        B1 = torch.randn(v_dim, k_dim, dtype=dtype, requires_grad=True)
        A2 = torch.randn(k_dim, k_dim, dtype=dtype, requires_grad=True)
        B2 = torch.randn(v_dim, k_dim, dtype=dtype, requires_grad=True)

        def compose_a_fn(
            a1: torch.Tensor, b1: torch.Tensor, a2: torch.Tensor, b2: torch.Tensor
        ) -> torch.Tensor:
            f1 = AffineScanOperator(a1, b1)
            f2 = AffineScanOperator(a2, b2)
            return f1.compose(f2).A

        def compose_b_fn(
            a1: torch.Tensor, b1: torch.Tensor, a2: torch.Tensor, b2: torch.Tensor
        ) -> torch.Tensor:
            f1 = AffineScanOperator(a1, b1)
            f2 = AffineScanOperator(a2, b2)
            return f1.compose(f2).B

        assert gradcheck(compose_a_fn, (A1, B1, A2, B2), eps=1e-6, atol=1e-5)
        assert gradcheck(compose_b_fn, (A1, B1, A2, B2), eps=1e-6, atol=1e-5)

    @pytest.mark.parametrize("chunk_size", [1, 2, 4])
    def test_sequential_vs_chunked_gradient_equivalence(self, chunk_size: int) -> None:
        """Verify gradients through sequential update match gradients through chunked scan."""
        dtype = torch.float64
        seq_len = 4
        k_dim, v_dim = 2, 2
        step_size = 0.3

        # Inputs for sequential path
        M0_seq = torch.randn(v_dim, k_dim, dtype=dtype, requires_grad=True)
        keys_seq = torch.randn(seq_len, k_dim, dtype=dtype, requires_grad=True)
        targets_seq = torch.randn(seq_len, v_dim, dtype=dtype, requires_grad=True)

        # Clone exactly for chunked path
        M0_chk = M0_seq.detach().clone().requires_grad_(True)
        keys_chk = keys_seq.detach().clone().requires_grad_(True)
        targets_chk = targets_seq.detach().clone().requires_grad_(True)

        # Path 1: sequential unrolling
        M_curr = M0_seq
        rule = DeltaRule(step_size=step_size)
        for t in range(seq_len):
            step_res = rule.step(M_curr, keys_seq[t], targets_seq[t])
            M_curr = step_res.new_memory
        loss_seq = (M_curr**2).sum()
        loss_seq.backward()

        # Path 2: chunked scan
        ops_chk = [
            delta_to_affine(keys_chk[t], targets_chk[t], step_size=step_size)
            for t in range(seq_len)
        ]
        res_chk = chunked_scan(M0_chk, ops_chk, chunk_size=chunk_size)
        loss_chk = (res_chk.final_memory**2).sum()
        loss_chk.backward()

        assert M0_seq.grad is not None and M0_chk.grad is not None
        assert torch.allclose(M0_seq.grad, M0_chk.grad, atol=1e-10)
        assert keys_seq.grad is not None and keys_chk.grad is not None
        assert torch.allclose(keys_seq.grad, keys_chk.grad, atol=1e-10)
        assert targets_seq.grad is not None and targets_chk.grad is not None
        assert torch.allclose(targets_seq.grad, targets_chk.grad, atol=1e-10)
