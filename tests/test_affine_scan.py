# ==============================================================================
# DeltaCore: tests/test_affine_scan.py
# Unit and property tests for AffineScanOperator and associative composition algebra.
# ==============================================================================

import pytest
import torch

from deltacore.scans.affine import AffineScanMetadata, AffineScanOperator


class TestAffineScanOperator:
    """Mathematical and property tests for affine scan algebra."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_associativity_property(self, dtype: torch.dtype) -> None:
        r"""Property: (F_1 \otimes F_2) \otimes F_3 == F_1 \otimes (F_2 \otimes F_3)."""
        gen = torch.Generator().manual_seed(42)
        v_dim = 3
        k_dim = 4

        A1 = torch.randn(k_dim, k_dim, dtype=dtype, generator=gen)
        B1 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)
        F1 = AffineScanOperator(A1, B1)

        A2 = torch.randn(k_dim, k_dim, dtype=dtype, generator=gen)
        B2 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)
        F2 = AffineScanOperator(A2, B2)

        A3 = torch.randn(k_dim, k_dim, dtype=dtype, generator=gen)
        B3 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)
        F3 = AffineScanOperator(A3, B3)

        # Left-associated
        F12_3 = (F1.compose(F2)).compose(F3)

        # Right-associated
        F1_23 = F1.compose(F2.compose(F3))

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(F12_3.A, F1_23.A, atol=tol, rtol=tol)
        assert torch.allclose(F12_3.B, F1_23.B, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_composition_matches_sequential_application(
        self, dtype: torch.dtype
    ) -> None:
        r"""Property: (F_1 \otimes F_2)(M) == F_2(F_1(M))."""
        gen = torch.Generator().manual_seed(101)
        v_dim = 3
        k_dim = 4

        A1 = torch.randn(k_dim, k_dim, dtype=dtype, generator=gen)
        B1 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)
        F1 = AffineScanOperator(A1, B1)

        A2 = torch.randn(k_dim, k_dim, dtype=dtype, generator=gen)
        B2 = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)
        F2 = AffineScanOperator(A2, B2)

        M = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)

        # Path 1: sequential application
        M_seq = F2.apply(F1.apply(M))

        # Path 2: apply composed operator
        F12 = F1.compose(F2)
        M_comp = F12.apply(M)

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(M_seq, M_comp, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_three_operator_composition_matches_sequential(
        self, dtype: torch.dtype
    ) -> None:
        r"""Property: ((F_1 \otimes F_2) \otimes F_3)(M) == F_3(F_2(F_1(M)))."""
        gen = torch.Generator().manual_seed(202)
        v_dim = 4
        k_dim = 4

        F1 = AffineScanOperator(
            torch.randn(k_dim, k_dim, dtype=dtype, generator=gen),
            torch.randn(v_dim, k_dim, dtype=dtype, generator=gen),
        )
        F2 = AffineScanOperator(
            torch.randn(k_dim, k_dim, dtype=dtype, generator=gen),
            torch.randn(v_dim, k_dim, dtype=dtype, generator=gen),
        )
        F3 = AffineScanOperator(
            torch.randn(k_dim, k_dim, dtype=dtype, generator=gen),
            torch.randn(v_dim, k_dim, dtype=dtype, generator=gen),
        )

        M = torch.randn(v_dim, k_dim, dtype=dtype, generator=gen)

        M_seq = F3.apply(F2.apply(F1.apply(M)))
        F123 = F1.compose(F2).compose(F3)
        M_comp = F123.apply(M)

        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(M_seq, M_comp, atol=tol, rtol=tol)

    def test_identity_operator_properties(self) -> None:
        r"""Property: I \otimes F == F, F \otimes I == F, and I(M) == M."""
        v_dim = 3
        k_dim = 4
        ident = AffineScanOperator.identity(k_dim=k_dim, v_dim=v_dim)

        F = AffineScanOperator(
            torch.randn(k_dim, k_dim),
            torch.randn(v_dim, k_dim),
        )
        M = torch.randn(v_dim, k_dim)

        # I(M) == M
        assert torch.allclose(ident.apply(M), M, atol=1e-7)

        # Left identity
        left_id = ident.compose(F)
        assert torch.allclose(left_id.A, F.A, atol=1e-7)
        assert torch.allclose(left_id.B, F.B, atol=1e-7)

        # Right identity
        right_id = F.compose(ident)
        assert torch.allclose(right_id.A, F.A, atol=1e-7)
        assert torch.allclose(right_id.B, F.B, atol=1e-7)

    def test_batched_affine_operator(self) -> None:
        """Verify batched [B, K, K] operator composition and application."""
        batch_size = 2
        v_dim = 3
        k_dim = 3

        A1 = torch.randn(batch_size, k_dim, k_dim)
        B1 = torch.randn(batch_size, v_dim, k_dim)
        F1 = AffineScanOperator(A1, B1)

        A2 = torch.randn(batch_size, k_dim, k_dim)
        B2 = torch.randn(batch_size, v_dim, k_dim)
        F2 = AffineScanOperator(A2, B2)

        M = torch.randn(batch_size, v_dim, k_dim)

        # Composed vs sequential
        M_seq = F2.apply(F1.apply(M))
        M_comp = F1.compose(F2).apply(M)

        assert torch.allclose(M_seq, M_comp, atol=1e-6)

    def test_stability_metadata_preservation(self) -> None:
        """Composed operators accumulate clipped_count and track min margin."""
        F1 = AffineScanOperator(
            torch.eye(3),
            torch.zeros(2, 3),
            metadata=AffineScanMetadata(
                min_stability_margin=1.2,
                max_normalized_step=0.8,
                clipped_count=1,
            ),
        )
        F2 = AffineScanOperator(
            torch.eye(3),
            torch.zeros(2, 3),
            metadata=AffineScanMetadata(
                min_stability_margin=0.9,
                max_normalized_step=1.1,
                clipped_count=2,
            ),
        )

        F12 = F1.compose(F2)
        assert F12.metadata.clipped_count == 3
        assert F12.metadata.min_stability_margin == 0.9
        assert F12.metadata.max_normalized_step == 1.1
