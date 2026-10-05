"""Unit and property tests for BoundaryRefreshChunkScan semantics.

Verifies:
1. Exact Equivalence Invariant at C=1:
   BoundaryRefreshChunkScan(C=1) == FiveMemorySystem.scan() exactly (token-for-token).
2. Algorithmic Sensitivity at C > 1:
   For C in {2, 4, T}, boundary staleness produces real computed trajectory differences.
3. Determinism:
   Repeated execution produces bitwise identical trajectories.
4. Non-mutation of input states and input tensors.
5. Boundary state refresh correctness.
"""

import pytest
import torch

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.scans.boundary_chunked import BoundaryRefreshChunkScan
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem


def _create_system_and_state(
    v_dim: int = 4,
    k_dim: int = 4,
    in_dim: int = 6,
    batch_size: int | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
) -> tuple[FiveMemorySystem, FiveMemoryState]:
    """Helper to initialize deterministic FiveMemorySystem and initial state."""
    torch.manual_seed(seed)
    cfg = FiveMemoryConfig(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        feat_dim=in_dim,
        lr_dim=2,
        ret_dim=2,
        eta_max=0.5,
        eta_bias=0.0,
        ret_min=0.1,
        ret_bias=1.5,
        eta_key=0.05,
        lambda_key=0.98,
        eta_val=0.05,
        lambda_val=0.98,
        rho_eta=0.05,
        lambda_eta=0.98,
        rho_ret=0.05,
        lambda_ret=0.98,
        beta=1.9,
        epsilon=1e-6,
        apply_stability_control=True,
    )
    system = FiveMemorySystem(cfg)
    state = FiveMemoryState.initialize(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        feat_dim=in_dim,
        lr_dim=2,
        ret_dim=2,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(seed),
        dtype=dtype,
    )
    return system, state


@pytest.mark.parametrize("is_batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_boundary_chunk_c1_exact_equivalence(
    is_batched: bool, dtype: torch.dtype
) -> None:
    """Verify BoundaryRefreshChunkScan(C=1) matches FiveMemorySystem.scan() token-for-token."""
    b_size = 2 if is_batched else None
    system, state = _create_system_and_state(batch_size=b_size, dtype=dtype, seed=101)

    T = 8
    in_dim = 6
    torch.manual_seed(202)
    shape = (2, T, in_dim) if is_batched else (T, in_dim)
    inputs = torch.randn(*shape, dtype=dtype)

    # 1. Reference sequential scan
    ref_result = system.scan(inputs, state)

    # 2. Boundary refresh chunk scan with C=1
    chunk_scan = BoundaryRefreshChunkScan(system, default_chunk_size=1)
    chunk_result = chunk_scan.scan(inputs, state, chunk_size=1)

    # Invariant: C=1 must match reference exactly
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(
        chunk_result.predictions, ref_result.predictions, atol=tol, rtol=tol
    )
    assert torch.allclose(chunk_result.errors, ref_result.errors, atol=tol, rtol=tol)
    assert torch.allclose(chunk_result.keys, ref_result.keys, atol=tol, rtol=tol)
    assert torch.allclose(chunk_result.values, ref_result.values, atol=tol, rtol=tol)
    assert torch.allclose(
        chunk_result.safe_learning_rates,
        ref_result.safe_learning_rates,
        atol=tol,
        rtol=tol,
    )
    assert torch.allclose(
        chunk_result.safe_retentions, ref_result.safe_retentions, atol=tol, rtol=tol
    )

    # Verify final memory states match exactly
    assert torch.allclose(
        chunk_result.final_state.content,
        ref_result.final_state.content,
        atol=tol,
        rtol=tol,
    )
    assert torch.allclose(
        chunk_result.final_state.key, ref_result.final_state.key, atol=tol, rtol=tol
    )
    assert torch.allclose(
        chunk_result.final_state.value, ref_result.final_state.value, atol=tol, rtol=tol
    )


def test_chunk_size_sensitivity_synthetic_sequence() -> None:
    """Verify chunk size visibly and systematically affects trajectory for C > 1.

    Construct a deliberately state-sensitive sequence where token representations
    cause adaptation of key, value, and meta-controllers.
    Compare C = 1, 2, 4, 16.
    """
    T = 16
    in_dim = 6
    v_dim = 4
    k_dim = 4
    system, state = _create_system_and_state(
        v_dim=v_dim,
        k_dim=k_dim,
        in_dim=in_dim,
        batch_size=2,
        dtype=torch.float64,
        seed=303,
    )

    # Deliberately dynamic synthetic inputs
    torch.manual_seed(404)
    inputs = torch.randn(2, T, in_dim, dtype=torch.float64)

    chunk_scan = BoundaryRefreshChunkScan(system)

    # Run C = 1, 2, 4, 16
    res_c1 = chunk_scan.scan(inputs, state, chunk_size=1)
    res_c2 = chunk_scan.scan(inputs, state, chunk_size=2)
    res_c4 = chunk_scan.scan(inputs, state, chunk_size=4)
    res_c16 = chunk_scan.scan(inputs, state, chunk_size=16)

    # 1. Verification of C=1 vs C=2
    # Predictions at t=0 should match because boundary state at t=0 is identical
    assert torch.allclose(
        res_c1.predictions[:, 0, :], res_c2.predictions[:, 0, :], atol=1e-12
    )
    # At t=1, C=1 updated boundary state for generating k_1, but C=2 still used S_0
    # Therefore, k_1 must differ between C=1 and C=2
    k1_c1 = res_c1.keys[:, 1, :]
    k1_c2 = res_c2.keys[:, 1, :]
    diff_k1 = torch.norm(k1_c1 - k1_c2).item()
    assert diff_k1 > 1e-4, f"Expected non-zero key difference at t=1, got {diff_k1}"

    # 2. Divergence of final content memory across chunk sizes
    diff_final_c2 = torch.norm(
        res_c1.final_state.content - res_c2.final_state.content
    ).item()
    diff_final_c4 = torch.norm(
        res_c1.final_state.content - res_c4.final_state.content
    ).item()
    diff_final_c16 = torch.norm(
        res_c1.final_state.content - res_c16.final_state.content
    ).item()

    assert diff_final_c2 > 1e-6, (
        f"C=2 final state must differ from C=1, got {diff_final_c2}"
    )
    assert diff_final_c4 > 1e-6, (
        f"C=4 final state must differ from C=1, got {diff_final_c4}"
    )
    assert diff_final_c16 > 1e-6, (
        f"C=16 final state must differ from C=1, got {diff_final_c16}"
    )

    # 3. Check number of boundary states recorded
    assert len(res_c1.boundary_states) == T
    assert len(res_c2.boundary_states) == T // 2
    assert len(res_c4.boundary_states) == T // 4
    assert len(res_c16.boundary_states) == 1

    # 4. Check chunk boundary indices
    assert res_c1.chunk_boundaries == list(range(T))
    assert res_c2.chunk_boundaries == [0, 2, 4, 6, 8, 10, 12, 14]
    assert res_c4.chunk_boundaries == [0, 4, 8, 12]
    assert res_c16.chunk_boundaries == [0]


def test_boundary_chunk_determinism_and_non_mutation() -> None:
    """Verify repeated execution is bitwise deterministic and does not mutate initial state."""
    system, state = _create_system_and_state(
        batch_size=2, dtype=torch.float64, seed=505
    )
    content_init_clone = state.content.clone()
    key_init_clone = state.key.clone()

    inputs = torch.randn(2, 8, 6, dtype=torch.float64)
    inputs_clone = inputs.clone()

    chunk_scan = BoundaryRefreshChunkScan(system, default_chunk_size=4)

    res1 = chunk_scan.scan(inputs, state)
    res2 = chunk_scan.scan(inputs, state)

    # Determinism
    assert torch.equal(res1.predictions, res2.predictions)
    assert torch.equal(res1.final_state.content, res2.final_state.content)

    # Non-mutation
    assert torch.equal(state.content, content_init_clone)
    assert torch.equal(state.key, key_init_clone)
    assert torch.equal(inputs, inputs_clone)
