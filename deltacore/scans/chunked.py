# ==============================================================================
# DeltaCore: deltacore/scans/chunked.py
# Chunked associative scan operator, boundary state execution, and prefix reconstruction.
# ==============================================================================

from dataclasses import dataclass

import torch

from deltacore.scans.affine import AffineScanOperator
from deltacore.scans.prefix import prefix_scan_memory


@dataclass(frozen=True)
class AffineChunk:
    r"""A contiguous block of affine transitions composed into a single operator.

    Attributes:
        start_index: Inclusive starting time index $t_{\text{start}}$.
        end_index: Exclusive ending time index $t_{\text{end}}$.
        operator: Composed AffineScanOperator $F_{\text{start}} \otimes \dots \otimes F_{\text{end}-1}$.
    """

    start_index: int
    end_index: int
    operator: AffineScanOperator

    @property
    def num_transitions(self) -> int:
        """Number of individual token transitions in this chunk."""
        return self.end_index - self.start_index

    @property
    def A(self) -> torch.Tensor:
        """Composed transition matrix."""
        return self.operator.A

    @property
    def B(self) -> torch.Tensor:
        """Composed affine drive matrix."""
        return self.operator.B

    @property
    def metadata(self):
        """Preserved stability diagnostics."""
        return self.operator.metadata


@dataclass(frozen=True)
class ChunkedScanResult:
    r"""Output of a chunked associative scan.

    Attributes:
        final_memory: Memory state $M_T$ after consuming all chunks.
        boundary_memories: List of memory states at chunk boundaries $[M_{t_1}, M_{t_2}, \dots, M_T]$.
        chunks: List of AffineChunk instances.
        trajectory: Full step-by-step reconstructed memory trajectory $[M_1, \dots, M_T]$ if requested.
    """

    final_memory: torch.Tensor
    boundary_memories: list[torch.Tensor]
    chunks: list[AffineChunk]
    trajectory: list[torch.Tensor] | None = None


def build_chunks(
    operators: list[AffineScanOperator],
    chunk_size: int,
) -> list[AffineChunk]:
    r"""Partition a sequence of affine operators into composed chunks.

    Supports arbitrary chunk sizes $C \ge 1$, including:
    - $C = 1$: identity chunking (single operator per chunk).
    - $C \ge T$: single chunk spanning the entire sequence.
    - Non-divisible $T$: final chunk has length $T \pmod C$.

    Args:
        operators: List of AffineScanOperator instances $[F_0, \dots, F_{T-1}]$.
        chunk_size: Number of transitions per chunk ($C \ge 1$).

    Returns:
        List of AffineChunk instances.

    Raises:
        ValueError: If chunk_size < 1.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    if not operators:
        return []

    t_total = len(operators)
    chunks: list[AffineChunk] = []

    for start_idx in range(0, t_total, chunk_size):
        end_idx = min(start_idx + chunk_size, t_total)
        slice_ops = operators[start_idx:end_idx]

        # Compose slice
        composed = slice_ops[0]
        for op in slice_ops[1:]:
            composed = composed.compose(op)

        chunks.append(
            AffineChunk(
                start_index=start_idx,
                end_index=end_idx,
                operator=composed,
            )
        )

    return chunks


def run_chunk(
    initial_state: torch.Tensor,
    chunk: AffineChunk | AffineScanOperator,
) -> torch.Tensor:
    r"""Reference execution path advancing a memory state across a chunk boundary.

    Mathematical contract:
        $$M_{\text{out}} = F_{\text{chunk}}(M_{\text{in}}) = M_{\text{in}} A_{\text{chunk}} + B_{\text{chunk}}$$

    Args:
        initial_state: Memory state tensor $M_{\text{in}} \in \mathbb{R}^{V \times K}$ or `[B, V, K]`.
        chunk: AffineChunk or AffineScanOperator.

    Returns:
        Advanced memory state tensor $M_{\text{out}}$ at the end of the chunk.
    """
    op = chunk.operator if isinstance(chunk, AffineChunk) else chunk
    return op.apply(initial_state)


def chunked_scan(
    initial_memory: torch.Tensor,
    operators: list[AffineScanOperator],
    chunk_size: int,
    return_trajectory: bool = False,
) -> ChunkedScanResult:
    r"""Execute chunked associative scan over affine operators.

    Execution strategy:
        1. Decompose $T$ operators into $N = \lceil T / C \rceil$ composed AffineChunks.
        2. Advance state across chunk boundaries: $M_{t_{k+1}} = \text{run\_chunk}(M_{t_k}, \text{chunk}_k)$.
        3. If return_trajectory=True, reconstruct intermediate states within each chunk via prefix scan.

    Args:
        initial_memory: Starting memory state $M_0$.
        operators: List of individual token AffineScanOperator instances.
        chunk_size: Chunk size $C \ge 1$.
        return_trajectory: If True, reconstructs all intermediate states $M_1, \dots, M_T$.

    Returns:
        ChunkedScanResult containing final state, boundary states, chunks, and optional full trajectory.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    if not operators:
        return ChunkedScanResult(
            final_memory=initial_memory,
            boundary_memories=[],
            chunks=[],
            trajectory=[] if return_trajectory else None,
        )

    chunks = build_chunks(operators, chunk_size=chunk_size)

    current_mem = initial_memory
    boundary_mems: list[torch.Tensor] = []
    full_trajectory: list[torch.Tensor] = [] if return_trajectory else []

    for chunk in chunks:
        # If trajectory is requested, reconstruct within-chunk intermediate memories
        if return_trajectory:
            slice_ops = operators[chunk.start_index : chunk.end_index]
            within_chunk_mems = prefix_scan_memory(
                current_mem, slice_ops, include_initial=False
            )
            full_trajectory.extend(within_chunk_mems)

        # Advance state to chunk boundary
        current_mem = run_chunk(current_mem, chunk)
        boundary_mems.append(current_mem)

    return ChunkedScanResult(
        final_memory=current_mem,
        boundary_memories=boundary_mems,
        chunks=chunks,
        trajectory=full_trajectory if return_trajectory else None,
    )
