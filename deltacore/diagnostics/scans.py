# ==============================================================================
# DeltaCore: deltacore/diagnostics/scans.py
# Observability and diagnostic metrics for chunked and associative scans.
# ==============================================================================

from dataclasses import dataclass

import torch

from deltacore.scans.chunked import AffineChunk


@dataclass(frozen=True)
class ScanDiagnostics:
    r"""Diagnostic summary of chunked associative scan execution.

    Attributes:
        chunk_count: Total number of chunks $N = \lceil T / C \rceil$.
        mean_chunk_length: Mean number of transitions per chunk.
        max_chunk_length: Maximum transitions in any chunk.
        boundary_count: Number of chunk boundaries traversed.
        composition_depth: Maximum number of sequential binary compositions within
            any chunk, defined as max(C_k - 1, 0) under linear composition chaining.
            This is linear composition depth, NOT logarithmic tree reduction depth.
        sequential_vs_chunk_max_error: Maximum absolute difference $\max \|M^{\text{seq}} - M^{\text{chunk}}\|_\infty$.
    """

    chunk_count: int
    mean_chunk_length: float
    max_chunk_length: int
    boundary_count: int
    composition_depth: int
    sequential_vs_chunk_max_error: float | None = None


def extract_scan_diagnostics(
    chunks: list[AffineChunk],
    sequential_memories: list[torch.Tensor] | torch.Tensor | None = None,
    chunked_memories: list[torch.Tensor] | torch.Tensor | None = None,
) -> ScanDiagnostics:
    r"""Compute diagnostic metrics for a chunked scan execution.

    Args:
        chunks: List of AffineChunk instances.
        sequential_memories: Sequential memory trajectory or final memory tensor.
        chunked_memories: Chunked reconstructed trajectory or final memory tensor.

    Returns:
        ScanDiagnostics object with structural and numerical error statistics.
    """
    chunk_count = len(chunks)
    if chunk_count > 0:
        lengths = [c.num_transitions for c in chunks]
        mean_len = float(sum(lengths) / len(lengths))
        max_len = int(max(lengths))
        comp_depth = max(max_len - 1, 0)
    else:
        mean_len = 0.0
        max_len = 0
        comp_depth = 0

    max_err: float | None = None
    if sequential_memories is not None and chunked_memories is not None:
        if isinstance(sequential_memories, list) and isinstance(chunked_memories, list):
            diffs = []
            for m_seq, m_chk in zip(
                sequential_memories, chunked_memories, strict=False
            ):
                diff = torch.max(
                    torch.abs(m_seq.detach().float() - m_chk.detach().float())
                ).item()
                diffs.append(diff)
            max_err = float(max(diffs)) if diffs else 0.0
        elif isinstance(sequential_memories, torch.Tensor) and isinstance(
            chunked_memories, torch.Tensor
        ):
            max_err = float(
                torch.max(
                    torch.abs(
                        sequential_memories.detach().float()
                        - chunked_memories.detach().float()
                    )
                ).item()
            )

    return ScanDiagnostics(
        chunk_count=chunk_count,
        mean_chunk_length=mean_len,
        max_chunk_length=max_len,
        boundary_count=chunk_count,
        composition_depth=comp_depth,
        sequential_vs_chunk_max_error=max_err,
    )
