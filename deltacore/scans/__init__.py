"""Scan operators and sequential/chunked execution strategies.

This module hosts recurrent sequential unrolling, affine operator representations,
and chunked associative scan implementations.
"""

from deltacore.scans.adaptive import AdaptiveScanResult, adaptive_scan
from deltacore.scans.affine import AffineScanMetadata, AffineScanOperator
from deltacore.scans.boundary_chunked import (
    BoundaryChunkResult,
    BoundaryRefreshChunkScan,
)
from deltacore.scans.chunked import (
    AffineChunk,
    ChunkedScanResult,
    build_chunks,
    chunked_scan,
    run_chunk,
)
from deltacore.scans.delta_affine import delta_to_affine
from deltacore.scans.prefix import affine_prefix_scan, prefix_scan_memory
from deltacore.scans.sequential import SequentialScanResult, sequential_scan

__all__: list[str] = [
    "sequential_scan",
    "SequentialScanResult",
    "adaptive_scan",
    "AdaptiveScanResult",
    "AffineScanOperator",
    "AffineScanMetadata",
    "delta_to_affine",
    "affine_prefix_scan",
    "prefix_scan_memory",
    "AffineChunk",
    "ChunkedScanResult",
    "build_chunks",
    "run_chunk",
    "chunked_scan",
    "BoundaryRefreshChunkScan",
    "BoundaryChunkResult",
]
