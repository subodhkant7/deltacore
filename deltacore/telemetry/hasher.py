"""Deterministic Telemetry Feature Hasher for DeltaCore.

Converts structured categorical and numerical event dictionaries into fixed-dimensional
float vectors using SHA-256 signed feature hashing (the hashing trick).

Mathematical Formulation:
    For each extracted feature token s:
        1. Compute SHA-256 digest of UTF-8 encoded string s:
           digest = sha256(s.encode("utf-8"))
        2. Bucket index:
           h(s) = uint64(digest[0:8]) mod D
        3. Sign:
           \\xi(s) = +1 if digest[8] & 1 == 0 else -1
        4. Accumulate:
           x[h(s)] += \\xi(s) * w
    Optionally L2-normalize:
        \\tilde{x} = x / (||x||_2 + \\epsilon)

Properties & Epistemic Distinctions:
    - NOT a semantic embedding: preserves no latent semantic topology.
    - Signed hashing guarantees that the expectation of inner products in hashed space
      unbiasedly estimates the inner product of the unhashed bag-of-features representation:
      E[\\langle \\phi(u), \\phi(v) \\rangle] = \\langle u, v \\rangle.
    - Due to finite dimension D, random hash collisions introduce variance and noise.
    - Cosine similarity in hashed space does NOT strictly equal Jaccard similarity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class CollisionStats:
    """Diagnostic statistics of hash collisions across a tested vocabulary."""

    total_tokens: int
    dimension: int
    distinct_buckets_used: int
    collisions: int
    collision_rate: float
    max_bucket_depth: int
    bucket_counts: dict[int, int]


@dataclass(frozen=True)
class TelemetryHasherConfig:
    """Configuration for DeterministicFeatureHasher.

    Attributes:
        dim: Output feature dimension D (e.g. 64, 128, 256, 512, 1024).
        normalize: Whether to L2-normalize output vector to unit norm.
        pair_interactions: Sequence of (field_a, field_b) pairs to hash as joint features.
        triple_interactions: Sequence of (field_a, field_b, field_c) triples to hash as joint features.
        include_unary: Whether to include individual field=value unary features.
        dtype: PyTorch float tensor data type (torch.float32 or torch.float64).
    """

    dim: int = 128
    normalize: bool = True
    pair_interactions: Sequence[tuple[str, str]] = field(
        default_factory=lambda: (
            ("service", "operation"),
            ("operation", "phase"),
            ("operation", "outcome"),
            ("provider", "operation"),
        )
    )
    triple_interactions: Sequence[tuple[str, str, str]] = field(default_factory=tuple)
    include_unary: bool = True
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"dtype must be torch.float32 or torch.float64, got {self.dtype}"
            )


class DeterministicFeatureHasher:
    """Converts structured telemetry event dictionaries into fixed-dimensional vectors."""

    def __init__(
        self, config: TelemetryHasherConfig | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            self.config = TelemetryHasherConfig(**kwargs)
        else:
            if kwargs:
                raise ValueError("Cannot pass both config and keyword arguments")
            self.config = config
        self.dim = self.config.dim

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Hash a single token string to (bucket_index, sign) deterministically via SHA-256."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # First 8 bytes as big-endian uint64
        val = int.from_bytes(digest[:8], byteorder="big", signed=False)
        bucket = val % self.dim
        # 9th byte parity for sign
        sign = 1.0 if (digest[8] & 1) == 0 else -1.0
        return bucket, sign

    def extract_tokens(self, event: Mapping[str, Any]) -> list[str]:
        """Extract deterministic feature token list from structured event dictionary."""
        tokens: list[str] = []

        # 1. Unary features
        if self.config.include_unary:
            for k in sorted(event.keys()):
                v = event[k]
                tokens.append(f"{k}={v}")

        # 2. Pair interactions
        for k1, k2 in self.config.pair_interactions:
            if k1 in event and k2 in event:
                tokens.append(f"{k1}={event[k1]}|{k2}={event[k2]}")

        # 3. Triple interactions
        for k1, k2, k3 in self.config.triple_interactions:
            if k1 in event and k2 in event and k3 in event:
                tokens.append(f"{k1}={event[k1]}|{k2}={event[k2]}|{k3}={event[k3]}")

        return tokens

    def encode(self, event: Mapping[str, Any]) -> torch.Tensor:
        """Encode a structured event dictionary into a fixed-dimensional float tensor in R^D.

        Args:
            event: Mapping of field name to scalar value (str, int, float, bool).

        Returns:
            Deterministic 1D torch.Tensor of shape (dim,).

        Raises:
            TypeError: If event is not a mapping.
            ValueError: If event produces no tokens or non-finite values.
        """
        if not isinstance(event, Mapping):
            raise TypeError(
                f"event must be a Mapping (e.g. dict), got {type(event).__name__}"
            )

        vec = torch.zeros(self.dim, dtype=self.config.dtype)
        tokens = self.extract_tokens(event)
        if not tokens:
            return vec

        for token in tokens:
            bucket, sign = self._hash_token(token)
            vec[bucket] += sign

        # Optional L2 normalization
        if self.config.normalize:
            norm = torch.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm

        if not torch.isfinite(vec).all():
            raise FloatingPointError(
                "Encoded feature vector contains non-finite values"
            )

        return vec

    def compute_collision_statistics(self, vocabulary: Sequence[str]) -> CollisionStats:
        """Analyze hash bucket distribution and collisions across a provided token vocabulary."""
        counts: dict[int, int] = {}
        for token in vocabulary:
            bucket, _ = self._hash_token(token)
            counts[bucket] = counts.get(bucket, 0) + 1

        total_tokens = len(vocabulary)
        distinct_buckets = len(counts)
        collisions = total_tokens - distinct_buckets
        collision_rate = collisions / max(1, total_tokens)
        max_depth = max(counts.values()) if counts else 0

        return CollisionStats(
            total_tokens=total_tokens,
            dimension=self.dim,
            distinct_buckets_used=distinct_buckets,
            collisions=collisions,
            collision_rate=collision_rate,
            max_bucket_depth=max_depth,
            bucket_counts=counts,
        )
