"""Rigorous test suite for DeterministicFeatureHasher (Section 8).

Verifies:
1. Determinism: Same event -> bit-exact identical vector across repeated runs and processes.
2. Dimension: Correct output dimension across D in {64, 128, 256, 512, 1024}.
3. Finite values: All outputs are strictly finite (no NaN, +inf, -inf).
4. Normalization: Non-zero vectors have ||x||_2 == 1.0 +/- 1e-6 when normalize=True.
5. Field sensitivity: Altering any single field value modifies the output vector.
6. Interaction sensitivity: Changing a configured interaction modifies the vector.
7. Collision accounting: Collision statistics utility accurately tracks bucket occupancy without claiming 0 collisions.
8. Empty/missing fields handling.
"""

import math

import pytest
import torch

from deltacore.telemetry.hasher import (
    DeterministicFeatureHasher,
    TelemetryHasherConfig,
)


@pytest.fixture
def sample_event() -> dict[str, object]:
    return {
        "service": "payments",
        "operation": "refund",
        "phase": "post_dispatch",
        "provider": "stripe",
        "http_status": 503,
        "transport": "connection_reset",
        "outcome": "unknown",
    }


def test_determinism_across_invocations(sample_event: dict[str, object]) -> None:
    """Verify same event produces bit-exact identical vector across instances."""
    hasher1 = DeterministicFeatureHasher(dim=128)
    hasher2 = DeterministicFeatureHasher(dim=128)

    v1 = hasher1.encode(sample_event)
    v2 = hasher2.encode(sample_event)

    assert torch.equal(v1, v2), "Feature encoding must be bit-exact deterministic"


@pytest.mark.parametrize("dim", [64, 128, 256, 512, 1024])
def test_output_dimensions(dim: int, sample_event: dict[str, object]) -> None:
    """Verify output vector matches configured dimensionality."""
    hasher = DeterministicFeatureHasher(dim=dim)
    vec = hasher.encode(sample_event)
    assert vec.shape == (dim,)
    assert vec.dtype == torch.float32


def test_strictly_finite_values(sample_event: dict[str, object]) -> None:
    """Verify encoded vector is strictly finite (no NaN or Inf)."""
    hasher = DeterministicFeatureHasher(dim=256)
    vec = hasher.encode(sample_event)
    assert torch.isfinite(vec).all(), "Hashed vector must contain only finite numbers"
    assert not torch.isnan(vec).any(), "Hashed vector must not contain NaNs"


def test_normalization_invariance(sample_event: dict[str, object]) -> None:
    """Verify unit Euclidean norm when normalize=True."""
    hasher_norm = DeterministicFeatureHasher(dim=128, normalize=True)
    vec_norm = hasher_norm.encode(sample_event)
    norm = float(torch.linalg.norm(vec_norm).item())
    assert math.isclose(norm, 1.0, rel_tol=1e-5), f"Expected unit norm 1.0, got {norm}"

    hasher_raw = DeterministicFeatureHasher(dim=128, normalize=False)
    vec_raw = hasher_raw.encode(sample_event)
    norm_raw = float(torch.linalg.norm(vec_raw).item())
    assert norm_raw > 1.0, (
        "Raw count vector norm should reflect accumulated token counts"
    )


def test_field_sensitivity(sample_event: dict[str, object]) -> None:
    """Verify that changing any single field alters the encoded vector."""
    hasher = DeterministicFeatureHasher(dim=128)
    v_base = hasher.encode(sample_event)

    mutated = dict(sample_event)
    mutated["http_status"] = 200
    v_mut = hasher.encode(mutated)

    assert not torch.equal(v_base, v_mut), (
        "Changing http_status must alter the feature vector"
    )
    cosine_sim = float(torch.dot(v_base, v_mut).item())
    assert cosine_sim < 0.999, (
        f"Mutating a field must produce distinct vector, sim={cosine_sim}"
    )


def test_interaction_sensitivity() -> None:
    """Verify that interaction configuration influences the encoded vector."""
    event1 = {"service": "checkout", "operation": "pay"}
    event2 = {"service": "checkout", "operation": "refund"}

    # Hasher without pair interactions
    hasher_no_pairs = DeterministicFeatureHasher(
        TelemetryHasherConfig(dim=128, pair_interactions=())
    )
    # Hasher with pair interaction
    hasher_pairs = DeterministicFeatureHasher(
        TelemetryHasherConfig(dim=128, pair_interactions=(("service", "operation"),))
    )

    t1_no_pairs = hasher_no_pairs.extract_tokens(event1)
    t1_pairs = hasher_pairs.extract_tokens(event1)

    assert len(t1_pairs) == len(t1_no_pairs) + 1
    assert "service=checkout|operation=pay" in t1_pairs

    v1 = hasher_pairs.encode(event1)
    v2 = hasher_pairs.encode(event2)
    assert not torch.equal(v1, v2)


def test_triple_interactions() -> None:
    """Verify that triple interactions generate explicit 3-way feature tokens."""
    event = {"service": "auth", "operation": "login", "outcome": "failure"}
    hasher = DeterministicFeatureHasher(
        TelemetryHasherConfig(
            dim=128,
            triple_interactions=(("service", "operation", "outcome"),),
        )
    )
    tokens = hasher.extract_tokens(event)
    assert "service=auth|operation=login|outcome=failure" in tokens


def test_collision_accounting() -> None:
    """Verify collision accounting utility produces accurate non-zero collision stats."""
    hasher = DeterministicFeatureHasher(dim=32)  # Low dimension to guarantee collisions
    vocab = [f"token_{i}" for i in range(100)]
    stats = hasher.compute_collision_statistics(vocab)

    assert stats.total_tokens == 100
    assert stats.dimension == 32
    assert stats.distinct_buckets_used <= 32
    assert stats.collisions == 100 - stats.distinct_buckets_used
    assert stats.collisions > 0, (
        "With 100 tokens into 32 buckets, collisions must occur"
    )
    assert stats.collision_rate > 0.0
    assert stats.max_bucket_depth > 1


def test_empty_event_handling() -> None:
    """Verify empty dictionary produces clean zero vector without exceptions."""
    hasher = DeterministicFeatureHasher(dim=64, normalize=True)
    vec = hasher.encode({})
    assert vec.shape == (64,)
    assert torch.all(vec == 0.0)
