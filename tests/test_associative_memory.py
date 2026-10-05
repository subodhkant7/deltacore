"""Tests for AssociativeMemory state abstraction and read operations.

Verifies:
- Read correctness on known matrices
- In-place mutation protection / cloning
- Dtype preservation (FP32, FP64)
- Device preservation
- Strict shape validation and error handling
"""

import pytest
import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.read import read


class TestReadOperation:
    """Test suite for the pure read operation: read(M, k) = M @ k."""

    def test_property_a_read_correctness_deterministic_matrix(self):
        """Property A: For known matrices, read(M, k) == M @ k exactly."""
        # 2x2 Identity
        m_ident = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        k = torch.tensor([3.0, 5.0], dtype=torch.float32)
        v_hat = read(m_ident, k)
        assert torch.equal(v_hat, k)

        # 3x2 Matrix
        # M = [[1, 2],
        #      [3, 4],
        #      [5, 6]]
        m_rect = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        k_2 = torch.tensor([2.0, -1.0], dtype=torch.float32)
        # Expected:
        # [1*2 + 2*(-1), 3*2 + 4*(-1), 5*2 + 6*(-1)] = [0, 2, 4]
        v_rect = read(m_rect, k_2)
        expected = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32)
        assert torch.allclose(v_rect, expected, atol=1e-7)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_read_dtype_preservation(self, dtype):
        """Property F: Dtype is preserved across FP32 and FP64."""
        m = torch.randn(4, 3, dtype=dtype)
        k = torch.randn(3, dtype=dtype)
        v = read(m, k)
        assert v.dtype == dtype

    def test_read_batched_keys_shared_memory(self):
        """Test reading with batched keys against a 2D memory: [B, K] against [V, K] -> [B, V]."""
        m = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        keys = torch.tensor([[1.0, 1.0], [2.0, 3.0], [0.0, 4.0]], dtype=torch.float32)
        # Expected:
        # [1, 2]
        # [2, 6]
        # [0, 8]
        expected = torch.tensor(
            [[1.0, 2.0], [2.0, 6.0], [0.0, 8.0]], dtype=torch.float32
        )
        out = read(m, keys)
        assert torch.allclose(out, expected, atol=1e-7)

    def test_read_batched_3d_memory(self):
        """Test batched memory [B, V, K] with batched keys [B, K] -> [B, V]."""
        m = torch.randn(3, 4, 2, dtype=torch.float32)
        keys = torch.randn(3, 2, dtype=torch.float32)
        out = read(m, keys)
        assert out.shape == (3, 4)
        for b in range(3):
            assert torch.allclose(out[b], m[b] @ keys[b], atol=1e-6)

    def test_read_invalid_types(self):
        """Validate informative TypeError on non-tensor inputs."""
        with pytest.raises(TypeError, match="memory must be a torch.Tensor"):
            read([[1.0, 0.0]], torch.tensor([1.0]))  # type: ignore

        with pytest.raises(TypeError, match="key must be a torch.Tensor"):
            read(torch.eye(2), [1.0, 2.0])  # type: ignore

    def test_read_dtype_mismatch_error(self):
        """Validate informative TypeError on mismatched dtypes."""
        m = torch.eye(2, dtype=torch.float32)
        k = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(TypeError, match="Dtype mismatch"):
            read(m, k)

    def test_read_dimension_mismatch_error(self):
        """Validate dimension mismatches raise ValueError with informative context."""
        m = torch.eye(3, dtype=torch.float32)
        k = torch.tensor([1.0, 2.0], dtype=torch.float32)  # size 2 vs 3
        with pytest.raises(ValueError, match="Key dimension mismatch"):
            read(m, k)


class TestAssociativeMemoryState:
    """Test suite for the AssociativeMemory state container."""

    def test_initialization_zeros(self):
        """Verify AssociativeMemory.zeros creates correct shapes and values."""
        mem = AssociativeMemory.zeros(v_dim=5, k_dim=3, dtype=torch.float32)
        assert mem.shape == (5, 3)
        assert mem.v_dim == 5
        assert mem.k_dim == 3
        assert mem.dtype == torch.float32
        assert torch.equal(mem.data, torch.zeros(5, 3))

    def test_initialization_from_tensor(self):
        """Verify initialization wraps existing tensor with proper properties."""
        t = torch.randn(4, 2, dtype=torch.float64)
        mem = AssociativeMemory.from_tensor(t)
        assert mem.shape == (4, 2)
        assert mem.v_dim == 4
        assert mem.k_dim == 2
        assert mem.dtype == torch.float64
        assert torch.equal(mem.data, t)

    def test_clone_independence(self):
        """Property E: Cloning produces independent state; mutating clone leaves original intact."""
        mem1 = AssociativeMemory.zeros(v_dim=2, k_dim=2)
        mem2 = mem1.clone()

        # Modify clone's underlying data
        mem2.data.add_(1.0)
        assert torch.equal(mem1.data, torch.zeros(2, 2))
        assert torch.equal(mem2.data, torch.ones(2, 2))

    def test_reset_zeros_memory(self):
        """Verify that reset sets contents to zero."""
        t = torch.ones(3, 3)
        mem = AssociativeMemory(t)
        mem.reset()
        assert torch.equal(mem.data, torch.zeros(3, 3))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_casting(self, dtype):
        """Verify .to() casts dtype cleanly without altering shapes."""
        mem = AssociativeMemory.zeros(3, 2, dtype=torch.float32)
        mem_cast = mem.to(dtype=dtype)
        assert mem_cast.dtype == dtype
        assert mem_cast.shape == (3, 2)

    def test_device_preservation(self):
        """Property G: Memory remains on CPU (or CUDA if explicitly specified)."""
        mem = AssociativeMemory.zeros(2, 2)
        assert mem.device.type == "cpu"
        k = torch.tensor([1.0, 0.0])
        out = mem.read(k)
        assert out.device.type == "cpu"

    def test_invalid_construction_raises(self):
        """Verify invalid constructor arguments raise appropriate exceptions."""
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            AssociativeMemory("invalid")  # type: ignore

        with pytest.raises(TypeError, match="requires floating point"):
            AssociativeMemory(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))

        with pytest.raises(ValueError, match="requires 2D.*or 3D"):
            AssociativeMemory(torch.ones(4))  # 1D is invalid

        with pytest.raises(ValueError, match="Dimensions must be positive"):
            AssociativeMemory.zeros(v_dim=0, k_dim=3)

        with pytest.raises(ValueError, match="strictly positive"):
            AssociativeMemory(torch.zeros(0, 3))
