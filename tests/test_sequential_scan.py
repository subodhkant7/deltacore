"""Tests for reference sequential associative scan."""

import pytest
import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.scans.sequential import SequentialScanResult, sequential_scan
from deltacore.updates.delta import DeltaRule
from deltacore.updates.hebbian import HebbianRule


class TestSequentialScan:
    """Verifies sequential state evolution across time steps."""

    def test_unbatched_sequential_scan(self):
        """Verify sequential scan over unbatched [T, K] keys and [T, V] targets."""
        t, k, v = 5, 3, 2
        keys = torch.randn(t, k, dtype=torch.float64)
        targets = torch.randn(t, v, dtype=torch.float64)

        preds, final_mem = sequential_scan(keys, targets, rule=DeltaRule(step_size=0.1))

        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (t, v)
        assert preds.dtype == torch.float64

        # Manual step-by-step verification
        m = torch.zeros(v, k, dtype=torch.float64)
        rule = DeltaRule(step_size=0.1)
        for i in range(t):
            res = rule.step(m, keys[i], targets[i])
            assert torch.allclose(preds[i], res.prediction, atol=1e-12)
            m = res.new_memory

        final_m_tensor = (
            final_mem.data if isinstance(final_mem, AssociativeMemory) else final_mem
        )
        assert torch.allclose(final_m_tensor, m, atol=1e-12)

    def test_batched_sequential_scan(self):
        """Verify sequential scan over batched [B, T, K] and [B, T, V] tensors."""
        b, t, k, v = 3, 4, 2, 3
        keys = torch.randn(b, t, k, dtype=torch.float32)
        targets = torch.randn(b, t, v, dtype=torch.float32)

        res = sequential_scan(keys, targets, rule=HebbianRule(step_size=0.2))
        assert isinstance(res, SequentialScanResult)
        assert res.predictions.shape == (b, t, v)

        # Verify against per-batch unbatched calls
        for batch_idx in range(b):
            single_res = sequential_scan(
                keys[batch_idx],
                targets[batch_idx],
                rule=HebbianRule(step_size=0.2),
            )
            assert torch.allclose(
                res.predictions[batch_idx], single_res.predictions, atol=1e-6
            )

    def test_custom_initial_memory_preservation(self):
        """Verify sequential_scan correctly starts from user-supplied initial memory."""
        t, k, v = 3, 2, 2
        keys = torch.ones(t, k)
        targets = torch.ones(t, v)

        init_m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scan_res = sequential_scan(keys, targets, initial_memory=init_m)

        # Initial prediction must match init_m @ keys[0] = [[1, 2], [3, 4]] @ [1, 1] = [3, 7]
        assert torch.allclose(scan_res.predictions[0], torch.tensor([3.0, 7.0]))

    def test_mismatched_sequence_length_raises(self):
        """Verify that differing T between keys and targets raises ValueError."""
        keys = torch.randn(5, 3)
        targets = torch.randn(4, 2)
        with pytest.raises(ValueError, match="Sequence length mismatch"):
            sequential_scan(keys, targets)
