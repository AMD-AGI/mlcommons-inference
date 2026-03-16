"""
Test fbgemm_gpu ops on GPU to verify they work (or segfault) on AMD MI355X.

Each test runs the fbgemm op on both CPU and GPU, comparing results against
a pure-PyTorch reference. A segfault on GPU indicates the fbgemm_gpu ROCm
build is broken for that op.

Usage:
    # Run all tests (will segfault if fbgemm is broken and no patch is applied)
    python tests/test_fbgemm_gpu_ops.py

    # Run with the compatibility patch applied
    DLRMV3_USE_PYTORCH_OPS=1 python tests/test_fbgemm_gpu_ops.py

    # Run a single test
    python tests/test_fbgemm_gpu_ops.py TestFbgemmOps.test_asynchronous_complete_cumsum_gpu
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fbgemm_gpu  # noqa: F401 — registers torch.ops.fbgemm


def skip_if_no_gpu(fn):
    return unittest.skipUnless(torch.cuda.is_available(), "No GPU available")(fn)


class TestFbgemmOps(unittest.TestCase):
    """Test each patched fbgemm op against a PyTorch reference."""

    # ── asynchronous_complete_cumsum ──────────────────────────────────

    def _ref_complete_cumsum(self, x):
        return torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device),
                          torch.cumsum(x, dim=0)])

    def test_asynchronous_complete_cumsum_cpu(self):
        x = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64)
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        expected = self._ref_complete_cumsum(x)
        torch.testing.assert_close(result, expected)

    @skip_if_no_gpu
    def test_asynchronous_complete_cumsum_gpu(self):
        x = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64, device="cuda")
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        expected = self._ref_complete_cumsum(x)
        torch.testing.assert_close(result, expected)

    @skip_if_no_gpu
    def test_asynchronous_complete_cumsum_gpu_large(self):
        x = torch.randint(1, 100, (1024,), dtype=torch.int64, device="cuda")
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        expected = self._ref_complete_cumsum(x)
        torch.testing.assert_close(result, expected)

    # ── jagged_to_padded_dense ────────────────────────────────────────

    def _ref_jagged_to_padded_dense(self, values, offsets, max_len, padding=0.0):
        batch_size = offsets.size(0) - 1
        if values.dim() == 1:
            out = torch.full((batch_size, max_len), padding,
                             dtype=values.dtype, device=values.device)
            for i in range(batch_size):
                s, e = offsets[i].item(), offsets[i + 1].item()
                length = min(e - s, max_len)
                if length > 0:
                    out[i, :length] = values[s:s + length]
        else:
            d = values.size(1)
            out = torch.full((batch_size, max_len, d), padding,
                             dtype=values.dtype, device=values.device)
            for i in range(batch_size):
                s, e = offsets[i].item(), offsets[i + 1].item()
                length = min(e - s, max_len)
                if length > 0:
                    out[i, :length, :] = values[s:s + length, :]
        return out

    def test_jagged_to_padded_dense_1d_cpu(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        offsets = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
        max_len = 3
        result = torch.ops.fbgemm.jagged_to_padded_dense(
            values=values, offsets=[offsets], max_lengths=[max_len], padding_value=0.0)
        expected = self._ref_jagged_to_padded_dense(values, offsets, max_len)
        torch.testing.assert_close(result, expected)

    @skip_if_no_gpu
    def test_jagged_to_padded_dense_1d_gpu(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        offsets = torch.tensor([0, 2, 3, 5], dtype=torch.int64, device="cuda")
        max_len = 3
        result = torch.ops.fbgemm.jagged_to_padded_dense(
            values=values, offsets=[offsets], max_lengths=[max_len], padding_value=0.0)
        expected = self._ref_jagged_to_padded_dense(values, offsets, max_len)
        torch.testing.assert_close(result, expected)

    @skip_if_no_gpu
    def test_jagged_to_padded_dense_2d_gpu(self):
        values = torch.randn(10, 4, device="cuda")
        offsets = torch.tensor([0, 3, 5, 10], dtype=torch.int64, device="cuda")
        max_len = 6
        result = torch.ops.fbgemm.jagged_to_padded_dense(
            values=values, offsets=[offsets], max_lengths=[max_len], padding_value=0.0)
        expected = self._ref_jagged_to_padded_dense(values, offsets, max_len)
        torch.testing.assert_close(result, expected)

    # ── dense_to_jagged ───────────────────────────────────────────────

    def _ref_dense_to_jagged(self, dense, offsets):
        batch_size = offsets.size(0) - 1
        lengths = offsets[1:] - offsets[:-1]
        total_L = int(lengths.sum().item())
        if dense.dim() == 2:
            out = torch.zeros(total_L, dtype=dense.dtype, device=dense.device)
            for i in range(batch_size):
                s = offsets[i].item()
                l = lengths[i].item()
                if l > 0:
                    out[s:s + l] = dense[i, :l]
        else:
            d = dense.size(2)
            out = torch.zeros(total_L, d, dtype=dense.dtype, device=dense.device)
            for i in range(batch_size):
                s = offsets[i].item()
                l = lengths[i].item()
                if l > 0:
                    out[s:s + l, :] = dense[i, :l, :]
        return out

    @skip_if_no_gpu
    def test_dense_to_jagged_3d_gpu(self):
        offsets = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device="cuda")
        dense = torch.randn(3, 5, 8, device="cuda")
        result, _ = torch.ops.fbgemm.dense_to_jagged(dense, [offsets])
        expected = self._ref_dense_to_jagged(dense, offsets)
        torch.testing.assert_close(result, expected)

    def test_dense_to_jagged_3d_cpu(self):
        offsets = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
        dense = torch.randn(3, 5, 8)
        result, _ = torch.ops.fbgemm.dense_to_jagged(dense, [offsets])
        expected = self._ref_dense_to_jagged(dense, offsets)
        torch.testing.assert_close(result, expected)

    # ── asynchronous_inclusive_cumsum ──────────────────────────────────

    def _ref_inclusive_cumsum(self, x):
        return torch.cumsum(x, dim=0)

    @skip_if_no_gpu
    def test_asynchronous_inclusive_cumsum_gpu(self):
        x = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64, device="cuda")
        result = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
        expected = self._ref_inclusive_cumsum(x)
        torch.testing.assert_close(result, expected)

    # ── asynchronous_exclusive_cumsum ─────────────────────────────────

    def _ref_exclusive_cumsum(self, x):
        return torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device),
                          torch.cumsum(x, dim=0)[:-1]])

    @skip_if_no_gpu
    def test_asynchronous_exclusive_cumsum_gpu(self):
        x = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64, device="cuda")
        result = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
        expected = self._ref_exclusive_cumsum(x)
        torch.testing.assert_close(result, expected)

    # ── jagged_dense_elementwise_add_jagged_output ────────────────────

    @skip_if_no_gpu
    def test_jagged_dense_add_gpu(self):
        offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="cuda")
        x_values = torch.randn(5, 4, device="cuda")
        y = torch.randn(2, 5, 4, device="cuda")
        result, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values, [offsets], y)
        # Reference: add corresponding dense rows to jagged values
        expected = x_values.clone()
        for i in range(2):
            s, e = offsets[i].item(), offsets[i + 1].item()
            expected[s:e] = x_values[s:e] + y[i, :e - s]
        torch.testing.assert_close(result, expected)


class TestFbgemmGPUSmoke(unittest.TestCase):
    """Smoke tests: just verify the ops don't segfault on GPU."""

    @skip_if_no_gpu
    def test_complete_cumsum_no_crash(self):
        x = torch.ones(100, dtype=torch.int64, device="cuda")
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        self.assertEqual(result.shape[0], 101)

    @skip_if_no_gpu
    def test_jagged_to_padded_dense_no_crash(self):
        values = torch.randn(50, 16, device="cuda")
        offsets = torch.tensor([0, 10, 20, 35, 50], dtype=torch.int64, device="cuda")
        result = torch.ops.fbgemm.jagged_to_padded_dense(
            values=values, offsets=[offsets], max_lengths=[20], padding_value=0.0)
        self.assertEqual(result.shape, (4, 20, 16))

    @skip_if_no_gpu
    def test_dense_to_jagged_no_crash(self):
        offsets = torch.tensor([0, 5, 12, 20], dtype=torch.int64, device="cuda")
        dense = torch.randn(3, 12, 8, device="cuda")
        result, _ = torch.ops.fbgemm.dense_to_jagged(dense, [offsets])
        self.assertEqual(result.shape[0], 20)

    @skip_if_no_gpu
    def test_roundtrip_jagged_dense_jagged(self):
        """jagged -> padded dense -> jagged should recover original values."""
        values = torch.randn(15, 8, device="cuda")
        offsets = torch.tensor([0, 4, 9, 15], dtype=torch.int64, device="cuda")
        max_len = 6
        padded = torch.ops.fbgemm.jagged_to_padded_dense(
            values=values, offsets=[offsets], max_lengths=[max_len], padding_value=0.0)
        recovered, _ = torch.ops.fbgemm.dense_to_jagged(padded, [offsets])
        torch.testing.assert_close(recovered, values)


if __name__ == "__main__":
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print()
    unittest.main(verbosity=2)
