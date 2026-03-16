"""
Test Triton vs PyTorch kernel backends on AMD MI355X GPUs.

Each test runs the same op with HammerKernel.TRITON and HammerKernel.PYTORCH,
comparing outputs. A segfault or mismatch with TRITON indicates the Triton
kernels don't work on AMD CDNA; matching results means they can be used
(and the HammerKernel.PYTORCH fallback in inference_modules.py can be removed).

Usage:
    # Run on a GPU node (will segfault if Triton kernels are broken on AMD)
    srun --jobid=<JOBID> bash -c \\
        "source .venv/bin/activate && DLRMV3_USE_PYTORCH_OPS=1 python tests/test_triton_ops.py"

    # Run a single test
    srun --jobid=<JOBID> bash -c \\
        "source .venv/bin/activate && DLRMV3_USE_PYTORCH_OPS=1 python tests/test_triton_ops.py TestTritonLayerNorm"
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fbgemm_gpu  # noqa: F401 — registers torch.ops.fbgemm

from generative_recommenders.common import HammerKernel


def skip_if_no_gpu(fn):
    return unittest.skipUnless(torch.cuda.is_available(), "No GPU available")(fn)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRITON = HammerKernel.TRITON
PYTORCH = HammerKernel.PYTORCH


class TestTritonLayerNorm(unittest.TestCase):
    """Test layer_norm, swish_layer_norm, rms_norm: Triton vs PyTorch."""

    @skip_if_no_gpu
    def test_layer_norm(self):
        from generative_recommenders.ops.layer_norm import layer_norm

        dim = 64
        x = torch.randn(8, dim, device=DEVICE, dtype=torch.float32)
        w = torch.ones(dim, device=DEVICE, dtype=torch.float32)
        b = torch.zeros(dim, device=DEVICE, dtype=torch.float32)

        ref = layer_norm(x, w, b, eps=1e-5, kernel=PYTORCH)
        out = layer_norm(x, w, b, eps=1e-5, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    @skip_if_no_gpu
    def test_swish_layer_norm(self):
        from generative_recommenders.ops.layer_norm import swish_layer_norm

        dim = 64
        x = torch.randn(8, dim, device=DEVICE, dtype=torch.float32)
        w = torch.ones(dim, device=DEVICE, dtype=torch.float32)
        b = torch.zeros(dim, device=DEVICE, dtype=torch.float32)

        ref = swish_layer_norm(x, w, b, eps=1e-5, kernel=PYTORCH)
        out = swish_layer_norm(x, w, b, eps=1e-5, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    @skip_if_no_gpu
    def test_rms_norm(self):
        from generative_recommenders.ops.layer_norm import rms_norm

        dim = 64
        x = torch.randn(8, dim, device=DEVICE, dtype=torch.float32)
        w = torch.ones(dim, device=DEVICE, dtype=torch.float32)

        ref = rms_norm(x, w, eps=1e-5, kernel=PYTORCH)
        out = rms_norm(x, w, eps=1e-5, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


class TestTritonAddmm(unittest.TestCase):
    """Test addmm: Triton vs PyTorch."""

    @skip_if_no_gpu
    def test_addmm(self):
        from generative_recommenders.ops.mm import addmm

        M, K, N = 16, 32, 64
        bias = torch.randn(N, device=DEVICE, dtype=torch.float32)
        mat1 = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        mat2 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

        ref = addmm(bias, mat1, mat2, kernel=PYTORCH)
        out = addmm(bias, mat1, mat2, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    @skip_if_no_gpu
    def test_addmm_large(self):
        from generative_recommenders.ops.mm import addmm

        M, K, N = 128, 512, 256
        bias = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
        mat1 = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        mat2 = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)

        ref = addmm(bias, mat1, mat2, kernel=PYTORCH)
        out = addmm(bias, mat1, mat2, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestTritonJaggedTensors(unittest.TestCase):
    """Test jagged tensor ops: Triton vs PyTorch."""

    @skip_if_no_gpu
    def test_concat_2D_jagged(self):
        from generative_recommenders.ops.jagged_tensors import concat_2D_jagged

        B, D = 4, 16
        lengths_left = torch.tensor([3, 2, 4, 1], device=DEVICE)
        lengths_right = torch.tensor([2, 3, 1, 2], device=DEVICE)
        offsets_left = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                                  torch.cumsum(lengths_left, 0)])
        offsets_right = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                                   torch.cumsum(lengths_right, 0)])
        total_left = int(lengths_left.sum().item())
        total_right = int(lengths_right.sum().item())
        values_left = torch.randn(total_left, D, device=DEVICE)
        values_right = torch.randn(total_right, D, device=DEVICE)
        max_left = int(lengths_left.max().item())
        max_right = int(lengths_right.max().item())
        max_seq = max_left + max_right

        ref = concat_2D_jagged(
            max_seq_len=max_seq, values_left=values_left, values_right=values_right,
            max_len_left=max_left, max_len_right=max_right,
            offsets_left=offsets_left, offsets_right=offsets_right, kernel=PYTORCH)
        out = concat_2D_jagged(
            max_seq_len=max_seq, values_left=values_left, values_right=values_right,
            max_len_left=max_left, max_len_right=max_right,
            offsets_left=offsets_left, offsets_right=offsets_right, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @skip_if_no_gpu
    def test_split_2D_jagged(self):
        from generative_recommenders.ops.jagged_tensors import split_2D_jagged

        B, D = 4, 16
        lengths_left = torch.tensor([3, 2, 4, 1], device=DEVICE)
        lengths_right = torch.tensor([2, 3, 1, 2], device=DEVICE)
        offsets_left = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                                  torch.cumsum(lengths_left, 0)])
        offsets_right = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                                   torch.cumsum(lengths_right, 0)])
        total_left = int(lengths_left.sum().item())
        total_right = int(lengths_right.sum().item())
        max_left = int(lengths_left.max().item())
        max_right = int(lengths_right.max().item())
        max_seq = max_left + max_right
        total = total_left + total_right
        values = torch.randn(total, D, device=DEVICE)

        ref_l, ref_r = split_2D_jagged(
            max_seq_len=max_seq, values=values,
            total_len_left=total_left, total_len_right=total_right,
            max_len_left=max_left, max_len_right=max_right,
            offsets_left=offsets_left, offsets_right=offsets_right, kernel=PYTORCH)
        out_l, out_r = split_2D_jagged(
            max_seq_len=max_seq, values=values,
            total_len_left=total_left, total_len_right=total_right,
            max_len_left=max_left, max_len_right=max_right,
            offsets_left=offsets_left, offsets_right=offsets_right, kernel=TRITON)
        torch.testing.assert_close(out_l, ref_l, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_r, ref_r, atol=1e-5, rtol=1e-5)

    @skip_if_no_gpu
    def test_jagged_dense_bmm_broadcast_add(self):
        from generative_recommenders.ops.jagged_tensors import jagged_dense_bmm_broadcast_add

        B, K, N = 3, 32, 16
        lengths = torch.tensor([4, 2, 5], device=DEVICE, dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                             torch.cumsum(lengths, 0)])
        total_L = int(lengths.sum().item())
        max_seq = int(lengths.max().item())
        jagged = torch.randn(total_L, K, device=DEVICE)
        dense = torch.randn(B, K, N, device=DEVICE)
        bias = torch.randn(B, N, device=DEVICE)

        ref = jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq, seq_offsets=offsets,
            jagged=jagged, dense=dense, bias=bias, kernel=PYTORCH)
        out = jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq, seq_offsets=offsets,
            jagged=jagged, dense=dense, bias=bias, kernel=TRITON)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


class TestTritonHSTUAttention(unittest.TestCase):
    """Test HSTU multi-head attention: Triton vs PyTorch."""

    @skip_if_no_gpu
    def test_hstu_mha(self):
        from generative_recommenders.ops.hstu_attention import hstu_mha

        B, H, D = 2, 4, 16
        lengths = torch.tensor([6, 8], device=DEVICE, dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                             torch.cumsum(lengths, 0)])
        total_L = int(lengths.sum().item())
        max_seq = int(lengths.max().item())

        q = torch.randn(total_L, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(total_L, H, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(total_L, H, D, device=DEVICE, dtype=torch.bfloat16)
        alpha = 1.0 / (D ** 0.5)

        ref = hstu_mha(
            max_seq_len=max_seq, alpha=alpha, q=q, k=k, v=v,
            seq_offsets=offsets, causal=True, dropout_pr=0.0,
            training=False, kernel=PYTORCH)
        out = hstu_mha(
            max_seq_len=max_seq, alpha=alpha, q=q, k=k, v=v,
            seq_offsets=offsets, causal=True, dropout_pr=0.0,
            training=False, kernel=TRITON)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


class TestTritonHSTUCompute(unittest.TestCase):
    """Test HSTU compute ops: Triton vs PyTorch."""

    @skip_if_no_gpu
    def test_hstu_compute_uqvk(self):
        from generative_recommenders.ops.hstu_compute import hstu_compute_uqvk

        total_L, input_dim = 20, 64
        num_heads, attn_dim, hidden_dim = 4, 16, 32
        out_dim = 2 * num_heads * (hidden_dim + attn_dim)

        x = torch.randn(total_L, input_dim, device=DEVICE, dtype=torch.bfloat16)
        norm_w = torch.ones(input_dim, device=DEVICE, dtype=torch.bfloat16)
        norm_b = torch.zeros(input_dim, device=DEVICE, dtype=torch.bfloat16)
        uvqk_w = torch.randn(input_dim, out_dim, device=DEVICE, dtype=torch.bfloat16)
        uvqk_b = torch.randn(out_dim, device=DEVICE, dtype=torch.bfloat16)

        ref = hstu_compute_uqvk(
            x, norm_w, norm_b, 1e-5, num_heads, attn_dim, hidden_dim,
            uvqk_w, uvqk_b, kernel=PYTORCH)
        out = hstu_compute_uqvk(
            x, norm_w, norm_b, 1e-5, num_heads, attn_dim, hidden_dim,
            uvqk_w, uvqk_b, kernel=TRITON)
        for r, o in zip(ref, out):
            torch.testing.assert_close(o.float(), r.float(), atol=1e-2, rtol=1e-2)


class TestTritonPosition(unittest.TestCase):
    """Test positional embedding: Triton vs PyTorch."""

    @skip_if_no_gpu
    def test_add_timestamp_positional_embeddings(self):
        from generative_recommenders.ops.position import add_timestamp_positional_embeddings

        B, D = 2, 64
        max_pos, max_ts = 128, 256
        lengths = torch.tensor([5, 8], device=DEVICE, dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int64),
                             torch.cumsum(lengths, 0)])
        total_L = int(lengths.sum().item())
        max_seq = int(lengths.max().item())

        pos_w = torch.randn(max_pos, D, device=DEVICE, dtype=torch.bfloat16)
        ts_w = torch.randn(max_ts, D, device=DEVICE, dtype=torch.bfloat16)

        seq_emb = torch.randn(total_L, D, device=DEVICE, dtype=torch.bfloat16)
        timestamps = torch.randint(0, 100, (total_L,), device=DEVICE, dtype=torch.int64)

        ref = add_timestamp_positional_embeddings(
            alpha=1.0, max_seq_len=max_seq, max_contextual_seq_len=0,
            position_embeddings_weight=pos_w, timestamp_embeddings_weight=ts_w,
            seq_offsets=offsets, seq_lengths=lengths,
            seq_embeddings=seq_emb.clone(), timestamps=timestamps,
            num_targets=None, interleave_targets=False, kernel=PYTORCH)
        out = add_timestamp_positional_embeddings(
            alpha=1.0, max_seq_len=max_seq, max_contextual_seq_len=0,
            position_embeddings_weight=pos_w, timestamp_embeddings_weight=ts_w,
            seq_offsets=offsets, seq_lengths=lengths,
            seq_embeddings=seq_emb.clone(), timestamps=timestamps,
            num_targets=None, interleave_targets=False, kernel=TRITON)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print()
    unittest.main(verbosity=2)
