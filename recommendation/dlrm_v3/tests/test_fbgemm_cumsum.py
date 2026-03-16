"""
Minimal test for torch.ops.fbgemm.asynchronous_complete_cumsum on GPU.

Usage:
    python tests/test_fbgemm_cumsum.py
"""

import torch
import fbgemm_gpu  # noqa: F401

assert torch.cuda.is_available(), "No GPU available"

x = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64, device="cuda")
print(f"input:    {x}")
result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
expected = torch.tensor([0, 3, 4, 8, 9, 14], dtype=torch.int64, device="cuda")
print(f"result:   {result}")
print(f"expected: {expected}")
assert torch.equal(result, expected), f"MISMATCH: {result} != {expected}"
print("PASSED")
