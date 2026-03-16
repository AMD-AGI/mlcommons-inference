# DLRMv3 Inference on AMD GPUs (ROCm)

The upstream MLPerf Inference DLRMv3 reference implementation was developed and tested exclusively on NVIDIA H100/B200 GPUs. This document describes the changes required to run the benchmark on AMD Instinct MI355X GPUs (CDNA4, gfx950) with ROCm 7.0.2.

## Issues and Fixes

### 1. Triton Kernels (gfx950 compiler failure)

**Problem:** The HSTU model uses custom Triton kernels (`generative_recommenders/ops/triton/`) that fail to compile on gfx950 with `RuntimeError: PassManager::run failed` in the AMD Triton backend's `make_ttgir` pass. This is a known Triton compiler bug (triton-lang/triton#9078).

**Fix:** Set `DLRMV3_USE_PYTORCH_OPS=1` (or pass `--pytorch-ops` to the launch scripts). This switches the `HammerKernel` dispatch from `TRITON` to `PYTORCH`, routing all ops through the pure-PyTorch implementations in `generative_recommenders/ops/pytorch/`.

**Files changed:**
- `inference_modules.py` — calls `model.set_hammer_kernel(HammerKernel.PYTORCH)` when the env var is set.

### 2. fbgemm_gpu (gfx950 architecture mismatch)

**Problem:** Pre-built `fbgemm_gpu` wheels lack native `gfx950` code. The `+rocm7.1` wheel has gfx950 but needs ROCm 7.1 runtime (system has 7.0.2). The `+rocm7.0` wheel matches the runtime but only has `gfx942` code.

**Fix:** Build `fbgemm_gpu` from source with `PYTORCH_ROCM_ARCH=gfx950` against the system ROCm:

```bash
git clone --depth 1 --recurse-submodules https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
export ROCM_PATH=/opt/rocm-7.0.2 PYTORCH_ROCM_ARCH=gfx950 BUILD_ROCM_VERSION=7.0
python setup.py --build-variant rocm --build-target default bdist_wheel
pip install dist/*.whl --force-reinstall --no-deps
```

### 3. `os.getlogin()` crash in Slurm

**Problem:** `main.py` uses `os.getlogin()` in an argparse default value, which crashes in Slurm jobs (no controlling terminal).

**Fix:** Replaced with `os.environ.get('USER', 'unknown')`.

### 4. Slow warmup with fallback ops

**Problem:** The autotune warmup loop runs 40 iterations of `predict()`. With the Python-based PyTorch fallback ops, each iteration is slow.

**Fix:** Added `--skip-warmup` flag to bypass the autotune warmup loop.

## New Files

| File | Purpose |
|---|---|
| `launch_slurm.sh` | Slurm job launcher |
| `launch_local.sh` | Direct execution launcher for allocated nodes |
| `tests/test_fbgemm_gpu_ops.py` | fbgemm GPU op tests |
| `tests/test_fbgemm_cumsum.py` | Minimal fbgemm cumsum test |
| `tests/test_triton_ops.py` | Triton vs PyTorch kernel tests |
| `tests/test_triton_basic.py` | Basic Triton compiler test |

## Environment Setup

Build a venv with ROCm-compatible packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1
pip install torchrec --extra-index-url https://download.pytorch.org/whl/rocm7.1
pip install gin_config pandas tensorboard
pip install ../../loadgen

# Build fbgemm_gpu from source (see issue #2 above)
```

## Quick Start

```bash
# Minimal test (8 GPUs, no checkpoint, PyTorch ops)
bash launch_slurm.sh --gpus 8 --pytorch-ops --skip-warmup --num-queries 10 --target-qps 10 --time 1:00:00

# Full benchmark with checkpoint
bash launch_slurm.sh --gpus 8 --pytorch-ops --model-path /path/to/checkpoint/ --time 4:00:00

# On an allocated node
srun --jobid=<JOBID> bash launch_local.sh --gpus 8 --pytorch-ops --skip-warmup --num-queries 10
```

## Verified Results

8x MI355X GPUs, random weights, 10 queries:

```
TestScenario.Server qps=10.72, avg_query_time=0.863s
  Sparse (CPU): 21.5 ms
  Dense (GPU):  153.9 ms
  P99 latency:  863.0 ms
```
