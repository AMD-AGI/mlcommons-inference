#!/usr/bin/env bash
# CLAUDE: This entire file was added for AMD ROCm / Slurm integration.
# Submit a 1-node DLRMv3 inference benchmark job via Slurm sbatch.
#
# Usage:
#   # 8 GPUs, default sampled dataset
#   bash launch_slurm.sh
#
#   # 4 GPUs, Offline scenario, custom model checkpoint
#   bash launch_slurm.sh --gpus 4 --scenario Offline --model-path /data/ckpts/streaming_100b/89/
#
#   # Quick 1-GPU smoke test, short time limit
#   bash launch_slurm.sh --gpus 1 --time 0:30:00 --target-qps 10
#
#   # Accuracy evaluation
#   bash launch_slurm.sh --compute-eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────
GPUS=8
PARTITION="amd-aig"
TIME="1:00:00"
JOB_NAME="dlrmv3_infer"
VENV="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"

# main.py arguments
DATASET="sampled-streaming-100b"
SCENARIO="Server"
BATCHSIZE=""
TARGET_QPS=""
MODEL_PATH=""
DATASET_PATH_PREFIX="/shared_aig/chcai/dlrmv3_dataset/"
DATASET_PERCENTAGE=""
NUM_QUERIES=""
COMPUTE_EVAL=""
SPARSE_QUANT=""
PYTORCH_OPS=""
SKIP_WARMUP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)                 GPUS="$2"; shift 2;;
        --partition)            PARTITION="$2"; shift 2;;
        --pytorch-ops)          PYTORCH_OPS="1"; shift;;
        --skip-warmup)          SKIP_WARMUP="1"; shift;;
        --time)                 TIME="$2"; shift 2;;
        --job-name)             JOB_NAME="$2"; shift 2;;
        --venv)                 VENV="$2"; shift 2;;
        --log-dir)              LOG_DIR="$2"; shift 2;;
        --dataset)              DATASET="$2"; shift 2;;
        --scenario)             SCENARIO="$2"; shift 2;;
        --batchsize)            BATCHSIZE="$2"; shift 2;;
        --target-qps)           TARGET_QPS="$2"; shift 2;;
        --model-path)           MODEL_PATH="$2"; shift 2;;
        --dataset-path-prefix)  DATASET_PATH_PREFIX="$2"; shift 2;;
        --dataset-percentage)   DATASET_PERCENTAGE="$2"; shift 2;;
        --num-queries)          NUM_QUERIES="$2"; shift 2;;
        --compute-eval)         COMPUTE_EVAL="1"; shift;;
        --sparse-quant)         SPARSE_QUANT="1"; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Error: venv not found at $VENV"
    echo "Create it first:  python3 -m venv $VENV && source $VENV/bin/activate && pip install -r requirements.txt && pip install ../../loadgen"
    exit 1
fi

mkdir -p "$LOG_DIR"

# Build optional main.py flags
EXTRA_ARGS=""
[[ -n "$BATCHSIZE" ]]           && EXTRA_ARGS+=" --batchsize $BATCHSIZE"
[[ -n "$TARGET_QPS" ]]          && EXTRA_ARGS+=" --target-qps $TARGET_QPS"
[[ -n "$MODEL_PATH" ]]          && EXTRA_ARGS+=" --model-path $MODEL_PATH"
[[ -n "$DATASET_PATH_PREFIX" ]] && EXTRA_ARGS+=" --dataset-path-prefix $DATASET_PATH_PREFIX"
[[ -n "$DATASET_PERCENTAGE" ]]  && EXTRA_ARGS+=" --dataset-percentage $DATASET_PERCENTAGE"
[[ -n "$NUM_QUERIES" ]]         && EXTRA_ARGS+=" --num-queries $NUM_QUERIES"
[[ -n "$COMPUTE_EVAL" ]]        && EXTRA_ARGS+=" --compute-eval True"
[[ -n "$SPARSE_QUANT" ]]        && EXTRA_ARGS+=" --sparse-quant True"
[[ -n "$SKIP_WARMUP" ]]         && EXTRA_ARGS+=" --skip-warmup"

SBATCH_SCRIPT=$(mktemp /tmp/dlrmv3_infer_sbatch_XXXXXX.sh)

cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$GPUS
#SBATCH --cpus-per-task=32
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm_%j.out
#SBATCH --error=$LOG_DIR/slurm_%j.err
#SBATCH --exclusive

set -euo pipefail

export NCCL_DEBUG=WARN

# AINIC RDMA transport
export NCCL_IB_GID_INDEX=1
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_USE_INLINE=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_ECE_ENABLE=0
export NCCL_CROSS_NIC=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DMABUF_ENABLE=1
export NCCL_GDRCOPY_ENABLE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_PXN_DISABLE=0
export NET_OPTIONAL_RECV_COMPLETION=1
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export RCCL_LL128_FORCE_ENABLE=1
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCL_ENABLE=0
export IONIC_LOCKFREE=all
export NCCL_CHECKS_DISABLE=1

echo "============================================"
echo "Job ID:        \$SLURM_JOB_ID"
echo "Node:          \$SLURM_NODELIST"
echo "GPUs:          $GPUS"
echo "Dataset:       $DATASET"
echo "Scenario:      $SCENARIO"
echo "============================================"

cd "$SCRIPT_DIR"

source "$VENV/bin/activate"

export WORLD_SIZE=$GPUS
[[ -n "$PYTORCH_OPS" ]] && export DLRMV3_USE_PYTORCH_OPS=1

python main.py \\
    --dataset $DATASET \\
    --scenario-name $SCENARIO \\
    $EXTRA_ARGS \\
    2>&1 | tee "$LOG_DIR/dlrmv3_infer_\${SLURM_JOB_ID}.log"

echo "Inference complete. Exit code: \$?"
SBATCH_EOF

echo "Generated sbatch script: $SBATCH_SCRIPT"
echo "---"
cat "$SBATCH_SCRIPT"
echo "---"

JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
echo ""
echo "Submitted Slurm job: $JOB_ID"
echo "  Monitor: squeue -j $JOB_ID"
echo "  Logs:    tail -f $LOG_DIR/slurm_${JOB_ID}.out"
echo "  Cancel:  scancel $JOB_ID"
