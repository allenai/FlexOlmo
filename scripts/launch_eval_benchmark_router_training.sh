#!/bin/bash
# =============================================================================
# Beaker launch script for supervised router training on EVAL BENCHMARKS
#
# This is for DEBUGGING purposes only - training on test data is cheating!
# The goal is to understand which training approach works best when training
# data is perfectly in-distribution.
#
# Three training modes:
#   MODE=llm       - Normal RT with LLM loss (language modeling)
#   MODE=classify  - Classification-based RT (domain labels → experts)
#   MODE=pertoken  - Per-token RT (requires pre-computed labels)
#
# PREREQUISITE: First run create_eval_benchmark_mix.py:
#   python src/scripts/train/create_eval_benchmark_mix.py --max_samples_per_task 1000
#
# For MODE=pertoken, also run label generation first:
#   ./scripts/launch_eval_benchmark_label_generation.sh
#
# Usage:
#   MODE=llm ./scripts/launch_eval_benchmark_router_training.sh
#   MODE=classify ./scripts/launch_eval_benchmark_router_training.sh
#   MODE=pertoken ./scripts/launch_eval_benchmark_router_training.sh
# =============================================================================

set -e

# Training mode: llm, classify, or pertoken
MODE="${MODE:-classify}"

# Common configuration
NUM_NODES=8
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code"

# Eval benchmark paths
EVAL_DATA_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data"
MIX="eval_benchmark_mix"
MIX_BASE_DIR="${EVAL_DATA_DIR}"
PER_TOKEN_LABELS_DIR="${EVAL_DATA_DIR}/per_token_labels"

# Training params
MAX_TOKENS=500000000  # 500M tokens
BATCH_SIZE=262144     # Global batch size
LR=2e-3
WARMUP_STEPS=100

echo "=== Supervised Router Training on EVAL BENCHMARKS ==="
echo ""
echo "⚠️  WARNING: This is for DEBUGGING only (training on test data)!"
echo ""
echo "Mode:       ${MODE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Mix:        ${MIX}"
echo "Mix base:   ${MIX_BASE_DIR}"
echo ""

# Check if eval benchmark data exists
if [[ ! -f "${EVAL_DATA_DIR}/${MIX}.txt" ]]; then
    echo "ERROR: Eval benchmark mix not found at ${EVAL_DATA_DIR}/${MIX}.txt"
    echo ""
    echo "First run:"
    echo "  python src/scripts/train/create_eval_benchmark_mix.py --max_samples_per_task 1000"
    exit 1
fi

# Generate experiment name based on mode
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
case ${MODE} in
    llm)
        EXPERIMENT_NAME="FlexOlmo-EvalBench-RT-LLMLoss-${TIMESTAMP}"
        SAVE_FOLDER="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/eval_benchmark_RT_llm_loss"
        TRAIN_SCRIPT="src/scripts/train/OLMoE-4x7B.py"
        EXTRA_ARGS=""
        ;;
    classify)
        EXPERIMENT_NAME="FlexOlmo-EvalBench-RT-Classify-${TIMESTAMP}"
        SAVE_FOLDER="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/eval_benchmark_RT_classify"
        TRAIN_SCRIPT="src/scripts/train/OLMoE-4x7B-supervised-router.py"
        EXTRA_ARGS="--train_module.router_loss_weight=1.0 --train_module.router_loss_only=true"
        ;;
    pertoken)
        EXPERIMENT_NAME="FlexOlmo-EvalBench-RT-PerToken-${TIMESTAMP}"
        SAVE_FOLDER="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/eval_benchmark_RT_pertoken"
        TRAIN_SCRIPT="src/scripts/train/OLMoE-4x7B-supervised-router.py"
        
        # Check if per-token labels exist
        if [[ ! -d "${PER_TOKEN_LABELS_DIR}" ]]; then
            echo "ERROR: Per-token labels not found at ${PER_TOKEN_LABELS_DIR}"
            echo ""
            echo "First run:"
            echo "  ./scripts/launch_eval_benchmark_label_generation.sh"
            exit 1
        fi
        
        LABELED_INDICES_FILE="${PER_TOKEN_LABELS_DIR}/labeled_indices.npy"
        EXTRA_ARGS="--train_module.router_loss_weight=1.0 --train_module.router_loss_only=true"
        EXTRA_ARGS="${EXTRA_ARGS} --data_loader.expert_labels_dir=${PER_TOKEN_LABELS_DIR}"
        EXTRA_ARGS="${EXTRA_ARGS} --data_loader.labeled_indices_file=${LABELED_INDICES_FILE}"
        ;;
    *)
        echo "ERROR: Unknown mode '${MODE}'. Use: llm, classify, or pertoken"
        exit 1
        ;;
esac

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Save to:    ${SAVE_FOLDER}"
echo ""

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=${EXPERIMENT_NAME} \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- ${TRAIN_SCRIPT} ${EXPERIMENT_NAME} \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix=${MIX} \
   --dataset.mix_base_dir=${MIX_BASE_DIR} \
   --dataset.include_instance_metadata=true \
   --trainer.max_duration.value=${MAX_TOKENS} \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${CHECKPOINT} \
   --trainer.save_folder=${SAVE_FOLDER} \
   --model.block.feed_forward_moe.num_experts=4 \
   --model.block.feed_forward_moe.router.top_k=4 \
   --train_module.rank_microbatch_size=4096 \
   --train_module.scheduler.warmup_steps=${WARMUP_STEPS} \
   --train_module.optim.lr=${LR} \
   --train_module.dp_config.num_replicas=16 \
   --train_module.ep_config.degree=4 \
   --data_loader.global_batch_size=${BATCH_SIZE} \
   ${EXTRA_ARGS}

echo ""
echo "Job submitted: ${EXPERIMENT_NAME}"
echo "Checkpoint will be saved to: ${SAVE_FOLDER}"
echo ""

