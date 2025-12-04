#!/bin/bash
# Beaker launch script for supervised router training with EVAL BENCHMARK ORACLE LABELS
#
# This script uses pre-computed per-token optimal expert labels generated from eval benchmarks.
# These are "oracle" labels that represent the expert that minimizes per-token loss.
#
# Prerequisites:
#   1. Run per-token label generation on eval benchmarks first:
#      ./scripts/launch_eval_benchmark_label_generation.sh  # For base model
#      ./scripts/launch_eval_benchmark_label_generation_sft.sh  # For SFT model (optional)
#   2. Wait for it to complete and produce the labels directory
#
# IMPORTANT: This is a very small dataset (180 sequences), so training will:
#   - Use fewer nodes (1-2 nodes instead of 8)
#   - Train for fewer tokens but may loop over data multiple times
#   - This is primarily for experimentation/analysis purposes

# Choose which labels to use:
# Option 1: Base model labels
EXPERT_LABELS_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels"

# Option 2: SFT model labels (uncomment to use instead)
# EXPERT_LABELS_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels_sft"

LABELED_INDICES_FILE="${EXPERT_LABELS_DIR}/labeled_indices.npy"

echo "=== Supervised Router Training with Eval Benchmark Oracle Labels ==="
echo "Expert labels directory: ${EXPERT_LABELS_DIR}"
echo "Labeled indices file: ${LABELED_INDICES_FILE}"
echo ""
echo "WARNING: This dataset contains only ~180 sequences from eval benchmarks."
echo "         This is primarily for experimentation/analysis, not production training."
echo "         Training will ONLY use sequences that have per-token labels."
echo ""

# Check if labels directory exists (only warn, since weka may not be mounted locally)
if [ ! -d "${EXPERT_LABELS_DIR}" ]; then
    echo "WARNING: Cannot verify labels directory exists locally: ${EXPERT_LABELS_DIR}"
    echo "         (This is normal if weka is not mounted on this machine)"
    echo "         The job will verify this on Beaker where weka is mounted."
else
    echo "✓ Labels directory found: ${EXPERT_LABELS_DIR}"
fi

if [ ! -f "${LABELED_INDICES_FILE}" ]; then
    echo "WARNING: Cannot verify labeled indices file exists locally: ${LABELED_INDICES_FILE}"
    echo "         (This is normal if weka is not mounted on this machine)"
else
    echo "✓ Labeled indices file found: ${LABELED_INDICES_FILE}"
fi
echo ""

# Generate unique experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MODEL_TYPE=$(basename $(dirname $(dirname ${EXPERT_LABELS_DIR})))
LABEL_TYPE=$(basename ${EXPERT_LABELS_DIR})
EXPERIMENT_NAME="FlexOlmo-4x7B-Supervised-RT-EvalOracle-${LABEL_TYPE}-${TIMESTAMP}"

# Use 2 nodes for better memory distribution (helps with OOM issues)
NUM_NODES=2
NUM_GPUS=8

# Reduced token budget since dataset is small (~180 sequences * 4096 tokens = ~737k tokens)
# Set to process data multiple times (e.g., 10 passes = ~7.3M tokens)
MAX_TOKENS=10000000  # ~10M tokens (about 13-14 passes through the data)

echo "Configuration:"
echo "  Nodes: ${NUM_NODES}"
echo "  GPUs per node: ${NUM_GPUS}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo ""

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=${EXPERIMENT_NAME} \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-4x7B-supervised-router-eval-benchmark.py FlexOlmo-4x7B-Supervised-RT-EvalOracle \
   --data_loader.expert_labels_dir=${EXPERT_LABELS_DIR} \
   --data_loader.labeled_indices_file=${LABELED_INDICES_FILE} \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/sanjaya/eval_benchmark_data \
   --dataset.include_instance_metadata=true \
   --trainer.max_duration.value=${MAX_TOKENS} \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code \
   --trainer.save_folder=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-RT-supervised-router-eval-oracle-${LABEL_TYPE} \
   --model.block.feed_forward_moe.num_experts=4 \
   --model.block.feed_forward_moe.router.top_k=4 \
   --train_module.rank_microbatch_size=4096 \
   --train_module.scheduler.warmup_steps=100 \
   --train_module.optim.lr=2e-3 \
   --train_module.router_loss_weight=1.0 \
   --train_module.router_loss_only=true \
   --train_module.dp_config.num_replicas=4 \
   --train_module.ep_config.degree=4 \
   --data_loader.global_batch_size=65536

echo ""
echo "Job submitted. Training with eval benchmark oracle (per-token optimal) expert labels."
echo "Only labeled sequences (~180) will be used."
echo ""
echo "NOTE: This is a small experimental dataset. Consider using training data labels"
echo "      for production supervised router training:"
echo "      ./scripts/launch_supervised_router_training_per_token.sh"

