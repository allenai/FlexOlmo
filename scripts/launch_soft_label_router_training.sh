#!/bin/bash
# =============================================================================
# Beaker launch script for SOFT LABEL router training
#
# This script trains the router using soft labels derived from expert losses
# via knowledge distillation (KL divergence), instead of hard argmin labels.
#
# Soft labels: q(e) = softmax(-β * (L_e - min(L)))
# Loss: KL(q || p_θ) = -Σ_e q(e) log p_θ(e) + const
#
# PREREQUISITE: First run label generation to get all_expert_losses:
#   ./scripts/launch_eval_benchmark_label_generation_sft.sh
#
# Usage:
#   ./scripts/launch_soft_label_router_training.sh           # β=1.0 (default)
#   BETA=0.5 ./scripts/launch_soft_label_router_training.sh  # β=0.5 (softer)
#   BETA=2.0 ./scripts/launch_soft_label_router_training.sh  # β=2.0 (sharper)
# =============================================================================

set -e

# Configuration
NUM_NODES=8
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-experts-sft-math-mixed"
LABELS_DIR="/weka/oe-training-default/sanjaya/flexolmo/expert_labels/optimal_labels_per_token_5B_sft_math_mixed"
MIX_BASE_DIR="/weka/oe-training-default/ai2-llm/"

# Soft label temperature (can be overridden via environment variable)
BETA="${BETA:-1.0}"

# Training params
MAX_TOKENS=5000000000  # 5B tokens (full RT mix)
BATCH_SIZE=262144      # Global batch size
LR=2e-3
WARMUP_STEPS=100

# Generate experiment name with timestamp and beta
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="FlexOlmo-SoftLabelRT-beta${BETA}-${TIMESTAMP}"
SAVE_FOLDER="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/soft_label_RT_beta${BETA}"

echo "=== Soft Label Router Training (Full RT Mix) ==="
echo ""
echo "Checkpoint:    ${CHECKPOINT}"
echo "Labels dir:    ${LABELS_DIR}"
echo "Mix base dir:  ${MIX_BASE_DIR}"
echo "Beta:          ${BETA}"
echo "Max tokens:    ${MAX_TOKENS}"
echo "Learning rate: ${LR}"
echo ""
echo "Experiment:    ${EXPERIMENT_NAME}"
echo "Save to:       ${SAVE_FOLDER}"
echo ""

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=${EXPERIMENT_NAME} \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-4x7B-soft-label-router.py ${EXPERIMENT_NAME} \
   --trainer.callbacks.profiler.enabled=false \
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
   --train_module.router_loss_weight=1.0 \
   --train_module.router_loss_only=true \
   --train_module.soft_label_beta=${BETA} \
   --train_module.expert_labels_dir=${LABELS_DIR} \
   --train_module.dp_config.num_replicas=16 \
   --train_module.ep_config.degree=4 \
   --data_loader.global_batch_size=${BATCH_SIZE}

echo ""
echo "Job submitted: ${EXPERIMENT_NAME}"
echo "Checkpoint will be saved to: ${SAVE_FOLDER}"
echo ""
