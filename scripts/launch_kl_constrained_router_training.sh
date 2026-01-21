#!/bin/bash
# =============================================================================
# Beaker launch script for CE-CONSTRAINED (Hard Label) router training
#
# This script implements a hybrid approach combining standard router training
# (learning from LM loss) with cross-entropy regularization towards hard
# domain-based labels (one-hot vectors).
#
# Total loss: L_total = L_LM + L_Z + λ * CE(hard_labels, router_logits)
#
# Hard labels:
#   - Math data → Expert 0: [1, 0, 0, 0]
#   - Code data → Expert 2: [0, 0, 1, 0]
#   - General data → Expert 1: [0, 1, 0, 0]
#
# Key differences:
#   - vs. Standard RT: Adds CE regularization for guidance
#   - vs. Supervised RT: LM loss is ACTIVE (not router_loss_only)
#   - vs. Soft Label RT: Uses HARD labels, no pre-computed expert losses needed
#
# Advantages:
#   - Simpler: No need to pre-compute expert losses
#   - Cleaner: Labels from domain metadata (already in batch)
#   - Flexible: Router can deviate from domain labels when LM loss benefits
#
# Usage:
#   ./scripts/launch_kl_constrained_router_training.sh           # λ=0.1 (default)
#   CE_WEIGHT=0.05 ./scripts/launch_kl_constrained_router_training.sh  # λ=0.05 (weaker)
#   CE_WEIGHT=0.5 ./scripts/launch_kl_constrained_router_training.sh   # λ=0.5 (stronger)
# =============================================================================

set -e

# Configuration (same as other RT methods for consistency)
NUM_NODES=8
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-experts-sft-math-mixed"
MIX_BASE_DIR="/weka/oe-training-default/ai2-llm/"

# CE-constrained training params (can be overridden via environment variable)
CE_WEIGHT="${CE_WEIGHT:-0.1}"     # λ in L_total = L_LM + λ*CE

# Training params
MAX_TOKENS=5000000000  # 5B tokens (full RT mix)
BATCH_SIZE=262144      # Global batch size (matches other RT methods)
LR=2e-3
WARMUP_STEPS=100

# Generate experiment name with timestamp and ce_weight
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="FlexOlmo-CE-Constrained-RT-lambda${CE_WEIGHT}-${TIMESTAMP}"
SAVE_FOLDER="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/ce_constrained_RT_lambda${CE_WEIGHT}"

echo "=== CE-Constrained Router Training (Hard Labels) ==="
echo ""
echo "Checkpoint:       ${CHECKPOINT}"
echo "Mix base dir:     ${MIX_BASE_DIR}"
echo "CE weight (λ):    ${CE_WEIGHT}"
echo "Max tokens:       ${MAX_TOKENS}"
echo "Learning rate:    ${LR}"
echo ""
echo "Training mode:    LM loss + λ*CE (hybrid with hard labels)"
echo "  - LM loss:      ACTIVE (learns from task performance)"
echo "  - CE term:      λ*CE(hard_labels, router) (domain guidance)"
echo "  - Labels:       HARD one-hot from domain metadata"
echo "  - Router:       Can deviate from domain labels when LM loss benefits"
echo ""
echo "Label mapping:"
echo "  - Math → Expert 0: [1, 0, 0, 0]"
echo "  - Code → Expert 2: [0, 0, 1, 0]"
echo "  - General → Expert 1: [0, 1, 0, 0]"
echo ""
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Save to:          ${SAVE_FOLDER}"
echo ""

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=${EXPERIMENT_NAME} \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-4x7B-kl-constrained-router.py ${EXPERIMENT_NAME} \
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
   --train_module.ce_loss_weight=${CE_WEIGHT} \
   --train_module.dp_config.num_replicas=16 \
   --train_module.ep_config.degree=4 \
   --data_loader.global_batch_size=${BATCH_SIZE}

echo ""
echo "Job submitted: ${EXPERIMENT_NAME}"
echo "Checkpoint will be saved to: ${SAVE_FOLDER}"
echo ""
echo "Hyperparameter sweep suggestions:"
echo "  CE weight (λ): 0.01, 0.05, 0.1, 0.5, 1.0"
echo ""
echo "Monitoring:"
echo "  - train/CE loss: LM loss (should decrease)"
echo "  - train/router CE loss: Domain guidance loss"
echo "  - Ratio CE_router/CE_lm: ~0.1-0.5 with λ=0.1"
echo ""
