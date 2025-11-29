#!/bin/bash
# Beaker launch script for generating PER-TOKEN optimal expert labels
#
# This script runs inference on the training data with each expert forced to be selected,
# computes the per-token loss, and stores the expert that minimizes loss for EACH TOKEN.
#
# IMPORTANT: This uses the default NumpyDataset behavior which samples proportionally
# to token count (file size), matching how --dataset.mix=router_training_mix works
# in training. This respects the intended ~uniform token distribution across domains
# that you designed in router_training_mix.txt.
#
# Output: A directory of numpy files, one per sequence, each containing per-token labels
#
# Usage:
#   ./scripts/launch_expert_label_generation_per_token.sh
#
# The output will be saved to:
#   /weka/oe-training-default/sanjaya/flexolmo/expert_labels/optimal_labels_per_token_5B_v2/

# Configuration
NUM_NODES=8
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code"
OUTPUT_DIR="/weka/oe-training-default/sanjaya/flexolmo/expert_labels/optimal_labels_per_token_5B"
MIX="router_training_mix"
MIX_BASE_DIR="/weka/oe-training-default/ai2-llm/"
BATCH_SIZE=1  # Per-GPU batch size (must be 1 to avoid MoE routing bug with forced experts)
MAX_TOKENS=5000000000  # 5B tokens
SEQUENCE_LENGTH=4096
SAVE_INTERVAL=1000  # Log progress every 1000 batches

echo "=== Per-Token Expert Label Generation (Token-Weighted Sampling) ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Mix: ${MIX}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Nodes: ${NUM_NODES}, GPUs per node: ${NUM_GPUS}"
echo ""
echo "NOTE: Sampling respects token counts (file sizes), matching training behavior."
echo ""

# Generate unique experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="FlexOlmo-PerToken-Label-Gen-${TIMESTAMP}"

echo "Experiment name: ${EXPERIMENT_NAME}"

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=${EXPERIMENT_NAME} \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/generate_expert_labels_per_token.py \
   --checkpoint ${CHECKPOINT} \
   --output_dir ${OUTPUT_DIR} \
   --mix ${MIX} \
   --mix_base_dir ${MIX_BASE_DIR} \
   --batch_size ${BATCH_SIZE} \
   --max_tokens ${MAX_TOKENS} \
   --sequence_length ${SEQUENCE_LENGTH} \
   --save_interval ${SAVE_INTERVAL} \
   --dtype bfloat16

echo ""
echo "Job submitted. Output will be saved to: ${OUTPUT_DIR}"
echo ""
echo "Token distribution will match training (~uniform across Math/Code/General)."
echo ""
echo "After completion, use the labels directory with supervised router training by setting:"
echo "  --data_loader.expert_labels_dir=${OUTPUT_DIR}"
echo ""
echo "Each rank saves stats_rank_XXX.json before distributed sync, so data is preserved"
echo "even if NCCL times out."

