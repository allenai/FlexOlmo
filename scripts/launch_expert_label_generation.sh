#!/bin/bash
# Beaker launch script for generating optimal expert labels
#
# This script runs inference on the training data with each expert forced to be selected,
# computes the per-sequence loss, and stores the expert that minimizes loss as the label.
#
# Output: A JSON file mapping sequence indices to optimal expert IDs
#
# Usage:
#   ./scripts/launch_expert_label_generation.sh
#
# The output will be saved to:
#   /weka/oe-training-default/sanjaya/flexolmo/expert_labels/optimal_labels_5B.json

# Configuration
NUM_NODES=8
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code"
OUTPUT_DIR="/weka/oe-training-default/sanjaya/flexolmo/expert_labels"
OUTPUT_FILE="${OUTPUT_DIR}/optimal_labels_5B.json"
MIX="router_training_mix"
MIX_BASE_DIR="/weka/oe-training-default/ai2-llm/"
BATCH_SIZE=4  # Per-GPU batch size (conservative for memory)
MAX_TOKENS=5000000000  # 5B tokens
SEQUENCE_LENGTH=4096
SAVE_INTERVAL=5000  # Save intermediate results every 5000 batches

echo "=== Expert Label Generation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_FILE}"
echo "Mix: ${MIX}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Nodes: ${NUM_NODES}, GPUs per node: ${NUM_GPUS}"
echo ""

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=FlexOlmo-Expert-Label-Generation \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=normal -- src/scripts/train/generate_expert_labels.py \
   --checkpoint ${CHECKPOINT} \
   --output ${OUTPUT_FILE} \
   --mix ${MIX} \
   --mix_base_dir ${MIX_BASE_DIR} \
   --batch_size ${BATCH_SIZE} \
   --max_tokens ${MAX_TOKENS} \
   --sequence_length ${SEQUENCE_LENGTH} \
   --save_interval ${SAVE_INTERVAL} \
   --dtype bfloat16

echo ""
echo "Job submitted. Output will be saved to: ${OUTPUT_FILE}"
echo ""
echo "After completion, use the labels file with supervised router training by setting:"
echo "  --expert_labels_file ${OUTPUT_FILE}"

