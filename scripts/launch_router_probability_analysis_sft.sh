#!/bin/bash
# =============================================================================
# Beaker launch script for analyzing router probabilities on EVAL BENCHMARKS
# Using the SFT model (math-code-experts-sft-math-mixed)
#
# This captures the router's softmax probability distribution directly,
# instead of computing losses by forcing each expert. This provides a more
# direct measure of what the router "wants" to do.
#
# Usage:
#   ./scripts/launch_router_probability_analysis_sft.sh
#
# For local execution (on a machine with GPUs and data access):
#   python src/scripts/train/analyze_router_probabilities.py \
#       --checkpoint /path/to/checkpoint \
#       --data_dir /path/to/eval_benchmark_data \
#       --output router_probability_analysis_sft.txt
# =============================================================================

# Configuration
NUM_NODES=1
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-experts-sft-math-mixed"

# Eval benchmark paths
EVAL_DATA_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data"
OUTPUT_FILE="router_probability_analysis_sft.txt"

BATCH_SIZE=1  # Can use larger batches since we're not forcing experts
MAX_SEQUENCES=0  # 0 = process all sequences
SEQUENCE_LENGTH=4096
NUM_EXPERTS=4

echo "=== Router Probability Analysis for EVAL BENCHMARKS (SFT Model) ==="
echo ""
echo "This analysis captures router softmax probabilities directly (single forward pass)"
echo "instead of computing losses by forcing each expert (3x forward passes)."
echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "Eval data:  ${EVAL_DATA_DIR}"
echo "Output:     ${OUTPUT_FILE}"
echo ""

# Generate unique experiment name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="FlexOlmo-RouterProbAnalysis-SFT-${TIMESTAMP}"

echo "Experiment: ${EXPERIMENT_NAME}"
echo ""

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.name=${EXPERIMENT_NAME} \
   --launch.num_nodes=${NUM_NODES} \
   --launch.num_gpus=${NUM_GPUS} \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/analyze_router_probabilities.py \
   --checkpoint ${CHECKPOINT} \
   --data_dir ${EVAL_DATA_DIR} \
   --output ${OUTPUT_FILE} \
   --batch_size ${BATCH_SIZE} \
   --max_sequences ${MAX_SEQUENCES} \
   --sequence_length ${SEQUENCE_LENGTH} \
   --num_experts ${NUM_EXPERTS} \
   --dtype bfloat16

echo ""
echo "Job submitted. Output will be saved to: ${OUTPUT_FILE}"
echo ""
echo "The analysis will show:"
echo "  - Overall expert probability distribution (router softmax)"
echo "  - Per-domain breakdown"
echo "  - Per-sequence details"
echo "  - Comparison with source-based expectations"
echo ""
