#!/bin/bash
# =============================================================================
# Beaker launch script for generating PER-TOKEN expert labels from EVAL BENCHMARKS
#
# This is for DEBUGGING purposes only - training on test data is cheating!
# The goal is to understand which training approach works best when training
# data is perfectly in-distribution.
#
# PREREQUISITE: First run create_eval_benchmark_mix.py to generate the data:
#   python src/scripts/train/create_eval_benchmark_mix.py --max_samples_per_task 1000
#
# Usage:
#   ./scripts/launch_eval_benchmark_label_generation.sh
# =============================================================================

# Configuration - try 1 node first to debug, can scale up later
# The 20B model fits on a single GPU, so we don't need 8 nodes for this small dataset
NUM_NODES=1
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code"

# Eval benchmark paths
EVAL_DATA_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data"
OUTPUT_DIR="${EVAL_DATA_DIR}/per_token_labels"
MIX="eval_benchmark_mix"
MIX_BASE_DIR="${EVAL_DATA_DIR}"

BATCH_SIZE=1  # Must be 1 to avoid MoE routing bug with forced experts
MAX_TOKENS=50000000  # 50M tokens (plenty for eval benchmarks ~2M, but gives buffer)
SEQUENCE_LENGTH=4096
SAVE_INTERVAL=100

echo "=== Per-Token Expert Label Generation for EVAL BENCHMARKS ==="
echo ""
echo "⚠️  WARNING: This is for DEBUGGING only (training on test data)!"
echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "Eval data:  ${EVAL_DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Mix:        ${MIX}"
echo "Mix base:   ${MIX_BASE_DIR}"
echo ""

# Note: We don't check if files exist locally since this launches on Beaker
# which has WEKA access. The mix file should be at ${EVAL_DATA_DIR}/${MIX}.txt

# Generate unique experiment name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="FlexOlmo-EvalBench-PerToken-Labels-${TIMESTAMP}"

echo "Experiment: ${EXPERIMENT_NAME}"
echo ""

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
   --dtype bfloat16 \
   --no_source_mixture

echo ""
echo "Job submitted. Output will be saved to: ${OUTPUT_DIR}"
echo ""
echo "After completion, use for supervised router training:"
echo "  --data_loader.expert_labels_dir=${OUTPUT_DIR}"
echo ""

