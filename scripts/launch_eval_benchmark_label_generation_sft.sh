#!/bin/bash
# =============================================================================
# Beaker launch script for generating PER-TOKEN expert labels from EVAL BENCHMARKS
# Using the SFT model (math-code-experts-sft-math-mixed)
#
# This is for DEBUGGING purposes only - training on test data is cheating!
# The goal is to compare label distributions between base and SFT models.
#
# Usage:
#   ./scripts/launch_eval_benchmark_label_generation_sft.sh
# =============================================================================

# Configuration - 1 node is enough for this small dataset
NUM_NODES=1
NUM_GPUS=8
CHECKPOINT="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-experts-sft-math-mixed"

# Eval benchmark paths - output to separate directory for SFT model
EVAL_DATA_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data"
OUTPUT_DIR="${EVAL_DATA_DIR}/per_token_labels_sft"
MIX="eval_benchmark_mix"
MIX_BASE_DIR="${EVAL_DATA_DIR}"

BATCH_SIZE=1  # Must be 1 to avoid MoE routing bug with forced experts
MAX_TOKENS=50000000  # 50M tokens (plenty for eval benchmarks ~2M, but gives buffer)
SEQUENCE_LENGTH=4096
SAVE_INTERVAL=100

echo "=== Per-Token Expert Label Generation for EVAL BENCHMARKS (SFT Model) ==="
echo ""
echo "⚠️  WARNING: This is for DEBUGGING only (training on test data)!"
echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "Eval data:  ${EVAL_DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Mix:        ${MIX}"
echo "Mix base:   ${MIX_BASE_DIR}"
echo ""

# Generate unique experiment name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="FlexOlmo-EvalBench-PerToken-Labels-SFT-${TIMESTAMP}"

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
echo "After completion, run analysis:"
echo "  python src/scripts/train/analyze_expert_labels.py \\"
echo "      --labels_dir ${OUTPUT_DIR} \\"
echo "      --data_dir ${EVAL_DATA_DIR} \\"
echo "      --output expert_label_analysis_sft.txt"
echo ""

