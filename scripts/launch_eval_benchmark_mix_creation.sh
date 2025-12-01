#!/bin/bash
# =============================================================================
# Create router training mix from evaluation benchmarks
# 
# This creates training data from test benchmarks for DEBUGGING purposes only.
# The goal is to understand which training approach works best when training
# data is perfectly in-distribution.
#
# Default output: /weka/oe-training-default/sanjaya/eval_benchmark_data/
#
# Usage:
#   ./scripts/launch_eval_benchmark_mix_creation.sh
#   ./scripts/launch_eval_benchmark_mix_creation.sh --max_samples 500
#   ./scripts/launch_eval_benchmark_mix_creation.sh --output_dir /my/path
# =============================================================================

set -e

DEFAULT_OUTPUT_DIR="/weka/oe-training-default/sanjaya/eval_benchmark_data"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
TOKENIZER="${TOKENIZER:-allenai/dolma2-tokenizer}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MIX_NAME="${MIX_NAME:-eval_benchmark_mix}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --mix_name) MIX_NAME="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Creating Router Training Mix from Evaluation Benchmarks"
echo "============================================================"
echo ""
echo "⚠️  WARNING: Training on TEST data - for DEBUGGING only!"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Max samples:      ${MAX_SAMPLES:-'all'}"
echo ""

CMD="python src/scripts/train/create_eval_benchmark_mix.py \
    --output_dir ${OUTPUT_DIR} \
    --tokenizer ${TOKENIZER} \
    --sequence_length ${SEQUENCE_LENGTH} \
    --mix_name ${MIX_NAME}"

[[ -n "${MAX_SAMPLES}" ]] && CMD="${CMD} --max_samples_per_task ${MAX_SAMPLES}"
[[ -n "${BENCHMARKS}" ]] && CMD="${CMD} --benchmarks ${BENCHMARKS}"

echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "============================================================"
echo "Next steps:"
echo "============================================================"
echo ""
echo "The mix file was auto-copied to src/flexolmo/data/mixes/${MIX_NAME}.txt"
echo ""
echo "Use in training scripts:"
echo "   --dataset.mix=${MIX_NAME} --dataset.mix_base_dir=${OUTPUT_DIR}"
echo ""
