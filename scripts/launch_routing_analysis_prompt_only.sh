#!/bin/bash
# =============================================================================
# Beaker launch script for prompt-only router probability analysis
#
# Analyzes expert routing patterns by running forward passes on task prompts
# and extracting router softmax probabilities at each MoE layer.
#
# Usage:
#   bash scripts/launch_routing_analysis_prompt_only.sh
# =============================================================================

MODEL_PATH="/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4/step66-hf"
CLUSTER="ai2/jupiter-cirrascale-2"
GPUS=1

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="routing-analysis-prompt-only-${TIMESTAMP}"

echo "=== Prompt-Only Router Analysis ==="
echo "Model:   ${MODEL_PATH}"
echo "Job:     ${JOB_NAME}"
echo "Cluster: ${CLUSTER}"
echo "GPUs:    ${GPUS}"
echo ""

gantry run \
    --name "${JOB_NAME}" \
    --weka oe-training-default:/weka/oe-training-default \
    --beaker-image jacobm/oe-eval-olmo3-official-test-4 \
    --budget ai2/oe-base \
    --workspace ai2/flex2 \
    --cluster "${CLUSTER}" \
    --priority urgent \
    --gpus "${GPUS}" \
    --env-secret HF_TOKEN=SANJAYA_HF_TOKEN \
    --env-secret AWS_ACCESS_KEY_ID=SANJAYA_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=SANJAYA_AWS_SECRET_ACCESS_KEY \
    -- \
    bash -c "cd /weka/oe-training-default/sanjaya/FlexOlmo && \
        pip install pyyaml matplotlib seaborn && \
        python routing_analysis_prompt_only.py \
            --config routing_analysis_prompt_only_config.yaml \
            --output-dir ./routing_analysis_output \
            --visualize"

echo ""
echo "Job submitted: ${JOB_NAME}"
echo "Output will be in: routing_analysis_output/"
