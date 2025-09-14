#!/bin/bash

# Script to launch beaker evaluations for all 3 models from the learning rate sweep
# Usage: bash scripts/eval/eval_lr_sweep_models.sh

set -e  # Exit on any error

# Configuration
BASE_MODEL_PATH="/weka/oe-training-default/sanjaya/flexolmo/checkpoints"
BASE_OUTPUT_DIR="s3://ai2-sewonm/sanjaya/eval_results"
BATCH_SIZE=4
CLUSTER="ai2/jupiter-cirrascale-2"
LIMIT=1000
MODEL_TYPE="hf"

# Define the 3 models from the learning rate sweep
# These correspond to the run names from the training script
MODELS=(
    "OLMo2-7b-flex-base-merged-math-code-RT-midtraining-lr2e-4"
    "OLMo2-7b-flex-base-merged-math-code-RT-midtraining-lr2e-3" 
    "OLMo2-7b-flex-base-merged-math-code-RT-midtraining-lr2e-2"
)

# Define all available tasks (same as original script)
TASKS=(
    # MC9 tasks
    arc_easy:mc::olmes
    arc_challenge:mc::olmes
    boolq:mc::olmes
    csqa:mc::olmes
    hellaswag:mc::olmes
    openbookqa:mc::olmes
    piqa:mc::olmes
    socialiqa:mc::olmes
    winogrande:mc::olmes
    
    # Gen5 tasks
    coqa::olmes
    squad::olmes
    naturalqs::olmes
    triviaqa::olmes
    drop::olmes

    # MMLU tasks
    mmlu:mc::olmes
    mmlu_pro_mc::none

    # AGI eval
    agi_eval_english:1shot::olmes

    # BBH
    bbh:cot-v1::olmes

    # Math2 tasks
    gsm8k::olmes
    minerva_math_algebra::olmes
    minerva_math_counting_and_probability::olmes
    minerva_math_geometry::olmes
    minerva_math_intermediate_algebra::olmes
    minerva_math_number_theory::olmes
    minerva_math_prealgebra::olmes
    minerva_math_precalculus::olmes

    # Code4 tasks
    codex_humaneval:temp0.8
    codex_humanevalplus:temp0.8
    mbpp::none
    mbppplus::none
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path//hf/${MODEL_TYPE}}"
}

# Function to find the latest checkpoint for a model
function find_latest_checkpoint {
    local model_name=$1
    local model_path="${BASE_MODEL_PATH}/${model_name}"
    
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path $model_path does not exist"
        return 1
    fi
    
    # Find the latest step checkpoint
    local latest_checkpoint=$(find "$model_path" -name "step*-hf" -type d | sort -V | tail -1)
    
    if [ -z "$latest_checkpoint" ]; then
        echo "Warning: No checkpoint found in $model_path"
        return 1
    fi
    
    echo "$latest_checkpoint"
}

# Function to launch evaluation for a single model and task
launch_evaluation() {
    local model_name=$1
    local task=$2
    local model_path=$3
    
    echo "Launching evaluation for model: $model_name, task: $task"
    
    # Get checkpoint name for output directory
    local checkpoint_name=$(get_checkpoint_name "$model_path")
    local output_dir="${BASE_OUTPUT_DIR}/${checkpoint_name}"
    
    # Determine GPU count and batch size
    local gpus=4
    local batch_size=$BATCH_SIZE
    
    # Batch size adjustment (matching original script)
    if [[ $task == *"cot"* || $task == "minerva_math_"* || $task == "mbpp"* || $task == "bigcodebench"* || $task == "ruler"* || $task == "sciriff"* ]]; then
        batch_size=1
    fi
    
    # Create a shorter, valid job name
    local safe_model_name=$(echo $checkpoint_name | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-20)
    local safe_task_name=$(echo $task | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-15)
    local job_name="eval-${safe_model_name}-${safe_task_name}"
    
    echo "  Model path: $model_path"
    echo "  Output dir: $output_dir"
    echo "  GPUs: $gpus"
    echo "  Batch size: $batch_size"
    echo "  Job name: $job_name"
    
    gantry run \
        --name $job_name \
        --weka oe-training-default:/weka/oe-training-default \
        --install "bash src/scripts/eval/setup_eval_env.sh;" \
        --budget ai2/oe-base \
        --workspace ai2/flex2 \
        --cluster $CLUSTER \
        --priority urgent \
        --gpus $gpus \
        --env-secret HF_TOKEN=SANJAYA_HF_TOKEN \
        --env-secret AWS_ACCESS_KEY_ID=SANJAYA_AWS_ACCESS_KEY_ID \
        --env-secret AWS_SECRET_ACCESS_KEY=SANJAYA_AWS_SECRET_ACCESS_KEY \
        -- \
        bash -c "PYTHONPATH=. python src/scripts/eval/launch_eval.py \
            --model $model_path \
            --model-type $MODEL_TYPE \
            --task $task \
            --limit $LIMIT \
            --remote-output-dir $output_dir \
            --batch-size $batch_size \
            --gpus $gpus"
    
    echo "✅ Launched evaluation for $model_name - $task"
    echo "----------------------------------------"
}

echo "🚀 Starting evaluation sweep for ${#MODELS[@]} models across ${#TASKS[@]} tasks"
echo "Models: ${MODELS[*]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Check if models exist and find their latest checkpoints
echo "Checking models and finding latest checkpoints..."
for model in "${MODELS[@]}"; do
    echo "Checking model: $model"
    checkpoint_path=$(find_latest_checkpoint "$model")
    if [ $? -eq 0 ]; then
        echo "✅ Found checkpoint: $checkpoint_path"
    else
        echo "❌ Skipping model: $model (checkpoint not found)"
    fi
    echo ""
done

# Launch evaluations for each model-task combination
total_jobs=0
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating model: $model"
    echo "=========================================="
    
    # Find checkpoint for this model
    checkpoint_path=$(find_latest_checkpoint "$model")
    if [ $? -eq 0 ]; then
        for task in "${TASKS[@]}"; do
            launch_evaluation "$model" "$task" "$checkpoint_path"
            ((total_jobs++))
        done
    else
        echo "❌ Skipping model: $model (checkpoint not found)"
    fi
done

echo "=========================================="
echo "🎉 Evaluation sweep completed!"
echo "Total jobs launched: $total_jobs"
echo "Check the beaker dashboard for job status."
echo "=========================================="