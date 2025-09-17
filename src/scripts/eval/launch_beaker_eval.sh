#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
MODELS=("/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-RT-experts-sft/step9537")
BASE_OUTPUT_DIR="s3://ai2-sewonm/sanjaya/eval_results"
BATCH_SIZE=4
CLUSTER="ai2/jupiter-cirrascale-2"
LIMIT=1000
model_type=hf

# Define all available tasks from run_eval.sh (ALL tasks from all groups)
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
    echo "${modified_path//hf/${model_type}}"
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASKS[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL_PATH in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_PATH"
    
    # For setting the output_dir (matching original script logic)
    if [[ $MODEL_PATH == "/"* ]]; then
        # internal model
        model=$(get_checkpoint_name $MODEL_PATH)
    else
        # HF model
        model=$(echo $MODEL_PATH | cut -d'/' -f2)
    fi
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"
    
    for TASK in "${TASKS[@]}"; do
        echo "Launching evaluation for model: $model, task: $TASK"
    
    gpus=4
    
    # Batch size adjustment (matching original script)
    if [[ $TASK == *"cot"* || $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "ruler"* || $TASK == "sciriff"* ]]; then
        batch_size=1
    else
        batch_size=4
    fi
    
    # Create a shorter, valid job name
    # Remove invalid characters and truncate long names
    safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-20)
    safe_task_name=$(echo $TASK | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-15)
    job_name="eval-${safe_model_name}-${safe_task_name}"
    
    echo "  Model name: $model"
    echo "  Output dir: $OUTPUT_DIR"
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
            --model $MODEL_PATH \
            --model-type hf \
            --task $TASK \
            --limit $LIMIT \
            --remote-output-dir $OUTPUT_DIR \
            --batch-size $batch_size \
            --gpus $gpus"
    
        echo "Launched evaluation for model: $model, task: $TASK"
        echo "----------------------------------------"
    done
    
    echo "Completed all tasks for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASKS[@]}))"
echo "Check the beaker dashboard for job status."