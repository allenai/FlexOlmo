#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
MODEL_PATH="/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-RT/step9537-hf"
BASE_OUTPUT_DIR= # insert S3 bucket path here
BATCH_SIZE=4
CLUSTER="ai2/oe-training"
LIMIT=1000
model_type="hf"

# Define all available tasks from run_eval.sh (matching the original script's active tasks)
TASKS=(
    # core9 tasks
    arc_easy:mc::olmes
    arc_challenge:mc::olmes
    boolq:mc::olmes
    csqa:mc::olmes
    hellaswag:mc::olmes
    openbookqa:mc::olmes
    piqa:mc::olmes
    socialiqa:mc::olmes
    winogrande:mc::olmes

    # MMLU and other math
    mmlu:mc::olmes
    mmlu_pro:mc::none
    gsm8k::olmes
    agi_eval_english:1shot::olmes
    
    # Generation
    drop::olmes
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path//hf/${model_type}}"
}

echo "Launching beaker evaluations for ${#TASKS[@]} tasks..."
echo "Model path: $MODEL_PATH"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each task
for TASK in "${TASKS[@]}"; do
    echo "Launching evaluation for task: $TASK"
    
    # For setting the output_dir (matching original script logic)
    if [[ $MODEL_PATH == "/"* ]]; then
        # internal model
        model=$(get_checkpoint_name $MODEL_PATH)
    else
        # HF model
        model=$(echo $MODEL_PATH | cut -d'/' -f2)
    fi
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"
    
    # GPU allocation (matching original script)
    if [[ $TASK == "news_gen" || $TASK == "poem_gen" ]]; then
        gpus=4
    else
        gpus=1
    fi
    
    # Batch size adjustment (matching original script)
    if [[ $TASK == *"cot"* || $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "ruler"* || $TASK == "sciriff"* ]]; then
        batch_size=1
    else
        batch_size=4
    fi
    
    echo "  Model name: $model"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  GPUs: $gpus"
    echo "  Batch size: $batch_size"
    
    PYTHONPATH=. python src/scripts/eval/launch.py \
        --model $MODEL_PATH \
        --model-args "model_path=$MODEL_PATH,model_type=hf" \
        --task $TASK \
        --limit $LIMIT \
        --remote-output-dir $OUTPUT_DIR \
        --use-gantry \
        --batch-size $batch_size \
        --gpus $gpus \
        --cluster $CLUSTER \
        --beaker-workspace ai2/flex2 \
        --beaker-budget ai2/oe-training \
        --beaker-priority urgent \
        --gantry-secret-aws-access-key-id SANJAYA_AWS_ACCESS_KEY_ID \
        --gantry-secret-aws-secret-access SANJAYA_AWS_SECRET_ACCESS_KEY \
        --gantry-args 'weka=oe-training-default:/data/input,preemptible=False,allow_dirty=true,hf_token=true'
    
    echo "Launched evaluation for $TASK"
    echo "----------------------------------------"
done

echo "All beaker evaluations have been launched!"
echo "Check the beaker dashboard for job status."