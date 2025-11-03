#!/bin/bash

# Learning rate sweep for router training
# This script runs the same training configuration with different learning rates

set -e  # Exit on any error

# Define the learning rates to sweep
LEARNING_RATES=("2e-4" "2e-3" "2e-2")

# Base command components
BASE_CMD="python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2"
LAUNCH_ARGS="--launch.num_nodes=8 --launch.num_gpus=8 --launch.budget=ai2/oe-base --launch.workspace=ai2/flex2 --launch.priority=urgent"
TRAIN_SCRIPT="src/scripts/train/OLMoE-4x7B.py"
BASE_RUN_NAME="FlexOlmo-4x7B-RT-midtraining"
TRAINER_ARGS="--trainer.callbacks.profiler.enabled=true"
DATASET_ARGS="--dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ --dataset.mix=router_training_mix_midtraining"
DURATION_ARGS="--trainer.max_duration.value=5_000_000_000 --trainer.max_duration.unit=tokens"
LOAD_PATH="--trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code"
MODEL_ARGS="--model.block.feed_forward_moe.router.top_k=4"
TRAIN_MODULE_ARGS="--train_module.rank_microbatch_size=4096 --train_module.scheduler.warmup_steps=100"

echo "Starting learning rate sweep for router training..."
echo "Learning rates to test: ${LEARNING_RATES[*]}"
echo ""

# Function to run training with a specific learning rate
run_training() {
    local lr=$1
    local run_name="${BASE_RUN_NAME}-lr${lr}"
    
    echo "=========================================="
    echo "Starting training with learning rate: $lr"
    echo "Run name: $run_name"
    echo "=========================================="
    
    # Construct the full command
    local full_cmd="$BASE_CMD $LAUNCH_ARGS -- $TRAIN_SCRIPT $run_name $TRAINER_ARGS $DATASET_ARGS $DURATION_ARGS $LOAD_PATH $MODEL_ARGS $TRAIN_MODULE_ARGS --train_module.optim.lr=$lr"
    
    echo "Command: $full_cmd"
    echo ""
    
    # Execute the command
    eval $full_cmd
    
    if [ $? -eq 0 ]; then
        echo "✅ Training job with LR=$lr completed successfully"
    else
        echo "❌ Training job with LR=$lr failed"
        exit 1
    fi
    
    echo ""
}

# Run training for each learning rate
for lr in "${LEARNING_RATES[@]}"; do
    run_training "$lr"
done

echo "=========================================="
echo "🎉 Learning rate sweep completed!"
echo "All training jobs have been launched."
echo "=========================================="