#!/bin/bash
# Midtraining script for 2x7B models
# This script trains router for both Math-2x7B and Code-2x7B models consecutively (stolen from sanjay)

set -e  # Exit on error

echo "=========================================="
echo "Starting Mid Training for 2x7B Models"
echo "=========================================="
echo ""

# Math-2x7B Mid Training
echo "🚀 Launching Math-2x7B Mid Training..."
echo ""

MATH_EXPERT=/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/flex-experiments/experts/math-base
CODE_EXPERT=/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/flex-experiments/experts/code-base

#    --launch.budget=ai2/oe-base \

python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.num_nodes=4 \
   --launch.num_gpus=8 \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal.py Flex-2x7B-math-mid-train \
   --trainer.callbacks.profiler.enabled=true \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=mj_finemath4plus \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${MATH_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=4096 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/flex-experiments/experts/math-midtrain


# torchrun --nproc-per-node=8 src/scripts/train/OLMoE-2x7B-anneal.py olmoe-2x7B-${EXPERT}_top2_grit_learnbias \
    # --dataset.mix_base_dir=${DATA_ROOT} \
    # --dataset.mix=${EXPERT} \