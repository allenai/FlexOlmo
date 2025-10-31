#!/bin/bash
# Router training script for 2x7B models
# This script trains router for both Math-2x7B and Code-2x7B models consecutively

set -e  # Exit on error

echo "=========================================="
echo "Starting Router Training for 2x7B Models"
echo "=========================================="
echo ""

# Math-2x7B Router Training
echo "🚀 Launching Math-2x7B Router Training..."
echo ""

python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oe-base \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B.py Math-2x7B-RT \
   --trainer.callbacks.profiler.enabled=true \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=math_general_rt_mix \
   --trainer.max_duration.value=5_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/merged-2x7B-general-math \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=4096 \
   --train_module.scheduler.warmup_steps=100 \
   --train_module.optim.lr=2e-3 \
   --trainer.save_folder=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-mixed-RT

if [ $? -ne 0 ]; then
    echo "❌ Math-2x7B Router Training launch failed!"
    exit 1
fi

echo ""
echo "✅ Math-2x7B Router Training launched successfully!"
echo ""
echo "=========================================="
echo ""

# Code-2x7B Router Training
echo "🚀 Launching Code-2x7B Router Training..."
echo ""

python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oe-base \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B.py Code-2x7B-RT \
   --trainer.callbacks.profiler.enabled=true \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=code_general_rt_mix \
   --trainer.max_duration.value=5_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/merged-2x7B-general-code \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=4096 \
   --train_module.scheduler.warmup_steps=100 \
   --train_module.optim.lr=2e-3 \
   --trainer.save_folder=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-code-RT

if [ $? -ne 0 ]; then
    echo "❌ Code-2x7B Router Training launch failed!"
    exit 1
fi

echo ""
echo "✅ Code-2x7B Router Training launched successfully!"
echo ""
echo "=========================================="
echo "✅ Both router training jobs launched successfully!"
echo "=========================================="

