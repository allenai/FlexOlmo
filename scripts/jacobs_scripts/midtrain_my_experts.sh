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

CODE_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-code-base
MATH_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base

MATH_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal-8k.py flex-2x7B-math_anneal-50b \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=mj_finemath4plus \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${MATH_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=8192 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-math_anneal-50b

# Code-2x7B Mid Training
echo "🚀 Launching Code-2x7B Mid Training..."
echo ""

uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal.py flex-2x7B-code_anneal-50b \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=olmo3_code \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${CODE_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=8192 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-code_anneal-50b


#### 7B anneal
echo "🚀 Launching Code 7B Mid Training..."
echo ""

EXPERT_7B=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/train
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=4 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/olmo-instruct \
   --launch.priority=urgent -- src/scripts/train/OLMo2-7B-fixed-anneal.py flex-7b-anneal-code-50b \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=olmo3_code \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.load_path=${EXPERT_7B} \
   --trainer.max_duration.unit=tokens \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-anneal-code-50b

#### BAseline 7B anneal
echo "🚀 Launching Mixed 7B Mid Training..."
echo ""

EXPERT_7B=/weka/oe-training-default/ai2-llm/checkpoints/akshitab/OLMo2-7B-stage1-step928646
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/olmo-instruct \
   --launch.priority=urgent -- src/scripts/train/OLMo2-7B-fixed-anneal.py flex-7b-full-mix-150b \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=baseline_mix_mixed \
   --trainer.max_duration.value=150_000_000_000 \
   --trainer.load_path=${EXPERT_7B} \
   --trainer.max_duration.unit=tokens \
   --train_module.float8_config.enabled=true --train_module.optim.lr=0.000061499 --train_module.scheduler.warmup_steps=0 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-full-mix-150b
