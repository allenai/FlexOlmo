#!/bin/bash
# Midtraining script for 2x7B models
# This script trains router for both Math-2x7B and Code-2x7B models consecutively (stolen from sanjay)

set -e  # Exit on error

echo "=========================================="
echo "Starting Mid Training for 2x7B Models"
echo "=========================================="
echo ""

# # Math-2x7B Mid Training
# echo "🚀 Launching Math-2x7B Mid Training..."
# echo ""

CODE_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base
# SANJAY_EXPERT=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/merged-2x7B-general-math
# TEST_EXPERT=/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/flex-experiments/experts/merged-2x7B-general-math-4
# SANJAY_2=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/merged-2x7B-general-math-3

#    --launch.budget=ai2/oe-base \

MATH_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal-8k.py flex-2x7B-olmo3_math_anneal-50b-8k \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=olmo3_math \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${MATH_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=8192 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_math_anneal-50b-8k

# # Code-2x7B Mid Training
# echo "🚀 Launching Code-2x7B Mid Training..."
# echo ""

uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal.py Flex-2x7B-olmo3-code-anneal-frozen-router-5B-1-active \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=olmo3_code \
   --trainer.max_duration.value=5_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${CODE_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=1 \
   --train_module.rank_microbatch_size=8192 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-frozen-router-5B-1-active


# torchrun --nproc-per-node=8 src/scripts/train/OLMoE-2x7B-anneal.py olmoe-2x7B-${EXPERT}_top2_grit_learnbias \
    # --dataset.mix_base_dir=${DATA_ROOT} \
    # --dataset.mix=${EXPERT} \


# Long-context 2x7B Mid training

-----
### reasoning
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal-8k.py flex-2x7B-olmo3_reasoning-fixed-20b-8k \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=olmo3_reasoning \
   --trainer.max_duration.value=20_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${MATH_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=8192 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_reasoning-fixed-20b-8k


-----


MATH_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base

MODEL_NAME=flex-2x7B-long_context-32k-5b
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=2 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal-long-context.py $MODEL_NAME \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=long_context \
   --trainer.max_duration.value=5_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${MATH_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/$MODEL_NAME 
   # --train_module.rank_microbatch_size=32768 \


# kevin_med_data
MATH_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-anneal-8k.py flex-2x7B-kevin_med_anneal-50b-8k \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=kevin_med_data \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=${MATH_EXPERT} \
   --model.block.feed_forward_moe.router.top_k=2 \
   --train_module.rank_microbatch_size=8192 \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-kevin_med_anneal-50b-8k
