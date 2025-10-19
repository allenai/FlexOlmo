#!/bin/bash
set -e
# Clone OLMo-core if not present
if [ ! -d "OLMo-core" ]; then
    git clone --branch cross-stage-training https://github.com/allenai/OLMo-core.git OLMo-core
fi
export PYTHONPATH=OLMo-core/src:$PYTHONPATH
python src/scripts/train/OLMoE-4x7B.py FlexOlmo-4x7B-RT-experts-sft-masked \
  --trainer.callbacks.profiler.enabled=true \
  --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
  --dataset.mix=router_training_mix \
  --trainer.max_duration.value=5_000_000_000 \
  --trainer.max_duration.unit=tokens \
  --trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-codee-xperts-sft-math-mixed \
  --model.block.feed_forward_moe.router.disabled_experts='[3]' \
  --model.block.feed_forward_moe.router.top_k=3 \
  --train_module.rank_microbatch_size=4096 \
  --train_module.scheduler.warmup_steps=100 \
  --train_module.optim.lr=2e-3 \
  --trainer.save_folder=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-RT-experts-sft-math-mixed-masked