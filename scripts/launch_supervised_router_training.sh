#!/bin/bash
# Beaker launch script for supervised router training (adjust values as needed)

PYTHONPATH=/weka/oe-training-default/sanjaya/FlexOlmo/src:$PYTHONPATH \
python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-4x7B-supervised-router.py FlexOlMo-4x7B-Supervised-RT \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.include_instance_metadata=true \
   --trainer.max_duration.value=100 \
   --trainer.max_duration.unit=tokens \
   --trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code \
   --trainer.save_folder=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-RT-supervised-small \
   --model.block.feed_forward_moe.num_experts=4 \
   --model.block.feed_forward_moe.router.top_k=4 \
   --train_module.rank_microbatch_size=4096 \
   --train_module.scheduler.warmup_steps=100 \
   --train_module.optim.lr=2e-3 \
   --train_module.router_loss_weight=1.0 \
   --train_module.router_loss_only=true \
   --train_module.dp_config.num_replicas=16 \
   --train_module.ep_config.degree=4 \
   --data_loader.global_batch_size=32768

