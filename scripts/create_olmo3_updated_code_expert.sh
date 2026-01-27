#!/bin/bash

CHECKPOINTS=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/
CHECKPOINT_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350/model_and_optim

python src/scripts/beaker/launch.py launch ai2/jupiter-cirrascale-2 \
  --launch.num_nodes=8 \
  --launch.num_gpus=8 \
  --launch.budget=ai2/oe-base \
  --launch.workspace=ai2/flex2 \
  --launch.priority=high -- \
  src/scripts/train/OLMo2-7B-finetune-nonFFN-frozen.py OLMo2-7B-from-posttrained-code-pretrained ${CHECKPOINT_PATH} \
  --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
  --dataset.mix=olmo3_code \
  --train_module.float8_config.enabled=true \
  --trainer.max_duration.value=50_000_000_000 \
  --trainer.max_duration.unit=tokens \
  --trainer.save_folder=${CHECKPOINTS}/OLMo2-7B-from-posttrained-olmo3-code-pretrainednonFFN-frozen
