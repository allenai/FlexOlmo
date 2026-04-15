
## this repo
# creating the base model
uv run python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $GENERAL_MODEL \
       $PRETRAINED_MODEL \
    -t /weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-base

# mid-train (when relevant)
flex: ...
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=8 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/flex2 \
   --launch.priority=urgent -- src/scripts/train/OLMoE-2x7B-mid-train.py flex-2x7B-code_anneal-50b \
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


7b: ...
EXPERT_7B=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/train
uv run python src/scripts/beaker/launch.py launch ai2/jupiter \
   --launch.num_nodes=4 \
   --launch.num_gpus=8 \
   --launch.budget=ai2/oceo \
   --launch.workspace=ai2/olmo-instruct \
   --launch.priority=urgent -- src/scripts/train/OLMo2-7B-mid-train.py flex-7b-anneal-code-50b \
   --trainer.callbacks.profiler.enabled=false \
   --dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/ \
   --dataset.mix=olmo3_code \
   --trainer.max_duration.value=50_000_000_000 \
   --trainer.load_path=${EXPERT_7B} \
   --trainer.max_duration.unit=tokens \
   --train_module.scheduler.warmup_steps=2000 \
   --train_module.optim.lr=9e-4 \
   --trainer.save_folder=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-anneal-code-50b

# sft (when relevant)
see olmo-core

# convert to hf

# rlvr
see open-instruct

# convert to olmo-core

# combining flex models
uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH $CODE $SAFETY $TOOL \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-retrain-final \
    --average_all_shared_params

# convert back to hf

# eval'ing (open-instruct and oe-eval-internal)
