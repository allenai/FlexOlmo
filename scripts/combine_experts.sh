# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
# export DATA_ROOT="http://flexolmo-data.org"
export CHECKPOINTS=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/ # /path/to/checkpoints

PUBLIC_EXPERT=${CHECKPOINTS}/OLMo2-7B-from-posttrained-math-pretrainednonFFN-frozen/step11921
EXPERT_1=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350
EXPERT_2=${CHECKPOINTS}/OLMo2-7B-from-posttrained-code-pretrainednonFFN-frozen/step11921
EXPERT_3=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350
# Add other experts

python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m ${PUBLIC_EXPERT} ${EXPERT_1} ${EXPERT_2} ${EXPERT_3} \
    -t ${CHECKPOINTS}/OLMo2-7b-flex-base-merged-math-code

# Optional router training on proxy data (provided by data owners)

# torchrun --nproc-per-node=8 src/scripts/train/OLMoE-4x7B.py FlexOlmo-4x7B-RT \
#    --trainer.callbacks.profiler.enabled=true \
#    --dataset.mix_base_dir=${DATA_ROOT} \
#    --dataset.mix=proxy_combined_public_math_code_news \
#    --trainer.max_duration.value=5_000_000_000 \
#    --trainer.max_duration.unit=tokens \
#    --trainer.load_path=${CHECKPOINTS}/FlexOlmo-4x7B \
#    --model.block.feed_forward_moe.router.top_k=4 \
#    --train_module.rank_microbatch_size=4096 \
#    --train_module.scheduler.warmup_steps=100 \
#    --train_module.optim.lr=2e-3 \
#    --trainer.save_folder=${CHECKPOINTS}/FlexOlmo-4x7B-RT
