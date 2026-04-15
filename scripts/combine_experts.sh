# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
export DATA_ROOT="http://flexolmo-data.org"
export CHECKPOINTS=  # /path/to/checkpoints

PUBLIC_EXPERT=${CHECKPOINTS}/olmoe-2x7b-public-public
EXPERT_1=${CHECKPOINTS}/olmoe-2x7B-news_top2_grit_learnbias/step11921
EXPERT_2=${CHECKPOINTS}/olmoe-2x7B-math_top2_grit_learnbias/step11921
EXPERT_3=${CHECKPOINTS}/olmoe-2x7B-code_top2_grit_learnbias/step11921
# Add other experts

python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m ${PUBLIC_EXPERT} -m ${EXPERT_1}-unsharded ${EXPERT_2}-unsharded ${EXPERT_2}-unsharded [OTHER EXPERTS]  \
    -t ${CHECKPOINTS}/FlexOlmo-4x7B

# Optional router training on proxy data (provided by data owners)
torchrun --nproc-per-node=8 src/scripts/train/OLMoE-4x7B.py FlexOlmo-4x7B-RT \
    --trainer.callbacks.profiler.enabled=true \
    --dataset.mix_base_dir=${DATA_ROOT} \
    --dataset.mix=proxy_combined_public_math_code_news \
    --trainer.max_duration.value=5_000_000_000 \
    --trainer.max_duration.unit=tokens \
    --trainer.load_path=${CHECKPOINTS}/FlexOlmo-4x7B \
    --model.block.feed_forward_moe.router.top_k=4 \
    --train_module.rank_microbatch_size=4096 \
    --train_module.scheduler.warmup_steps=100 \
    --train_module.optim.lr=2e-3 \
    --trainer.save_folder=${CHECKPOINTS}/FlexOlmo-4x7B-RT