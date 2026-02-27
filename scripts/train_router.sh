# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
export DATA_ROOT=/weka/oe-training-default/ai2-llm
export CHECKPOINTS=/weka/oe-training-default/akshitab/scratch-work/check-flex-v1/models/final


# Optional router training on proxy data (provided by data owners)
# torchrun --nproc-per-node=8 src/scripts/train/OLMoE-7x7B.py FlexOlmo-7x7B-1T-RT \
#     --trainer.callbacks.profiler.enabled=true \
#     --dataset.mix_base_dir=${DATA_ROOT} \
#     --dataset.mix=proxy_combined_public_math_code_news_reddit_pes2o_creative \
#     --trainer.max_duration.value=5_000_000_000 \
#     --trainer.max_duration.unit=tokens \
#     --trainer.load_path=${CHECKPOINTS}/FlexOlmo-7x7B-1T \
#     --model.block.feed_forward_moe.router.top_k=7 \
#     --train_module.rank_microbatch_size=4096 \
#     --train_module.scheduler.warmup_steps=100 \
#     --train_module.optim.lr=6e-4 \
#     --trainer.save_folder=${CHECKPOINTS}/FlexOlmo-7x7B-1T-RT

python src/scripts/beaker/launch.py launch ai2/jupiter --launch.allow-dirty \
    --launch.num_nodes=7 \
    --launch.workspace=ai2/flex2 \
    --launch.priority=urgent -- src/scripts/train/OLMoE-7x7B.py FlexOlmo-7x7B-1T-RT \
    --trainer.callbacks.profiler.enabled=true \
    --dataset.mix_base_dir=${DATA_ROOT} \
    --dataset.mix=proxy_combined_public_math_code_news_reddit_pes2o_creative \
    --trainer.max_duration.value=5_000_000_000 \
    --trainer.max_duration.unit=tokens \
    --trainer.load_path=${CHECKPOINTS}/FlexOlmo-7x7B-1T \
    --model.block.feed_forward_moe.router.top_k=7 \
    --train_module.rank_microbatch_size=4096 \
    --train_module.scheduler.warmup_steps=100 \
    --train_module.optim.lr=6e-4 \
    --trainer.save_folder=${CHECKPOINTS}/FlexOlmo-7x7B-1T-RT \
    --train_module.dp_config.num_replicas=8 \
    --train_module.ep_config.degree=7