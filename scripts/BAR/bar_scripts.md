# BAR Training Recipe

Scripts used to train [BAR](https://huggingface.co/collections/allenai/bar) (Branch-Adapt-Route) models. BAR trains independent domain experts through their own full post-training pipelines and composes them into a single Mixture-of-Experts via lightweight router training.

## Repositories

BAR training spans three repos:

1. **[FlexOlmo](https://github.com/allenai/FlexOlmo)** — upcycling dense → MoE, mid-training, merging experts, HF ↔ olmo-core conversion.
2. **[BAR-olmo-core](https://github.com/jacob-morrison/BAR-olmo-core)** — SFT and router training.
3. **[open-instruct](https://github.com/allenai/open-instruct)** — RLVR and evaluation.

Released checkpoints: [allenai BAR collection](https://huggingface.co/collections/allenai/bar).

## Pipeline overview

For each domain expert (math, code, tool use, safety):

```
upcycle → [mid-train] → SFT → [RLVR] → merge into 5x7B → router SFT
```

Mid-training is used for math and code; RLVR is used for math and code. Tool use and safety are SFT only. RLVR runs on HF-format checkpoints (via open-instruct), so experts are converted HF ↔ olmo-core as they move between training and RL stages.

---

## 1. Create the initial 2-expert MoE

Upcycle a post-trained dense anchor + a pretrained dense model into a 2-expert MoE. The first model's FFN initializes the frozen anchor expert; the second initializes the trainable domain expert.

```bash
# Repo: FlexOlmo
uv run python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $ANCHOR_MODEL \       # fully post-trained dense 7B (e.g., allenai/BAR-7B)
       $PRETRAINED_MODEL \   # pretrained dense 7B  (e.g., allenai/OLMo-2-1124-7B)
    -t $OUTPUT_DIR
```

## 2. Mid-training (math and code experts only)

```bash
# Repo: FlexOlmo
uv run python src/scripts/train/OLMoE-2x7B-mid-train.py $RUN_NAME \
    --trainer.callbacks.profiler.enabled=false \
    --dataset.mix_base_dir=$DATA_ROOT \
    --dataset.mix=olmo3_code \
    --trainer.max_duration.value=50_000_000_000 \
    --trainer.max_duration.unit=tokens \
    --trainer.load_path=$INITIAL_MOE_CKPT \
    --model.block.feed_forward_moe.router.top_k=2 \
    --train_module.rank_microbatch_size=8192 \
    --train_module.scheduler.warmup_steps=2000 \
    --train_module.optim.lr=9e-4 \
    --trainer.save_folder=$OUTPUT_DIR
```

Swap `--dataset.mix=olmo3_code` for the math mid-training mix for the math expert.

## 3. SFT

SFT is run in the BAR-olmo-core fork. Start from the mid-trained expert (for math/code) or the upcycled MoE (for tool use/safety).

```bash
# Repo: BAR-olmo-core
uv run python src/scripts/train/sft/FlexOlmo-SFT.py train $RUN_NAME \
    $INPUT_CKPT \                      # e.g., mid-trained expert in olmo-core format
    --seq_len=4096 \
    --num_nodes=4 \
    --global_batch_size=1048576 \
    --model_name=olmoe-2x7b \
    --dataset_path=$SFT_DATASET \      # domain SFT data mixed with general SFT data
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.num_gpus=8
```

See the BAR-olmo-core README for dataset prep and launcher details.

## 4. Convert olmo-core → HF (for RLVR)

```bash
# Repo: FlexOlmo
uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i $MODEL_PATH \
    -o $MODEL_PATH-hf \
    --skip-validation \
    --max-sequence-length 65536
```

## 5. RLVR (math and code experts only)

GRPO with verifiable rewards, via open-instruct. The example below shows the math RLVR run; swap the datasets and tasks for the code version.

```bash
# Repo: open-instruct
python open_instruct/grpo_fast.py \
    --exp_name $EXP_NAME \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 64 \
    --num_mini_batches 4 \
    --num_epochs 1 \
    --learning_rate 6e-7 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list \
        hamishivi/omega-combined-no-boxed_filtered 20000 \
        hamishivi/rlvr_orz_math_57k_collected_filtered 14000 \
        hamishivi/polaris_53k 14000 \
        hamishivi/MathSub-30K_filtered 9000 \
        hamishivi/DAPO-Math-17k-Processed_filtered 7000 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list \
        hamishivi/omega-combined 4 \
        allenai/IF_multi_constraints_upto5 4 \
        saurabh5/rlvr_acecoder_filtered 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --pack_length 7168 \
    --model_name_or_path allenai/BAR-2x7B-Math-SFT \
    --output_dir $OUTPUT_DIR \
    --chat_template_name olmo123 \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 1024000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 24 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --gradient_checkpointing \
    --seed 1 \
    --local_eval_every 50 \
    --save_freq 50 \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --oe_eval_max_length 32768 \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek \
    --code_pass_rate_reward_threshold 0.99 \
    --inflight_updates true \
    --async_steps 1 \
    --advantage_normalization_type centered \
    --no_resampling_pass_rate 0.875
```

For multi-node Ray + code-sandbox setup, see the open-instruct docs.

## 6. Convert HF → olmo-core (after RLVR)

Bring the RL-trained expert back into olmo-core format for merging.

```bash
# Repo: FlexOlmo
uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
    -i $MODEL_PATH \
    -o $MODEL_PATH-oc \
    -c $REFERENCE_CONFIG_JSON \    # config.json from a matching olmo-core checkpoint
    --skip-validation
```

## 7. Merge experts into a single 5x7B

```bash
# Repo: FlexOlmo
uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $ANCHOR $MATH $CODE $SAFETY $TOOL \
    -t $OUTPUT_DIR \
    --average_all_shared_params
```

`--average_all_shared_params` averages shared parameters that diverged across expert runs (from SFT/RLVR stages with unfrozen shared layers).

## 8. Router training

Train only the router on a 5% stratified SFT sample from all domains; all other weights frozen.

```bash
# Repo: BAR-olmo-core
BASE_CKPT=$MERGED_5X7B_CKPT
AMOUNT=0.05
LR=1e-4
SFT_DATASET=$ROUTER_SFT_DATASET   # 5% stratified sample across all domains

uv run python src/scripts/train/sft/FlexOlmo-SFT-5x7B.py train \
    FlexOlmo-5x7B-router-$AMOUNT-$LR \
    $BASE_CKPT \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --seq_len=2048 \
    --num_nodes=5 \
    --model_name=olmoe-5x7b \
    --dataset_path=$SFT_DATASET \
    --launch.num_gpus=8
```

## 9. Convert final model to HF

```bash
# Repo: FlexOlmo
uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i $MODEL_PATH \
    -o $MODEL_PATH-hf \
    --skip-validation \
    --max-sequence-length 65536
```

## Evaluation

Evaluations are run via [open-instruct](https://github.com/allenai/open-instruct); see that repo for task definitions and launch scripts.