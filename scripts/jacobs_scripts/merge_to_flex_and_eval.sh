cd /weka/oe-adapt-default/jacobm/flexolmo/FlexOlmo

MATH_BASE=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-unsharded
CODE_BASE=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base-unsharded
MATH_ANNEAL=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368-unsharded
CODE_ANNEAL=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95368-unsharded
CODE_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150-unsharded
MATH_MIX_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-unsharded
CODE_MIX_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620-unsharded
MATH_MIX_SFT_NO_ANNEAL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed-on-base-no-anneal/step1062
CODE_MIX_SFT_NO_ANNEAL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620-unsharded
CKPT_DIR=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft
OLMO3_CODE=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-5B/step620
OLMO3_CODE_20B=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-20B/step620

CONFIGS=(
    "${MATH_BASE},${MATH_MIX_SFT},${OLMO3_CODE}|${CKPT_DIR}/FlexOlmo-3x7B-math_base-math_mixed-code_mixed_olmo3_5b_anneal"
)

for config in "${CONFIGS[@]}"; do
    EXPERTS="${config%|*}"
    OUTPUT="${config#*|}"
    
    IFS=',' read -r E1 E2 E3 <<< "$EXPERTS"
    
    python src/scripts/upcycle/merge_experts_to_flexolmo.py \
        -m "$E1" "$E2" "$E3" \
        -t "$OUTPUT"
done

cd ../Olmo-core


"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use-average_all-no_rt"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl_x4"
MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_all-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_4-math_rl-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_4-code_rl-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-math_code_rl-0.05-1e-4/step66"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Processing: $MODEL_PATH"
    
    mkdir -p "${MODEL_PATH}/model_and_optim" && \
    cp -f "${MODEL_PATH}"/*.distcp "${MODEL_PATH}/model_and_optim/" && \
    cp -f "${MODEL_PATH}/.metadata" "${MODEL_PATH}/model_and_optim/" && \
    uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
        -i "$MODEL_PATH" \
        -o "${MODEL_PATH}-hf" \
        --skip-validation \
        --max-sequence-length 65536 && \
    cp /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router-hf/chat_template.jinja "${MODEL_PATH}"
    
    if [ $? -ne 0 ]; then
        echo "Failed: $MODEL_PATH"
    fi
done

cd ../open-instruct

MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-router_sft_all_mixed-2k/step1128-hf"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/-hf$//')
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "$MODEL_NAME" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done

cd ../FlexOlmo