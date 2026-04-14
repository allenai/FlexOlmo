MODEL_PATHS=(
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-tool-use-sft-unfrozen/step888-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-safety-sft/step534-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-tool-use-sft/step888-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-safety-sft/step534-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-mid-train-sft/step1856-hf"

"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_50"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_100"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_150"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_200"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_250"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_300"
# "/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_350"
# "/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base-hf/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-7b-math-sft-6e-7__1__1775598277_checkpoints/step_400"

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_300"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_350"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_400"

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_300"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_350"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_400"

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_300"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_350"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_400"

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7__1__1775624750_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7__1__1775624750_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7__1__1775624750_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7__1__1775624750_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7__1__1775624750_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-post-train-sft-6e-7__1__1775624750_checkpoints/step_300"

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7__1__1775624758_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7__1__1775624758_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7__1__1775624758_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7__1__1775624758_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7__1__1775624758_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856-hf/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7/grpo_mixed_olmo2-7b-continued-post-train-sft-6e-7__1__1775624758_checkpoints/step_300"

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-mid-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-mid-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-mid-train-sft-6e-7__1__1775667837_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-mid-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-mid-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-mid-train-sft-6e-7__1__1775667837_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-mid-train-sft/step1856-hf/grpo_mixed_olmo2-7b-retrain-mid-train-sft-6e-7/grpo_mixed_olmo2-7b-retrain-mid-train-sft-6e-7__1__1775667837_checkpoints/step_150"
)
all_but_safety="mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning,gpqa:0shot_cot::qwen3-instruct,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,bfcl_all::std"
# coding_tasks="codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"
# all_but_safety="codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi

    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $all_but_safety \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi
    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks "minerva_math::hamish_zs_reasoning_deepseek" \
        --gpu_multiplier 2 \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done


safety_tasks="harmbench::default,do_anything_now::default,wildguardtest::default,wildjailbreak::benign,trustllm_jailbreaktrigger::default"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi
    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $safety_tasks \
        --beaker_image maliam/flexolmo-libraries-safety \
        --gpu_multiplier 2
done

if_ood="ifeval_ood::tulu-thinker"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi
    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $if_ood \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5 
done