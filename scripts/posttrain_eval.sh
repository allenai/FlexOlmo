# # need to set budget correctly
# # can update to not run aime (?)

# # MODEL_PATH=/weka/oe-adapt-default/sanjaya/flexolmo/checkpoints/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr/test-if-rlvr-flex-olmo__1__1754720424_checkpoints/step_350
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-hf
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base-hf
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368-hf
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95250-hf
# # MODEL_PATH=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7B-from-posttrained-math-pretrainednonFFN-frozen/step11921-hf
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step594-hf
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed-on-base-no-anneal/step1062-hf
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-on-base-no-anneal/step594-hf
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-frozen-router-mixed-sft/step1062-hf
# # MODEL_NAME=flexolmo-2x7b-5b-math-anneal-frozen-router-mixed-sft
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-NO-frozen-router-mixed-sft/step1062-hf
# # MODEL_NAME=flexolmo-2x7b-5b-math-anneal-NO-frozen-router-mixed-sft
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-on-base-no-anneal/step150-hf
# # MODEL_NAME=flexolmo-2x7b-code-sft-on-base-no-anneal
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150-hf
# # MODEL_NAME=flexolmo-2x7b-code-sft



# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620-hf
# MODEL_NAME=flexolmo-2x7b-code-sft-mixed-on-base-no-anneal
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620-hf
# MODEL_NAME=flexolmo-2x7b-code-sft-mixed

# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95250-hf
# MODEL_NAME=flex_olmo_2x7b_code_anneal

# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-hf/
# # MODEL_NAME=FlexOlmo-3x7B-sft-only-test

# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router-hf
# MODEL_NAME=FlexOlmo-3x7B-sft-only-test-router

# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed-hf
# MODEL_NAME=FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed
# uv run python scripts/submit_eval_jobs.py \
#     --model_name $MODEL_NAME \
#     --location $MODEL_PATH \
#     --cluster ai2/saturn \
#     --is_tuned \
#     --workspace ai2/flex2 \
#     --priority urgent \
#     --preemptible \
#     --use_hf_tokenizer_template \
#     --run_oe_eval_experiments \
#     --evaluate_on_weka \
#     --run_id placeholder \
#     --oe_eval_max_length 4096 \
#     --process_output r1_style \
#     --skip_oi_evals \
#     --beaker_image jacobm/oe-eval-flex-olmo-9-29-5 


# # 32768

MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-math_mixed-code_mixed_no_ann-hf"
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