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

# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_5b-router_sft_all_mixed/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_20b-router_sft_all_mixed/step1128-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test-math_base-math_5b_sft-code_5b_sft-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test_frozen_router-math_base-math_5b_sft-code_5b_sft-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-NO-frozen-router-mixed-sft-router/step1062-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-frozen-router-mixed-sft-router/step1062-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-code-frozen-router-mixed-sft-router/step620-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test-math_base-math_50b_sft-code_50b_sft-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_50/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_100/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_150/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_200/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_50/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_100/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_150/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_200/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_250/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_300/"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_olmo3_code_anneal-tool_use_only/step422-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-tool_use_general_mix/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-tool_use_general_mix/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_anneal-general_sft/step470-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-olmo3_math-mixed-sft/step1062-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal_mixed_SFT_TEST/step620-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_50"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_100"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_150"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_200"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_250"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_300"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_350"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_400"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_450"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_500"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_general_only/step394-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_50"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_100"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_150"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_200"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_250"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_300"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_350"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_400"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_50"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_100"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_150"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_200"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_250"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_300"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_350"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_400"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-4k-test/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_anneal-general-olmo3_math-mix/step966-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_0.25_mix/step536-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_math_code_mix/step1224-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3-reasoning_sft_0.75/step842-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-reasoning_anneal-general-olmo3_reasoning-mix/step784-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-olmo3_code-general-mix-unf-lm-head/step782-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-unf-lm-head-embed/step1128-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed-1-active/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b_code_1a-mix_sft/step782-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-unf-rt-4-domain/step1128-hf"
#     "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/router-sft-only-newcode-sft-experts/step1212-hf"
#     "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/router-sft-newcode-pretrained/step1212-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3-reasoning_sft_0.75/step842-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.05/step56-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.1/step112-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.25/step280-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.5/step562-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.75/step842-hf"
# "/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/general-model-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-olmo3_code-general-mix-unf-lm-head/step782-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed-1-active/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-unf-lm-head-embed/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-unf-rt-4-domain/step1128-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_1.0/step1128-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_10"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_20"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_30"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_50"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-redux/step562-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-reasoning_anneal-FIXED-general-olmo3_reasoning-mix/step784-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-old-seeds/step562-hf"
MODEL_PATHS=(
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_300"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_350"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_400"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_450"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_500"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_550"
)

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
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
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
        --oe_eval_tasks "minerva_math::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek" \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done