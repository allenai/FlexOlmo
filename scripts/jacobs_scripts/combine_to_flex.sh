# # PUBLIC_EXPERT=
# MATH_BASE=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-unsharded
# CODE_BASE=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base-unsharded
# MATH_EXPERT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-unsharded
# CODE_EXPERT_1=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620-unsharded
# CODE_EXPERT_2=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150-hf/
# python src/scripts/upcycle/merge_experts_to_flexolmo.py \
#     -m ${GENERAL_EXPERT} ${CODE_EXPERT_1} ${MATH_EXPERT}  \
#     -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed

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
# OLMO3_CODE_20B=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-20B/step38147
OLMO3_CODE_20B_WITH_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-20B/step620
OLMO3_CODE_50B=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
OLMO3_CODE_50B_WITH_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620

# router training tests
MATH_5B_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-NO-frozen-router-mixed-sft-router/step1062
MATH_5B_SFT_FROZEN_ROUTER=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-frozen-router-mixed-sft-router/step1062
CODE_5B_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-5B/step620/
CODE_5B_SFT_FROZEN_ROUTER=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-code-frozen-router-mixed-sft-router/step620

MATH_RL_UNF_LM_EMBED=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500-oc

TOOL_USE_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head/step888

uv run python src/scripts/upcycle/dense_to_expert_moe.py \
    -m  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_tool_use-mix/step888 \
        /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350 \
        /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_400-oc \
        /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-50b_ol3_code_ann-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-6e-7-unf__1__1773949222_checkpoints/step_100-oc \
        /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_safety-mix/step534 \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first

#### NEED TO FIX BTX w/ math expert


# Smaller list of experts:
BASE=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-unsharded
MATH_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-unsharded
MATH_RL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500-oc
OLMO3_MATH_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_math_anneal-olmo3_math-mix-4k/step500
OLMO3_MATH_RL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_math_anneal-olmo3_math-mix-4k/step500-hf/grpo_math_only_flex-2x7b-50b_ol3_ann-ol3_sft_math-6e-7-unf/grpo_math_only_flex-2x7b-50b_ol3_ann-ol3_sft_math-6e-7-unf__1__1773370912_checkpoints/step_200-oc
OLMO2_CODE_MIX_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620-unsharded
CODE_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620
CODE_RL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_200-oc
TOOL_USE_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head/step888
SAFETY_SFT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_safety-general-mix/step534-hf-oc

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $CODE_RL \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-final-no-safety-tool \
    --average_all_shared_params &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $CODE_RL $TOOL_USE_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-final-no-safety \
    --average_all_shared_params

-----

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_SFT $CODE_SFT $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-sft-only \
    --average_all_shared_params

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $OLMO2_CODE_MIX_SFT $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo2_code_math \
    --average_all_shared_params

---

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $CODE_RL $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo3_code_rl-olmo2_math \
    --average_all_shared_params &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $OLMO3_MATH_RL $OLMO2_CODE_MIX_SFT $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo2_code-olmo3_math \
    --average_all_shared_params

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $OLMO3_MATH_SFT $CODE_SFT $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_all \
    --average_all_shared_params &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $OLMO3_MATH_RL $CODE_SFT $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_4-math_rl \
    --average_all_shared_params &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $OLMO3_MATH_SFT $CODE_RL $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_4-code_rl \
    --average_all_shared_params &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $OLMO3_MATH_RL $CODE_RL $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-math_code_rl \
    --average_all_shared_params

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $OLMO2_CODE_MIX_SFT $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-math_rl-olmo2_code_sft-tool_use-safety_sft \
    --average_all_shared_params

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $CODE_RL $TOOL_USE_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl-code_rl-tool_use \
    --average_all_shared_params;

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $BASE $MATH_RL $CODE_RL $TOOL_USE_SFT $SAFETY_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-math_rl-code_rl-tool_use-safety_sft \
    --average_all_shared_params


uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_MIX_SFT $OLMO3_CODE_50B_WITH_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test-math_base-math_50b_sft-code_50b_sft 
    &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_5B_SFT_FROZEN_ROUTER $CODE_5B_SFT_FROZEN_ROUTER \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test_frozen_router-math_base-math_5b_sft-code_5b_sft

# uv sync --extra all
# uv pip install -e ../Olmo-core
uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_MIX_SFT $OLMO3_CODE_20B_WITH_SFT $MATH_BASE \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_20b_ann_with_sft-math_base_again &&

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_MIX_SFT $OLMO3_CODE_50B_WITH_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann_with_sft

# MERGING LM HEAD AND EMBEDDINGS
uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_RL_UNF_LM_EMBED $OLMO3_CODE_50B_WITH_SFT $MATH_BASE \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-math_base \
    --average_shared_params lm_head embeddings


uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_RL_UNF_LM_EMBED $OLMO3_CODE_50B_WITH_SFT $TOOL_USE_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use \
    --average_shared_params lm_head embeddings

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_BASE $MATH_RL_UNF_LM_EMBED $OLMO3_CODE_50B_WITH_SFT $TOOL_USE_SFT \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use-average_all-no_rt \
    --average_all_shared_params

uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m $MATH_RL_UNF_LM_EMBED $MATH_RL_UNF_LM_EMBED $MATH_RL_UNF_LM_EMBED $MATH_RL_UNF_LM_EMBED \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl_x4 \
    --average_all_shared_params

# Define configurations as "expert1,expert2,expert3|output_path"
CONFIGS=(
    "${MATH_BASE},${MATH_MIX_SFT},${OLMO3_CODE}|${CKPT_DIR}/FlexOlmo-3x7B-math_base-math_mixed-code_mixed_olmo3_5b_ann"
)

for config in "${CONFIGS[@]}"; do
    EXPERTS="${config%|*}"
    OUTPUT="${config#*|}"
    
    IFS=',' read -r E1 E2 E3 <<< "$EXPERTS"
    
    python src/scripts/upcycle/merge_experts_to_flexolmo.py \
        -m "$E1" "$E2" "$E3" \
        -t "$OUTPUT"
done