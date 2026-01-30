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