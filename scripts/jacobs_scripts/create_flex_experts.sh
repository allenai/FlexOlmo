GENERAL_MODEL=/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/general-model-resized
PRETRAINED_MODEL=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921

# math expert
uv run python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $GENERAL_MODEL \
       $PRETRAINED_MODEL \
    -e src/data/domain_embeddings/grit/public.npy \
       src/data/domain_embeddings/grit/math.npy \
    -t /weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base

# code expert
uv run python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $GENERAL_MODEL \
       $PRETRAINED_MODEL \
    -e src/data/domain_embeddings/grit/public.npy \
       src/data/domain_embeddings/grit/code.npy \
    -t /weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-code-base


# PUBLIC_EXPERT=/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/math_expert_sft_mixed/step1062
# EXPERT_1=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350
# EXPERT_2=/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/code_expert/step150
# EXPERT_3=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350
# # Add other experts

# python src/scripts/upcycle/dense_to_expert_moe.py \
#     -m ${PUBLIC_EXPERT} ${EXPERT_1} \
#     -t /weka/oe-adapt-default/jacobm/flexolmo/checkpoints/flex-experiments/experts/math-base-test-new-conda-env

# python src/scripts/upcycle/dense_to_expert_moe.py \
#     -m ${PUBLIC_EXPERT} ${EXPERT_1} \
#     -e src/data/domain_embeddings/grit/public.npy src/data/domain_embeddings/grit/math.npy \
#     -t ../checkpoints/flex-experiments/experts/test-my-sanjaya-ckpt-with-embedding