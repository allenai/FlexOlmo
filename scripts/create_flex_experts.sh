GENERAL_MODEL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350
PRETRAINED_MODEL=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921-unsharded

# math expert
python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $GENERAL_MODEL \
       $PRETRAINED_MODEL \
    -e src/data/domain_embeddings/grit/public.npy \
       src/data/domain_embeddings/grit/math.npy \
    -t ../checkpoints/flex-experiments/experts/math-base

# code expert
# python src/scripts/upcycle/dense_to_expert_moe.py \
#     -m $GENERAL_MODEL \
#        $PRETRAINED_MODEL \
#     -e src/data/domain_embeddings/grit/public.npy \
#        src/data/domain_embeddings/grit/code.npy \
#     -t ../checkpoints/flex-experiments/experts/code-base