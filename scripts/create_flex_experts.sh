GENERAL_MODEL=/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/general-model-resized
PRETRAINED_MODEL=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921-unsharded

# # math expert
# python src/scripts/upcycle/dense_to_expert_moe.py \
#     -m $GENERAL_MODEL \
#        $PRETRAINED_MODEL \
#     -e src/data/domain_embeddings/grit/public.npy \
#        src/data/domain_embeddings/grit/math.npy \
#     -t ../checkpoints/flex-experiments/experts/math-base-test

# # code expert
# python src/scripts/upcycle/dense_to_expert_moe.py \
#     -m $GENERAL_MODEL \
#        $PRETRAINED_MODEL \
#     -e src/data/domain_embeddings/grit/public.npy \
#        src/data/domain_embeddings/grit/code.npy \
#     -t ../checkpoints/flex-experiments/experts/code-base

python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $GENERAL_MODEL \
       $GENERAL_MODEL \
    -t ../checkpoints/flex-experiments/experts/general-general-test

python src/scripts/upcycle/dense_to_expert_moe.py \
    -m $PRETRAINED_MODEL \
       $PRETRAINED_MODEL \
    -t ../checkpoints/flex-experiments/experts/pre-pre-test