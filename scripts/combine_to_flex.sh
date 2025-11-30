# PUBLIC_EXPERT=
GENERAL_EXPERT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-unsharded
MATH_EXPERT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-unsharded
CODE_EXPERT_1=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620-unsharded
python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m ${GENERAL_EXPERT} ${MATH_EXPERT} ${CODE_EXPERT_1}  \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test