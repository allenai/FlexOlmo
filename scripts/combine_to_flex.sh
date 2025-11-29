# PUBLIC_EXPERT=
CODE_EXPERT_2=
MATH_EXPERT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062
CODE_EXPERT_1=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620
uv run python src/scripts/upcycle/merge_experts_to_flexolmo.py \
    -m ${MATH_EXPERT} -m ${CODE_EXPERT_1}  \
    -t /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test