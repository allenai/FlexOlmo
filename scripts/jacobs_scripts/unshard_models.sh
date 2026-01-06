# PUBLIC_EXPERT=
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base

    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150"
    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620"
    # Add more model paths here...
    "/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368"
    "/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95368"
MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed-on-base-no-anneal/step1062"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Unsharding: $MODEL_PATH"
    python src/scripts/utils/unshard.py \
        -i "$MODEL_PATH" \
        -o "${MODEL_PATH}-unsharded"
done