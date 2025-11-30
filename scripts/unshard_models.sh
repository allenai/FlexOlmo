# PUBLIC_EXPERT=
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062
MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
python src/scripts/utils/unshard.py \
    -i $MODEL_PATH \
    -o ${MODEL_PATH}-unsharded