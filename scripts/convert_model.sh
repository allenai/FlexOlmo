# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base
# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95250
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-on-base-no-anneal/step150
# gantry run --cluster ai2/saturn -y --budget ai2/oceo --workspace ai2/flex2 \
#         --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
#         --weka=oe-adapt-default:/weka/oe-adapt-default \
#         --weka=oe-training-default:/weka/oe-training-default \
#         --priority urgent \
#         --gpus 8 \
#         -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
#             -i $MODEL_PATH \
#             -o $MODEL_PATH-hf \
#             --skip-validation \
#             --max-sequence-length 65536