# need to set budget correctly
# can update to not run aime (?)

# CKPT=/weka/oe-adapt-default/sanjaya/flexolmo/checkpoints/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr/test-if-rlvr-flex-olmo__1__1754720424_checkpoints/step_350
# CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-hf
# CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base-hf
# CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368-hf
# CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95250-hf
# CKPT=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7B-from-posttrained-math-pretrainednonFFN-frozen/step11921-hf
CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step594-hf
MODEL_NAME=flexolmo-2x7b-math-sft
uv run python scripts/submit_eval_jobs.py \
    --model_name $MODEL_NAME \
    --location $CKPT \
    --cluster ai2/saturn \
    --is_tuned \
    --workspace ai2/flex2 \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 4096 \
    --process_output r1_style \
    --skip_oi_evals \
    --beaker_image jacobm/oe-eval-flex-olmo-9-29-5 


# 32768