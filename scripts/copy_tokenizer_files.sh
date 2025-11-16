OLD_CKPT=/weka/oe-adapt-default/sanjaya/flexolmo/checkpoints/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr/test-if-rlvr-flex-olmo__1__1754720424_checkpoints/step_350
# NEW_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-hf
# NEW_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base-hf
NEW_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368-hf

cp $OLD_CKPT/tokenizer_config.json $NEW_CKPT
cp $OLD_CKPT/chat_template.jinja $NEW_CKPT
cp $OLD_CKPT/tokenizer.json $NEW_CKPT
cp $OLD_CKPT/generation_config.json $NEW_CKPT
cp $OLD_CKPT/vocab.json $NEW_CKPT
