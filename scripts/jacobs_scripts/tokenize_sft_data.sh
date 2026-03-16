# uv run gantry run \
#         --cluster ai2/neptune-cirrascale \
#         --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
#         --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
#         --weka=oe-training-default:/weka/oe-training-default \
#         --env-secret HF_TOKEN=jacobm_HF_TOKEN \
#         -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
#             --dataset_mixer_list ai2-adapt-dev/personahub_code_v2_34999 1.0 ai2-adapt-dev/evol_codealpaca_heval_decontaminated 1.0 ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 ai2-adapt-dev/flan_v2_converted 1.0 ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 ai2-adapt-dev/oasst1_converted 1.0 ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
#             --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
#             --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/code-general-mix \
#             --visualize True \
#             --chat_template_name olmo \
#             --max_seq_length 4096

uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list jacobmorrison/Dolci-Instruct-SFT-Safety 1.0 \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
                ai2-adapt-dev/flan_v2_converted 1.0 \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/olmo3_safety-general-mix \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096


uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list allenai/Dolci-Instruct-SFT-Tool-Use 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096


# tool + some general
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list allenai/Dolci-Instruct-SFT-Tool-Use 1.0 \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
                ai2-adapt-dev/flan_v2_converted 1.0 \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
                allenai/tulu-3-sft-personas-math-grade 1.0 \
                ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k 1.0 \
                ai2-adapt-dev/evol_codealpaca_heval_decontaminated 1.0 \
                ai2-adapt-dev/numinamath_tir_math_decontaminated 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k 1.0 \
                ai2-adapt-dev/personahub_math_v5_regen_149960 1.0 \
                ai2-adapt-dev/personahub_code_v2_34999 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-math-code-mix-fixed \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096

# tool use
allenai/Dolci-Instruct-SFT-Tool-Use 1.0 \
# general
ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
ai2-adapt-dev/flan_v2_converted 1.0 \
ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
ai2-adapt-dev/oasst1_converted 1.0 \
ai2-adapt-dev/oasst1_converted 1.0 \
ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
# math
allenai/tulu-3-sft-personas-math-grade 1.0 \
ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k 1.0 \
ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k 1.0 \
ai2-adapt-dev/numinamath_tir_math_decontaminated 1.0 \
ai2-adapt-dev/personahub_math_v5_regen_149960 1.0 \
# code
ai2-adapt-dev/evol_codealpaca_heval_decontaminated 1.0 \
ai2-adapt-dev/personahub_code_v2_34999 1.0 \
# safety
jacobmorrison/Dolci-Instruct-SFT-Safety 1.0 \

FRACTION=0.05
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k $FRACTION \
                ai2-adapt-dev/flan_v2_converted $FRACTION \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 $FRACTION \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k $FRACTION \
                ai2-adapt-dev/oasst1_converted $FRACTION \
                ai2-adapt-dev/tulu_v3.9_aya_100k $FRACTION \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 $FRACTION \
                jacobmorrison/Dolci-Instruct-SFT-Math $FRACTION \
                jacobmorrison/Dolci-Instruct-SFT-Tool-Use $FRACTION \
                jacobmorrison/Dolci-Instruct-SFT-Safety $FRACTION \
                ai2-adapt-dev/evol_codealpaca_heval_decontaminated $FRACTION \
                ai2-adapt-dev/personahub_code_v2_34999 $FRACTION \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_olmo2_code_tool_use_safety-$FRACTION \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096

uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                jacobmorrison/Dolci-Instruct-SFT-Math 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Tool-Use 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Safety 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Coding 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/olmo3_math \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096



# Make dataset fractions:
FRACTION=0.50
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --weka=oe-training-default:/weka/oe-adapt-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list ai2-adapt-dev/tulu_v3.9_wildchat_100k $FRACTION \
                ai2-adapt-dev/personahub_math_v5_regen_149960 $FRACTION \
                ai2-adapt-dev/flan_v2_converted $FRACTION \
                ai2-adapt-dev/tulu_v3.9_aya_100k $FRACTION \
                ai2-adapt-dev/personahub_code_v2_34999 $FRACTION \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 $FRACTION \
                allenai/tulu-3-sft-personas-math-grade $FRACTION \
                ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k $FRACTION \
                ai2-adapt-dev/evol_codealpaca_heval_decontaminated $FRACTION \
                ai2-adapt-dev/numinamath_tir_math_decontaminated $FRACTION \
                ai2-adapt-dev/oasst1_converted $FRACTION \
                ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k $FRACTION \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k $FRACTION \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 $FRACTION \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-math-code-${FRACTION}-redux \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096

# General + Olmo 3 Code
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
                ai2-adapt-dev/flan_v2_converted 1.0 \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Coding 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_code-mix \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096

# General + Olmo 3 Science + Biomed data
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
                ai2-adapt-dev/flan_v2_converted 1.0 \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Science 1.0 \
                jacobmorrison/medical-SFT 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_science-biomed-mix \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096

# General + Olmo 3 Math
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
                ai2-adapt-dev/flan_v2_converted 1.0 \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Math 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_math-mix \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096

# General + Olmo 3 Reasoning (NOT THINKING)
uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                ai2-adapt-dev/tulu_v3.9_wildchat_100k 1.0 \
                ai2-adapt-dev/flan_v2_converted 1.0 \
                ai2-adapt-dev/tulu_hard_coded_repeated_10 1.0 \
                ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
                ai2-adapt-dev/oasst1_converted 1.0 \
                ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 1.0 \
                jacobmorrison/Dolci-Instruct-SFT-Reasoning 1.0 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_reasoning-mix \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096


uv run gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=jacobm_HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list \
                ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 10000 \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/10k-instance-test-dataset \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 4096