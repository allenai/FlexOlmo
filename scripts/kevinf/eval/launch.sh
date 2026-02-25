#!/bin/bash

# Script to launch beaker evaluations for FlexOlmo models
# Usage: bash scripts/kevinf/eval/launch.sh

# Configuration
MODELS=(
    # "/data/input/ai2-llm/checkpoints/FlexOlmo_final/Flex-math-2x7B-1T"
    # "/data/input/ai2-llm/checkpoints/FlexOlmo_final/Flex-code-2x7B-1T"
    # "/data/input/akshitab/scratch-work/check-flex-v1/models/FlexOlmo-7x7B-1T/"
    # "/data/input/ai2-llm/checkpoints/kevinfarhat/OLMo2-7B-anneal-public-mix/step11921-hf/"
    # "allenai/Flex-math-2x7B-1T"
    # "allenai/Flex-code-2x7B-1T"
    # "allenai/Flex-public-7B-1T"
    # "/data/input/akshitab/scratch-work/check-flex-v1/models/final/fixshard-FlexOlmo-7x7B-1T"
    # "/data/input/ai2-llm/checkpoints/FlexOlmo_final/Flex-news-2x7B-1T"
    # "/data/input/ai2-llm/checkpoints/FlexOlmo_final/Flex-pes2o-2x7B-1T/"
    # "allenai/Flex-creative-2x7B-1T" # note that we are putting this one in old_flex even though its the ONLY HF one we are evaling
    # "allenai/Flex-math-2x7B-1T"
    # "allenai/Flex-code-2x7B-1T"
    # "allenai/Flex-creative-2x7B-1T"
    # "allenai/Flex-news-2x7B-1T"
    # "allenai/Flex-pes2o-2x7B-1T"
    # "allenai/Flex-reddit-2x7B-1T"
    # "allenai/Flex-public-7B-1T"
    # "/data/input/ai2-llm/checkpoints/sewonm/model/merge/7e_optout_lb-hf"
    "/data/input/akshitab/scratch-work/check-flex-v1/models/final/FlexOlmo-7x7B-1T-RT/step10900-hf"
)

BASE_OUTPUT_DIR="/data/input/kevinf/7x7B_1000_final_model"
CLUSTER="ai2/titan"
LIMIT=1000

# Define all available tasks (comment/uncomment as needed)
TASKS=(
    # # # # MC9 tasks
    # arc_easy:mc::olmes
    # arc_challenge:mc::olmes
    # boolq:mc::olmes
    # csqa:mc::olmes
    # hellaswag:mc::olmes
    # openbookqa:mc::olmes
    # piqa:mc::olmes
    # socialiqa:mc::olmes
    # winogrande:mc::olmes

    # # Gen5 tasks
    # coqa::olmes
    # squad::olmes
    # naturalqs::olmes
    # triviaqa::olmes
    # drop::olmes

    # # MMLU tasks
    # mmlu:mc::olmes
    # mmlu_pro:mc

    # # # # # AGI eval
    # agi_eval_english:1shot::olmes

    # # # # BBH
    # bbh:cot-v1::olmes

    # # # # Math2 tasks
    # gsm8k::olmes
    # minerva_math_algebra::olmes
    # minerva_math_counting_and_probability::olmes
    # minerva_math_geometry::olmes
    # minerva_math_intermediate_algebra::olmes
    # minerva_math_number_theory::olmes
    # minerva_math_prealgebra::olmes
    # minerva_math_precalculus::olmes

    # # # Code4 tasks
    # codex_humaneval:temp0.8
    # codex_humanevalplus:temp0.8
    # mbpp::none
    # mbppplus::none

    news_gen
    poem_gen
    # sciriff5
)

# Extract a human-readable name from a model path
function get_checkpoint_name {
    local path=$1
    local leaf=$(basename "$path")
    local parent=$(basename "$(dirname "$path")")
    # If the leaf looks like a step (e.g. step7153-hf), include parent run name
    if [[ $leaf == step* ]]; then
        echo "${parent}_${leaf}"
    else
        echo "${leaf}"
    fi
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASKS[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

for MODEL_PATH in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_PATH"

    if [[ $MODEL_PATH == "/"* ]]; then
        # internal model path
        model=$(get_checkpoint_name "$MODEL_PATH")
    else
        # HF model name (e.g. allenai/FlexOlmo-7x7B)
        model=$(echo "$MODEL_PATH" | cut -d'/' -f2)
    fi

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${model}/4_act"

    for TASK in "${TASKS[@]}"; do
        echo "Launching evaluation for model: $model, task: $TASK"

        gpus=1
        nodes=1

        # GPU overrides
        if [[ ($TASK == *"gsm8k"* || $TASK == *"humaneval"* || $TASK == *"bbh"*) && $MODEL_PATH == *"7x7B"* ]]; then
            gpus=4
        fi



        # Batch size rules
        if [[ $TASK == *"cot"* || $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "ruler"* || $TASK == "sciriff"* ]]; then
            batch_size=1
        else
            batch_size=4
        fi

        if [[ ($TASK == *"poem_gen"*) && $MODEL_PATH == *"7x7B"* ]]; then
            gpus=8
            batch_size=1
        fi
        
        if [[ ( $TASK == *"news_gen"*) && $MODEL_PATH == *"7x7B"* ]]; then
            gpus=8
            batch_size=1
        fi


        safe_model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9_-]//g')
        safe_task_name=$(echo "$TASK" | sed 's/[^a-zA-Z0-9_-]//g')
        job_name="eval-${safe_model_name}-${safe_task_name}"

        echo "  Output dir: $OUTPUT_DIR"
        echo "  GPUs: $gpus, Batch size: $batch_size"
        echo "  Job name: $job_name"

        gantry run \
            --name "$job_name" \
            --weka oe-training-default:/data/input \
            --install "pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval]' && uv pip uninstall transformers && uv pip install 'transformers==4.57.1' 'datasets==2.18.0'" \
            --budget ai2/oe-adapt \
            --workspace ai2/flex2 \
            --cluster "$CLUSTER" \
            --priority urgent \
            --gpus "$gpus" \
            --env-secret HF_TOKEN=KEVINF_HF_TOKEN \
            --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
            --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
            --allow-dirty \
            -- \
            bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
                --model $MODEL_PATH \
                --model-type hf \
                --task $TASK \
                --limit $LIMIT \
                --output-dir $OUTPUT_DIR \
                --batch-size $batch_size \
                --gpus $gpus \
                "

        echo "Launched: $job_name"
        echo "----------------------------------------"
    done


    echo "Completed all tasks for model: $model"
    echo "========================================"
done

echo "All beaker evaluations launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASKS[@]}))"



# this for 2x7B local:  --install "pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval]' && uv pip uninstall transformers && uv pip install 'transformers@git+https://github.com/swj0419/transformers' 'datasets==2.18.0'" \


# this for 7x7B and 2x7B hf: --install "pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval]' && uv pip uninstall transformers && uv pip install 'transformers==4.57.1' 'datasets==2.18.0'" \
#we were using datasets == 3.6.0 but it caused list error   2026-02-25T00:20:46.734Z ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation',                         
  # 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf']  

