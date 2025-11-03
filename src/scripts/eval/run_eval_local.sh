#! /bin/bash

## Script for running evaluations on local models (not from HuggingFace)
## Usage: bash src/scripts/eval/run_eval_local.sh <MODEL_PATH> <TASK_NAME> <BASE_OUTPUT_DIR> <GPUS>

MODEL_PATH=$1
TASK_NAME=$2
BASE_OUTPUT_DIR=$3
GPUS=$4

# Define all available tasks
if [[ $TASK_NAME == "all" ]] ; then
	TASKS=(
		# Multiple choice tasks
		arc_easy:mc::olmes
		arc_challenge:mc::olmes
		boolq:mc::olmes
		csqa:mc::olmes
		hellaswag:mc::olmes
		openbookqa:mc::olmes
		piqa:mc::olmes
		socialiqa:mc::olmes
		winogrande:mc::olmes
		# Generation tasks
		coqa::olmes
		squad::olmes
		naturalqs::olmes
		triviaqa::olmes
		drop::olmes
		# Math tasks
		gsm8k::olmes
		minerva_math_algebra::olmes
		minerva_math_counting_and_probability::olmes
		minerva_math_geometry::olmes
		minerva_math_intermediate_algebra::olmes
		minerva_math_number_theory::olmes
		minerva_math_prealgebra::olmes
		minerva_math_precalculus::olmes
		# Code tasks
		codex_humaneval:temp0.8
		codex_humanevalplus:temp0.8
		mbpp::none
		mbppplus::none
		# Other tasks
		mmlu:mc::olmes
		mmlu_pro_mc::none
		agi_eval_english:1shot::olmes
		bbh:cot-v1::olmes
	)
else
	# Use predefined task suites or custom task
	if [[ $TASK_NAME == "mc9" ]] ; then
		TASKS=(
			arc_easy:mc::olmes
			arc_challenge:mc::olmes
			boolq:mc::olmes
			csqa:mc::olmes
			hellaswag:mc::olmes
			openbookqa:mc::olmes
			piqa:mc::olmes
			socialiqa:mc::olmes
			winogrande:mc::olmes
		)
	elif [[ $TASK_NAME == "gen5" ]] ; then
		TASKS=(
			coqa::olmes
			squad::olmes
			naturalqs::olmes
			triviaqa::olmes
			drop::olmes
		)
	elif [[ $TASK_NAME == "math2" ]] ; then
		TASKS=(
			gsm8k::olmes
			minerva_math_algebra::olmes
			minerva_math_counting_and_probability::olmes
			minerva_math_geometry::olmes
			minerva_math_intermediate_algebra::olmes
			minerva_math_number_theory::olmes
			minerva_math_prealgebra::olmes
			minerva_math_precalculus::olmes
		)
	elif [[ $TASK_NAME == "code4" ]] ; then
		TASKS=(
			codex_humaneval:temp0.8
			codex_humanevalplus:temp0.8
			mbpp::none
			mbppplus::none
		)
	else
		TASKS=($TASK_NAME)
	fi
fi

# Extract model name from path for output directory naming
MODEL_NAME=$(basename $(dirname $MODEL_PATH))

for TASK in "${TASKS[@]}"; do
	echo "Running task: $TASK"
	
	# OOM with some tasks, so batch size to be 1
	if [[ $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "sciriff"* ]] ; then
		batch_size=1
	else
		batch_size=4
	fi

	PYTHONPATH=. python src/offline_evals/run_eval.py \
	--model-path $MODEL_PATH \
	--model-type hf \
	--task $TASK \
	--limit 1000 \
	--output-dir ${BASE_OUTPUT_DIR}/${MODEL_NAME} \
	--batch-size $batch_size \
	--gpus $GPUS
done 