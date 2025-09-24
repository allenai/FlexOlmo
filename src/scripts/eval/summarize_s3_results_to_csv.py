import argparse
import csv
import json
from typing import Dict, List, Set, Tuple, Optional

import boto3


def list_model_prefixes(s3_client, bucket: str, prefix: str) -> List[str]:
    """
    Return a list of immediate child prefixes under the given prefix (i.e., model names).
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    prefixes: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            p = cp["Prefix"]
            if p.endswith("/"):
                prefixes.append(p)
    return prefixes


def list_task_metrics_files(s3_client, bucket: str, model_prefix: str) -> List[str]:
    """
    Return a list of task-*-metrics.json files for a given model.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    task_files: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=model_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("-metrics.json") and "task-" in key:
                task_files.append(key)
    return task_files


def read_metrics_json(s3_client, bucket: str, key: str) -> Dict:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return json.loads(data)


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def calculate_mc7(scores: Dict[str, float]) -> Optional[float]:
    """Calculate MC7 average: arc_challenge:mc, arc_easy:mc, boolq:mc, csqa:mc, hellaswag:mc, openbookqa:mc, winogrande:mc"""
    mc7_tasks = ["arc_challenge:mc", "arc_easy:mc", "boolq:mc", "csqa:mc", "hellaswag:mc", "openbookqa:mc", "winogrande:mc"]
    values = []
    for task in mc7_tasks:
        if task in scores and scores[task] is not None:
            values.append(scores[task])
        else:
            print(f"Warning: Missing {task} for MC7 calculation")
            return None
    return sum(values) / len(values)


def calculate_gen5(scores: Dict[str, float]) -> Optional[float]:
    """Calculate Gen5 average: coqa, drop, naturalqs, squad, triviaqa"""
    gen5_tasks = ["coqa", "drop", "naturalqs", "squad", "triviaqa"]
    values = []
    for task in gen5_tasks:
        if task in scores and scores[task] is not None:
            values.append(scores[task])
        else:
            print(f"Warning: Missing {task} for Gen5 calculation")
            return None
    return sum(values) / len(values)


def calculate_code_avg(scores: Dict[str, float]) -> Optional[float]:
    """Calculate Code Average: codex_humaneval:temp0.8, codex_humanevalplus:temp0.8, mbpp, mbppplus"""
    code_tasks = ["codex_humaneval:temp0.8", "codex_humanevalplus:temp0.8", "mbpp", "mbppplus"]
    values = []
    for task in code_tasks:
        if task in scores and scores[task] is not None:
            values.append(scores[task])
        else:
            print(f"Warning: Missing {task} for Code Average calculation")
            return None
    return sum(values) / len(values)


def calculate_math_avg(scores: Dict[str, float]) -> Optional[float]:
    """Calculate Math Average: average of minerva math tasks and gsm8k"""
    minerva_tasks = [
        "minerva_math_algebra", "minerva_math_counting_and_probability", "minerva_math_geometry",
        "minerva_math_intermediate_algebra", "minerva_math_number_theory", 
        "minerva_math_prealgebra", "minerva_math_precalculus"
    ]
    
    # Calculate minerva average
    minerva_values = []
    for task in minerva_tasks:
        if task in scores and scores[task] is not None:
            minerva_values.append(scores[task])
        else:
            print(f"Warning: Missing {task} for Math Average calculation")
            return None
    
    minerva_avg = sum(minerva_values) / len(minerva_values)
    
    # Check gsm8k
    if "gsm8k" not in scores or scores["gsm8k"] is None:
        print(f"Warning: Missing gsm8k for Math Average calculation")
        return None
    
    # Return average of minerva average and gsm8k
    return (minerva_avg + scores["gsm8k"]) / 2


def validate_required_columns(scores: Dict[str, Optional[float]]) -> bool:
    """Validate that all required columns are present and populated"""
    required_columns = [
        "MC7", "Gen5", "MMLU", "AGI Eval", "BBH", "Code Avg.", "Math Avg.",
        "agi_eval_english:1shot", "arc_challenge:mc", "arc_easy:mc", "bbh:cot-v1::olmes",
        "boolq:mc", "coqa", "csqa:mc", "drop", "hellaswag:mc", "mmlu:mc",
        "naturalqs", "openbookqa:mc", "squad", "triviaqa", "winogrande:mc",
        "codex_humaneval:temp0.8", "codex_humanevalplus:temp0.8", "mbpp", "mbppplus",
        "minerva_math_algebra", "minerva_math_counting_and_probability", "minerva_math_geometry",
        "minerva_math_intermediate_algebra", "minerva_math_number_theory",
        "minerva_math_prealgebra", "minerva_math_precalculus", "gsm8k"
    ]
    
    missing_columns = []
    empty_columns = []
    
    for col in required_columns:
        if col not in scores:
            missing_columns.append(col)
        elif scores[col] is None or scores[col] == "":
            empty_columns.append(col)
    
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        return False
    
    if empty_columns:
        print(f"ERROR: Empty required columns: {empty_columns}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results from S3 into CSV")
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="s3://ai2-sewonm/sanjaya/eval_results/",
        help="S3 prefix where eval results are stored (each model is a subfolder)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="eval_summary_s3.csv",
        help="Path to write the CSV summary",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="primary_metric",
        help="Metric field to use per task (default: primary_metric)",
    )
    args = parser.parse_args()

    bucket, base_key = parse_s3_uri(args.s3_prefix)
    if base_key and not base_key.endswith("/"):
        base_key += "/"

    s3 = boto3.client("s3")

    model_prefixes = list_model_prefixes(s3, bucket, base_key)
    if not model_prefixes:
        raise SystemExit(f"No model prefixes found under {args.s3_prefix}")

    # Collect scores per model
    all_tasks: Set[str] = set()
    model_to_task_scores: Dict[str, Dict[str, Optional[float]]] = {}

    for model_prefix in model_prefixes:
        # Model name is the last non-empty segment
        parts = [p for p in model_prefix.split("/") if p]
        model_name = parts[-1]
        metrics_key = f"{model_prefix}metrics.json"

        # Get all task-*-metrics.json files for this model
        task_metrics_files = list_task_metrics_files(s3, bucket, model_prefix)
        print(f"DEBUG: Found {len(task_metrics_files)} task metrics files for {model_name}")
        
        # Collect all scores for each task bucket (for averaging)
        task_scores_raw: Dict[str, List[float]] = {}
        
        for task_file in task_metrics_files:
            try:
                task_metrics = read_metrics_json(s3, bucket, task_file)
                print(f"DEBUG: Read {task_file}")
                
                # Extract task name from filename and normalize it to match the 26 task buckets
                # e.g., "task-000-arc_challenge:mc-metrics.json" -> "arc_challenge:mc"
                filename = task_file.split("/")[-1]
                
                # Remove task-XXX- prefix and -metrics.json suffix
                if filename.startswith("task-") and "-metrics.json" in filename:
                    # Find the position after the task-XXX- prefix
                    prefix_end = filename.find("-", 5)  # Start after "task-"
                    if prefix_end != -1:
                        raw_task_name = filename[prefix_end + 1:-len("-metrics.json")]
                    else:
                        raw_task_name = filename.replace("-metrics.json", "")
                else:
                    raw_task_name = filename.replace("-metrics.json", "")
                
                # Map to the 26 task buckets from launch_beaker_eval.sh
                # Use EXACT string matching to avoid ambiguity
                if raw_task_name == "mmlu_abstract_algebra:mc" or raw_task_name.startswith("mmlu_") and ":mc" in raw_task_name:
                    # All MMLU subject tasks -> mmlu:mc
                    task_name = "mmlu:mc"
                elif raw_task_name.startswith("agi_eval_") and ":mc" in raw_task_name:
                    # All AGI eval tasks -> agi_eval_english:1shot
                    task_name = "agi_eval_english:1shot"
                elif raw_task_name.startswith("bbh_"):
                    # All BBH subtasks -> bbh:cot-v1::olmes
                    task_name = "bbh:cot-v1::olmes"
                elif raw_task_name == "minerva_math_algebra":
                    task_name = "minerva_math_algebra"
                elif raw_task_name == "minerva_math_counting_and_probability":
                    task_name = "minerva_math_counting_and_probability"
                elif raw_task_name == "minerva_math_geometry":
                    task_name = "minerva_math_geometry"
                elif raw_task_name == "minerva_math_intermediate_algebra":
                    task_name = "minerva_math_intermediate_algebra"
                elif raw_task_name == "minerva_math_number_theory":
                    task_name = "minerva_math_number_theory"
                elif raw_task_name == "minerva_math_prealgebra":
                    task_name = "minerva_math_prealgebra"
                elif raw_task_name == "minerva_math_precalculus":
                    task_name = "minerva_math_precalculus"
                elif raw_task_name == "codex_humaneval":
                    task_name = "codex_humaneval:temp0.8"
                elif raw_task_name == "codex_humanevalplus":
                    task_name = "codex_humanevalplus:temp0.8"
                elif raw_task_name == "mbpp":
                    task_name = "mbpp"
                elif raw_task_name == "mbppplus":
                    task_name = "mbppplus"
                elif raw_task_name == "naturalqs_open":
                    task_name = "naturalqs"
                elif raw_task_name == "arc_easy:mc":
                    task_name = "arc_easy:mc"
                elif raw_task_name == "arc_challenge:mc":
                    task_name = "arc_challenge:mc"
                elif raw_task_name == "boolq:mc":
                    task_name = "boolq:mc"
                elif raw_task_name == "csqa:mc":
                    task_name = "csqa:mc"
                elif raw_task_name == "hellaswag:mc":
                    task_name = "hellaswag:mc"
                elif raw_task_name == "openbookqa:mc":
                    task_name = "openbookqa:mc"
                elif raw_task_name == "winogrande:mc":
                    task_name = "winogrande:mc"
                elif raw_task_name == "coqa":
                    task_name = "coqa"
                elif raw_task_name == "squad":
                    task_name = "squad"
                elif raw_task_name == "triviaqa":
                    task_name = "triviaqa"
                elif raw_task_name == "drop":
                    task_name = "drop"
                elif raw_task_name == "gsm8k":
                    task_name = "gsm8k"
                else:
                    # Fallback for any unexpected tasks
                    print(f"WARNING: Unexpected task name '{raw_task_name}', using as-is")
                    task_name = raw_task_name
                
                # Get the primary_score directly from metrics
                value = task_metrics.get("metrics", {}).get("primary_score")
                if value is None:
                    print(f"DEBUG: No primary_score in metrics for {task_file}")
                    continue
                
                # Ensure numeric when possible
                try:
                    value = float(value)
                except Exception:
                    pass
                
                # Collect all scores for this task bucket
                if task_name not in task_scores_raw:
                    task_scores_raw[task_name] = []
                task_scores_raw[task_name].append(value)
                all_tasks.add(task_name)
                print(f"DEBUG: Added score {value} for task {task_name}")
                
            except Exception as e:
                print(f"Warning: failed to read {task_file}: {e}")
                continue

        # Now average the scores for each task bucket
        task_scores: Dict[str, float] = {}
        for task_name, scores in task_scores_raw.items():
            if len(scores) > 1:
                avg_score = sum(scores) / len(scores)
                print(f"DEBUG: Averaged {len(scores)} scores for {task_name}: {avg_score}")
                task_scores[task_name] = avg_score
            else:
                task_scores[task_name] = scores[0]
                print(f"DEBUG: Single score for {task_name}: {scores[0]}")

        # Calculate derived metrics
        enhanced_scores: Dict[str, Optional[float]] = {k: v for k, v in task_scores.items()}
        
        # Calculate MC7
        mc7_score = calculate_mc7(task_scores)
        if mc7_score is not None:
            enhanced_scores["MC7"] = mc7_score
        else:
            print(f"ERROR: Could not calculate MC7 for model {model_name}")
            enhanced_scores["MC7"] = None
        
        # Calculate Gen5
        gen5_score = calculate_gen5(task_scores)
        if gen5_score is not None:
            enhanced_scores["Gen5"] = gen5_score
        else:
            print(f"ERROR: Could not calculate Gen5 for model {model_name}")
            enhanced_scores["Gen5"] = None
        
        # Copy MMLU
        if "mmlu:mc" in task_scores:
            enhanced_scores["MMLU"] = task_scores["mmlu:mc"]
        else:
            print(f"ERROR: Missing mmlu:mc for model {model_name}")
            enhanced_scores["MMLU"] = None
        
        # Copy AGI Eval
        if "agi_eval_english:1shot" in task_scores:
            enhanced_scores["AGI Eval"] = task_scores["agi_eval_english:1shot"]
        else:
            print(f"ERROR: Missing agi_eval_english:1shot for model {model_name}")
            enhanced_scores["AGI Eval"] = None
        
        # Copy BBH
        if "bbh:cot-v1::olmes" in task_scores:
            enhanced_scores["BBH"] = task_scores["bbh:cot-v1::olmes"]
        else:
            print(f"ERROR: Missing bbh:cot-v1::olmes for model {model_name}")
            enhanced_scores["BBH"] = None
        
        # Calculate Code Average
        code_avg_score = calculate_code_avg(task_scores)
        if code_avg_score is not None:
            enhanced_scores["Code Avg."] = code_avg_score
        else:
            print(f"ERROR: Could not calculate Code Average for model {model_name}")
            enhanced_scores["Code Avg."] = None
        
        # Calculate Math Average
        math_avg_score = calculate_math_avg(task_scores)
        if math_avg_score is not None:
            enhanced_scores["Math Avg."] = math_avg_score
        else:
            print(f"ERROR: Could not calculate Math Average for model {model_name}")
            enhanced_scores["Math Avg."] = None

        model_to_task_scores[model_name] = enhanced_scores

    # Define the exact column order as requested
    column_order = [
        "MC7", "Gen5", "MMLU", "AGI Eval", "BBH", "Code Avg.", "Math Avg.",
        "agi_eval_english:1shot", "arc_challenge:mc", "arc_easy:mc", "bbh:cot-v1::olmes",
        "boolq:mc", "coqa", "csqa:mc", "drop", "hellaswag:mc", "mmlu:mc",
        "naturalqs", "openbookqa:mc", "squad", "triviaqa", "winogrande:mc",
        "codex_humaneval:temp0.8", "codex_humanevalplus:temp0.8", "mbpp", "mbppplus",
        "minerva_math_algebra", "minerva_math_counting_and_probability", "minerva_math_geometry",
        "minerva_math_intermediate_algebra", "minerva_math_number_theory",
        "minerva_math_prealgebra", "minerva_math_precalculus", "gsm8k"
    ]
    
    header = ["model"] + column_order

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for model_name, scores in sorted(model_to_task_scores.items()):
            # Validate required columns
            if not validate_required_columns(scores):
                raise SystemExit(f"Validation failed for model {model_name}")
            
            row: Dict[str, object] = {"model": model_name}
            for col in column_order:
                row[col] = scores.get(col, "")
            writer.writerow(row)

    print(f"Wrote CSV with {len(model_to_task_scores)} models and {len(column_order)} columns to {args.output_csv}")
    
    # Report validation results
    print(f"\nValidation Summary:")
    print(f"- Total models processed: {len(model_to_task_scores)}")
    print(f"- Required columns: {len(column_order)}")
    print(f"- All models passed validation")


if __name__ == "__main__":
    main()

