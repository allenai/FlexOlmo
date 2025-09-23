import argparse
import csv
import json
from typing import Dict, List, Set, Tuple

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
    model_to_task_scores: Dict[str, Dict[str, float]] = {}

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

        model_to_task_scores[model_name] = task_scores

    # Create CSV header: model, sorted task names
    task_columns = sorted(all_tasks)
    header = ["model"] + task_columns

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for model_name, scores in sorted(model_to_task_scores.items()):
            row: Dict[str, object] = {"model": model_name}
            for t in task_columns:
                row[t] = scores.get(t, "")
            writer.writerow(row)

    print(f"Wrote CSV with {len(model_to_task_scores)} models and {len(task_columns)} tasks to {args.output_csv}")
    
    # Expected tasks from launch_beaker_eval.sh (excluding mmlu_pro_mc, socialiqa, piqa)
    expected_tasks = {
        'arc_easy:mc', 'arc_challenge:mc', 'boolq:mc', 'csqa:mc', 'hellaswag:mc', 
        'openbookqa:mc', 'winogrande:mc', 'coqa', 'squad', 'naturalqs', 'triviaqa', 
        'drop', 'mmlu:mc', 'agi_eval_english:1shot', 'gsm8k', 'minerva_math_algebra', 
        'minerva_math_counting_and_probability', 'minerva_math_geometry', 
        'minerva_math_intermediate_algebra', 'minerva_math_number_theory', 
        'minerva_math_prealgebra', 'minerva_math_precalculus', 'codex_humaneval:temp0.8', 
        'codex_humanevalplus:temp0.8', 'mbpp', 'mbppplus', 'bbh:cot-v1::olmes'
    }
    
    print(f"\nExpected tasks: {len(expected_tasks)}")
    print(f"Found tasks: {len(task_columns)}")
    
    missing_tasks = expected_tasks - set(task_columns)
    if missing_tasks:
        print(f"Missing tasks: {sorted(missing_tasks)}")
    
    extra_tasks = set(task_columns) - expected_tasks
    if extra_tasks:
        print(f"Extra tasks found: {sorted(extra_tasks)}")
    
    print(f"Task coverage: {len(task_columns)}/{len(expected_tasks)} = {len(task_columns)/len(expected_tasks)*100:.1f}%")


if __name__ == "__main__":
    main()

