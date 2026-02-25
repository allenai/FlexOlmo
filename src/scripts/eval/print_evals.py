#!/usr/bin/env python3
"""
Evaluation Results Aggregator

This script collects and displays evaluation results from local or S3 model directories.
It expects task-*metrics.json files containing evaluation scores.

The --base-dir argument can be:
  1. A parent directory containing multiple model directories, where each model directory contains task-*metrics.json files
  2. A single model directory containing metrics files directly

Usage:
    # Multiple models (parent directory)
    python print_evals.py --base-dir /path/to/evaluations/

    # Single model (direct path)
    python print_evals.py --base-dir /path/to/evaluations/model1/

    # S3 paths work the same way
    python print_evals.py --base-dir s3://bucket/evaluations/ --export-csv results.csv
"""

import argparse
import csv
import glob
import json
import os
import pickle as pkl
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import smart_open
from fsspec import AbstractFileSystem, get_filesystem_class
from prettytable import PrettyTable
from termcolor import colored
from tqdm import tqdm

MMLU_SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

MMLU_CATEGORIES = {
    "physics": "STEM",
    "chemistry": "STEM",
    "biology": "STEM",
    "computer science": "STEM",
    "math": "STEM",
    "engineering": "STEM",
    "history": "humanities",
    "philosophy": "humanities",
    "law": "humanities",
    "politics": "social sciences",
    "culture": "social sciences",
    "economics": "social sciences",
    "geography": "social sciences",
    "psychology": "social sciences",
    "other": "other (business, health, misc.)",
    "business": "other (business, health, misc.)",
    "health": "other (business, health, misc.)",
}

FS_KWARGS: Dict[str, Dict[str, Any]] = {
    "": {"auto_mkdir": True},
}

RE_GLOB_STAR_ESCAPE = re.compile(r"(?<!\\)\*")
RE_GLOB_ONE_ESCAPE = re.compile(r"(?<!\\)\?")
RE_GLOB_OPEN_ESCAPE = re.compile(r"(?<!\\)\[")
RE_GLOB_CLOSE_ESCAPE = re.compile(r"(?<!\\)\]")
ESCAPE_SYMBOLS_MAP = {"*": "\u2581", "?": "\u2582", "[": "\u2583", "]": "\u2584"}
REVERSE_ESCAPE_SYMBOLS_MAP = {v: k for k, v in ESCAPE_SYMBOLS_MAP.items()}
PATCHED_GLOB = False


def _get_fs(path: Union[Path, str]) -> AbstractFileSystem:
    """Get the filesystem class for a given path."""
    path = str(path)
    protocol = urlparse(path).scheme
    fs = get_filesystem_class(protocol)(**FS_KWARGS.get(protocol, {}))

    global PATCHED_GLOB
    # patch glob method to support recursive globbing
    if protocol == "" and not PATCHED_GLOB:
        fs.glob = partial(glob.glob, recursive=True)
        PATCHED_GLOB = True

    return fs


def _escape_glob(s: Union[str, Path]) -> str:
    """Escape glob characters in a string."""
    s = str(s)
    s = RE_GLOB_STAR_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["*"], s)
    s = RE_GLOB_ONE_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["?"], s)
    s = RE_GLOB_OPEN_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["["], s)
    s = RE_GLOB_CLOSE_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["]"], s)
    return s


def _unescape_glob(s: Union[str, Path]) -> str:
    """Unescape glob characters in a string."""
    s = str(s)
    for k, v in REVERSE_ESCAPE_SYMBOLS_MAP.items():
        s = s.replace(k, v)
    return s


def _pathify(path: Union[Path, str]) -> Tuple[str, Path]:
    """Return the protocol and path of a given path."""
    path = _escape_glob(str(path))
    parsed = urlparse(path)
    path = Path(f"{parsed.netloc}/{parsed.path}") if parsed.netloc else Path(parsed.path)
    return parsed.scheme, path


def join_path(protocol: Union[str, None], *parts: Union[str]) -> str:
    """Join a path from its protocol and path components."""
    from itertools import chain

    all_prots, all_parts = zip(
        *(_pathify(p) for p in chain.from_iterable([p] if isinstance(p, str) else p for p in parts))
    )
    path = str(Path(*all_parts)).rstrip("/")
    protocol = protocol or str(all_prots[0])

    if protocol:
        path = f"{protocol}://{path.lstrip('/')}"
    return _unescape_glob(path)


def glob_path(
    path: Union[Path, str],
    hidden_files: bool = False,
    autoglob_dirs: bool = True,
    recursive_dirs: bool = False,
    yield_dirs: bool = True,
) -> Iterator[str]:
    """Expand a glob path into a list of paths."""
    protocol, parsed_path = _pathify(path)
    fs = _get_fs(path)

    if fs.isdir(path) and autoglob_dirs:
        path = join_path(protocol, _unescape_glob(parsed_path), "*")

    for gl in fs.glob(path):
        gl = str(gl)

        if not hidden_files and Path(gl).name.startswith("."):
            continue

        if fs.isdir(gl):
            if recursive_dirs:
                yield from glob_path(
                    gl,
                    hidden_files=hidden_files,
                    autoglob_dirs=autoglob_dirs,
                    recursive_dirs=recursive_dirs,
                    yield_dirs=yield_dirs,
                )
            if yield_dirs:
                yield join_path(protocol, gl)
        else:
            yield join_path(protocol, gl)


def map_mmlu_cat(subtask):
    subtask = subtask.split("mmlu_")[-1].split(":")[0]
    return MMLU_CATEGORIES[MMLU_SUBCATEGORIES[subtask][0]]


def map_mmlu_subcat(subtask):
    subtask = subtask.split("mmlu_")[-1].split(":")[0]
    return MMLU_SUBCATEGORIES[subtask][0]


def avg_tasks(results, label, task_names):
    """Average scores across multiple tasks into a single metric."""
    for model_name in results:
        if np.all([task_name in results[model_name] for task_name in task_names]):
            results[model_name][label] = np.mean(
                [results[model_name][task_name] for task_name in task_names]
            )
        for task_name in task_names:
            if task_name in results[model_name]:
                del results[model_name][task_name]
    return results


def get_task_names(results):
    """Get all unique task names across all models."""
    task_names = set()
    for dic in results.values():
        for task_name in dic:
            task_names.add(task_name)
    return task_names


def main(args):
    # Cache handling
    if not os.path.exists("cached_results.pkl") or args.reset_cache:
        assert not args.load_cache
        results: dict = defaultdict(dict)
        visited_paths = set()
    else:
        with open("cached_results.pkl", "rb") as f:
            results, visited_paths = pkl.load(f)

    if not args.load_cache:
        updated = False

        def _add_to_results(model_name, task_name, score):
            if model_name not in results:
                results[model_name] = {}
            if task_name in results[model_name]:
                if results[model_name][task_name] != score:
                    print(
                        f"Warning: Duplicated results not identical for {task_name} in {model_name}: "
                        f"{results[model_name][task_name]} vs {score}"
                    )
                    exit()
            else:
                results[model_name][task_name] = score

        # Model directory discovery
        # Check if base_dir itself contains metrics files
        # if this is not empty, the user is likely looking at results for a single model in this directory
        direct_metrics = list(glob_path(os.path.join(args.base_dir, "task-*metrics.json")))

        if direct_metrics:
            # base_dir is a model directory itself
            model_dirs = [args.base_dir]
        else:
            # if we reach here, we are looking at results for multiple models in this directory
            # base_dir contains model directories
            model_dirs = list(glob_path(os.path.join(args.base_dir, "*")))

        if args.show_models:
            show_models = args.show_models.split(",")
            model_dirs = [dir for dir in model_dirs if any(t in dir for t in show_models)]

        if args.hide_models:
            hide_models = args.hide_models.split(",")
            model_dirs = [dir for dir in model_dirs if not any(t in dir for t in hide_models)]

        for model_dir in tqdm(model_dirs, desc="Processing models"):
            model_name = model_dir.split("/")[-1]

            # Find all metric files
            metric_paths = list(glob_path(os.path.join(model_dir, "task-*metrics.json")))

            if args.show_tasks:
                show_tasks = args.show_tasks.split(",")
                metric_paths = [path for path in metric_paths if any(t in path for t in show_tasks)]

            if args.hide_tasks:
                hide_tasks = args.hide_tasks.split(",")
                metric_paths = [
                    path for path in metric_paths if not any(t in path for t in hide_tasks)
                ]

            for metric_path in metric_paths:
                if metric_path in visited_paths:
                    continue

                with smart_open.open(metric_path, "r") as f:
                    metric = json.load(f)

                task_name = metric["task_name"]
                metrics = metric["metrics"]
                score = metrics["primary_score"]
                _add_to_results(model_name, task_name, score)

                visited_paths.add(metric_path)
                updated = True

        if updated:
            with open("cached_results.pkl", "wb") as f_out:
                pkl.dump([results, visited_paths], f_out)

    # Common task groupings
    core9_tasks = [
        "arc_easy:mc",
        "arc_challenge:mc",
        "boolq:mc",
        "csqa:mc",
        "hellaswag:mc",
        "openbookqa:mc",
        "piqa:mc",
        "socialiqa:mc",
        "winogrande:mc",
    ]
    gen5_tasks = ["coqa", "squad", "naturalqs_open", "triviaqa", "drop"]

    task_names = get_task_names(results)

    # Task averaging options
    if args.avg_core:
        results = avg_tasks(results, "core9:mc", core9_tasks)

    if args.avg_mmlu_pro:
        mmlu_pro_tasks = [
            task_name for task_name in task_names if task_name.startswith("mmlu_pro_")
        ]
        if mmlu_pro_tasks:
            results = avg_tasks(results, "mmlu_pro:mc", mmlu_pro_tasks)

    if args.avg_mmlu:
        mmlu_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("mmlu_") and not task_name.startswith("mmlu_pro_")
        ]
        if mmlu_tasks:
            results = avg_tasks(results, "mmlu:mc", mmlu_tasks)

    if args.avg_mmlu_cat or args.avg_mmlu_subcat:
        mmlu_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("mmlu_") and not task_name.startswith("mmlu_pro_")
        ]

        mmlu_cats = defaultdict(list)
        for task in mmlu_tasks:
            if args.avg_mmlu_cat:
                mmlu_cats[map_mmlu_cat(task)].append(task)
            else:
                mmlu_cats[map_mmlu_subcat(task)].append(task)

        for mmlu_cat, tasks in mmlu_cats.items():
            results = avg_tasks(results, f"mmlu_{mmlu_cat.split(' ')[0]}:mc", tasks)

    if args.avg_bbh:
        bbh_tasks = [task_name for task_name in task_names if task_name.startswith("bbh_")]
        if bbh_tasks:
            results = avg_tasks(results, "bbh", bbh_tasks)

    if args.avg_gen:
        results = avg_tasks(results, "gen5", gen5_tasks)

    if args.avg_agi_eval:
        agi_eval_tasks = [
            task_name for task_name in task_names if task_name.startswith("agi_eval_")
        ]
        if agi_eval_tasks:
            results = avg_tasks(results, "agi_eval:mc", agi_eval_tasks)

    if args.avg_mm:
        mm_tasks = [task_name for task_name in task_names if task_name.startswith("minerva_math_")]
        if mm_tasks:
            results = avg_tasks(results, "minerva_math", mm_tasks)

    if args.avg_math2:
        mm_tasks = [task_name for task_name in task_names if task_name.startswith("minerva_math_")]
        for model_name in results:
            model = results[model_name]
            gsm8k_score = model.get("gsm8k")
            # Use already-aggregated minerva_math if avg_mm ran first, otherwise compute from subtasks
            if "minerva_math" in model:
                minerva_avg = model.pop("minerva_math")
            elif mm_tasks:
                minerva_scores = [model[t] for t in mm_tasks if t in model]
                minerva_avg = np.mean(minerva_scores) if minerva_scores else None
            else:
                minerva_avg = None
            if gsm8k_score is not None and minerva_avg is not None:
                model["math2"] = np.mean([gsm8k_score, minerva_avg])
            for t in mm_tasks + ["gsm8k"]:
                model.pop(t, None)

    if args.avg_ruler:
        ruler_4k_tasks = [
            task_name for task_name in task_names if task_name.startswith("ruler_4k_")
        ]
        if ruler_4k_tasks:
            results = avg_tasks(results, "ruler_4k", ruler_4k_tasks)

    if args.avg_sciriff:
        sciriff_tasks = [task_name for task_name in task_names if task_name.startswith("sciriff_")]
        if sciriff_tasks:
            results = avg_tasks(results, "sciriff5", sciriff_tasks)

    if args.avg_code:
        code_tasks = [
            task_name
            for task_name in task_names
            if ("mbpp" in task_name or "codex" in task_name) and not task_name.endswith("@10")
        ]
        if code_tasks:
            results = avg_tasks(results, "code_4tasks", code_tasks)

    # Filter results based on task types
    if args.stem_only:
        stem_tasks = [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
            "health",
            "arc",
            "drop",
            "bbh",
            "agi_eval",
            "gsm8k",
            "openbook",
            "piqa",
        ]
        results = {
            model_name: {
                k: v for k, v in result.items() if any(keyword in k for keyword in stem_tasks)
            }
            for model_name, result in results.items()
        }

    # Model filtering
    if args.show_models:
        show_models = args.show_models.split(",")
        results = {
            key: value for key, value in results.items() if any(m in key for m in show_models)
        }

    if args.hide_models:
        hide_models = args.hide_models.split(",")
        results = {
            key: value for key, value in results.items() if not any(m in key for m in hide_models)
        }

    # Prepare task list for display
    pt = PrettyTable()
    task_names = sorted(get_task_names(results))

    def _is_code(task_name):
        return (
            "mbpp" in task_name
            or "bigcodebench" in task_name
            or "codex" in task_name
            or "code_4tasks" in task_name
        )

    # Task filtering
    if args.code_only:
        task_names = [task_name for task_name in task_names if _is_code(task_name)]
    elif args.hide_code:
        task_names = [task_name for task_name in task_names if not _is_code(task_name)]
    else:
        # Sort so that code tasks go last
        task_names = sorted(task_names, key=_is_code)

    if args.show_tasks:
        show_tasks = args.show_tasks.split(",")
        task_names = [
            task_name for task_name in task_names if any(t in task_name for t in show_tasks)
        ]

    if args.hide_tasks:
        hide_tasks = args.hide_tasks.split(",")
        task_names = [
            task_name for task_name in task_names if not any(t in task_name for t in hide_tasks)
        ]

    if args.core_and_gen_only:
        task_names = [
            task_name for task_name in task_names if task_name in core9_tasks + gen5_tasks
        ]

    # Compute best scores for highlighting
    best_results = {}
    for task_name in task_names:
        best_results[task_name] = np.max(
            [_results.get(task_name, -1) for _results in results.values()]
        )

    def format_model_name(model):
        """Format model name for display. Override this for custom formatting."""
        return model

    def format_number(v, task_name):
        """Format score with best result highlighting."""
        is_best = np.abs(v - best_results[task_name]) < 0.001 if v is not None else False
        v = f"{v:.3f}" if v is not None else "-"
        return colored(v, "red") if is_best else v

    def format_task(name):
        """Format task name for display."""
        name = name.replace("codex_humaneval", "humaneval")
        name = name.replace("bigcodebench", "bcb")
        name = name.replace("plus", "+")
        if "medmcqa" not in name:
            name = name.replace(":mc", "")
        return name

    # Build and display table
    if args.transpose:
        formatted_task_names = [format_task(name) for name in task_names]
        pt.field_names = ["Model"] + formatted_task_names
        pt.align["Model"] = "l"
        for task_name in formatted_task_names:
            pt.align[task_name] = "r"
        for model_name in sorted(results.keys()):
            pt.add_row(
                [format_model_name(model_name)]
                + [
                    format_number(results[model_name].get(task_name, None), task_name)
                    for task_name in task_names
                ]
            )
    else:
        model_names = sorted(list(results.keys()))
        formatted_model_names = [format_model_name(model_name) for model_name in model_names]

        model_names = [name for i, name in enumerate(model_names) if formatted_model_names[i]]
        formatted_model_names = [name for i, name in enumerate(formatted_model_names) if name]

        pt.align["Model"] = "l"

        pt.field_names = ["Model"] + formatted_model_names
        pt.align["Model"] = "l"
        for model_name in formatted_model_names:
            pt.align[model_name] = "r"
        for task_name in task_names:
            pt.add_row(
                [format_task(task_name)]
                + [
                    format_number(results[model_name].get(task_name, None), task_name)
                    for model_name in model_names
                ]
            )

    print(pt)

    # Export to CSV if requested
    if args.export_csv:

        def export_csv(results, task_names, model_names, args):
            with open(args.export_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                if args.transpose:
                    header = ["Model"] + [format_task(name) for name in task_names]
                    writer.writerow(header)
                    for model in model_names:
                        row = [format_model_name(model)]
                        for task in task_names:
                            value = results[model].get(task, None)
                            row.append(f"{value:.3f}" if value is not None else "-")
                        writer.writerow(row)
                else:
                    header = ["Task"] + [format_model_name(m) for m in model_names]
                    writer.writerow(header)
                    for task in task_names:
                        row = [format_task(task)]
                        for m in model_names:
                            value = results[m].get(task, None)
                            row.append(f"{value:.3f}" if value is not None else "-")
                        writer.writerow(row)

            print(f"Exported results to {args.export_csv}")

        model_names = sorted(results.keys())
        export_csv(results, task_names, model_names, args)

    # Exclude specific tasks from cache (useful for re-running certain evaluations)
    # Example: exclude_tasks = ["mmlu", "arc"] will remove all MMLU and ARC results from cache
    exclude_tasks: list = []
    if len(exclude_tasks) > 0:
        results = {
            k: {t: r for t, r in v.items() if np.any([et not in t for et in exclude_tasks])}
            for k, v in results.items()
        }
        visited_paths = {
            path
            for path in visited_paths
            if np.any([et not in path.split("/")[-1] for et in exclude_tasks])
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and display evaluation results from multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display results from all models in a parent directory
  python print_evals.py --base-dir ./evaluations/

  # Display results from a single model directory
  python print_evals.py --base-dir ./evaluations/model1/

  # Display results from S3 (works for both parent and model directories)
  python print_evals.py --base-dir s3://bucket/evaluations/

  # Export to CSV and transpose the table
  python print_evals.py --base-dir ./evaluations/ --export-csv results.csv -t

  # Show only specific models and tasks
  python print_evals.py --show-models model1,model2 --show-tasks arc,boolq
        """,
    )

    parser.add_argument(
        "-b",
        "--base-dir",
        type=str,
        required=True,
        help="Base directory path. Can be either: 1) A parent directory containing model folders, or 2) A single model directory with task-*metrics.json files",
    )

    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Transpose the table (models as rows, tasks as columns)",
    )

    parser.add_argument(
        "-c",
        "--load-cache",
        action="store_true",
        help="Load results from cache without re-scanning directories",
    )

    parser.add_argument(
        "-r",
        "--reset-cache",
        action="store_true",
        help="Clear cache and re-scan all directories",
    )

    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file",
    )

    # Task averaging options
    parser.add_argument("--avg-core", action="store_true", help="Average core 9 tasks")
    parser.add_argument("--avg-mmlu", action="store_true", help="Average MMLU tasks")
    parser.add_argument("--avg-mmlu-cat", action="store_true", help="Average MMLU by category")
    parser.add_argument(
        "--avg-mmlu-subcat", action="store_true", help="Average MMLU by subcategory"
    )
    parser.add_argument("--avg-gen", action="store_true", help="Average generation tasks")
    parser.add_argument("--avg-mmlu-pro", action="store_true", help="Average MMLU Pro tasks")
    parser.add_argument("--avg-agi-eval", action="store_true", help="Average AGI Eval tasks")
    parser.add_argument("--avg-bbh", action="store_true", help="Average BBH tasks")
    parser.add_argument("--avg-mm", action="store_true", help="Average Minerva Math tasks")
    parser.add_argument("--avg-math2", action="store_true", help="Average gsm8k and Minerva Math equally into math2")
    parser.add_argument("--avg-ruler", action="store_true", help="Average RULER tasks")
    parser.add_argument("--avg-sciriff", action="store_true", help="Average SciRIFF tasks")
    parser.add_argument("--avg-code", action="store_true", help="Average coding tasks")

    # Task filtering
    parser.add_argument("--stem-only", action="store_true", help="Show only STEM tasks")
    parser.add_argument("--code-only", action="store_true", help="Show only coding tasks")
    parser.add_argument("--hide-code", action="store_true", help="Hide coding tasks")
    parser.add_argument(
        "--core-and-gen-only", action="store_true", help="Show only core and generation tasks"
    )

    # Model/task filtering
    parser.add_argument("--hide-models", help="Comma-separated list of model patterns to hide")
    parser.add_argument("--show-models", help="Comma-separated list of model patterns to show")
    parser.add_argument("--hide-tasks", help="Comma-separated list of task patterns to hide")
    parser.add_argument("--show-tasks", help="Comma-separated list of task patterns to show")

    args = parser.parse_args()

    # Validate arguments
    if args.avg_mmlu and args.avg_mmlu_cat:
        parser.error("Cannot use both --avg-mmlu and --avg-mmlu-cat")

    main(args)
