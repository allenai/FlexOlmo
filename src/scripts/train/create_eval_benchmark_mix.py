#!/usr/bin/env python3
"""
Create router training mix from evaluation benchmarks.

This script downloads evaluation datasets (MMLU, GSM8K, HumanEval, etc.),
tokenizes them using the dolma2 tokenizer, and saves them as numpy files
that can be used for router training.

This is intended for DEBUGGING purposes only - training on test data is 
cheating and cannot be used for final results. The goal is to understand 
which training approach works best when the training data is perfectly 
in-distribution.

Output structure:
    <output_dir>/
    ├── mj_finemath_gsm8k/part-00-00000.npy   # Math benchmarks
    ├── starcoder_humaneval/part-00-00000.npy  # Code benchmarks  
    ├── mmlu/part-00-00000.npy                 # General benchmarks
    ├── eval_benchmark_mix.txt                 # Mix file (copy to src/flexolmo/data/mixes/)
    └── metadata.json

Usage:
    python src/scripts/train/create_eval_benchmark_mix.py --max_samples_per_task 1000
    
    # Then in training scripts:
    --dataset.mix=eval_benchmark_mix --dataset.mix_base_dir=/weka/oe-training-default/sanjaya/eval_benchmark_data

Author: FlexOlmo Team
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Default output directory (in sanjaya's workspace)
DEFAULT_OUTPUT_DIR = "/weka/oe-training-default/sanjaya/eval_benchmark_data"


# =============================================================================
# BENCHMARK DEFINITIONS
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark dataset."""
    name: str                   # Short name (used as domain_label in mix file)
    hf_path: str               # HuggingFace dataset path
    hf_name: Optional[str]     # HuggingFace dataset config (optional)
    split: str                 # Split to use
    text_fields: List[str]     # Fields to concatenate for text
    answer_field: Optional[str] # Field containing the answer
    expert_category: str       # "math", "code", or "general" (for expert_label_utils mapping)
    prompt_template: Optional[str] = None
    

# Domain label prefixes that expert_label_utils.py recognizes:
#   - "mj_finemath*" → Math expert (0)
#   - "starcoder*" or "*code*" → Code expert (2)  
#   - everything else → General expert (1)

BENCHMARK_CONFIGS = {
    # =========================
    # MATH BENCHMARKS (use mj_finemath_ prefix for expert_label_utils mapping)
    # =========================
    "mj_finemath_gsm8k": BenchmarkConfig(
        name="mj_finemath_gsm8k",
        hf_path="gsm8k",  # Correct: gsm8k, not openai/gsm8k
        hf_name="main",
        split="test",
        text_fields=["question"],
        answer_field="answer",
        expert_category="math",
        prompt_template="Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer: {answer}",
    ),
    "mj_finemath_minerva": BenchmarkConfig(
        name="mj_finemath_minerva",
        hf_path="EleutherAI/hendrycks_math",
        hf_name="all",  # Load all subsets together
        split="test",
        text_fields=["problem"],
        answer_field="solution",
        expert_category="math",
        prompt_template="Solve the following math problem.\n\nProblem: {problem}\n\nSolution: {solution}",
    ),
    "mj_finemath_aime": BenchmarkConfig(
        name="mj_finemath_aime",
        hf_path="allenai/aime-2021-2025",
        hf_name=None,
        split="train",  # Filtered by year during eval, but we use all for training
        text_fields=["problem"],
        answer_field="answer",
        expert_category="math",
        prompt_template="Solve the following AIME problem.\n\nProblem: {problem}\n\nAnswer: {answer}",
    ),
    
    # =========================
    # CODE BENCHMARKS (use starcoder_ prefix for expert_label_utils mapping)
    # =========================
    "starcoder_humaneval": BenchmarkConfig(
        name="starcoder_humaneval",
        hf_path="evalplus/humanevalplus",
        hf_name=None,
        split="test",
        text_fields=["prompt"],
        answer_field="canonical_solution",
        expert_category="code",
        prompt_template="{prompt}\n{canonical_solution}",
    ),
    "starcoder_mbpp": BenchmarkConfig(
        name="starcoder_mbpp",
        hf_path="evalplus/mbppplus",
        hf_name=None,
        split="test",
        text_fields=["prompt"],
        answer_field="code",
        expert_category="code",
        prompt_template="# Task: {prompt}\n\n{code}",
    ),
    
    # =========================
    # GENERAL/REASONING BENCHMARKS
    # =========================
    "mmlu": BenchmarkConfig(
        name="mmlu",
        hf_path="cais/mmlu",
        hf_name="all",
        split="test",
        text_fields=["question"],
        answer_field=None,
        expert_category="general",
        prompt_template=None,  # Special handling for MC
    ),
    "bbh": BenchmarkConfig(
        name="bbh",
        hf_path="lukaemon/bbh",
        hf_name=None,  # Will iterate over configs
        split="test",
        text_fields=["input"],
        answer_field="target",
        expert_category="general",
        prompt_template="Question: {input}\n\nAnswer: {target}",
    ),
    "popqa": BenchmarkConfig(
        name="popqa",
        hf_path="akariasai/PopQA",
        hf_name=None,
        split="test",
        text_fields=["question"],
        answer_field="possible_answers",
        expert_category="general",
        prompt_template="Question: {question}\n\nAnswer: {answer}",
    ),
    "simpleqa": BenchmarkConfig(
        name="simpleqa",
        hf_path="lighteval/SimpleQA",
        hf_name=None,
        split="test",
        text_fields=["problem"],
        answer_field="answer",
        expert_category="general",
        prompt_template="Question: {problem}\n\nAnswer: {answer}",
    ),
    "gpqa": BenchmarkConfig(
        name="gpqa",
        hf_path="Idavidrein/gpqa",
        hf_name="gpqa_main",  # Correct: gpqa_main, not gpqa_extended
        split="train",
        text_fields=["Question"],
        answer_field="Correct Answer",
        expert_category="general",
        prompt_template="Question: {Question}\n\nAnswer: {Correct Answer}",
    ),
    "zebralogic": BenchmarkConfig(
        name="zebralogic",
        hf_path="allenai/ZebraLogicBench-private",
        hf_name="grid_mode",
        split="test",
        text_fields=["puzzle"],
        answer_field="solution",
        expert_category="general",
        prompt_template="Solve this logic puzzle:\n\n{puzzle}\n\nSolution: {solution}",
    ),
    "ifeval": BenchmarkConfig(
        name="ifeval",
        hf_path="HuggingFaceH4/ifeval",  # Correct: HuggingFaceH4/ifeval, not google/IFEval
        hf_name=None,
        split="train",
        text_fields=["prompt"],
        answer_field=None,
        expert_category="general",
        prompt_template="{prompt}",
    ),
    # NOTE: alpaca_eval uses deprecated HF loading script - skipped
    # NOTE: agi_eval_english uses local files - skipped
}


# =============================================================================
# TOKENIZATION UTILITIES
# =============================================================================

def load_tokenizer(tokenizer_name: str):
    """Load the tokenizer from HuggingFace."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    log.info(f"Loaded tokenizer: {tokenizer_name}")
    log.info(f"  Vocab size: {tokenizer.vocab_size}")
    log.info(f"  EOS token ID: {tokenizer.eos_token_id}")
    return tokenizer


def format_text_from_example(example: Dict[str, Any], config: BenchmarkConfig) -> str:
    """Format text from a dataset example."""
    
    # Special handling for MMLU (multiple choice)
    if config.name == "mmlu":
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer_idx = example.get("answer", 0)
        
        choice_letters = ["A", "B", "C", "D"]
        choices_text = "\n".join([f"{choice_letters[i]}. {c}" for i, c in enumerate(choices)])
        answer = choice_letters[answer_idx] if isinstance(answer_idx, int) and answer_idx < len(choice_letters) else str(answer_idx)
        
        return f"Question: {question}\n\n{choices_text}\n\nAnswer: {answer}"
    
    # Use prompt template if available
    if config.prompt_template:
        try:
            format_dict = {}
            for field in config.text_fields:
                format_dict[field] = example.get(field, "")
            
            if config.answer_field:
                answer = example.get(config.answer_field, "")
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                format_dict[config.answer_field] = answer
                format_dict["answer"] = answer
            
            return config.prompt_template.format(**format_dict)
        except KeyError as e:
            log.warning(f"Missing field {e} in example for {config.name}")
    
    # Fallback: concatenate text fields
    text_parts = []
    for field in config.text_fields:
        if field in example and example[field]:
            text_parts.append(str(example[field]))
    
    if config.answer_field and config.answer_field in example:
        answer = example[config.answer_field]
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        text_parts.append(str(answer))
    
    return "\n\n".join(text_parts)


def tokenize_and_save(
    texts: List[str],
    tokenizer,
    output_dir: Path,
    domain_label: str,
    sequence_length: int,
) -> Tuple[List[Path], int, int]:
    """
    Tokenize texts and save as numpy files.
    
    Returns:
        List of saved file paths, number of sequences, total tokens
    """
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 100257
    
    # Tokenize all texts
    all_token_ids = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_token_id)
        all_token_ids.extend(tokens)
    
    total_tokens = len(all_token_ids)
    
    # Pack into sequences
    sequences = []
    for i in range(0, len(all_token_ids), sequence_length):
        seq = all_token_ids[i:i + sequence_length]
        if len(seq) < sequence_length:
            seq = seq + [eos_token_id] * (sequence_length - len(seq))
        sequences.append(np.array(seq, dtype=np.uint16))
    
    # Save to numpy files
    benchmark_dir = output_dir / domain_label
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    max_seqs_per_file = 1000
    for file_idx, start_idx in enumerate(range(0, len(sequences), max_seqs_per_file)):
        end_idx = min(start_idx + max_seqs_per_file, len(sequences))
        file_sequences = sequences[start_idx:end_idx]
        
        # Concatenate into single array (standard format like router_training_mix)
        data = np.concatenate(file_sequences)
        
        file_path = benchmark_dir / f"part-{file_idx:02d}-00000.npy"
        np.save(file_path, data)
        saved_files.append(file_path)
    
    return saved_files, len(sequences), total_tokens


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_benchmark_dataset(config: BenchmarkConfig, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a benchmark dataset from HuggingFace."""
    from datasets import load_dataset, get_dataset_config_names
    
    log.info(f"Loading dataset: {config.hf_path} (config: {config.hf_name}, split: {config.split})")
    
    try:
        # Special handling for BBH which has multiple configs
        if config.hf_path == "lukaemon/bbh" and config.hf_name is None:
            all_examples = []
            try:
                configs = get_dataset_config_names(config.hf_path)
                for cfg_name in configs[:10]:  # First 10 BBH tasks
                    try:
                        ds = load_dataset(config.hf_path, cfg_name, split=config.split)
                        all_examples.extend(list(ds))
                    except Exception as e:
                        log.warning(f"Failed to load BBH config {cfg_name}: {e}")
            except Exception as e:
                log.warning(f"Failed to get BBH configs: {e}")
                ds = load_dataset(config.hf_path, split=config.split)
                all_examples = list(ds)
            
            if max_samples and len(all_examples) > max_samples:
                all_examples = all_examples[:max_samples]
            return all_examples
        
        # Standard loading (no trust_remote_code - deprecated in newer HF versions)
        if config.hf_name:
            dataset = load_dataset(config.hf_path, config.hf_name, split=config.split)
        else:
            dataset = load_dataset(config.hf_path, split=config.split)
        
        examples = list(dataset)
        if max_samples and len(examples) > max_samples:
            examples = examples[:max_samples]
        
        log.info(f"  Loaded {len(examples)} examples")
        return examples
        
    except Exception as e:
        log.error(f"Failed to load {config.name}: {e}")
        return []


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create router training mix from evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python %(prog)s --max_samples_per_task 1000
  python %(prog)s --output_dir /my/custom/path --max_samples_per_task 500

After running, copy the mix file to src/flexolmo/data/mixes/:
  cp {DEFAULT_OUTPUT_DIR}/eval_benchmark_mix.txt src/flexolmo/data/mixes/

Then use in training:
  --dataset.mix=eval_benchmark_mix --dataset.mix_base_dir={DEFAULT_OUTPUT_DIR}
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (also used as mix_base_dir). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/dolma2-tokenizer",
        help="HuggingFace tokenizer to use",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=4096,
        help="Sequence length for packed data",
    )
    parser.add_argument(
        "--max_samples_per_task",
        type=int,
        default=None,
        help="Maximum samples per benchmark (None = all)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmarks to process (default: all)",
    )
    parser.add_argument(
        "--mix_name",
        type=str,
        default="eval_benchmark_mix",
        help="Name for the generated mix file",
    )
    args = parser.parse_args()
    
    # output_dir is also used as mix_base_dir
    # Paths in mix file are relative to output_dir (e.g., "mj_finemath_gsm8k/part-00-00000.npy")
    output_dir = Path(args.output_dir)
    mix_base_dir = output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("Creating Router Training Mix from Evaluation Benchmarks")
    log.info("=" * 60)
    log.info(f"Output directory: {output_dir}")
    log.info(f"  (this is also mix_base_dir for training)")
    log.info(f"Tokenizer:        {args.tokenizer}")
    log.info(f"Sequence length:  {args.sequence_length}")
    log.info(f"Max samples/task: {args.max_samples_per_task or 'all'}")
    log.info("")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Determine which benchmarks to process
    if args.benchmarks:
        benchmark_names = args.benchmarks
    else:
        benchmark_names = list(BENCHMARK_CONFIGS.keys())
    
    log.info(f"Processing {len(benchmark_names)} benchmarks")
    log.info("")
    
    # Process each benchmark
    mix_entries = []  # (domain_label, relative_path_from_mix_base)
    stats = {"math": 0, "code": 0, "general": 0}
    total_sequences = 0
    total_tokens = 0
    
    for bench_name in benchmark_names:
        if bench_name not in BENCHMARK_CONFIGS:
            log.warning(f"Unknown benchmark: {bench_name}, skipping")
            continue
        
        config = BENCHMARK_CONFIGS[bench_name]
        log.info(f"Processing: {bench_name} ({config.expert_category})")
        
        # Load dataset
        examples = load_benchmark_dataset(config, args.max_samples_per_task)
        if not examples:
            continue
        
        # Format texts
        texts = []
        for ex in tqdm(examples, desc=f"Formatting {bench_name}", leave=False):
            text = format_text_from_example(ex, config)
            if text.strip():
                texts.append(text)
        
        if not texts:
            log.warning(f"No valid texts for {bench_name}, skipping")
            continue
        
        log.info(f"  Formatted {len(texts)} texts")
        
        # Tokenize and save
        saved_files, num_seqs, num_tokens = tokenize_and_save(
            texts, tokenizer, output_dir, config.name, args.sequence_length
        )
        
        # Add to mix entries (path relative to output_dir/mix_base_dir)
        for f in saved_files:
            relative_path = f"{config.name}/{f.name}"
            mix_entries.append((config.name, relative_path))
        
        stats[config.expert_category] += num_seqs
        total_sequences += num_seqs
        total_tokens += num_tokens
        
        log.info(f"  Saved {len(saved_files)} files ({num_seqs} sequences, {num_tokens:,} tokens)")
        log.info("")
    
    # Write mix file
    mix_file = output_dir / f"{args.mix_name}.txt"
    mix_content = []
    mix_content.append(f"# Router training mix from evaluation benchmarks")
    mix_content.append(f"# WARNING: Training on test data - for debugging only!")
    mix_content.append(f"# Format: domain_label,path/to/file.npy")
    mix_content.append(f"# mix_base_dir: {mix_base_dir}")
    mix_content.append(f"#")
    for domain_label, path in mix_entries:
        mix_content.append(f"{domain_label},{path}")
    
    mix_text = "\n".join(mix_content) + "\n"
    
    with open(mix_file, "w") as f:
        f.write(mix_text)
    log.info(f"Created mix file: {mix_file}")
    
    # Also copy to src/flexolmo/data/mixes/ if we can find the repo root
    try:
        script_path = Path(__file__).resolve()
        # Script is at src/scripts/train/create_eval_benchmark_mix.py
        # Repo root is 4 levels up
        repo_root = script_path.parent.parent.parent.parent
        mixes_dir = repo_root / "src" / "flexolmo" / "data" / "mixes"
        if mixes_dir.exists():
            repo_mix_file = mixes_dir / f"{args.mix_name}.txt"
            with open(repo_mix_file, "w") as f:
                f.write(mix_text)
            log.info(f"Also copied to: {repo_mix_file}")
    except Exception as e:
        log.warning(f"Could not copy to repo mixes dir: {e}")
        log.info(f"Manually copy: cp {mix_file} src/flexolmo/data/mixes/")
    
    # Save metadata
    metadata = {
        "mix_name": args.mix_name,
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "domain_distribution": stats,  # Based on benchmark category, not actual expert routing
        "output_dir": str(output_dir),
        "mix_base_dir": str(mix_base_dir),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    log.info(f"Total sequences: {total_sequences}")
    log.info(f"Total tokens: {total_tokens:,}")
    log.info("")
    log.info("Domain distribution (by sequences):")
    log.info("  (This is based on benchmark category, NOT actual expert routing)")
    for domain, count in stats.items():
        pct = count / total_sequences * 100 if total_sequences > 0 else 0
        log.info(f"  {domain}: {count} ({pct:.1f}%)")
    log.info("")
    log.info("=" * 60)
    log.info("Usage Instructions")
    log.info("=" * 60)
    log.info("")
    log.info("1. Use in training:")
    log.info(f"   --dataset.mix={args.mix_name} --dataset.mix_base_dir={mix_base_dir}")
    log.info("")
    log.info("2. For per-token labels, run:")
    log.info(f"   torchrun --nproc-per-node=8 src/scripts/train/generate_expert_labels_per_token.py \\")
    log.info(f"       --checkpoint /path/to/checkpoint \\")
    log.info(f"       --mix {args.mix_name} \\")
    log.info(f"       --mix_base_dir {mix_base_dir} \\")
    log.info(f"       --output_dir {output_dir}/per_token_labels")
    log.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
