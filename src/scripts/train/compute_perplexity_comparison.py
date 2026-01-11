#!/usr/bin/env python3
"""
Compute and compare perplexity across different data sources and experts.

This script computes:
1. Perplexity on eval benchmark data (using existing per-token labels)
2. Perplexity on router training mix (pre-training data) - requires running inference

Perplexity = exp(average cross-entropy loss)

Usage:
    # Just analyze existing eval labels (fast, no GPU needed):
    python src/scripts/train/compute_perplexity_comparison.py \
        --eval_labels_dir /weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels_sft \
        --eval_data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data
    
    # Also compute perplexity on router training mix (requires GPU):
    torchrun --nproc-per-node=1 src/scripts/train/compute_perplexity_comparison.py \
        --eval_labels_dir /weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels_sft \
        --eval_data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data \
        --compute_training_ppl \
        --checkpoint /weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-experts-sft-math-mixed \
        --training_mix router_training_mix \
        --training_mix_base_dir /weka/oe-training-default/ai2-llm/ \
        --num_training_seqs 100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

EXPERT_NAMES = {0: "Math", 1: "General", 2: "Code"}


def compute_perplexity(losses: np.ndarray) -> float:
    """Compute perplexity from losses. PPL = exp(mean(loss))"""
    mean_loss = np.mean(losses)
    return np.exp(mean_loss)


def analyze_eval_labels(labels_dir: Path, data_dir: Path) -> Dict:
    """Analyze existing eval labels to compute perplexity per expert and domain."""
    
    log.info(f"Analyzing eval labels from: {labels_dir}")
    
    # Load mix file for domain mapping
    mix_file = data_dir / "eval_benchmark_mix.txt"
    domain_files = []
    if mix_file.exists():
        with open(mix_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        domain_label, file_path = parts[0], parts[1]
                        domain_files.append((domain_label, file_path))
    
    # Build seq_idx -> domain mapping
    seq_to_domain = {}
    current_seq_idx = 0
    for domain_label, file_path in domain_files:
        full_path = data_dir / file_path
        if full_path.exists():
            data = np.load(full_path)
            num_seqs = len(data) // 4096
            for i in range(num_seqs):
                seq_to_domain[current_seq_idx + i] = domain_label
            current_seq_idx += num_seqs
    
    # Collect losses by domain and expert
    domain_losses = defaultdict(lambda: defaultdict(list))  # domain -> expert -> [losses]
    overall_losses = defaultdict(list)  # expert -> [all losses]
    
    # Load labeled indices
    labeled_indices_path = labels_dir / "labeled_indices.npy"
    if labeled_indices_path.exists():
        labeled_indices = np.load(labeled_indices_path)
    else:
        labeled_indices = []
        for f in labels_dir.glob("seq_*.npz"):
            idx = int(f.stem.split("_")[1])
            labeled_indices.append(idx)
        labeled_indices = sorted(labeled_indices)
    
    num_experts = 3
    total_tokens = 0
    
    for seq_idx in labeled_indices:
        label_file = labels_dir / f"seq_{seq_idx:08d}.npz"
        if not label_file.exists():
            continue
        
        data = np.load(label_file)
        if "all_expert_losses" not in data:
            continue
        
        all_losses = data["all_expert_losses"]  # (seq_len-1, num_experts)
        if np.isnan(all_losses).any():
            continue
        
        num_experts = all_losses.shape[1]
        domain = seq_to_domain.get(seq_idx, "unknown")
        
        # Accumulate losses
        for e in range(num_experts):
            expert_losses = all_losses[:, e]
            domain_losses[domain][e].extend(expert_losses.tolist())
            overall_losses[e].extend(expert_losses.tolist())
        
        total_tokens += all_losses.shape[0]
    
    log.info(f"Processed {len(labeled_indices)} sequences, {total_tokens:,} tokens")
    
    # Compute perplexities
    results = {
        "total_sequences": len(labeled_indices),
        "total_tokens": total_tokens,
        "overall": {},
        "per_domain": {},
    }
    
    # Overall perplexity per expert
    for e in range(num_experts):
        losses = np.array(overall_losses[e])
        ppl = compute_perplexity(losses)
        avg_loss = np.mean(losses)
        results["overall"][EXPERT_NAMES.get(e, f"Expert{e}")] = {
            "perplexity": float(ppl),
            "avg_loss": float(avg_loss),
            "num_tokens": len(losses),
        }
    
    # Per-domain perplexity
    for domain in sorted(domain_losses.keys()):
        results["per_domain"][domain] = {}
        for e in range(num_experts):
            losses = np.array(domain_losses[domain][e])
            if len(losses) > 0:
                ppl = compute_perplexity(losses)
                avg_loss = np.mean(losses)
                results["per_domain"][domain][EXPERT_NAMES.get(e, f"Expert{e}")] = {
                    "perplexity": float(ppl),
                    "avg_loss": float(avg_loss),
                    "num_tokens": len(losses),
                }
    
    return results


def get_domain_category(domain_label: str) -> str:
    """Map domain label to category (Math, Code, General)."""
    domain_lower = domain_label.lower()
    if domain_lower.startswith("mj_finemath"):
        return "Math"
    elif domain_lower.startswith("starcoder") or "code" in domain_lower:
        return "Code"
    else:
        return "General"


def compute_training_perplexity(
    checkpoint_path: str,
    mix_name: str,
    mix_base_dir: str,
    num_seqs_per_domain: int = 50,
    expert_indices: Tuple[int, ...] = (0, 1, 2),
) -> Dict:
    """Compute perplexity on training data by running inference, broken down by domain."""
    
    import torch
    import torch.nn.functional as F
    
    # Add FlexOlmo to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    
    from olmo_core.data import NumpyDatasetConfig, TokenizerConfig
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.moe.router import MoERouter
    from flexolmo.data.mixes import CustomDataMix
    
    # Import model_utils to register olmoe_nx7b on TransformerConfig
    import flexolmo.internal.model_utils  # noqa: F401 - side effect import
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    log.info(f"Loading model from {checkpoint_path}")
    
    # Build model
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=100352,
        num_experts=4,
        top_k=4,
    )
    model = model_config.build(init_device="cpu")
    
    # Load checkpoint
    checkpoint_dir = Path(checkpoint_path)
    
    # Check for distributed checkpoint format (.distcp files)
    distcp_files = list(checkpoint_dir.glob("*.distcp"))
    if distcp_files:
        # Distributed checkpoint - use load_model_and_optim_state directly on the directory
        log.info(f"Loading distributed checkpoint from {checkpoint_dir}")
        load_model_and_optim_state(checkpoint_dir, model)
    elif (checkpoint_dir / "model").exists():
        # Checkpoint with model subdirectory
        load_model_and_optim_state(checkpoint_dir, model)
    elif checkpoint_dir.is_file():
        # Single file checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Could not determine checkpoint format for: {checkpoint_path}")
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    log.info(f"Model loaded, loading data by domain...")
    
    # Load mix file to get domain-specific files
    mix_file_path = Path(__file__).parent.parent.parent / "flexolmo" / "data" / "mixes" / f"{mix_name}.txt"
    if not mix_file_path.exists():
        # Try alternate path
        mix_file_path = Path("src/flexolmo/data/mixes") / f"{mix_name}.txt"
    
    # Parse mix file by domain category
    domain_files = {"Math": [], "Code": [], "General": []}
    
    if mix_file_path.exists():
        with open(mix_file_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        domain_label, file_path = parts[0], parts[1]
                        category = get_domain_category(domain_label)
                        domain_files[category].append((domain_label, file_path))
        
        log.info(f"Found files by category:")
        for cat, files in domain_files.items():
            log.info(f"  {cat}: {len(files)} files")
    else:
        log.warning(f"Mix file not found: {mix_file_path}, will sample randomly")
    
    # ForcedExpertRouter context manager
    class ForcedExpertRouter:
        def __init__(self, model, forced_expert_idx):
            self.model = model
            self.forced_expert_idx = forced_expert_idx
            self.original_forwards = {}
            self.routers = []
        
        def __enter__(self):
            for name, module in self.model.named_modules():
                if isinstance(module, MoERouter):
                    self.routers.append((name, module))
                    self.original_forwards[name] = module.forward
                    
                    forced_idx = self.forced_expert_idx
                    num_experts = module.num_experts
                    
                    def make_forced_forward(forced_expert, n_experts):
                        def forced_forward(x, *, loss_div_factor=None):
                            batch_size, seq_len = x.shape[:2]
                            top_k = 4
                            
                            forced_indices = torch.full(
                                (batch_size, seq_len, top_k), 
                                forced_expert, 
                                dtype=torch.int32, 
                                device=x.device
                            )
                            forced_weights = torch.zeros(
                                (batch_size, seq_len, top_k), 
                                dtype=x.dtype, 
                                device=x.device
                            )
                            forced_weights[..., 0] = 1.0
                            
                            batch_size_per_expert = torch.zeros(
                                n_experts, dtype=torch.int32, device=x.device
                            )
                            batch_size_per_expert[forced_expert] = batch_size * seq_len * top_k
                            
                            return forced_weights, forced_indices, batch_size_per_expert, torch.tensor(0.0, device=x.device)
                        return forced_forward
                    
                    module.forward = make_forced_forward(forced_idx, num_experts)
            return self
        
        def __exit__(self, *args):
            for name, module in self.routers:
                module.forward = self.original_forwards[name]
    
    # Collect losses per domain category and expert
    results = {
        "num_seqs_per_domain": num_seqs_per_domain,
        "per_domain": {},
        "overall": {},
    }
    
    overall_losses = {e: [] for e in expert_indices}
    
    for category in ["Math", "Code", "General"]:
        log.info(f"\nProcessing {category} data...")
        
        if not domain_files[category]:
            log.warning(f"No files found for {category}, skipping")
            continue
        
        # Sample files for this category
        import random
        random.seed(42)
        sampled_files = random.sample(
            domain_files[category], 
            min(len(domain_files[category]), num_seqs_per_domain)
        )
        
        category_losses = {e: [] for e in expert_indices}
        seqs_processed = 0
        
        for domain_label, file_path in sampled_files:
            full_path = Path(mix_base_dir) / file_path
            if not full_path.exists():
                continue
            
            try:
                data = np.load(full_path)
                num_seqs_in_file = len(data) // 4096
                
                if num_seqs_in_file == 0:
                    continue
                
                # Take first sequence from each file
                input_ids = torch.tensor(data[:4096], dtype=torch.long).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    for expert_idx in expert_indices:
                        with ForcedExpertRouter(model, expert_idx):
                            output = model(input_ids)
                            logits = output.logits if hasattr(output, 'logits') else output
                            
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = input_ids[:, 1:].contiguous()
                            
                            per_token_loss = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction='none',
                            )
                            
                            losses_list = per_token_loss.cpu().numpy().tolist()
                            category_losses[expert_idx].extend(losses_list)
                            overall_losses[expert_idx].extend(losses_list)
                
                seqs_processed += 1
                if seqs_processed % 10 == 0:
                    log.info(f"  Processed {seqs_processed}/{len(sampled_files)} sequences")
                
                if seqs_processed >= num_seqs_per_domain:
                    break
                    
            except Exception as e:
                log.warning(f"Error loading {file_path}: {e}")
                continue
        
        # Compute perplexity for this category
        results["per_domain"][category] = {
            "num_sequences": seqs_processed,
            "num_tokens": len(category_losses[expert_indices[0]]) if category_losses[expert_indices[0]] else 0,
            "per_expert": {},
        }
        
        for e in expert_indices:
            if category_losses[e]:
                losses = np.array(category_losses[e])
                ppl = compute_perplexity(losses)
                avg_loss = np.mean(losses)
                results["per_domain"][category]["per_expert"][EXPERT_NAMES.get(e, f"Expert{e}")] = {
                    "perplexity": float(ppl),
                    "avg_loss": float(avg_loss),
                }
        
        log.info(f"  {category}: {seqs_processed} sequences, {results['per_domain'][category]['num_tokens']:,} tokens")
    
    # Compute overall perplexity
    results["overall"]["num_tokens"] = len(overall_losses[expert_indices[0]]) if overall_losses[expert_indices[0]] else 0
    results["overall"]["per_expert"] = {}
    
    for e in expert_indices:
        if overall_losses[e]:
            losses = np.array(overall_losses[e])
            ppl = compute_perplexity(losses)
            avg_loss = np.mean(losses)
            results["overall"]["per_expert"][EXPERT_NAMES.get(e, f"Expert{e}")] = {
                "perplexity": float(ppl),
                "avg_loss": float(avg_loss),
            }
    
    return results


def print_results(eval_results: Dict, training_results: Optional[Dict] = None):
    """Print formatted results."""
    
    print("\n" + "=" * 100)
    print("PERPLEXITY COMPARISON REPORT")
    print("=" * 100)
    
    print("\n" + "-" * 80)
    print("EVAL BENCHMARK DATA")
    print("-" * 80)
    print(f"Total sequences: {eval_results['total_sequences']}")
    print(f"Total tokens: {eval_results['total_tokens']:,}")
    
    print("\nOverall Perplexity by Expert:")
    print(f"  {'Expert':<12} | {'Perplexity':>12} | {'Avg Loss':>12} | {'Tokens':>12}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for expert, stats in eval_results["overall"].items():
        print(f"  {expert:<12} | {stats['perplexity']:>12.2f} | {stats['avg_loss']:>12.4f} | {stats['num_tokens']:>12,}")
    
    print("\nPerplexity by Domain and Expert:")
    for domain in sorted(eval_results["per_domain"].keys()):
        domain_stats = eval_results["per_domain"][domain]
        print(f"\n  {domain}:")
        for expert, stats in domain_stats.items():
            print(f"    {expert:<10}: PPL={stats['perplexity']:>8.2f}, AvgLoss={stats['avg_loss']:.4f}")
    
    if training_results:
        print("\n" + "-" * 80)
        print("ROUTER TRAINING MIX (PRE-TRAINING DATA) - BY DOMAIN")
        print("-" * 80)
        print(f"Sequences per domain: {training_results['num_seqs_per_domain']}")
        print(f"Total tokens: {training_results['overall'].get('num_tokens', 0):,}")
        
        # Overall perplexity
        print("\nOverall Perplexity by Expert:")
        print(f"  {'Expert':<12} | {'Perplexity':>12} | {'Avg Loss':>12}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}")
        for expert, stats in training_results["overall"].get("per_expert", {}).items():
            print(f"  {expert:<12} | {stats['perplexity']:>12.2f} | {stats['avg_loss']:>12.4f}")
        
        # Per-domain breakdown
        print("\nPerplexity by Domain and Expert (TRAINING DATA):")
        for domain in ["Math", "Code", "General"]:
            if domain in training_results.get("per_domain", {}):
                domain_stats = training_results["per_domain"][domain]
                print(f"\n  {domain} (training data):")
                print(f"    Sequences: {domain_stats.get('num_sequences', 0)}, Tokens: {domain_stats.get('num_tokens', 0):,}")
                for expert, stats in domain_stats.get("per_expert", {}).items():
                    print(f"    {expert:<10}: PPL={stats['perplexity']:>8.2f}, AvgLoss={stats['avg_loss']:.4f}")
    
    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find which expert has lowest perplexity on each eval domain
    print("\nBest expert (lowest perplexity) per EVAL domain:")
    for domain in sorted(eval_results["per_domain"].keys()):
        domain_stats = eval_results["per_domain"][domain]
        best_expert = min(domain_stats.keys(), key=lambda e: domain_stats[e]["perplexity"])
        best_ppl = domain_stats[best_expert]["perplexity"]
        print(f"  {domain:<25}: {best_expert} (PPL={best_ppl:.2f})")
    
    if training_results and "per_domain" in training_results:
        print("\nBest expert (lowest perplexity) per TRAINING domain:")
        for domain in ["Math", "Code", "General"]:
            if domain in training_results["per_domain"]:
                domain_stats = training_results["per_domain"][domain].get("per_expert", {})
                if domain_stats:
                    best_expert = min(domain_stats.keys(), key=lambda e: domain_stats[e]["perplexity"])
                    best_ppl = domain_stats[best_expert]["perplexity"]
                    print(f"  {domain:<25}: {best_expert} (PPL={best_ppl:.2f})")
        
        # Specialization analysis
        print("\n" + "-" * 60)
        print("SPECIALIZATION CHECK:")
        print("-" * 60)
        print("Expected: Math expert should have lowest PPL on Math data,")
        print("          Code expert on Code data, General expert on General data.")
        print("")
        
        expected_best = {"Math": "Math", "Code": "Code", "General": "General"}
        for domain in ["Math", "Code", "General"]:
            if domain in training_results["per_domain"]:
                domain_stats = training_results["per_domain"][domain].get("per_expert", {})
                if domain_stats:
                    best_expert = min(domain_stats.keys(), key=lambda e: domain_stats[e]["perplexity"])
                    expected = expected_best[domain]
                    match = "✓" if best_expert == expected else "✗"
                    print(f"  {domain} data: Best={best_expert}, Expected={expected} {match}")


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity comparison across data sources")
    
    # Eval data arguments
    parser.add_argument("--eval_labels_dir", type=str, required=True,
                        help="Directory containing per-token labels for eval data")
    parser.add_argument("--eval_data_dir", type=str, required=True,
                        help="Directory containing eval benchmark data")
    
    # Training data arguments (optional)
    parser.add_argument("--compute_training_ppl", action="store_true",
                        help="Also compute perplexity on training data (requires GPU)")
    parser.add_argument("--checkpoint", type=str,
                        help="Model checkpoint path (required if --compute_training_ppl)")
    parser.add_argument("--training_mix", type=str, default="router_training_mix",
                        help="Training mix name")
    parser.add_argument("--training_mix_base_dir", type=str, default="/weka/oe-training-default/ai2-llm/",
                        help="Training mix base directory")
    parser.add_argument("--num_seqs_per_domain", type=int, default=50,
                        help="Number of sequences to sample PER DOMAIN (Math, Code, General)")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Analyze eval labels
    eval_results = analyze_eval_labels(
        Path(args.eval_labels_dir),
        Path(args.eval_data_dir)
    )
    
    # Optionally compute training perplexity
    training_results = None
    if args.compute_training_ppl:
        if not args.checkpoint:
            parser.error("--checkpoint is required when --compute_training_ppl is set")
        
        training_results = compute_training_perplexity(
            args.checkpoint,
            args.training_mix,
            args.training_mix_base_dir,
            args.num_seqs_per_domain,
        )
    
    # Print results
    print_results(eval_results, training_results)
    
    # Save to file if requested
    if args.output:
        output_data = {
            "eval_benchmark": eval_results,
            "training_mix": training_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

