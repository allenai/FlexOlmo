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


def compute_training_perplexity(
    checkpoint_path: str,
    mix_name: str,
    mix_base_dir: str,
    num_seqs: int = 100,
    expert_indices: Tuple[int, ...] = (0, 1, 2),
) -> Dict:
    """Compute perplexity on training data by running inference."""
    
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
    if (checkpoint_dir / "model").exists():
        load_model_and_optim_state(checkpoint_dir, model)
    else:
        # Try loading as a single file
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    log.info(f"Model loaded, building dataset...")
    
    # Build dataset
    dataset_config = NumpyDatasetConfig(
        sequence_length=4096,
        tokenizer=TokenizerConfig.dolma2(),
        mix=CustomDataMix(mix_name),
        mix_base_dir=mix_base_dir,
        include_instance_metadata=False,
    )
    dataset = dataset_config.build()
    
    log.info(f"Dataset built, computing perplexity on {num_seqs} sequences...")
    
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
                            top_k = 4  # Assuming top_k=4
                            
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
    
    # Collect losses per expert
    expert_losses = {e: [] for e in expert_indices}
    
    with torch.no_grad():
        for seq_idx in range(min(num_seqs, len(dataset))):
            batch = dataset[seq_idx]
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            
            for expert_idx in expert_indices:
                with ForcedExpertRouter(model, expert_idx):
                    output = model(input_ids)
                    logits = output.logits if hasattr(output, 'logits') else output
                    
                    # Compute per-token loss
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction='none',
                    )
                    
                    expert_losses[expert_idx].extend(per_token_loss.cpu().numpy().tolist())
            
            if (seq_idx + 1) % 10 == 0:
                log.info(f"Processed {seq_idx + 1}/{num_seqs} sequences")
    
    # Compute perplexities
    results = {
        "num_sequences": num_seqs,
        "num_tokens": len(expert_losses[expert_indices[0]]),
        "per_expert": {},
    }
    
    for e in expert_indices:
        losses = np.array(expert_losses[e])
        ppl = compute_perplexity(losses)
        avg_loss = np.mean(losses)
        results["per_expert"][EXPERT_NAMES.get(e, f"Expert{e}")] = {
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
        print("ROUTER TRAINING MIX (PRE-TRAINING DATA)")
        print("-" * 80)
        print(f"Sequences sampled: {training_results['num_sequences']}")
        print(f"Total tokens: {training_results['num_tokens']:,}")
        
        print("\nPerplexity by Expert:")
        print(f"  {'Expert':<12} | {'Perplexity':>12} | {'Avg Loss':>12}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}")
        for expert, stats in training_results["per_expert"].items():
            print(f"  {expert:<12} | {stats['perplexity']:>12.2f} | {stats['avg_loss']:>12.4f}")
    
    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find which expert has lowest perplexity on each domain
    print("\nBest expert (lowest perplexity) per eval domain:")
    for domain in sorted(eval_results["per_domain"].keys()):
        domain_stats = eval_results["per_domain"][domain]
        best_expert = min(domain_stats.keys(), key=lambda e: domain_stats[e]["perplexity"])
        best_ppl = domain_stats[best_expert]["perplexity"]
        print(f"  {domain:<25}: {best_expert} (PPL={best_ppl:.2f})")


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
    parser.add_argument("--num_training_seqs", type=int, default=100,
                        help="Number of training sequences to sample")
    
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
            args.num_training_seqs,
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

