#!/usr/bin/env python3
"""
Analyze router probabilities for expert distribution analysis.

This script captures the router's softmax probability distribution over experts
directly during a single forward pass, providing a more direct measure of expert
preferences compared to the loss-based approach.

Unlike generate_expert_labels_per_token.py which runs 3 forward passes (one per expert)
and takes argmin of losses, this script:
1. Runs a single forward pass with normal routing
2. Captures the router's softmax probabilities at each MoE layer
3. Aggregates probabilities to analyze expert distribution

Usage:
    python src/scripts/train/analyze_router_probabilities.py \
        --checkpoint /path/to/checkpoint \
        --data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data \
        --output router_probability_analysis.txt \
        --batch_size 1

For distributed inference:
    torchrun --nproc-per-node=8 src/scripts/train/analyze_router_probabilities.py \
        --checkpoint /path/to/checkpoint \
        --data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data \
        --output router_probability_analysis.txt
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

# Add FlexOlmo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from olmo_core.config import DType
from olmo_core.data import NumpyDatasetConfig, TokenizerConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all

# Import flexolmo model_utils to register olmoe_nx7b on TransformerConfig
from flexolmo.internal.model_utils import *  # noqa: F401, F403

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Expert mapping
# Note: Expert 3 is also a General expert, so we combine 1+3 for "General" category
EXPERT_NAMES = {0: "Math", 1: "General", 2: "Code", 3: "General2"}

# Logical domain mapping (for reporting combined categories)
DOMAIN_NAMES = {"Math": [0], "General": [1, 3], "Code": [2]}


def get_expected_expert(domain_label: str) -> int:
    """Get expected expert based on domain label.
    
    Returns the primary expert index for the domain.
    For General domains, returns 1 (but note Expert 3 is also General).
    """
    domain_lower = domain_label.lower()
    if domain_lower.startswith("mj_finemath"):
        return 0  # Math
    elif domain_lower.startswith("starcoder") or "code" in domain_lower:
        return 2  # Code
    else:
        return 1  # General (primary)


def get_expected_domain(domain_label: str) -> str:
    """Get expected domain category (Math, General, or Code)."""
    domain_lower = domain_label.lower()
    if domain_lower.startswith("mj_finemath"):
        return "Math"
    elif domain_lower.startswith("starcoder") or "code" in domain_lower:
        return "Code"
    else:
        return "General"


@dataclass
class RouterProbabilityConfig:
    """Configuration for router probability analysis."""
    
    checkpoint_path: str
    data_dir: str
    output_path: str
    batch_size: int = 1
    max_sequences: int = 1000
    sequence_length: int = 4096
    num_experts: int = 4  # Total experts in the model
    seed: int = 42
    dtype: str = "bfloat16"
    aggregate_layers: bool = True  # If True, average across layers; if False, report per-layer


class RouterProbabilityCapture:
    """Context manager to capture router probabilities during forward pass."""
    
    def __init__(self, model: torch.nn.Module, num_experts: int = 4):
        self.model = model
        self.num_experts = num_experts
        self.hooks = []
        self.router_probs: List[torch.Tensor] = []  # Per-layer probabilities
        self.router_logits: List[torch.Tensor] = []  # Per-layer logits
        
    def __enter__(self):
        """Register forward hooks on all MoERouter modules."""
        self.router_probs = []
        self.router_logits = []
        
        def make_hook(layer_idx: int):
            def hook(module: MoERouter, input, output):
                # The router stores the latest logits in _latest_router_logits
                # Shape: (batch_size, seq_len, num_experts)
                if hasattr(module, '_latest_router_logits') and module._latest_router_logits is not None:
                    logits = module._latest_router_logits.detach()
                    # Apply softmax to get probabilities
                    probs = F.softmax(logits.float(), dim=-1)
                    self.router_logits.append(logits.cpu())
                    self.router_probs.append(probs.cpu())
            return hook
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                hook = module.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
                layer_idx += 1
        
        log.info(f"Registered hooks on {layer_idx} MoERouter modules")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_aggregated_probs(self) -> Optional[torch.Tensor]:
        """Get probabilities aggregated across all layers.
        
        Returns:
            Tensor of shape (batch_size, seq_len, num_experts) averaged across layers,
            or None if no probabilities were captured.
        """
        if not self.router_probs:
            return None
        
        # Stack along new dimension and average
        # Each tensor is (batch, seq, num_experts)
        stacked = torch.stack(self.router_probs, dim=0)  # (num_layers, batch, seq, num_experts)
        return stacked.mean(dim=0)  # (batch, seq, num_experts)
    
    def get_per_layer_probs(self) -> List[torch.Tensor]:
        """Get per-layer probabilities."""
        return self.router_probs
    
    def clear(self):
        """Clear captured probabilities for next batch."""
        self.router_probs = []
        self.router_logits = []


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load the model from checkpoint."""
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    
    log.info(f"Loading model from {checkpoint_path}")
    
    # Build model config (same as training)
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=100352,
        num_experts=4,
        top_k=4,
    )
    
    # Build model on CPU first, then move to device
    model = model_config.build(init_device="cpu")
    
    # Load checkpoint
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        load_model_and_optim_state(checkpoint_dir, model)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    log.info("Model loaded successfully")
    return model


def build_domain_mapping(data_dir: Path) -> Tuple[Dict[int, str], List[Tuple[str, str]], int]:
    """Build mapping from sequence index to domain label.
    
    Returns:
        seq_to_domain: Dict mapping sequence index to domain label
        domain_files: List of (domain_label, file_path) tuples
        total_sequences: Total number of sequences across all files
    """
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
    
    # Build sequence index to domain mapping
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
    
    return seq_to_domain, domain_files, current_seq_idx


def load_sequences(data_dir: Path, domain_files: List[Tuple[str, str]], 
                   sequence_length: int = 4096) -> List[Tuple[int, str, np.ndarray]]:
    """Load all sequences from the data files.
    
    Returns:
        List of (global_idx, domain_label, token_ids) tuples
    """
    sequences = []
    current_idx = 0
    
    for domain_label, file_path in domain_files:
        full_path = data_dir / file_path
        if full_path.exists():
            data = np.load(full_path)
            num_seqs = len(data) // sequence_length
            for i in range(num_seqs):
                start = i * sequence_length
                end = start + sequence_length
                token_ids = data[start:end].astype(np.int64)
                sequences.append((current_idx + i, domain_label, token_ids))
            current_idx += num_seqs
    
    return sequences


def analyze_router_probabilities(config: RouterProbabilityConfig):
    """Main analysis function."""
    
    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        log.info(f"Starting router probability analysis with {world_size} GPUs")
        log.info(f"Checkpoint: {config.checkpoint_path}")
        log.info(f"Data dir: {config.data_dir}")
    
    # Set seed
    seed_all(config.seed)
    
    # Get dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[config.dtype]
    
    # Load model
    model = load_model(config.checkpoint_path, device, dtype)
    
    # CUDA warmup
    if rank == 0:
        log.info("Running CUDA warmup...")
    torch.cuda.synchronize(device)
    with torch.no_grad():
        dummy_input = torch.ones(1, 16, dtype=torch.long, device=device)
        try:
            _ = model(dummy_input)
            torch.cuda.synchronize(device)
        except Exception as e:
            log.warning(f"Warmup forward pass failed: {e}")
        del dummy_input
        torch.cuda.empty_cache()
    
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        log.info("CUDA warmup complete")
    
    # Build domain mapping
    data_dir = Path(config.data_dir)
    seq_to_domain, domain_files, total_sequences = build_domain_mapping(data_dir)
    
    if rank == 0:
        log.info(f"Found {total_sequences} sequences across {len(domain_files)} domain files")
    
    # Load sequences
    sequences = load_sequences(data_dir, domain_files, config.sequence_length)
    
    # Limit to max_sequences
    if config.max_sequences > 0 and len(sequences) > config.max_sequences:
        sequences = sequences[:config.max_sequences]
    
    if rank == 0:
        log.info(f"Processing {len(sequences)} sequences")
    
    # Split sequences across ranks
    sequences_per_rank = len(sequences) // world_size
    start_idx = rank * sequences_per_rank
    end_idx = start_idx + sequences_per_rank if rank < world_size - 1 else len(sequences)
    local_sequences = sequences[start_idx:end_idx]
    
    # Results storage
    results = []
    domain_stats = defaultdict(lambda: {
        "total_tokens": 0, 
        "expert_probs": {i: 0.0 for i in range(config.num_experts)}
    })
    overall_stats = {
        "total_tokens": 0, 
        "expert_probs": {i: 0.0 for i in range(config.num_experts)}
    }
    
    # Process sequences
    with RouterProbabilityCapture(model, config.num_experts) as capture:
        progress_bar = tqdm(
            range(0, len(local_sequences), config.batch_size),
            desc=f"Rank {rank}",
            disable=(rank != 0),
        )
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + config.batch_size, len(local_sequences))
            batch_sequences = local_sequences[batch_start:batch_end]
            
            # Prepare batch
            batch_input_ids = []
            batch_indices = []
            batch_domains = []
            
            for global_idx, domain_label, token_ids in batch_sequences:
                input_ids = torch.tensor(token_ids, dtype=torch.long)
                batch_input_ids.append(input_ids)
                batch_indices.append(global_idx)
                batch_domains.append(domain_label)
            
            input_ids_batch = torch.stack(batch_input_ids).to(device)
            
            # Clear previous capture
            capture.clear()
            
            # Forward pass
            with torch.no_grad():
                try:
                    _ = model(input_ids_batch)
                except Exception as e:
                    log.error(f"Error in forward pass: {e}")
                    continue
            
            # Get aggregated probabilities
            if config.aggregate_layers:
                probs = capture.get_aggregated_probs()
            else:
                # For now, just use aggregated; per-layer can be added later
                probs = capture.get_aggregated_probs()
            
            if probs is None:
                log.warning(f"No probabilities captured for batch starting at {batch_start}")
                continue
            
            # Process each sequence in batch
            for i, (global_idx, domain_label) in enumerate(zip(batch_indices, batch_domains)):
                seq_probs = probs[i]  # (seq_len, num_experts)
                
                # Sum probabilities across tokens for each expert
                # This gives us the "expected number of tokens" routed to each expert
                expert_prob_sums = seq_probs.sum(dim=0).numpy()  # (num_experts,)
                
                # Also compute mean probability per expert (for percentages)
                expert_prob_means = seq_probs.mean(dim=0).numpy()  # (num_experts,)
                
                total_tokens = seq_probs.shape[0]
                expected_domain = get_expected_domain(domain_label)
                
                # Compute combined domain probabilities (Math, General=1+3, Code)
                domain_prob_means = {
                    "Math": float(expert_prob_means[0]),
                    "General": float(expert_prob_means[1] + expert_prob_means[3]),  # Combine experts 1 and 3
                    "Code": float(expert_prob_means[2]),
                }
                
                # Determine "majority" domain by highest combined probability
                majority_domain = max(domain_prob_means, key=domain_prob_means.get)
                
                # Check agreement (based on domain, not individual expert)
                agrees = (majority_domain == expected_domain)
                
                # Store result
                result = {
                    "seq_idx": global_idx,
                    "domain": domain_label,
                    "expected_domain": expected_domain,
                    "majority_domain": majority_domain,
                    "agrees": agrees,
                    "expert_prob_pcts": {
                        e: float(expert_prob_means[e] * 100) for e in range(config.num_experts)
                    },
                    "domain_prob_pcts": {
                        d: v * 100 for d, v in domain_prob_means.items()
                    },
                    "total_tokens": total_tokens,
                }
                results.append(result)
                
                # Accumulate stats
                for e in range(config.num_experts):
                    domain_stats[domain_label]["expert_probs"][e] += expert_prob_sums[e]
                    overall_stats["expert_probs"][e] += expert_prob_sums[e]
                domain_stats[domain_label]["total_tokens"] += total_tokens
                overall_stats["total_tokens"] += total_tokens
    
    # Gather results from all ranks (if distributed)
    if world_size > 1:
        # Gather results list
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        if rank == 0:
            results = []
            for r in all_results:
                results.extend(r)
        
        # Reduce stats
        for domain in list(domain_stats.keys()):
            stats_tensor = torch.tensor([
                domain_stats[domain]["total_tokens"],
                *[domain_stats[domain]["expert_probs"][e] for e in range(config.num_experts)]
            ], device=device, dtype=torch.float64)
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            if rank == 0:
                domain_stats[domain]["total_tokens"] = int(stats_tensor[0].item())
                for e in range(config.num_experts):
                    domain_stats[domain]["expert_probs"][e] = stats_tensor[1 + e].item()
        
        overall_tensor = torch.tensor([
            overall_stats["total_tokens"],
            *[overall_stats["expert_probs"][e] for e in range(config.num_experts)]
        ], device=device, dtype=torch.float64)
        dist.all_reduce(overall_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            overall_stats["total_tokens"] = int(overall_tensor[0].item())
            for e in range(config.num_experts):
                overall_stats["expert_probs"][e] = overall_tensor[1 + e].item()
    
    # Write output (rank 0 only)
    if rank == 0:
        write_analysis_output(
            config.output_path, 
            results, 
            domain_stats, 
            overall_stats, 
            config.num_experts
        )
        log.info(f"Analysis written to {config.output_path}")
    
    # Cleanup
    try:
        cleanup_distributed()
    except Exception as e:
        log.warning(f"Cleanup warning: {e}")


def write_analysis_output(
    output_path: str, 
    results: List[Dict], 
    domain_stats: Dict, 
    overall_stats: Dict,
    num_experts: int
):
    """Write analysis results to file."""
    
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("ROUTER PROBABILITY ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("This analysis captures the router's softmax probability distribution directly,\n")
        f.write("instead of computing losses by forcing each expert. This provides a more direct\n")
        f.write("measure of what the router 'wants' to do.\n\n")
        
        f.write("Expert Mapping:\n")
        f.write("  Expert 0 = Math\n")
        f.write("  Expert 1 = General\n")
        f.write("  Expert 2 = Code\n")
        f.write("  Expert 3 = General (combined with Expert 1 for 'General' category)\n")
        f.write("\n")
        f.write("Domain Categories (for agreement analysis):\n")
        f.write("  Math    = Expert 0\n")
        f.write("  General = Expert 1 + Expert 3 (combined)\n")
        f.write("  Code    = Expert 2\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        total_tokens = overall_stats["total_tokens"]
        f.write(f"Total sequences analyzed: {len(results)}\n")
        f.write(f"Total tokens: {total_tokens:,}\n\n")
        
        # Individual expert distribution
        f.write("Per-Expert Probability Distribution:\n")
        for e in range(num_experts):
            prob_sum = overall_stats["expert_probs"][e]
            pct = prob_sum / total_tokens * 100 if total_tokens > 0 else 0
            name = EXPERT_NAMES.get(e, f"Expert{e}")
            f.write(f"  {name:10} (Expert {e}): {pct:5.2f}%\n")
        
        # Combined domain distribution
        f.write("\nCombined Domain Probability Distribution:\n")
        math_pct = overall_stats["expert_probs"][0] / total_tokens * 100 if total_tokens > 0 else 0
        general_pct = (overall_stats["expert_probs"][1] + overall_stats["expert_probs"][3]) / total_tokens * 100 if total_tokens > 0 else 0
        code_pct = overall_stats["expert_probs"][2] / total_tokens * 100 if total_tokens > 0 else 0
        f.write(f"  Math       (Expert 0):     {math_pct:5.2f}%\n")
        f.write(f"  General    (Expert 1+3):   {general_pct:5.2f}%\n")
        f.write(f"  Code       (Expert 2):     {code_pct:5.2f}%\n")
        
        # Agreement analysis
        valid_results = [r for r in results if r.get("agrees") is not None]
        agree_count = sum(1 for r in valid_results if r["agrees"])
        disagree_count = len(valid_results) - agree_count
        
        f.write(f"\nSource-based vs Router-based Agreement (using combined domains):\n")
        if valid_results:
            f.write(f"  Agree:    {agree_count:>4} sequences ({agree_count/len(valid_results)*100:.1f}%)\n")
            f.write(f"  Disagree: {disagree_count:>4} sequences ({disagree_count/len(valid_results)*100:.1f}%)\n")
        else:
            f.write("  No valid results to analyze\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("PER-DOMAIN STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        for domain in sorted(domain_stats.keys()):
            stats = domain_stats[domain]
            total_d = stats["total_tokens"]
            expected_domain = get_expected_domain(domain)
            
            f.write(f"Domain: {domain}\n")
            f.write(f"  Expected Category: {expected_domain}\n")
            f.write(f"  Total tokens: {total_d:,}\n")
            
            # Per-expert distribution
            f.write(f"  Per-expert distribution:\n")
            for e in range(num_experts):
                prob_sum = stats["expert_probs"][e]
                pct = prob_sum / total_d * 100 if total_d > 0 else 0
                name = EXPERT_NAMES.get(e, f"Expert{e}")
                f.write(f"    {name:10} (E{e}): {pct:5.2f}%\n")
            
            # Combined domain distribution
            math_pct_d = stats["expert_probs"][0] / total_d * 100 if total_d > 0 else 0
            general_pct_d = (stats["expert_probs"][1] + stats["expert_probs"][3]) / total_d * 100 if total_d > 0 else 0
            code_pct_d = stats["expert_probs"][2] / total_d * 100 if total_d > 0 else 0
            
            f.write(f"  Combined domain distribution:\n")
            f.write(f"    Math    (E0):   {math_pct_d:5.2f}%{' <-- expected' if expected_domain == 'Math' else ''}\n")
            f.write(f"    General (E1+3): {general_pct_d:5.2f}%{' <-- expected' if expected_domain == 'General' else ''}\n")
            f.write(f"    Code    (E2):   {code_pct_d:5.2f}%{' <-- expected' if expected_domain == 'Code' else ''}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("PER-SEQUENCE DETAILS\n")
        f.write("=" * 100 + "\n\n")
        
        # Header - show both individual experts and combined domains
        f.write(f"{'Seq':>6} | {'Source Domain':25} | {'Expected':8} | {'Majority':8} | {'Match':5} |  Math% |   Gen% |  Code% | (E0)%  | (E1)%  | (E2)%  | (E3)%\n")
        f.write("-" * 130 + "\n")
        
        for r in sorted(results, key=lambda x: x["seq_idx"]):
            match_str = "✓" if r["agrees"] else "✗"
            domain_probs = r.get("domain_prob_pcts", {})
            expert_probs = r.get("expert_prob_pcts", {})
            line = f"{r['seq_idx']:>6} | {r['domain']:25} | {r['expected_domain']:8} | {r['majority_domain']:8} | {match_str:^5} "
            line += f"| {domain_probs.get('Math', 0):5.1f}% | {domain_probs.get('General', 0):5.1f}% | {domain_probs.get('Code', 0):5.1f}% "
            line += f"| {expert_probs.get(0, 0):5.1f}% | {expert_probs.get(1, 0):5.1f}% | {expert_probs.get(2, 0):5.1f}% | {expert_probs.get(3, 0):5.1f}%"
            f.write(line + "\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("DISAGREEMENT ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        disagreements = [r for r in results if not r.get("agrees", True)]
        
        if disagreements:
            f.write(f"Found {len(disagreements)} sequences where router majority differs from source-based expectation:\n\n")
            
            for r in disagreements[:50]:  # Limit to first 50
                f.write("-" * 100 + "\n")
                f.write(f"Seq {r['seq_idx']}: {r['domain']}\n")
                f.write(f"  Expected: {r['expected_domain']}, Got: {r['majority_domain']}\n")
                domain_probs = r.get("domain_prob_pcts", {})
                f.write(f"  Combined Domain Probs: Math={domain_probs.get('Math', 0):.1f}%, General={domain_probs.get('General', 0):.1f}%, Code={domain_probs.get('Code', 0):.1f}%\n")
                expert_probs = r.get("expert_prob_pcts", {})
                f.write(f"  Per-Expert Probs: E0={expert_probs.get(0, 0):.1f}%, E1={expert_probs.get(1, 0):.1f}%, E2={expert_probs.get(2, 0):.1f}%, E3={expert_probs.get(3, 0):.1f}%\n\n")
            
            if len(disagreements) > 50:
                f.write(f"\n... and {len(disagreements) - 50} more disagreements\n")
        else:
            f.write("All sequences agree between source-based and router-based labels!\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze router probabilities for expert distribution")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing eval benchmark data and mix file")
    parser.add_argument("--output", type=str, default="router_probability_analysis.txt",
                        help="Output file for analysis results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--max_sequences", type=int, default=0,
                        help="Maximum sequences to process (0 for all)")
    parser.add_argument("--sequence_length", type=int, default=4096,
                        help="Sequence length")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="Number of experts in the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")
    
    args = parser.parse_args()
    
    config = RouterProbabilityConfig(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        max_sequences=args.max_sequences,
        sequence_length=args.sequence_length,
        num_experts=args.num_experts,
        seed=args.seed,
        dtype=args.dtype,
    )
    
    analyze_router_probabilities(config)


if __name__ == "__main__":
    main()
