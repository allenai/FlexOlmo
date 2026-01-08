#!/usr/bin/env python3
"""
Analyze per-token expert labels using AGGREGATE-THEN-SOFTMAX approach.

This script differs from analyze_expert_labels.py in how it computes expert distribution:

OLD approach (argmin-then-aggregate):
  - For each token: pick expert with min loss (winner-take-all)
  - Aggregate: count how many tokens each expert "won"
  - Problem: loses granularity when losses are close

NEW approach (aggregate-then-softmax):
  - For each token: keep all expert losses
  - Aggregate: sum losses per expert across all tokens
  - Convert: softmax(-summed_losses) to get distribution
  - Benefit: captures cumulative advantage, preserves magnitude differences

Usage:
    python src/scripts/train/analyze_expert_labels_aggregate.py \
        --labels_dir /weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels_sft \
        --data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data \
        --output expert_label_analysis_aggregate.txt
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np


# Expert mapping
# Note: Expert 3 is also General, so we combine 1+3 for "General" category
EXPERT_NAMES = {0: "Math", 1: "General", 2: "Code", 3: "General2"}
DOMAIN_NAMES = {"Math": [0], "General": [1, 3], "Code": [2]}


def get_expected_domain(domain_label: str) -> str:
    """Get expected domain category (Math, General, or Code)."""
    domain_lower = domain_label.lower()
    if domain_lower.startswith("mj_finemath"):
        return "Math"
    elif domain_lower.startswith("starcoder") or "code" in domain_lower:
        return "Code"
    else:
        return "General"


def losses_to_distribution(loss_sums):
    """Convert summed losses to probability distribution via inverse normalization.
    
    Uses 1/loss normalized - probability is inversely proportional to loss.
    Lower loss = higher probability, preserves relative differences without
    exponential amplification.
    """
    losses = np.array(loss_sums)
    
    # Handle edge case of zero or negative losses
    losses = np.maximum(losses, 1e-10)
    
    # Inverse loss: lower loss = higher weight
    inv_losses = 1.0 / losses
    
    # Normalize to sum to 1
    return inv_losses / inv_losses.sum()


def analyze_labels_aggregate(labels_dir: str, data_dir: str, output_path: str):
    """Analyze per-token labels using aggregate-then-inverse approach."""
    
    labels_dir = Path(labels_dir)
    data_dir = Path(data_dir)
    
    print(f"Analyzing labels from: {labels_dir}")
    print("Using inverse loss normalization (1/loss / sum(1/loss))")
    
    # Load labeled indices
    labeled_indices_path = labels_dir / "labeled_indices.npy"
    if labeled_indices_path.exists():
        labeled_indices = np.load(labeled_indices_path)
        print(f"Found {len(labeled_indices)} labeled indices")
    else:
        # Fall back to finding seq_*.npz files
        labeled_indices = []
        for f in labels_dir.glob("seq_*.npz"):
            idx = int(f.stem.split("_")[1])
            labeled_indices.append(idx)
        labeled_indices = sorted(labeled_indices)
        print(f"Found {len(labeled_indices)} sequence files")
    
    # Load the mix file to understand domain mapping
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
    
    print(f"Mapped {len(seq_to_domain)} sequences to domains")
    
    # Determine number of experts from first valid file
    num_experts = 3  # Default
    for seq_idx in labeled_indices[:10]:
        label_file = labels_dir / f"seq_{seq_idx:08d}.npz"
        if label_file.exists():
            data = np.load(label_file)
            if "all_expert_losses" in data:
                num_experts = data["all_expert_losses"].shape[1]
                print(f"Detected {num_experts} experts from data")
                break
    
    # Storage for aggregate losses
    # Per-domain: sum of losses for each expert
    domain_loss_sums = defaultdict(lambda: {e: 0.0 for e in range(num_experts)})
    domain_token_counts = defaultdict(int)
    
    # Overall: sum of losses for each expert
    overall_loss_sums = {e: 0.0 for e in range(num_experts)}
    overall_token_count = 0
    
    # Per-sequence results (for detailed output)
    results = []
    
    # Track sequences with missing all_expert_losses
    missing_all_losses = 0
    
    for seq_idx in sorted(labeled_indices):
        label_file = labels_dir / f"seq_{seq_idx:08d}.npz"
        if not label_file.exists():
            continue
        
        # Load data
        data = np.load(label_file)
        
        # Check if all_expert_losses is available
        if "all_expert_losses" not in data:
            missing_all_losses += 1
            continue
        
        all_expert_losses = data["all_expert_losses"]  # Shape: (seq_len-1, num_experts)
        
        # Skip if NaN (corrupted sequence)
        if np.isnan(all_expert_losses).any():
            continue
        
        # Get domain for this sequence
        domain = seq_to_domain.get(seq_idx, "unknown")
        expected_domain = get_expected_domain(domain)
        
        # Sum losses per expert for this sequence
        seq_loss_sums = all_expert_losses.sum(axis=0)  # Shape: (num_experts,)
        total_tokens = all_expert_losses.shape[0]
        
        # Convert to distribution for this sequence
        seq_distribution = losses_to_distribution(seq_loss_sums)
        
        # Compute combined domain distribution (Math, General=E1+E3, Code)
        if num_experts >= 4:
            domain_dist = {
                "Math": float(seq_distribution[0]),
                "General": float(seq_distribution[1] + seq_distribution[3]),
                "Code": float(seq_distribution[2]),
            }
        else:
            domain_dist = {
                "Math": float(seq_distribution[0]),
                "General": float(seq_distribution[1]),
                "Code": float(seq_distribution[2]) if num_experts > 2 else 0.0,
            }
        
        # Determine majority domain
        majority_domain = max(domain_dist, key=domain_dist.get)
        agrees = (majority_domain == expected_domain)
        
        # Store per-sequence result
        result = {
            "seq_idx": seq_idx,
            "domain": domain,
            "expected_domain": expected_domain,
            "majority_domain": majority_domain,
            "agrees": agrees,
            "expert_dist_pcts": {e: float(seq_distribution[e] * 100) for e in range(num_experts)},
            "domain_dist_pcts": {d: v * 100 for d, v in domain_dist.items()},
            "total_tokens": total_tokens,
        }
        results.append(result)
        
        # Accumulate losses for domain-level analysis
        for e in range(num_experts):
            domain_loss_sums[domain][e] += seq_loss_sums[e]
            overall_loss_sums[e] += seq_loss_sums[e]
        domain_token_counts[domain] += total_tokens
        overall_token_count += total_tokens
    
    if missing_all_losses > 0:
        print(f"WARNING: {missing_all_losses} sequences missing 'all_expert_losses' (need to regenerate labels)")
    
    print(f"Processed {len(results)} valid sequences")
    
    # Write output
    write_analysis_output(
        output_path, 
        results, 
        domain_loss_sums, 
        domain_token_counts,
        overall_loss_sums, 
        overall_token_count,
        num_experts
    )
    
    print(f"\nAnalysis written to: {output_path}")


def write_analysis_output(
    output_path: str, 
    results: list,
    domain_loss_sums: dict,
    domain_token_counts: dict,
    overall_loss_sums: dict,
    overall_token_count: int,
    num_experts: int
):
    """Write analysis results to file."""
    
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-TOKEN EXPERT LABEL ANALYSIS (AGGREGATE-THEN-NORMALIZE)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("This analysis uses the AGGREGATE-THEN-INVERSE approach:\n")
        f.write("  1. Sum losses per expert across all tokens\n")
        f.write("  2. Convert to distribution via inverse normalization: (1/loss) / sum(1/loss)\n")
        f.write("  3. Lower loss = higher probability (inversely proportional)\n")
        f.write("  4. No exponential amplification - preserves actual proportions\n\n")
        
        f.write("Expert Mapping:\n")
        f.write("  Expert 0 = Math\n")
        f.write("  Expert 1 = General\n")
        f.write("  Expert 2 = Code\n")
        if num_experts >= 4:
            f.write("  Expert 3 = General (combined with Expert 1 for 'General' category)\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total sequences analyzed: {len(results)}\n")
        f.write(f"Total tokens: {overall_token_count:,}\n\n")
        
        # Convert overall loss sums to distribution
        overall_dist = losses_to_distribution(
            [overall_loss_sums[e] for e in range(num_experts)]
        )
        
        f.write("Per-Expert Distribution (from aggregate softmax):\n")
        for e in range(num_experts):
            name = EXPERT_NAMES.get(e, f"Expert{e}")
            f.write(f"  {name:10} (Expert {e}): {overall_dist[e]*100:5.2f}%\n")
        
        # Combined domain distribution
        if num_experts >= 4:
            math_pct = overall_dist[0] * 100
            general_pct = (overall_dist[1] + overall_dist[3]) * 100
            code_pct = overall_dist[2] * 100
        else:
            math_pct = overall_dist[0] * 100
            general_pct = overall_dist[1] * 100
            code_pct = overall_dist[2] * 100 if num_experts > 2 else 0
        
        f.write("\nCombined Domain Distribution:\n")
        f.write(f"  Math       (E0):     {math_pct:5.2f}%\n")
        f.write(f"  General    (E1+E3):  {general_pct:5.2f}%\n")
        f.write(f"  Code       (E2):     {code_pct:5.2f}%\n")
        
        # Agreement analysis
        valid_results = [r for r in results if r.get("agrees") is not None]
        agree_count = sum(1 for r in valid_results if r["agrees"])
        disagree_count = len(valid_results) - agree_count
        
        f.write(f"\nSource-based vs Loss-based Agreement:\n")
        if valid_results:
            f.write(f"  Agree:    {agree_count:>4} sequences ({agree_count/len(valid_results)*100:.1f}%)\n")
            f.write(f"  Disagree: {disagree_count:>4} sequences ({disagree_count/len(valid_results)*100:.1f}%)\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("PER-DOMAIN STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        for domain in sorted(domain_loss_sums.keys()):
            loss_sums = domain_loss_sums[domain]
            token_count = domain_token_counts[domain]
            expected = get_expected_domain(domain)
            
            # Convert domain loss sums to distribution
            domain_dist = losses_to_distribution(
                [loss_sums[e] for e in range(num_experts)]
            )
            
            f.write(f"Domain: {domain}\n")
            f.write(f"  Expected Category: {expected}\n")
            f.write(f"  Total tokens: {token_count:,}\n")
            
            f.write(f"  Per-expert distribution:\n")
            for e in range(num_experts):
                name = EXPERT_NAMES.get(e, f"Expert{e}")
                f.write(f"    {name:10} (E{e}): {domain_dist[e]*100:5.2f}%\n")
            
            # Combined domain distribution
            if num_experts >= 4:
                math_d = domain_dist[0] * 100
                general_d = (domain_dist[1] + domain_dist[3]) * 100
                code_d = domain_dist[2] * 100
            else:
                math_d = domain_dist[0] * 100
                general_d = domain_dist[1] * 100
                code_d = domain_dist[2] * 100 if num_experts > 2 else 0
            
            f.write(f"  Combined domain distribution:\n")
            f.write(f"    Math    (E0):   {math_d:5.2f}%{' <-- expected' if expected == 'Math' else ''}\n")
            f.write(f"    General (E1+3): {general_d:5.2f}%{' <-- expected' if expected == 'General' else ''}\n")
            f.write(f"    Code    (E2):   {code_d:5.2f}%{' <-- expected' if expected == 'Code' else ''}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("PER-SEQUENCE DETAILS\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write(f"{'Seq':>6} | {'Source Domain':25} | {'Expected':8} | {'Majority':8} | {'Match':5} |  Math% |   Gen% |  Code%")
        if num_experts >= 4:
            f.write(" | (E0)%  | (E1)%  | (E2)%  | (E3)%")
        f.write("\n")
        f.write("-" * 130 + "\n")
        
        for r in sorted(results, key=lambda x: x["seq_idx"]):
            match_str = "✓" if r["agrees"] else "✗"
            domain_probs = r.get("domain_dist_pcts", {})
            expert_probs = r.get("expert_dist_pcts", {})
            
            line = f"{r['seq_idx']:>6} | {r['domain']:25} | {r['expected_domain']:8} | {r['majority_domain']:8} | {match_str:^5} "
            line += f"| {domain_probs.get('Math', 0):5.1f}% | {domain_probs.get('General', 0):5.1f}% | {domain_probs.get('Code', 0):5.1f}%"
            if num_experts >= 4:
                line += f" | {expert_probs.get(0, 0):5.1f}% | {expert_probs.get(1, 0):5.1f}% | {expert_probs.get(2, 0):5.1f}% | {expert_probs.get(3, 0):5.1f}%"
            f.write(line + "\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("DISAGREEMENT ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        disagreements = [r for r in results if not r.get("agrees", True)]
        
        if disagreements:
            f.write(f"Found {len(disagreements)} sequences where aggregate-softmax majority differs from expected:\n\n")
            
            for r in disagreements[:50]:
                f.write("-" * 100 + "\n")
                f.write(f"Seq {r['seq_idx']}: {r['domain']}\n")
                f.write(f"  Expected: {r['expected_domain']}, Got: {r['majority_domain']}\n")
                domain_probs = r.get("domain_dist_pcts", {})
                f.write(f"  Domain Dist: Math={domain_probs.get('Math', 0):.1f}%, General={domain_probs.get('General', 0):.1f}%, Code={domain_probs.get('Code', 0):.1f}%\n\n")
            
            if len(disagreements) > 50:
                f.write(f"\n... and {len(disagreements) - 50} more disagreements\n")
        else:
            f.write("All sequences agree between source-based and aggregate-softmax labels!\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-token expert labels using aggregate-then-inverse normalization")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory containing per-token labels (seq_*.npz files with all_expert_losses)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the eval benchmark data and mix file")
    parser.add_argument("--output", type=str, default="expert_label_analysis_aggregate.txt",
                        help="Output file for analysis results")
    args = parser.parse_args()
    
    analyze_labels_aggregate(args.labels_dir, args.data_dir, args.output)


if __name__ == "__main__":
    main()
