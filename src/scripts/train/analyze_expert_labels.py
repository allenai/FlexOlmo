#!/usr/bin/env python3
"""
Analyze per-token expert labels and compare with source-based labels.

This script loads the generated per-token labels and compares them with
what we'd expect based on the source domain (e.g., mj_finemath_* → Math expert).

Usage:
    python src/scripts/train/analyze_expert_labels.py \
        --labels_dir /weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels \
        --data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data \
        --output analysis_results.txt
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np


# Expert mapping
EXPERT_NAMES = {0: "Math", 1: "General", 2: "Code"}

# Source domain to expected expert mapping
# Based on expert_label_utils.py conventions:
#   - "mj_finemath*" → Math expert (0)
#   - "starcoder*" or "*code*" → Code expert (2)
#   - everything else → General expert (1)
def get_expected_expert(domain_label: str) -> int:
    """Get expected expert based on domain label."""
    domain_lower = domain_label.lower()
    if domain_lower.startswith("mj_finemath"):
        return 0  # Math
    elif domain_lower.startswith("starcoder") or "code" in domain_lower:
        return 2  # Code
    else:
        return 1  # General


def analyze_labels(labels_dir: str, data_dir: str, output_path: str):
    """Analyze per-token labels and compare with source-based expectations."""
    
    labels_dir = Path(labels_dir)
    data_dir = Path(data_dir)
    
    # Load metadata
    metadata_path = labels_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata.get('num_labeled_sequences', 'unknown')} sequences")
    else:
        metadata = {}
        print("No metadata.json found")
    
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
    # Each numpy file contains multiple sequences (file_size / (4096 * 4 bytes))
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
    
    # Analyze each labeled sequence
    results = []
    domain_stats = defaultdict(lambda: {"total_tokens": 0, "expert_counts": {0: 0, 1: 0, 2: 0}})
    overall_stats = {"total_tokens": 0, "expert_counts": {0: 0, 1: 0, 2: 0}}
    agreement_stats = {"agree": 0, "disagree": 0}
    
    for seq_idx in sorted(labeled_indices):
        label_file = labels_dir / f"seq_{seq_idx:08d}.npz"
        if not label_file.exists():
            continue
        
        # Load per-token labels
        data = np.load(label_file)
        token_labels = data["labels"]
        token_losses = data.get("losses", None)
        
        # Get domain for this sequence
        domain = seq_to_domain.get(seq_idx, "unknown")
        expected_expert = get_expected_expert(domain)
        
        # Count expert distribution for this sequence
        expert_counts = {0: 0, 1: 0, 2: 0}
        for label in token_labels:
            if label in expert_counts:
                expert_counts[label] += 1
        
        total_tokens = len(token_labels)
        
        # Determine majority expert from loss-based labels
        majority_expert = max(expert_counts, key=expert_counts.get)
        majority_pct = expert_counts[majority_expert] / total_tokens * 100 if total_tokens > 0 else 0
        
        # Check agreement
        agrees = (majority_expert == expected_expert)
        if agrees:
            agreement_stats["agree"] += 1
        else:
            agreement_stats["disagree"] += 1
        
        # Calculate percentages
        pcts = {e: expert_counts[e] / total_tokens * 100 if total_tokens > 0 else 0 for e in [0, 1, 2]}
        
        # Average loss if available
        avg_loss = np.nanmean(token_losses) if token_losses is not None else 0
        
        results.append({
            "seq_idx": seq_idx,
            "domain": domain,
            "expected_expert": expected_expert,
            "expected_expert_name": EXPERT_NAMES[expected_expert],
            "majority_expert": majority_expert,
            "majority_expert_name": EXPERT_NAMES[majority_expert],
            "majority_pct": majority_pct,
            "agrees": agrees,
            "expert_pcts": pcts,
            "total_tokens": total_tokens,
            "avg_loss": avg_loss,
        })
        
        # Accumulate stats
        for e in [0, 1, 2]:
            domain_stats[domain]["expert_counts"][e] += expert_counts[e]
            overall_stats["expert_counts"][e] += expert_counts[e]
        domain_stats[domain]["total_tokens"] += total_tokens
        overall_stats["total_tokens"] += total_tokens
    
    # Write output
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-TOKEN EXPERT LABEL ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Expert Mapping:\n")
        f.write("  0 = Math (expected for mj_finemath_* sources)\n")
        f.write("  1 = General (expected for mmlu, bbh, popqa, simpleqa, gpqa, ifeval)\n")
        f.write("  2 = Code (expected for starcoder_* sources)\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        total = overall_stats["total_tokens"]
        f.write(f"Total sequences analyzed: {len(results)}\n")
        f.write(f"Total tokens: {total:,}\n\n")
        
        f.write("Overall Expert Distribution (per-token, loss-based):\n")
        for e in [0, 1, 2]:
            count = overall_stats["expert_counts"][e]
            pct = count / total * 100 if total > 0 else 0
            f.write(f"  {EXPERT_NAMES[e]:10} (Expert {e}): {count:>10,} tokens ({pct:5.2f}%)\n")
        
        f.write(f"\nSource-based vs Loss-based Agreement:\n")
        agree = agreement_stats["agree"]
        disagree = agreement_stats["disagree"]
        total_seqs = agree + disagree
        f.write(f"  Agree:    {agree:>4} sequences ({agree/total_seqs*100:.1f}%)\n")
        f.write(f"  Disagree: {disagree:>4} sequences ({disagree/total_seqs*100:.1f}%)\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("PER-DOMAIN STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        for domain in sorted(domain_stats.keys()):
            stats = domain_stats[domain]
            total_d = stats["total_tokens"]
            expected = get_expected_expert(domain)
            
            f.write(f"Domain: {domain}\n")
            f.write(f"  Expected Expert: {EXPERT_NAMES[expected]} (Expert {expected})\n")
            f.write(f"  Total tokens: {total_d:,}\n")
            f.write(f"  Expert distribution:\n")
            for e in [0, 1, 2]:
                count = stats["expert_counts"][e]
                pct = count / total_d * 100 if total_d > 0 else 0
                marker = " <-- expected" if e == expected else ""
                f.write(f"    {EXPERT_NAMES[e]:10}: {pct:5.2f}%{marker}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("PER-SEQUENCE DETAILS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Seq':>6} | {'Domain':25} | {'Expected':10} | {'Majority':10} | {'Match':5} | {'Math%':>6} | {'Gen%':>6} | {'Code%':>6} | {'Loss':>6}\n")
        f.write("-" * 100 + "\n")
        
        for r in results:
            match_str = "✓" if r["agrees"] else "✗"
            f.write(f"{r['seq_idx']:>6} | {r['domain']:25} | {r['expected_expert_name']:10} | {r['majority_expert_name']:10} | {match_str:^5} | {r['expert_pcts'][0]:>5.1f}% | {r['expert_pcts'][1]:>5.1f}% | {r['expert_pcts'][2]:>5.1f}% | {r['avg_loss']:>6.2f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("DISAGREEMENT ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        disagreements = [r for r in results if not r["agrees"]]
        if disagreements:
            f.write(f"Found {len(disagreements)} sequences where loss-based majority differs from source-based expectation:\n\n")
            for r in disagreements:
                f.write(f"Seq {r['seq_idx']}: {r['domain']}\n")
                f.write(f"  Expected: {r['expected_expert_name']}, Got: {r['majority_expert_name']} ({r['majority_pct']:.1f}%)\n")
                f.write(f"  Distribution: Math={r['expert_pcts'][0]:.1f}%, General={r['expert_pcts'][1]:.1f}%, Code={r['expert_pcts'][2]:.1f}%\n")
                f.write(f"  Avg Loss: {r['avg_loss']:.2f}\n\n")
        else:
            f.write("All sequences agree between source-based and loss-based labels!\n")
    
    print(f"\nAnalysis written to: {output_path}")
    print(f"\nQuick summary:")
    print(f"  Total sequences: {len(results)}")
    print(f"  Agreement rate: {agreement_stats['agree']}/{agreement_stats['agree']+agreement_stats['disagree']} ({agreement_stats['agree']/(agreement_stats['agree']+agreement_stats['disagree'])*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-token expert labels")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory containing per-token labels (seq_*.npz files)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the eval benchmark data and mix file")
    parser.add_argument("--output", type=str, default="expert_label_analysis.txt",
                        help="Output file for analysis results")
    args = parser.parse_args()
    
    analyze_labels(args.labels_dir, args.data_dir, args.output)


if __name__ == "__main__":
    main()

