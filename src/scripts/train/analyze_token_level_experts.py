#!/usr/bin/env python3
"""
Analyze per-token expert advantages to understand WHERE each expert wins.

This helps diagnose issues like:
- Code expert losing on natural language prompts in code benchmarks
- Math expert winning on algorithmic descriptions

Usage:
    python src/scripts/train/analyze_token_level_experts.py \
        --labels_dir /weka/oe-training-default/sanjaya/eval_benchmark_data/per_token_labels_sft \
        --data_dir /weka/oe-training-default/sanjaya/eval_benchmark_data \
        --domain starcoder_humaneval
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_tokenizer():
    """Load tokenizer for decoding."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


EXPERT_NAMES = {0: "Math", 1: "General", 2: "Code"}


def analyze_domain(labels_dir: Path, data_dir: Path, domain: str, tokenizer, max_seqs: int = 5):
    """Analyze token-level expert performance for a specific domain."""
    
    # Load mix file to find domain files
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
    
    # Find sequences for this domain
    seq_to_domain = {}
    current_seq_idx = 0
    target_seq_indices = []
    
    for domain_label, file_path in domain_files:
        full_path = data_dir / file_path
        if full_path.exists():
            data = np.load(full_path)
            num_seqs = len(data) // 4096
            for i in range(num_seqs):
                seq_idx = current_seq_idx + i
                seq_to_domain[seq_idx] = domain_label
                if domain_label == domain:
                    target_seq_indices.append(seq_idx)
            current_seq_idx += num_seqs
    
    if not target_seq_indices:
        print(f"No sequences found for domain: {domain}")
        return
    
    print(f"\n{'='*100}")
    print(f"TOKEN-LEVEL ANALYSIS FOR: {domain}")
    print(f"{'='*100}")
    print(f"Found {len(target_seq_indices)} sequences for this domain")
    
    # Analyze a few sequences in detail
    for seq_idx in target_seq_indices[:max_seqs]:
        label_file = labels_dir / f"seq_{seq_idx:08d}.npz"
        if not label_file.exists():
            continue
        
        data = np.load(label_file)
        if "all_expert_losses" not in data:
            continue
        
        all_losses = data["all_expert_losses"]  # Shape: (seq_len-1, num_experts)
        num_tokens = all_losses.shape[0]
        num_experts = all_losses.shape[1]
        
        # Find winning expert at each position
        winning_experts = all_losses.argmin(axis=1)
        
        # Load token IDs for this sequence
        tokens = None
        for domain_label, file_path in domain_files:
            if domain_label == domain:
                full_path = data_dir / file_path
                if full_path.exists():
                    all_data = np.load(full_path)
                    local_idx = seq_idx - (current_seq_idx - len(all_data) // 4096)
                    # Need to recalculate proper index
                    break
        
        print(f"\n{'-'*80}")
        print(f"Sequence {seq_idx} ({num_tokens} tokens)")
        print(f"{'-'*80}")
        
        # Count wins by position in sequence
        # Split into quartiles
        q_size = num_tokens // 4
        quartile_wins = {q: defaultdict(int) for q in range(4)}
        
        for i, winner in enumerate(winning_experts):
            q = min(i // q_size, 3)  # Quartile 0-3
            quartile_wins[q][winner] += 1
        
        print("\nExpert wins by sequence position:")
        print(f"  {'Position':<20} | {'Math':>8} | {'General':>8} | {'Code':>8}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        for q in range(4):
            start_pct = q * 25
            end_pct = (q + 1) * 25
            total = sum(quartile_wins[q].values())
            if total > 0:
                math_pct = quartile_wins[q][0] / total * 100
                gen_pct = quartile_wins[q][1] / total * 100
                code_pct = quartile_wins[q][2] / total * 100
                print(f"  {start_pct:>3}%-{end_pct:<3}% tokens    | {math_pct:>7.1f}% | {gen_pct:>7.1f}% | {code_pct:>7.1f}%")
        
        # Overall stats
        total_wins = defaultdict(int)
        for winner in winning_experts:
            total_wins[winner] += 1
        
        print(f"\n  Overall token wins:")
        for e in range(num_experts):
            pct = total_wins[e] / num_tokens * 100
            name = EXPERT_NAMES.get(e, f"E{e}")
            print(f"    {name}: {total_wins[e]:,} tokens ({pct:.1f}%)")
        
        # Compute loss differential
        print(f"\n  Average loss by expert:")
        for e in range(num_experts):
            avg_loss = all_losses[:, e].mean()
            name = EXPERT_NAMES.get(e, f"E{e}")
            print(f"    {name}: {avg_loss:.4f}")
        
        # Show where Code expert has biggest advantage/disadvantage
        code_advantage = all_losses[:, 0] - all_losses[:, 2]  # Math - Code (positive = Code better)
        
        best_code_positions = np.argsort(code_advantage)[-10:]  # Where Code does best
        worst_code_positions = np.argsort(code_advantage)[:10]  # Where Code does worst
        
        print(f"\n  Code expert's biggest advantages (vs Math):")
        for pos in best_code_positions:
            adv = code_advantage[pos]
            print(f"    Token {pos}: Code better by {adv:.3f} loss")
        
        print(f"\n  Code expert's biggest disadvantages (vs Math):")
        for pos in worst_code_positions:
            adv = code_advantage[pos]
            print(f"    Token {pos}: Math better by {-adv:.3f} loss")
    
    # Aggregate analysis across all sequences
    print(f"\n{'='*80}")
    print(f"AGGREGATE ANALYSIS ACROSS ALL {len(target_seq_indices)} SEQUENCES")
    print(f"{'='*80}")
    
    all_quartile_wins = {q: defaultdict(int) for q in range(4)}
    total_quartile_tokens = {q: 0 for q in range(4)}
    
    for seq_idx in target_seq_indices:
        label_file = labels_dir / f"seq_{seq_idx:08d}.npz"
        if not label_file.exists():
            continue
        
        data = np.load(label_file)
        if "all_expert_losses" not in data:
            continue
        
        all_losses = data["all_expert_losses"]
        num_tokens = all_losses.shape[0]
        winning_experts = all_losses.argmin(axis=1)
        
        q_size = num_tokens // 4
        for i, winner in enumerate(winning_experts):
            q = min(i // q_size, 3)
            all_quartile_wins[q][winner] += 1
            total_quartile_tokens[q] += 1
    
    print("\nExpert wins by position (aggregated):")
    print(f"  {'Position':<20} | {'Math':>8} | {'General':>8} | {'Code':>8} | {'Total Tokens':>12}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")
    
    for q in range(4):
        start_pct = q * 25
        end_pct = (q + 1) * 25
        total = total_quartile_tokens[q]
        if total > 0:
            math_pct = all_quartile_wins[q][0] / total * 100
            gen_pct = all_quartile_wins[q][1] / total * 100
            code_pct = all_quartile_wins[q][2] / total * 100
            print(f"  {start_pct:>3}%-{end_pct:<3}% tokens    | {math_pct:>7.1f}% | {gen_pct:>7.1f}% | {code_pct:>7.1f}% | {total:>12,}")


def main():
    parser = argparse.ArgumentParser(description="Analyze token-level expert performance")
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain to analyze (e.g., starcoder_humaneval, starcoder_mbpp)")
    parser.add_argument("--max_seqs", type=int, default=5,
                        help="Max sequences to show in detail")
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    data_dir = Path(args.data_dir)
    tokenizer = load_tokenizer()
    
    analyze_domain(labels_dir, data_dir, args.domain, tokenizer, args.max_seqs)


if __name__ == "__main__":
    main()

