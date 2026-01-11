#!/usr/bin/env python3
"""
Analyze per-token expert labels using AGGREGATE-THEN-REPORT approach.

This script reports RAW TOTAL LOSSES per expert (no normalization):
  - For each token: keep all expert losses
  - Aggregate: sum losses per expert across all tokens
  - Report: raw total losses (lower loss = better expert for that data)

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


def load_tokenizer():
    """Load tokenizer for decoding sequences."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
        print("Loaded tokenizer: allenai/dolma2-tokenizer")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


def load_sequence_text(data_dir: Path, seq_idx: int, domain_files: list, tokenizer, sequence_length: int = 4096) -> str:
    """Load and decode the text for a specific sequence index."""
    if tokenizer is None:
        return "[Tokenizer not available]"
    
    try:
        # Find which file and offset this sequence is in
        current_seq_idx = 0
        for domain_label, file_path in domain_files:
            full_path = data_dir / file_path
            if full_path.exists():
                data = np.load(full_path)
                num_seqs = len(data) // sequence_length
                if current_seq_idx <= seq_idx < current_seq_idx + num_seqs:
                    # Found the file, extract the sequence
                    local_idx = seq_idx - current_seq_idx
                    start = local_idx * sequence_length
                    end = start + sequence_length
                    token_ids = data[start:end].astype(np.int64)
                    # Decode to text
                    text = tokenizer.decode(token_ids, skip_special_tokens=False)
                    return text
                current_seq_idx += num_seqs
    except Exception as e:
        return f"[Error loading text: {e}]"
    return "[Sequence not found]"


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


def get_raw_losses(loss_sums):
    """Return raw losses as numpy array."""
    return np.array(loss_sums)


def analyze_labels_aggregate(labels_dir: str, data_dir: str, output_path: str, show_text: bool = True):
    """Analyze per-token labels - reporting raw total losses per expert."""
    
    labels_dir = Path(labels_dir)
    data_dir = Path(data_dir)
    
    print(f"Analyzing labels from: {labels_dir}")
    print("Reporting RAW TOTAL LOSSES per expert (no normalization)")
    
    # Load tokenizer if we want to show text
    tokenizer = load_tokenizer() if show_text else None
    
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
        
        # Compute combined domain losses (Math, General=E1+E3, Code)
        if num_experts >= 4:
            domain_losses = {
                "Math": float(seq_loss_sums[0]),
                "General": float(seq_loss_sums[1] + seq_loss_sums[3]),
                "Code": float(seq_loss_sums[2]),
            }
        else:
            domain_losses = {
                "Math": float(seq_loss_sums[0]),
                "General": float(seq_loss_sums[1]),
                "Code": float(seq_loss_sums[2]) if num_experts > 2 else 0.0,
            }
        
        # Determine best domain (lowest loss = best)
        best_domain = min(domain_losses, key=domain_losses.get)
        agrees = (best_domain == expected_domain)
        
        # Store per-sequence result
        result = {
            "seq_idx": seq_idx,
            "domain": domain,
            "expected_domain": expected_domain,
            "best_domain": best_domain,
            "agrees": agrees,
            "expert_losses": {e: float(seq_loss_sums[e]) for e in range(num_experts)},
            "domain_losses": domain_losses,
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
        num_experts,
        data_dir,
        domain_files,
        tokenizer
    )
    
    print(f"\nAnalysis written to: {output_path}")


def write_analysis_output(
    output_path: str, 
    results: list,
    domain_loss_sums: dict,
    domain_token_counts: dict,
    overall_loss_sums: dict,
    overall_token_count: int,
    num_experts: int,
    data_dir: Path = None,
    domain_files: list = None,
    tokenizer = None
):
    """Write analysis results to file."""
    
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-TOKEN EXPERT LABEL ANALYSIS (RAW TOTAL LOSSES)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("This analysis reports RAW TOTAL LOSSES per expert:\n")
        f.write("  1. Sum losses per expert across all tokens\n")
        f.write("  2. Report raw totals (lower loss = better expert for that data)\n")
        f.write("  3. No normalization applied\n\n")
        
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
        
        # Get raw overall losses
        overall_losses = get_raw_losses([overall_loss_sums[e] for e in range(num_experts)])
        
        f.write("Per-Expert Total Losses:\n")
        for e in range(num_experts):
            name = EXPERT_NAMES.get(e, f"Expert{e}")
            f.write(f"  {name:10} (Expert {e}): {overall_losses[e]:,.2f}\n")
        
        # Combined domain losses
        if num_experts >= 4:
            math_loss = overall_losses[0]
            general_loss = overall_losses[1] + overall_losses[3]
            code_loss = overall_losses[2]
        else:
            math_loss = overall_losses[0]
            general_loss = overall_losses[1]
            code_loss = overall_losses[2] if num_experts > 2 else 0
        
        f.write("\nCombined Domain Total Losses:\n")
        f.write(f"  Math       (E0):     {math_loss:,.2f}\n")
        f.write(f"  General    (E1+E3):  {general_loss:,.2f}\n")
        f.write(f"  Code       (E2):     {code_loss:,.2f}\n")
        
        # Also show average loss per token for easier comparison
        f.write("\nAverage Loss Per Token:\n")
        for e in range(num_experts):
            name = EXPERT_NAMES.get(e, f"Expert{e}")
            avg_loss = overall_losses[e] / overall_token_count if overall_token_count > 0 else 0
            f.write(f"  {name:10} (Expert {e}): {avg_loss:.4f}\n")
        
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
            
            # Get raw domain losses
            domain_raw_losses = get_raw_losses([loss_sums[e] for e in range(num_experts)])
            
            f.write(f"Domain: {domain}\n")
            f.write(f"  Expected Category: {expected}\n")
            f.write(f"  Total tokens: {token_count:,}\n")
            
            f.write(f"  Per-expert total losses:\n")
            for e in range(num_experts):
                name = EXPERT_NAMES.get(e, f"Expert{e}")
                f.write(f"    {name:10} (E{e}): {domain_raw_losses[e]:,.2f}\n")
            
            f.write(f"  Per-expert avg loss per token:\n")
            for e in range(num_experts):
                name = EXPERT_NAMES.get(e, f"Expert{e}")
                avg = domain_raw_losses[e] / token_count if token_count > 0 else 0
                f.write(f"    {name:10} (E{e}): {avg:.4f}\n")
            
            # Combined domain losses
            if num_experts >= 4:
                math_l = domain_raw_losses[0]
                general_l = domain_raw_losses[1] + domain_raw_losses[3]
                code_l = domain_raw_losses[2]
            else:
                math_l = domain_raw_losses[0]
                general_l = domain_raw_losses[1]
                code_l = domain_raw_losses[2] if num_experts > 2 else 0
            
            # Find best (lowest loss)
            domain_losses_dict = {"Math": math_l, "General": general_l, "Code": code_l}
            best = min(domain_losses_dict, key=domain_losses_dict.get)
            
            f.write(f"  Combined domain total losses:\n")
            f.write(f"    Math    (E0):   {math_l:,.2f}{' <-- lowest (best)' if best == 'Math' else ''}{' <-- expected' if expected == 'Math' else ''}\n")
            f.write(f"    General (E1+3): {general_l:,.2f}{' <-- lowest (best)' if best == 'General' else ''}{' <-- expected' if expected == 'General' else ''}\n")
            f.write(f"    Code    (E2):   {code_l:,.2f}{' <-- lowest (best)' if best == 'Code' else ''}{' <-- expected' if expected == 'Code' else ''}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("PER-SEQUENCE DETAILS\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write(f"{'Seq':>6} | {'Source Domain':25} | {'Expected':8} | {'Best':8} | {'Match':5} | {'Math Loss':>12} | {'Gen Loss':>12} | {'Code Loss':>12}")
        if num_experts >= 4:
            f.write(f" | {'E0':>10} | {'E1':>10} | {'E2':>10} | {'E3':>10}")
        f.write("\n")
        f.write("-" * 160 + "\n")
        
        for r in sorted(results, key=lambda x: x["seq_idx"]):
            match_str = "✓" if r["agrees"] else "✗"
            domain_losses = r.get("domain_losses", {})
            expert_losses = r.get("expert_losses", {})
            
            line = f"{r['seq_idx']:>6} | {r['domain']:25} | {r['expected_domain']:8} | {r['best_domain']:8} | {match_str:^5} "
            line += f"| {domain_losses.get('Math', 0):12.2f} | {domain_losses.get('General', 0):12.2f} | {domain_losses.get('Code', 0):12.2f}"
            if num_experts >= 4:
                line += f" | {expert_losses.get(0, 0):10.2f} | {expert_losses.get(1, 0):10.2f} | {expert_losses.get(2, 0):10.2f} | {expert_losses.get(3, 0):10.2f}"
            f.write(line + "\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("DISAGREEMENT ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        disagreements = [r for r in results if not r.get("agrees", True)]
        
        if disagreements:
            f.write(f"Found {len(disagreements)} sequences where lowest-loss expert differs from expected:\n\n")
            
            # Show first 20 disagreements with full text
            for r in disagreements[:20]:
                f.write("-" * 100 + "\n")
                f.write(f"Seq {r['seq_idx']}: {r['domain']}\n")
                f.write(f"  Expected: {r['expected_domain']}, Got: {r['best_domain']}\n")
                domain_losses = r.get("domain_losses", {})
                f.write(f"  Domain Losses: Math={domain_losses.get('Math', 0):.2f}, General={domain_losses.get('General', 0):.2f}, Code={domain_losses.get('Code', 0):.2f}\n")
                
                # Load and display sequence text
                if tokenizer is not None and data_dir is not None and domain_files is not None:
                    text = load_sequence_text(data_dir, r['seq_idx'], domain_files, tokenizer)
                    # Show first 800 chars and last 400 chars
                    if len(text) > 1400:
                        text_preview = text[:800] + "\n\n  [...middle truncated...]\n\n" + text[-400:]
                    else:
                        text_preview = text
                    # Indent the text
                    text_lines = text_preview.split('\n')
                    indented = '\n'.join('    ' + line for line in text_lines)
                    f.write(f"\n  SEQUENCE TEXT:\n{indented}\n")
                f.write("\n")
            
            if len(disagreements) > 20:
                f.write(f"\n... and {len(disagreements) - 20} more disagreements (text not shown)\n")
                f.write("\nRemaining disagreement summaries:\n")
                for r in disagreements[20:]:
                    domain_losses = r.get("domain_losses", {})
                    f.write(f"  Seq {r['seq_idx']}: {r['domain']} - Expected: {r['expected_domain']}, Got: {r['best_domain']} ")
                    f.write(f"(Math={domain_losses.get('Math', 0):.2f}, Gen={domain_losses.get('General', 0):.2f}, Code={domain_losses.get('Code', 0):.2f})\n")
        else:
            f.write("All sequences agree between source-based and aggregate labels!\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-token expert labels - report raw total losses per expert")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory containing per-token labels (seq_*.npz files with all_expert_losses)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the eval benchmark data and mix file")
    parser.add_argument("--output", type=str, default="expert_label_analysis_aggregate.txt",
                        help="Output file for analysis results")
    parser.add_argument("--show_text", action="store_true", default=True,
                        help="Show decoded text for disagreement sequences (default: True)")
    parser.add_argument("--no_text", action="store_true",
                        help="Don't show decoded text for disagreement sequences")
    args = parser.parse_args()
    
    show_text = args.show_text and not args.no_text
    analyze_labels_aggregate(args.labels_dir, args.data_dir, args.output, show_text=show_text)


if __name__ == "__main__":
    main()
