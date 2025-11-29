"""
Generate optimal PER-TOKEN expert labels for supervised router training.

This script runs inference on the training data with each expert forced to be selected,
computes the per-token loss for each expert, and stores the expert that minimizes
the loss for EACH TOKEN as the ground truth label.

The output is a directory of numpy files mapping sequence indices to per-token optimal expert IDs.

Usage:
    torchrun --nproc-per-node=8 src/scripts/train/generate_expert_labels_per_token.py \
        --checkpoint /path/to/checkpoint \
        --output_dir /path/to/labels_dir \
        --mix router_training_mix \
        --mix_base_dir /weka/oe-training-default/ai2-llm/ \
        --batch_size 4 \
        --max_tokens 5000000000

The script supports multi-node distributed inference for efficiency.
Output format: numpy arrays of shape (seq_len-1,) with values in {0, 1, 2} for each sequence.
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


@dataclass
class LabelGenerationConfig:
    """Configuration for per-token label generation."""
    
    checkpoint_path: str
    output_dir: str
    mix_name: str
    mix_base_dir: str
    batch_size: int = 1  # Use batch_size=1 to avoid MoE routing bugs with larger batches
    max_tokens: int = 5_000_000_000
    sequence_length: int = 4096
    num_experts: int = 3  # Only 3 active experts (expert 3 is masked)
    seed: int = 42
    dtype: str = "bfloat16"
    
    # Expert indices to evaluate (0=Math, 1=General, 2=Code)
    expert_indices: Tuple[int, ...] = (0, 1, 2)


class ForcedExpertRouter:
    """Context manager to force router to select a specific expert."""
    
    def __init__(self, model: torch.nn.Module, forced_expert_idx: int):
        self.model = model
        self.forced_expert_idx = forced_expert_idx
        self.original_forwards = {}
        self.routers = []
        
    def __enter__(self):
        """Patch all routers to force selection of a specific expert."""
        for name, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                self.routers.append((name, module))
                self.original_forwards[name] = module.forward
                
                # Create patched forward that forces expert selection
                forced_idx = self.forced_expert_idx
                original_forward = module.forward
                num_experts = module.num_experts
                
                def make_forced_forward(orig_fn, forced_expert, n_experts):
                    def forced_forward(x, *, loss_div_factor=None):
                        # Call original to get proper shapes
                        expert_weights, expert_indices, batch_size_per_expert, aux_loss = orig_fn(
                            x, loss_div_factor=loss_div_factor
                        )
                        
                        # Force all tokens to route to the specified expert with weight 1.0
                        batch_size, seq_len, top_k = expert_indices.shape
                        
                        # Create forced indices (all pointing to forced_expert)
                        forced_indices = torch.full_like(expert_indices, forced_expert)
                        
                        # Create forced weights (1.0 for the forced expert, 0 for others)
                        forced_weights = torch.zeros_like(expert_weights)
                        forced_weights[..., 0] = 1.0  # Only first top-k slot gets weight
                        
                        # Update batch_size_per_expert to reflect forced routing
                        total_tokens = batch_size * seq_len * top_k
                        new_batch_size_per_expert = torch.zeros(n_experts, dtype=batch_size_per_expert.dtype, device=batch_size_per_expert.device)
                        new_batch_size_per_expert[forced_expert] = total_tokens
                        
                        return forced_weights, forced_indices, new_batch_size_per_expert, aux_loss
                    
                    return forced_forward
                
                module.forward = make_forced_forward(original_forward, forced_idx, num_experts)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original router forwards."""
        for name, module in self.routers:
            module.forward = self.original_forwards[name]


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
    from olmo_core.nn.transformer import TransformerConfig
    
    log.info(f"Loading model from {checkpoint_path}")
    
    # Build model config (same as training)
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=100352,  # Standard vocab size
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
    
    log.info(f"Model loaded successfully")
    return model


def build_dataset(config: LabelGenerationConfig):
    """
    Build the dataset for label generation.
    
    Uses the default NumpyDataset behavior which samples proportionally to token count
    (file size), matching how --dataset.mix=router_training_mix works in training.
    This respects the intended token distribution from file repetitions in the mix.
    """
    from flexolmo.data.mixes import CustomDataMix
    
    # Create dataset config - use CustomDataMix directly (no source_mixture_config)
    # This matches the behavior of --dataset.mix=router_training_mix in training
    dataset_config = NumpyDatasetConfig(
        sequence_length=config.sequence_length,
        tokenizer=TokenizerConfig.dolma2(),
        mix=CustomDataMix(config.mix_name),  # Convert to CustomDataMix enum
        mix_base_dir=config.mix_base_dir,
        include_instance_metadata=True,
    )
    
    # Don't set source_mixture_config - let it use default behavior
    # which samples uniformly by index, where each file contributes
    # instances proportional to its token count (file size)
    
    # Build the dataset
    dataset = dataset_config.build()
    
    log.info(f"Dataset built with {len(dataset)} total instances")
    log.info(f"Token distribution follows file sizes (token counts), not domain labels")
    
    return dataset, dataset_config


def compute_per_token_losses(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute per-token cross-entropy losses.
    
    Args:
        model: The transformer model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        device: Device to run on
        ignore_index: Index to ignore in loss computation
    
    Returns:
        Per-token loss tensor of shape (batch_size, seq_len-1)
    """
    input_ids = input_ids.to(device)
    batch_size, seq_len = input_ids.shape
    
    with torch.no_grad():
        # Forward pass to get logits
        output = model(input_ids)
        
        # Handle different output formats
        if hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Get local tensor if distributed
        if hasattr(logits, 'full_tensor'):
            logits = get_local_tensor(logits)
        
        # Shift for next-token prediction
        # logits: (batch_size, seq_len, vocab_size)
        # We predict token[i+1] from position i
        shift_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
        shift_labels = input_ids[:, 1:].contiguous()   # (B, S-1)
        
        # Compute per-token cross-entropy (no reduction)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # (B*(S-1), V)
            shift_labels.view(-1),                          # (B*(S-1),)
            reduction='none',
        ).view(batch_size, seq_len - 1)  # (B, S-1)
        
    return per_token_loss


def generate_per_token_labels_for_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    expert_indices: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate optimal per-token expert labels for a batch of sequences.
    
    Args:
        model: The transformer model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        device: Device to run on
        expert_indices: Tuple of expert indices to evaluate
    
    Returns:
        Tuple of:
            - optimal_expert_ids: shape (batch_size, seq_len-1), values in expert_indices
            - min_losses: shape (batch_size, seq_len-1), minimum loss per token
    """
    batch_size, seq_len = input_ids.shape
    num_experts = len(expert_indices)
    token_len = seq_len - 1  # After shift for next-token prediction
    
    # Store losses for each expert: (num_experts, batch_size, seq_len-1)
    all_losses = torch.zeros(num_experts, batch_size, token_len, device=device)
    
    for i, expert_idx in enumerate(expert_indices):
        with ForcedExpertRouter(model, expert_idx):
            per_token_loss = compute_per_token_losses(model, input_ids, device)
            all_losses[i] = per_token_loss
    
    # Find expert with minimum loss for each token
    # all_losses: (num_experts, batch_size, seq_len-1)
    min_losses, min_indices = all_losses.min(dim=0)  # (batch_size, seq_len-1)
    
    # Map indices back to actual expert IDs
    expert_indices_tensor = torch.tensor(expert_indices, device=device)
    optimal_experts = expert_indices_tensor[min_indices]  # (batch_size, seq_len-1)
    
    return optimal_experts, min_losses


def main():
    parser = argparse.ArgumentParser(description="Generate optimal per-token expert labels for router training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for labels")
    parser.add_argument("--mix", type=str, default="router_training_mix", help="Data mix name")
    parser.add_argument("--mix_base_dir", type=str, default="/weka/oe-training-default/ai2-llm/", help="Base directory for data mix")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--max_tokens", type=int, default=5_000_000_000, help="Maximum tokens to process")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--save_interval", type=int, default=1000, help="Save progress every N batches")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        log.info(f"Starting PER-TOKEN label generation with {world_size} GPUs")
        log.info(f"Config: checkpoint={args.checkpoint}, output_dir={args.output_dir}")
        log.info(f"        mix={args.mix}, batch_size={args.batch_size}, max_tokens={args.max_tokens}")
    
    # Set seed
    seed_all(args.seed)
    
    # Create config
    config = LabelGenerationConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        mix_name=args.mix,
        mix_base_dir=args.mix_base_dir,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
        seed=args.seed,
        dtype=args.dtype,
    )
    
    # Get dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[config.dtype]
    
    # Create output directory
    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Load model
    model = load_model(config.checkpoint_path, device, dtype)
    
    # Build dataset
    if rank == 0:
        log.info("Building dataset...")
    dataset, dataset_config = build_dataset(config)
    
    # Calculate number of sequences to process
    max_sequences = config.max_tokens // config.sequence_length
    total_sequences = min(len(dataset), max_sequences)
    
    if rank == 0:
        log.info(f"Dataset size: {len(dataset)} sequences")
        log.info(f"Processing up to {total_sequences} sequences ({config.max_tokens} tokens)")
    
    # Split indices across ranks
    indices = list(range(total_sequences))
    indices_per_rank = len(indices) // world_size
    start_idx = rank * indices_per_rank
    end_idx = start_idx + indices_per_rank if rank < world_size - 1 else len(indices)
    local_indices = indices[start_idx:end_idx]
    
    if rank == 0:
        log.info(f"Each rank processing ~{indices_per_rank} sequences")
    
    # Statistics tracking
    expert_token_counts = {0: 0, 1: 0, 2: 0}
    total_tokens_processed = 0
    total_loss = 0.0
    
    # Process in batches
    num_batches = (len(local_indices) + config.batch_size - 1) // config.batch_size
    
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Rank {rank}",
        disable=(rank != 0),
    )
    
    for batch_idx in progress_bar:
        batch_start = batch_idx * config.batch_size
        batch_end = min(batch_start + config.batch_size, len(local_indices))
        batch_indices = local_indices[batch_start:batch_end]
        
        # Get batch data
        batch_input_ids = []
        
        for global_idx in batch_indices:
            item = dataset[global_idx]
            if isinstance(item, dict):
                input_ids = item["input_ids"]
            else:
                input_ids = item
            
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            
            batch_input_ids.append(input_ids)
        
        # Stack into batch
        input_ids_batch = torch.stack(batch_input_ids)
        
        # Generate per-token labels for this batch
        try:
            optimal_experts, min_losses = generate_per_token_labels_for_batch(
                model,
                input_ids_batch,
                device,
                config.expert_indices,
            )
            
            # Save each sequence's labels as a numpy file
            for i, global_idx in enumerate(batch_indices):
                # Get per-token labels for this sequence
                token_labels = optimal_experts[i].cpu().numpy().astype(np.uint8)
                # Keep losses as float32 for accuracy, convert to float16 only for storage
                token_losses_f32 = min_losses[i].cpu().float().numpy()
                token_losses = token_losses_f32.astype(np.float16)
                
                # Save to file
                label_path = output_dir / f"seq_{global_idx:08d}.npz"
                np.savez_compressed(
                    label_path,
                    labels=token_labels,
                    losses=token_losses,
                )
                
                # Update statistics (use float32 for accumulation to avoid overflow)
                for expert_id in config.expert_indices:
                    expert_token_counts[expert_id] += int((token_labels == expert_id).sum())
                total_tokens_processed += len(token_labels)
                total_loss += float(token_losses_f32.sum())  # Use f32 for stats
        
        except Exception as e:
            log.error(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Save fallback (general expert for all tokens) with NaN losses to indicate error
            for global_idx in batch_indices:
                token_labels = np.ones(config.sequence_length - 1, dtype=np.uint8)  # All General
                token_losses = np.full(config.sequence_length - 1, np.nan, dtype=np.float16)
                label_path = output_dir / f"seq_{global_idx:08d}.npz"
                np.savez_compressed(label_path, labels=token_labels, losses=token_losses)
                # Don't update total_loss for failed batches (NaN would propagate)
        
        # Update progress bar
        if rank == 0:
            progress_bar.set_postfix({
                "seqs": len(batch_indices) * (batch_idx + 1),
                "tokens": total_tokens_processed,
            })
        
        # Log progress periodically
        if (batch_idx + 1) % args.save_interval == 0 and rank == 0:
            log.info(f"Progress: {batch_idx + 1}/{num_batches} batches, {total_tokens_processed} tokens")
            log.info(f"Expert distribution so far: {expert_token_counts}")
    
    # Save local statistics BEFORE any distributed sync (to avoid losing data if sync fails)
    local_stats = {
        "rank": rank,
        "expert_token_counts": expert_token_counts,
        "total_tokens_processed": total_tokens_processed,
        "total_loss": total_loss,
        "num_sequences_processed": len(local_indices),
    }
    local_stats_path = output_dir / f"stats_rank_{rank:03d}.json"
    with open(local_stats_path, "w") as f:
        json.dump(local_stats, f, indent=2)
    log.info(f"Rank {rank}: Saved local stats to {local_stats_path}")
    
    # Try to gather statistics from all ranks (with timeout handling)
    global_stats_gathered = False
    if world_size > 1:
        try:
            # Set a reasonable timeout for the barrier
            dist.barrier()
            
            # Convert to tensors for all_reduce
            counts_tensor = torch.tensor(
                [expert_token_counts[0], expert_token_counts[1], expert_token_counts[2], 
                 total_tokens_processed, total_loss],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
            
            if rank == 0:
                expert_token_counts = {
                    0: int(counts_tensor[0].item()),
                    1: int(counts_tensor[1].item()),
                    2: int(counts_tensor[2].item()),
                }
                total_tokens_processed = int(counts_tensor[3].item())
                total_loss = counts_tensor[4].item()
                global_stats_gathered = True
        except Exception as e:
            log.warning(f"Rank {rank}: Failed to gather global statistics: {e}")
            log.warning(f"Rank {rank}: Local data is saved. Run combine_stats.py to merge.")
    else:
        global_stats_gathered = True
    
    # Save metadata (rank 0 only, or if single GPU)
    if rank == 0:
        if not global_stats_gathered:
            # If global gather failed, use local stats and note it
            log.warning("Global stats gathering failed. Saving local rank 0 stats only.")
            log.warning("To get full stats, combine stats_rank_*.json files manually.")
        
        metadata = {
            "type": "per_token",
            "checkpoint": config.checkpoint_path,
            "mix": config.mix_name,
            "max_tokens": config.max_tokens,
            "sequence_length": config.sequence_length,
            "token_length": config.sequence_length - 1,  # After shift
            "num_sequences": total_sequences,
            "total_tokens": total_tokens_processed,
            "global_stats_complete": global_stats_gathered,
            "expert_mapping": {
                "0": "Math",
                "1": "General",
                "2": "Code",
            },
            "statistics": {
                "expert_distribution": expert_token_counts,
                "expert_percentages": {
                    str(k): v / total_tokens_processed * 100 if total_tokens_processed > 0 else 0
                    for k, v in expert_token_counts.items()
                },
                "average_loss_per_token": total_loss / total_tokens_processed if total_tokens_processed > 0 else 0,
            },
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"Saved {total_sequences} sequences with per-token labels to {output_dir}")
        log.info(f"Total tokens: {total_tokens_processed}")
        log.info(f"Expert distribution: {expert_token_counts}")
        log.info(f"Expert percentages: {metadata['statistics']['expert_percentages']}")
        log.info(f"Average loss per token: {metadata['statistics']['average_loss_per_token']:.4f}")
    
    # Cleanup (don't fail if already cleaned up)
    try:
        cleanup_distributed()
    except Exception as e:
        log.warning(f"Rank {rank}: Cleanup warning: {e}")
    
    if rank == 0:
        log.info("Per-token label generation complete!")


if __name__ == "__main__":
    main()

