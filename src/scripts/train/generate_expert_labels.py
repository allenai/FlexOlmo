"""
Generate optimal expert labels for supervised router training.

This script runs inference on the training data with each expert forced to be selected,
computes the per-sequence loss for each expert, and stores the expert that minimizes
the loss as the ground truth label.

The output is a JSON file mapping global sequence indices to optimal expert IDs.

Usage:
    torchrun --nproc-per-node=8 src/scripts/train/generate_expert_labels.py \
        --checkpoint /path/to/checkpoint \
        --output /path/to/labels.json \
        --mix router_training_mix \
        --mix_base_dir /weka/oe-training-default/ai2-llm/ \
        --batch_size 8 \
        --max_tokens 5000000000

The script supports multi-node distributed inference for efficiency.
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
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
    """Configuration for label generation."""
    
    checkpoint_path: str
    output_path: str
    mix_name: str
    mix_base_dir: str
    batch_size: int = 8
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
    # We need to infer the config from the checkpoint or use a known config
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=100352,  # Standard vocab size
        num_experts=4,
        top_k=4,
    )
    
    model = model_config.build(device="cpu", max_seq_len=4096)
    
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
    """Build the dataset for label generation."""
    from flexolmo.data.mixes import CustomDataMix, get_mixture_dataset_config_by_domain
    
    # Create dataset config
    dataset_config = NumpyDatasetConfig(
        sequence_length=config.sequence_length,
        tokenizer=TokenizerConfig.dolma2(),
        mix=config.mix_name,
        mix_base_dir=config.mix_base_dir,
        include_instance_metadata=True,
    )
    
    # Get source mixture config split by domain
    source_mixture_config = get_mixture_dataset_config_by_domain(dataset_config, validate_files=True)
    dataset_config.source_mixture_config = source_mixture_config
    dataset_config.mix = None
    
    # Build the dataset
    dataset = dataset_config.build()
    
    return dataset, dataset_config


def compute_sequence_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute per-sequence loss (sum of token losses per sequence).
    
    Args:
        model: The transformer model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        device: Device to run on
        ignore_index: Index to ignore in loss computation
    
    Returns:
        Per-sequence loss tensor of shape (batch_size,)
    """
    input_ids = input_ids.to(device)
    batch_size, seq_len = input_ids.shape
    
    # Create labels (shifted input_ids)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index  # Last token has no target
    
    with torch.no_grad():
        # Forward pass with loss_reduction="none" to get per-token losses
        output = model(
            input_ids,
            labels=labels,
            ignore_index=ignore_index,
            loss_reduction="none",
            return_logits=False,
        )
        
        # output is LMOutputWithLoss: (logits, loss, ce_loss, z_loss)
        # With reduction="none", ce_loss has shape (batch_size, seq_len)
        if hasattr(output, '__iter__') and len(output) >= 3:
            _, _, ce_loss, _ = output
        else:
            # Fallback: compute loss manually
            logits = output if isinstance(output, torch.Tensor) else output[0]
            logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
            targets = labels[:, 1:].contiguous()  # (B, S-1)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
                reduction="none",
            ).view(batch_size, -1)
        
        # Get local tensor if distributed
        ce_loss = get_local_tensor(ce_loss) if hasattr(ce_loss, 'full_tensor') else ce_loss
        
        # Sum losses per sequence (excluding ignored tokens)
        mask = (labels != ignore_index).float()
        if ce_loss.dim() == 1:
            ce_loss = ce_loss.view(batch_size, -1)
        
        # Per-sequence sum of losses
        sequence_losses = (ce_loss * mask[:, :ce_loss.size(1)]).sum(dim=1)
        
    return sequence_losses


def generate_labels_for_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    expert_indices: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate optimal expert labels for a batch of sequences.
    
    Args:
        model: The transformer model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        device: Device to run on
        expert_indices: Tuple of expert indices to evaluate
    
    Returns:
        Tuple of (optimal_expert_ids, min_losses) each of shape (batch_size,)
    """
    batch_size = input_ids.shape[0]
    num_experts = len(expert_indices)
    
    # Store losses for each expert
    all_losses = torch.zeros(batch_size, num_experts, device=device)
    
    for i, expert_idx in enumerate(expert_indices):
        with ForcedExpertRouter(model, expert_idx):
            losses = compute_sequence_loss(model, input_ids, device)
            all_losses[:, i] = losses
    
    # Find expert with minimum loss for each sequence
    min_losses, min_indices = all_losses.min(dim=1)
    
    # Map back to actual expert indices
    optimal_experts = torch.tensor(
        [expert_indices[idx.item()] for idx in min_indices],
        device=device,
    )
    
    return optimal_experts, min_losses


def main():
    parser = argparse.ArgumentParser(description="Generate optimal expert labels for router training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for labels JSON file")
    parser.add_argument("--mix", type=str, default="router_training_mix", help="Data mix name")
    parser.add_argument("--mix_base_dir", type=str, default="/weka/oe-training-default/ai2-llm/", help="Base directory for data mix")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--max_tokens", type=int, default=5_000_000_000, help="Maximum tokens to process")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--save_interval", type=int, default=10000, help="Save intermediate results every N batches")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        log.info(f"Starting label generation with {world_size} GPUs")
        log.info(f"Config: checkpoint={args.checkpoint}, output={args.output}")
        log.info(f"        mix={args.mix}, batch_size={args.batch_size}, max_tokens={args.max_tokens}")
    
    # Set seed
    seed_all(args.seed)
    
    # Create config
    config = LabelGenerationConfig(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
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
    
    # Create distributed sampler for even distribution across GPUs
    # Use sequential sampling (no shuffle) to maintain deterministic index mapping
    indices = list(range(total_sequences))
    
    # Split indices across ranks
    indices_per_rank = len(indices) // world_size
    start_idx = rank * indices_per_rank
    end_idx = start_idx + indices_per_rank if rank < world_size - 1 else len(indices)
    local_indices = indices[start_idx:end_idx]
    
    if rank == 0:
        log.info(f"Each rank processing ~{indices_per_rank} sequences")
    
    # Results storage: global_index -> optimal_expert_id
    local_results: Dict[int, Dict[str, Any]] = {}
    
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
        batch_metadata = []
        
        for global_idx in batch_indices:
            item = dataset[global_idx]
            if isinstance(item, dict):
                input_ids = item["input_ids"]
                metadata = item.get("metadata", {})
            else:
                input_ids = item
                metadata = {}
            
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            
            batch_input_ids.append(input_ids)
            batch_metadata.append(metadata)
        
        # Stack into batch
        input_ids_batch = torch.stack(batch_input_ids)
        
        # Generate labels for this batch
        try:
            optimal_experts, min_losses = generate_labels_for_batch(
                model,
                input_ids_batch,
                device,
                config.expert_indices,
            )
            
            # Store results
            for i, global_idx in enumerate(batch_indices):
                local_results[global_idx] = {
                    "expert_id": optimal_experts[i].item(),
                    "loss": min_losses[i].item(),
                    "source_name": batch_metadata[i].get("source_name", "unknown"),
                }
        
        except Exception as e:
            log.error(f"Error processing batch {batch_idx}: {e}")
            # Store fallback (general expert)
            for global_idx in batch_indices:
                local_results[global_idx] = {
                    "expert_id": 1,  # General expert as fallback
                    "loss": float("inf"),
                    "source_name": "error",
                }
        
        # Update progress bar
        if rank == 0:
            progress_bar.set_postfix({
                "processed": len(local_results),
                "last_expert": optimal_experts[-1].item() if 'optimal_experts' in dir() else -1,
            })
        
        # Save intermediate results periodically
        if (batch_idx + 1) % args.save_interval == 0:
            intermediate_path = f"{config.output_path}.rank{rank}.partial"
            with open(intermediate_path, "w") as f:
                json.dump(local_results, f)
            if rank == 0:
                log.info(f"Saved intermediate results to {intermediate_path}")
    
    # Gather results from all ranks
    if world_size > 1:
        # Save local results
        local_path = f"{config.output_path}.rank{rank}.json"
        with open(local_path, "w") as f:
            json.dump(local_results, f)
        
        dist.barrier()
        
        # Rank 0 combines all results
        if rank == 0:
            all_results = {}
            for r in range(world_size):
                rank_path = f"{config.output_path}.rank{r}.json"
                with open(rank_path, "r") as f:
                    rank_results = json.load(f)
                    # Convert string keys back to int
                    for k, v in rank_results.items():
                        all_results[int(k)] = v
                # Clean up rank file
                os.remove(rank_path)
            
            local_results = all_results
    
    # Save final results (rank 0 only)
    if rank == 0:
        # Create output directory if needed
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare final output
        output_data = {
            "metadata": {
                "checkpoint": config.checkpoint_path,
                "mix": config.mix_name,
                "max_tokens": config.max_tokens,
                "sequence_length": config.sequence_length,
                "num_sequences": len(local_results),
                "expert_mapping": {
                    "0": "Math",
                    "1": "General", 
                    "2": "Code",
                },
            },
            "labels": {str(k): v for k, v in sorted(local_results.items())},
        }
        
        # Compute statistics
        expert_counts = {0: 0, 1: 0, 2: 0}
        total_loss = 0.0
        for v in local_results.values():
            expert_counts[v["expert_id"]] = expert_counts.get(v["expert_id"], 0) + 1
            total_loss += v["loss"]
        
        output_data["metadata"]["statistics"] = {
            "expert_distribution": expert_counts,
            "average_loss": total_loss / len(local_results) if local_results else 0,
        }
        
        with open(config.output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        log.info(f"Saved {len(local_results)} labels to {config.output_path}")
        log.info(f"Expert distribution: {expert_counts}")
        log.info(f"Average loss: {total_loss / len(local_results) if local_results else 0:.4f}")
    
    # Cleanup
    cleanup_distributed()
    
    if rank == 0:
        log.info("Label generation complete!")


if __name__ == "__main__":
    main()

