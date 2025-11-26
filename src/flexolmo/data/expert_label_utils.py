"""
Utilities for handling expert labels in supervised router training.

This module provides two labeling strategies:
1. Domain-based labeling: Assigns expert based on data source (e.g., "starcoder" -> Code expert)
2. Loss-based labeling: Uses pre-computed optimal labels from generate_expert_labels.py

The loss-based approach generates labels by running inference with each expert and
selecting the one that minimizes the per-sequence loss.

Usage:
    # Domain-based (default):
    label = get_expert_label_tensor("mj_finemath4plus")  # Returns [1, 0, 0, 0]
    
    # Optimal loss-based (via collator):
    # Pass expert_labels_file to DataCollator, which handles label lookup automatically
"""

import torch

__all__ = [
    "get_expert_label_tensor",
    "expert_id_to_one_hot",
]


def get_expert_label_tensor(domain_label: str) -> torch.Tensor:
    """
    Map domain label to expert label tensor (domain-based heuristic).
    
    Args:
        domain_label: Domain label string (e.g., "starcoder", "mj_finemath4plus")
    
    Returns:
        One-hot encoded tensor of shape (4,) for 3-expert setup:
        - Expert 0 (Math): [1, 0, 0, 0] - mj_finemath4plus
        - Expert 1 (General): [0, 1, 0, 0] - everything else
        - Expert 2 (Code): [0, 0, 1, 0] - starcoder
        - Expert 3: Unused/masked
    """
    domain_lower = domain_label.lower().strip()
    
    # Math expert (expert 0)
    if domain_lower.startswith("mj_finemath4plus") or domain_lower.startswith("mj_finemath"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Code expert (expert 2)
    if domain_lower.startswith("starcoder") or "code" in domain_lower:
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    
    # General expert (expert 1) - everything else
    return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def expert_id_to_one_hot(expert_id: int, num_experts: int = 4) -> torch.Tensor:
    """
    Convert expert ID to one-hot encoded tensor.
    
    Args:
        expert_id: Integer expert ID (0, 1, or 2 for 3-expert setup)
        num_experts: Total number of experts (default 4, with expert 3 unused)
    
    Returns:
        One-hot encoded tensor of shape (num_experts,)
    """
    one_hot = torch.zeros(num_experts, dtype=torch.float32)
    if 0 <= expert_id < num_experts:
        one_hot[expert_id] = 1.0
    else:
        one_hot[1] = 1.0  # Fallback to general expert
    return one_hot
