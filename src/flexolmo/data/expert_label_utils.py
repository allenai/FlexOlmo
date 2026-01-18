"""
Utilities for handling expert labels in supervised router training.

This module provides labeling strategies:
1. Domain-based hard labeling: Assigns expert based on data source (one-hot)
2. Domain-based soft labeling: Assigns soft probability distribution based on data source
3. Loss-based labeling: Uses pre-computed optimal labels from generate_expert_labels.py

Usage:
    # Domain-based hard labels (default):
    label = get_expert_label_tensor("mj_finemath4plus")  # Returns [1, 0, 0, 0]
    
    # Domain-based soft labels:
    label = get_soft_expert_label_tensor("mj_finemath4plus")  # Returns [0.5, 0.5, 0, 0]
    
    # Custom soft priors:
    priors = {"math": [0.7, 0.3, 0.0, 0.0], "code": [0.0, 0.3, 0.7, 0.0]}
    label = get_soft_expert_label_tensor("mj_finemath4plus", soft_priors=priors)
    
    # Optimal loss-based (via collator):
    # Pass expert_labels_file to DataCollator, which handles label lookup automatically
"""

import torch
from typing import Dict, List, Optional

__all__ = [
    "get_expert_label_tensor",
    "get_soft_expert_label_tensor",
    "expert_id_to_one_hot",
    "DEFAULT_SOFT_PRIORS",
]

# Default soft priors: maps domain type to probability distribution [math, general, code, unused]
DEFAULT_SOFT_PRIORS: Dict[str, List[float]] = {
    "math": [0.5, 0.5, 0.0, 0.0],      # Math: 50% math expert, 50% general expert
    "code": [0.0, 0.5, 0.5, 0.0],      # Code: 50% general expert, 50% code expert
    "general": [0.0, 1.0, 0.0, 0.0],   # General: 100% general expert
}


def _get_domain_type(domain_label: str) -> str:
    """Classify domain label into domain type (math, code, or general)."""
    domain_lower = domain_label.lower().strip()
    
    if domain_lower.startswith("mj_finemath4plus") or domain_lower.startswith("mj_finemath"):
        return "math"
    
    if domain_lower.startswith("starcoder") or "code" in domain_lower:
        return "code"
    
    return "general"


def get_expert_label_tensor(domain_label: str) -> torch.Tensor:
    """
    Map domain label to HARD expert label tensor (one-hot encoded).
    
    Args:
        domain_label: Domain label string (e.g., "starcoder", "mj_finemath4plus")
    
    Returns:
        One-hot encoded tensor of shape (4,) for 3-expert setup:
        - Expert 0 (Math): [1, 0, 0, 0] - mj_finemath4plus
        - Expert 1 (General): [0, 1, 0, 0] - everything else
        - Expert 2 (Code): [0, 0, 1, 0] - starcoder
        - Expert 3: Unused/masked
    """
    domain_type = _get_domain_type(domain_label)
    
    if domain_type == "math":
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    elif domain_type == "code":
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    else:
        return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def get_soft_expert_label_tensor(
    domain_label: str,
    soft_priors: Optional[Dict[str, List[float]]] = None,
) -> torch.Tensor:
    """
    Map domain label to SOFT expert label tensor (probability distribution).
    
    This allows training the router with soft targets instead of hard one-hot labels,
    which can lead to smoother routing behavior.
    
    Args:
        domain_label: Domain label string (e.g., "starcoder", "mj_finemath4plus")
        soft_priors: Optional dict mapping domain type to probability distribution.
            If None, uses DEFAULT_SOFT_PRIORS:
            - math: [0.5, 0.5, 0.0, 0.0] (50% math, 50% general)
            - code: [0.0, 0.5, 0.5, 0.0] (50% general, 50% code)
            - general: [0.0, 1.0, 0.0, 0.0] (100% general)
    
    Returns:
        Soft probability tensor of shape (4,) for 3-expert setup:
        - Expert 0 (Math)
        - Expert 1 (General)
        - Expert 2 (Code)
        - Expert 3: Unused/masked
    
    Example:
        >>> get_soft_expert_label_tensor("mj_finemath4plus")
        tensor([0.5, 0.5, 0.0, 0.0])
        
        >>> get_soft_expert_label_tensor("starcoder")
        tensor([0.0, 0.5, 0.5, 0.0])
        
        >>> custom = {"math": [0.7, 0.3, 0.0, 0.0]}
        >>> get_soft_expert_label_tensor("mj_finemath4plus", soft_priors=custom)
        tensor([0.7, 0.3, 0.0, 0.0])
    """
    if soft_priors is None:
        soft_priors = DEFAULT_SOFT_PRIORS
    
    domain_type = _get_domain_type(domain_label)
    
    if domain_type in soft_priors:
        return torch.tensor(soft_priors[domain_type], dtype=torch.float32)
    
    # Fallback to general if domain type not in priors
    if "general" in soft_priors:
        return torch.tensor(soft_priors["general"], dtype=torch.float32)
    
    # Ultimate fallback: 100% general expert
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
