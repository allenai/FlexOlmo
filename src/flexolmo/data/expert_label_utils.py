"""
Utilities for handling expert labels in supervised router training.
"""

import torch


def get_expert_label_tensor(domain_label: str) -> torch.Tensor:
    """
    Map domain label to expert label tensor.
    
    Args:
        domain_label: Domain label string (e.g., "starcoder", "mj_finemath4plus")
    
    Returns:
        One-hot encoded tensor of shape (4,) for 4 experts:
        - Expert 0 (Math): [1, 0, 0, 0]
        - Expert 1 (General): [0, 1, 0, 0]
        - Expert 2 (Code): [0, 0, 1, 0]
        - Expert 3: [0, 0, 0, 0] (masked)
    """
    domain_lower = domain_label.lower().strip()
    
    # Code expert (expert 2)
    if domain_lower.startswith("starcoder"):
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    
    # Math expert (expert 0)
    if domain_lower.startswith("mj_finemath4plus") or domain_lower.startswith("mj_finemath"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    # General expert (expert 1) - default
    return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

