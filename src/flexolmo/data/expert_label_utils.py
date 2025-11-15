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
        - Expert 0 (Math): [1, 0, 0, 0] - mj_finemath4plus
        - Expert 1 (General): [0, 1, 0, 0] - default/fallback
        - Expert 2 (Code): [0, 0, 1, 0] - starcoder
        - Expert 3 (Academic/Technical): [0, 0, 0, 1] - academic_writing, technical_writing, etc.
    """
    domain_lower = domain_label.lower().strip()
    
    # Code expert (expert 2)
    if domain_lower.startswith("starcoder") or "code" in domain_lower:
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    
    # Math expert (expert 0)
    if domain_lower.startswith("mj_finemath4plus") or domain_lower.startswith("mj_finemath"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Academic/Technical expert (expert 3)
    # Technical writing, academic content, knowledge articles, tutorials
    if any(keyword in domain_lower for keyword in [
        "academic", "technical_writing", "knowledge_article", 
        "tutorial", "nonfiction", "faq"
    ]):
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    
    # General expert (expert 1) - default for everything else
    # Personal blogs, discussion forums, interviews, product pages, etc.
    return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

