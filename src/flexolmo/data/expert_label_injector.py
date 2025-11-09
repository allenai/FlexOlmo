"""
Data loader wrapper that injects expert labels into batches based on domain labels.

This module provides a wrapper around the olmo-core data loader that:
1. Tracks domain labels from the dataset
2. Converts domain labels to expert labels
3. Injects expert labels into batches
"""

import logging
from typing import Dict, Any, Optional, Iterator
import torch

from flexolmo.data.expert_label_utils import get_expert_label_tensor

log = logging.getLogger(__name__)


class ExpertLabelDataLoaderWrapper:
    """
    Wrapper around a data loader that injects expert labels into batches.
    
    This works by:
    1. Extracting domain labels from batch metadata (if available)
    2. Converting domain labels to expert labels
    3. Adding expert_labels to the batch dictionary
    
    The wrapper tries multiple methods to extract source/domain information:
    - From batch["domain_labels"] or batch["source_name"]
    - From dataset source configurations
    - From batch metadata
    """
    
    def __init__(self, data_loader, use_domain_labels: bool = True, dataset=None):
        """
        Args:
            data_loader: The underlying data loader to wrap
            use_domain_labels: If True, extract expert labels from domain labels
            dataset: Optional dataset reference to extract source information
        """
        self.data_loader = data_loader
        self.use_domain_labels = use_domain_labels
        self.dataset = dataset
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches and inject expert labels."""
        for batch in self.data_loader:
            batch = self._inject_expert_labels(batch)
            yield batch
    
    def __len__(self) -> int:
        """Return length of underlying data loader."""
        return len(self.data_loader)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying data loader."""
        return getattr(self.data_loader, name)
    
    def _inject_expert_labels(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject expert labels into batch based on available metadata.
        
        This tries multiple methods to extract domain/source information:
        1. batch["domain_labels"] - direct domain labels
        2. batch["source_name"] - source name from dataset
        3. batch metadata fields
        """
        batch_size = batch["input_ids"].shape[0]
        
        # Try to extract domain labels
        domain_labels = None
        
        # Method 1: Extract from batch["metadata"] - PRIMARY METHOD
        # When using add_source_name_metadata, each instance has metadata with source_name
        if "metadata" in batch:
            metadata_list = batch["metadata"]
            if isinstance(metadata_list, list) and len(metadata_list) == batch_size:
                domain_labels = []
                for meta in metadata_list:
                    if isinstance(meta, dict) and "source_name" in meta:
                        domain_labels.append(meta["source_name"])
                    else:
                        # Fallback to general if metadata doesn't have source_name
                        domain_labels.append("general")
                # If we got domain_labels, use them
                if not domain_labels:
                    domain_labels = None
        
        # Method 2: Check for domain_labels directly in batch (fallback)
        if domain_labels is None and "domain_labels" in batch:
            domain_labels = batch["domain_labels"]
            if isinstance(domain_labels, str):
                domain_labels = [domain_labels] * batch_size
            elif not isinstance(domain_labels, (list, tuple)):
                domain_labels = None
        
        # Method 3: Check for source_name directly in batch (fallback)
        if domain_labels is None and "source_name" in batch:
            source_name = batch["source_name"]
            if isinstance(source_name, str):
                domain_labels = [source_name] * batch_size
            elif isinstance(source_name, (list, tuple)):
                domain_labels = list(source_name)
            else:
                domain_labels = None
        
        # Method 4: Check if batch has source info from dataset metadata (another fallback)
        if domain_labels is None and hasattr(self.data_loader, 'dataset') and hasattr(self.data_loader.dataset, '_current_source'):
            # Some dataset implementations might track current source
            current_source = self.data_loader.dataset._current_source
            if current_source:
                domain_labels = [current_source] * batch_size
            else:
                domain_labels = None
        
        # Method 5: Check batch metadata (object attributes)
        if domain_labels is None and hasattr(batch, "__dict__"):
            if hasattr(batch, "domain_labels"):
                domain_labels = batch.domain_labels
            elif hasattr(batch, "source_name"):
                source_name = batch.source_name
                if isinstance(source_name, str):
                    domain_labels = [source_name] * batch_size
                else:
                    domain_labels = None
        
        # Convert domain labels to expert labels
        if domain_labels and self.use_domain_labels:
            expert_labels_list = []
            for domain_label in domain_labels:
                expert_label = get_expert_label_tensor(str(domain_label))
                expert_labels_list.append(expert_label)
            
            # Stack into tensor: (batch_size, num_experts)
            expert_labels = torch.stack(expert_labels_list, dim=0)
            batch["expert_labels"] = expert_labels
            
            # Also store domain labels for debugging
            batch["domain_labels"] = domain_labels if not isinstance(domain_labels, str) else [domain_labels] * batch_size
        else:
            # Fallback: use general expert if we couldn't extract domain labels
            if self.use_domain_labels:
                log.debug(
                    f"Could not extract domain labels from batch, using default general expert labels. "
                    f"Batch keys: {list(batch.keys())}, "
                    f"Has metadata: {'metadata' in batch}, "
                    f"Metadata type: {type(batch.get('metadata'))}"
                )
            batch["expert_labels"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * batch_size, dtype=torch.float32)
        
        return batch


def wrap_data_loader_with_expert_labels(data_loader, use_domain_labels: bool = True, dataset=None):
    """
    Convenience function to wrap a data loader with expert label injection.
    
    Args:
        data_loader: The data loader to wrap
        use_domain_labels: If True, extract expert labels from domain labels
        dataset: Optional dataset reference to extract source information
    
    Returns:
        ExpertLabelDataLoaderWrapper instance
    """
    return ExpertLabelDataLoaderWrapper(data_loader, use_domain_labels=use_domain_labels, dataset=dataset)

