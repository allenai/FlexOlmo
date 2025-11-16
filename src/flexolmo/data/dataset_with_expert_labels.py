"""
Dataset wrapper that injects expert labels directly into dataset items.

This is the SIMPLEST approach: modify what the dataset returns in __getitem__.
"""

import logging
from typing import Dict, Any
import torch

from flexolmo.data.expert_label_utils import get_expert_label_tensor

log = logging.getLogger(__name__)


class DatasetWithExpertLabels:
    """
    Wraps a dataset to inject expert labels based on metadata.
    
    This works at the dataset level (not data loader level), so it can't be bypassed.
    """
    
    def __init__(self, dataset):
        """
        Args:
            dataset: The underlying dataset with metadata
        """
        self.dataset = dataset
        self._logged_first = False
        
        # Check if dataset has metadata
        if hasattr(dataset, 'metadata') and dataset.metadata:
            log.info(f"DatasetWithExpertLabels: Dataset has {len(dataset.metadata)} metadata entries")
            if len(dataset.metadata) > 0:
                log.info(f"  First metadata entry: {dataset.metadata[0]}")
        else:
            log.warning("DatasetWithExpertLabels: Dataset has no metadata - will use fallback labels")
    
    def __getitem__(self, idx):
        """Get item and inject expert label."""
        # Get original item from dataset
        item = self.dataset[idx]
        
        # Log first item for debugging
        if not self._logged_first:
            self._logged_first = True
            log.info(f"DatasetWithExpertLabels.__getitem__({idx}): First item")
            log.info(f"  Item type: {type(item)}")
            log.info(f"  Item keys: {list(item.keys()) if isinstance(item, dict) else 'not a dict'}")
            if isinstance(item, dict) and 'metadata' in item:
                log.info(f"  Has metadata: {item['metadata']}")
        
        # Inject expert label
        if isinstance(item, dict):
            # Get source name from metadata
            source_name = 'general'  # default
            if 'metadata' in item and isinstance(item['metadata'], dict):
                source_name = item['metadata'].get('source_name', 'general')
            
            # Convert to expert label tensor
            expert_label = get_expert_label_tensor(source_name)
            
            # Add to item
            item['expert_label'] = expert_label
            item['domain_label'] = source_name
            
            if not self._logged_first:
                log.info(f"  Injected expert_label for source: {source_name}")
                log.info(f"  Expert label: {expert_label}")
        
        return item
    
    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying dataset."""
        return getattr(self.dataset, name)

