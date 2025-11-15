"""
Wrapper for DataCollator that injects expert labels from metadata.

This wraps the existing DataCollator to preserve its behavior while adding expert label injection.
"""

import logging
from typing import List, Dict, Any
import torch

from flexolmo.data.expert_label_utils import get_expert_label_tensor

log = logging.getLogger(__name__)


class ExpertLabelCollatorWrapper:
    """
    Wraps an existing collator to inject expert labels from metadata.
    
    This preserves all the original collator's behavior (padding, etc.) 
    while adding expert label injection.
    """
    
    def __init__(self, base_collator):
        """
        Args:
            base_collator: The original DataCollator to wrap
        """
        self.base_collator = base_collator
        self._first_batch = True
        log.info(f"ExpertLabelCollatorWrapper: Wrapping {type(base_collator).__name__}")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch and inject expert labels.
        
        Args:
            batch: List of items from dataset (each has 'input_ids' and 'metadata')
        
        Returns:
            Collated batch with expert_labels added
        """
        # First, use the base collator to handle tensors/padding
        collated = self.base_collator(batch)
        
        # Log first batch for debugging
        if self._first_batch:
            self._first_batch = False
            log.info(f"ExpertLabelCollatorWrapper: Processing first batch")
            log.info(f"  Input batch length: {len(batch)}")
            log.info(f"  Input item 0 keys: {list(batch[0].keys()) if batch else 'empty'}")
            log.info(f"  Collated batch keys (before injection): {list(collated.keys())}")
            if 'metadata' in collated:
                log.info(f"  Metadata preserved: {len(collated['metadata'])} entries")
            else:
                log.warning(f"  Metadata NOT preserved by base collator!")
        
        # Extract metadata from original batch (before collation)
        # The base collator might drop metadata, so we extract it from the original items
        metadata_list = []
        for item in batch:
            if isinstance(item, dict) and 'metadata' in item:
                metadata_list.append(item['metadata'])
            else:
                metadata_list.append(None)
        
        # If we have metadata, inject expert labels
        if any(m is not None for m in metadata_list):
            expert_labels_list = []
            domain_labels_list = []
            
            for metadata in metadata_list:
                if metadata and isinstance(metadata, dict) and 'source_name' in metadata:
                    source_name = metadata['source_name']
                    expert_label = get_expert_label_tensor(source_name)
                    expert_labels_list.append(expert_label)
                    domain_labels_list.append(source_name)
                else:
                    # Fallback to general if no metadata
                    expert_label = get_expert_label_tensor('general')
                    expert_labels_list.append(expert_label)
                    domain_labels_list.append('general')
            
            # Add expert labels and metadata to collated batch
            if expert_labels_list:
                collated['expert_labels'] = torch.stack(expert_labels_list, dim=0)
                collated['domain_labels'] = domain_labels_list
                collated['metadata'] = metadata_list  # Re-add metadata since base collator might drop it
                
                if self._first_batch:
                    log.info(f"  Injected expert_labels: shape={collated['expert_labels'].shape}")
                    log.info(f"  Domain labels (first 3): {domain_labels_list[:3]}")
                    log.info(f"  Expert assignments (first 3): {collated['expert_labels'][:3].argmax(dim=1).tolist()}")
        else:
            if self._first_batch:
                log.warning("  No metadata found in batch items - using general expert labels")
            # Fallback: use general expert for all
            batch_size = collated['input_ids'].shape[0] if 'input_ids' in collated else len(batch)
            collated['expert_labels'] = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * batch_size, dtype=torch.float32)
            collated['domain_labels'] = ['general'] * batch_size
        
        if self._first_batch:
            log.info(f"  Final batch keys: {list(collated.keys())}")
        
        return collated
    
    def __getattr__(self, name):
        """Delegate all other attributes to the base collator."""
        return getattr(self.base_collator, name)

