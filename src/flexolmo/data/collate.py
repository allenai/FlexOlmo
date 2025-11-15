"""
Custom collate function that preserves metadata for supervised router training.
"""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data._utils.collate import default_collate


def collate_with_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that preserves metadata fields.
    
    This handles the 'metadata' field specially by keeping it as a list
    rather than trying to stack/concatenate it like tensors.
    
    Args:
        batch: List of dictionaries, each containing 'input_ids' and optionally 'metadata'
    
    Returns:
        Batched dictionary with metadata preserved as a list
    """
    # Separate metadata from other fields
    metadata_list = []
    batch_without_metadata = []
    
    for item in batch:
        if isinstance(item, dict):
            # Extract metadata if present
            if 'metadata' in item:
                metadata_list.append(item['metadata'])
            else:
                metadata_list.append(None)
            
            # Create copy without metadata for default collation
            item_copy = {k: v for k, v in item.items() if k != 'metadata'}
            batch_without_metadata.append(item_copy)
        else:
            batch_without_metadata.append(item)
            metadata_list.append(None)
    
    # Use default collation for tensors/arrays
    collated = default_collate(batch_without_metadata)
    
    # Add metadata back as a list (not collated)
    if any(m is not None for m in metadata_list):
        collated['metadata'] = metadata_list
    
    return collated


def collate_with_expert_labels(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that also injects expert labels from metadata.
    
    This is the recommended approach: extract expert labels during collation
    rather than wrapping the data loader.
    
    Args:
        batch: List of dictionaries with 'input_ids' and 'metadata'
    
    Returns:
        Batched dictionary with 'metadata' and 'expert_labels' fields
    """
    from flexolmo.data.expert_label_utils import get_expert_label_tensor
    
    # First, preserve metadata
    collated = collate_with_metadata(batch)
    
    # Extract expert labels from metadata
    if 'metadata' in collated:
        expert_labels_list = []
        domain_labels_list = []
        
        for metadata in collated['metadata']:
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
        
        # Stack expert labels into tensor
        if expert_labels_list:
            collated['expert_labels'] = torch.stack(expert_labels_list, dim=0)
            collated['domain_labels'] = domain_labels_list
    
    return collated

