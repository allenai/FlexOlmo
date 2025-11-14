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
        
        # Try to get dataset from data_loader if not provided
        if self.dataset is None and hasattr(data_loader, 'dataset'):
            self.dataset = data_loader.dataset
        
        # Build source mapping from dataset if available (for fallback)
        self._source_mapping = None
        self._build_source_mapping()
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches and inject expert labels."""
        log.info("ExpertLabelDataLoaderWrapper: Starting iteration, will inject expert labels into batches")
        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx == 0:
                log.info(f"ExpertLabelDataLoaderWrapper: Processing first batch. Batch keys before injection: {list(batch.keys())}")
            batch = self._inject_expert_labels(batch)
            if batch_idx == 0:
                log.info(f"ExpertLabelDataLoaderWrapper: After injection. Batch keys: {list(batch.keys())}, Has expert_labels: {'expert_labels' in batch}")
            yield batch
    
    def __len__(self) -> int:
        """Return length of underlying data loader."""
        return len(self.data_loader)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying data loader."""
        return getattr(self.data_loader, name)
    
    def _build_source_mapping(self):
        """Build a mapping from instance indices to source names if dataset is available."""
        if self.dataset is None:
            return
        
        try:
            # Try to extract source information from SourceMixtureDataset
            if hasattr(self.dataset, 'sources'):
                # This is a SourceMixtureDataset - we can map indices to sources
                source_mapping = {}
                current_idx = 0
                for source in self.dataset.sources:
                    source_name = getattr(source, 'source_name', None)
                    if source_name is None and hasattr(source, 'source_config'):
                        source_name = getattr(source.source_config, 'source_name', None)
                    
                    if source_name:
                        # Get the number of instances in this source
                        if hasattr(source, 'path_tokens'):
                            num_instances = len(source.path_tokens)
                        elif hasattr(source, '__len__'):
                            num_instances = len(source)
                        else:
                            num_instances = 0
                        
                        # Map all indices for this source
                        for i in range(num_instances):
                            source_mapping[current_idx + i] = source_name
                        current_idx += num_instances
                
                if source_mapping:
                    self._source_mapping = source_mapping
                    log.debug(f"Built source mapping for {len(source_mapping)} instances")
        except Exception as e:
            log.debug(f"Could not build source mapping from dataset: {e}")
            self._source_mapping = None
    
    def _inject_expert_labels(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject expert labels into batch based on available metadata.
        
        This tries multiple methods to extract domain/source information:
        1. batch["domain_labels"] - direct domain labels
        2. batch["source_name"] - source name from dataset
        3. batch metadata fields
        """
        batch_size = batch["input_ids"].shape[0]
        
        # Log first batch for debugging
        if not hasattr(self, '_first_batch_logged'):
            self._first_batch_logged = True
            log.info(f"ExpertLabelDataLoaderWrapper._inject_expert_labels: Processing batch with {batch_size} samples")
            log.info(f"  Batch keys: {list(batch.keys())}")
            log.info(f"  Has metadata: {'metadata' in batch}")
            log.info(f"  Has dataset: {self.dataset is not None}")
            log.info(f"  Has source_mapping: {self._source_mapping is not None}")
            if self._source_mapping:
                log.info(f"  Source mapping size: {len(self._source_mapping)}")
        
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
        
        # Method 6: Extract from dataset source mapping using instance indices
        if domain_labels is None and self._source_mapping is not None:
            # Try to get instance indices from batch
            if "instance_indices" in batch:
                instance_indices = batch["instance_indices"]
                if isinstance(instance_indices, torch.Tensor):
                    instance_indices = instance_indices.cpu().tolist()
                elif isinstance(instance_indices, (list, tuple)):
                    pass
                else:
                    instance_indices = None
                
                if instance_indices and len(instance_indices) == batch_size:
                    domain_labels = []
                    for idx in instance_indices:
                        source_name = self._source_mapping.get(idx, "general")
                        domain_labels.append(source_name)
                    if domain_labels:
                        log.debug(f"Extracted domain labels from dataset source mapping for {len(domain_labels)} instances")
        
        # Method 7: If we have dataset and can query it directly
        if domain_labels is None and self.dataset is not None:
            # Try to get source from dataset metadata if available
            if hasattr(self.dataset, 'metadata') and self.dataset.metadata:
                # If we can get instance indices, use them to look up metadata
                if "instance_indices" in batch:
                    instance_indices = batch["instance_indices"]
                    if isinstance(instance_indices, torch.Tensor):
                        instance_indices = instance_indices.cpu().tolist()
                    
                    if instance_indices and len(instance_indices) == batch_size:
                        try:
                            domain_labels = []
                            for idx in instance_indices:
                                if idx < len(self.dataset.metadata):
                                    meta = self.dataset.metadata[idx]
                                    if isinstance(meta, dict) and "source_name" in meta:
                                        domain_labels.append(meta["source_name"])
                                    else:
                                        domain_labels.append("general")
                                else:
                                    domain_labels.append("general")
                            if domain_labels:
                                log.debug(f"Extracted domain labels from dataset.metadata for {len(domain_labels)} instances")
                        except Exception as e:
                            log.debug(f"Error extracting from dataset.metadata: {e}")
        
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
            # Always set expert_labels (even if use_domain_labels is False) to ensure batches are consistent
            if self.use_domain_labels:
                # Log warning (not just debug) since this indicates a potential issue
                log.warning(
                    f"⚠️ Could not extract domain labels from batch, using default general expert labels. "
                    f"Batch keys: {list(batch.keys())}, "
                    f"Has metadata: {'metadata' in batch}, "
                    f"Metadata type: {type(batch.get('metadata'))}, "
                    f"Has dataset: {self.dataset is not None}, "
                    f"Has source_mapping: {self._source_mapping is not None}"
                )
                # If we have instance_indices, log them for debugging
                if "instance_indices" in batch:
                    instance_indices = batch["instance_indices"]
                    if isinstance(instance_indices, torch.Tensor):
                        instance_indices = instance_indices.cpu().tolist()[:5]  # First 5 for logging
                    log.warning(f"  Instance indices (first 5): {instance_indices}")
            else:
                log.debug(
                    f"use_domain_labels=False, using default general expert labels. "
                    f"Batch keys: {list(batch.keys())}"
                )
            # Always inject expert_labels to ensure consistency
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

