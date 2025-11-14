"""
Utility to build dataset config with source_name metadata for supervised router training.

This module automatically extracts source_name from SourceMixtureDataset and adds it
as metadata so it can be accessed in batches via batch["metadata"].

Key insight: We iterate through mixture.sources and path_tokens in the SAME ORDER as
to_paths() to ensure metadata[i] corresponds to paths[i].
"""

import logging
from typing import List, Dict, Any, Optional

from olmo_core.data import NumpyDatasetConfig
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig

log = logging.getLogger(__name__)


def add_source_name_metadata(
    dataset_config: NumpyDatasetConfig,
    source_mixture_config: Optional[SourceMixtureDatasetConfig] = None,
) -> NumpyDatasetConfig:
    """
    Add source_name metadata to dataset config so it appears in batches.
    
    This function creates metadata directly from source_mixture_config.source_configs
    WITHOUT building the full mixture (which would count tokens for all files and be slow).
    
    The metadata order matches the order that to_paths() returns: iterate through
    source_configs, then through paths within each source.
    
    Args:
        dataset_config: The dataset config to modify
        source_mixture_config: Optional source mixture config (if None, uses dataset_config.source_mixture_config)
    
    Returns:
        Modified dataset_config with metadata containing source_name
    """
    if source_mixture_config is None:
        source_mixture_config = dataset_config.source_mixture_config
    
    if source_mixture_config is None:
        log.warning("No source_mixture_config found, cannot add source_name metadata")
        return dataset_config
    
    # Create metadata directly from source_configs (fast, no token counting)
    # This matches the order that SourceMixtureDataset.to_paths() returns:
    # chain.from_iterable([outcome.path_tokens for outcome in self.sources])
    metadata: List[Dict[str, Any]] = []
    
    # Iterate through source_configs in order, then through paths within each source
    # This matches the order that the built mixture would return
    for source_config in source_mixture_config.source_configs:
        source_name = source_config.source_name
        # Create one metadata entry per path in this source
        for _ in source_config.paths:
            metadata.append({"source_name": source_name})
    
    # Update dataset config
    dataset_config.metadata = metadata
    dataset_config.include_instance_metadata = True
    
    # Get unique sources for logging
    unique_sources = set(source_config.source_name for source_config in source_mixture_config.source_configs)
    total_paths = sum(len(source_config.paths) for source_config in source_mixture_config.source_configs)
    
    log.info(f"Added source_name metadata for {total_paths} paths (fast path, no token counting)")
    log.info(f"Unique sources ({len(unique_sources)}): {sorted(unique_sources)}")
    
    # Warn that dataset build will be slow
    if total_paths > 1000:
        log.info(
            f"⚠️  Note: Dataset build will gather instance indices for {total_paths} files. "
            f"This may take several minutes but is a one-time cost. "
            f"Metadata creation completed quickly (no token counting performed)."
        )
    
    # Log some examples for debugging
    if len(metadata) > 0:
        log.debug(f"First 5 metadata entries: {metadata[:5]}")
        if len(source_mixture_config.source_configs) > 0:
            first_source = source_mixture_config.source_configs[0]
            log.debug(f"First 5 paths from first source: {first_source.paths[:5]}")
    
    return dataset_config

