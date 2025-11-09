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
    
    This function:
    1. Builds the SourceMixtureDataset to get source information
    2. Creates a mapping from paths to source names
    3. Creates metadata list with source_name for each path
    4. Sets include_instance_metadata=True
    
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
    
    # Build the mixture to get source information
    mixture = source_mixture_config.build()
    
    # Create metadata by iterating in the same order as to_paths()
    # to_paths() does: chain.from_iterable([outcome.path_tokens for outcome in self.sources])
    # So we need to match that exact order to ensure metadata[i] corresponds to paths[i]
    metadata: List[Dict[str, Any]] = []
    
    # Iterate in the same order as to_paths() - through sources, then path_tokens
    # to_paths() implementation: chain.from_iterable([outcome.path_tokens for outcome in self.sources])
    for outcome in mixture.sources:
        source_name = outcome.name
        for path_token in outcome.path_tokens:
            # Create metadata for this path_token in the same order as to_paths() returns
            metadata.append({"source_name": source_name})
    
    # Verify: paths should match the order we just created metadata
    paths = mixture.to_paths()
    if len(paths) != len(metadata):
        log.warning(
            f"Mismatch: {len(paths)} paths but {len(metadata)} metadata entries. "
            "This should not happen!"
        )
    
    # Update dataset config
    dataset_config.metadata = metadata
    dataset_config.include_instance_metadata = True
    
    # Get unique sources for logging
    unique_sources = set(outcome.name for outcome in mixture.sources)
    
    log.info(f"Added source_name metadata for {len(metadata)} paths")
    log.info(f"Unique sources ({len(unique_sources)}): {sorted(unique_sources)}")
    
    # Log some examples for debugging
    if len(metadata) > 0:
        log.debug(f"First 5 metadata entries: {metadata[:5]}")
        log.debug(f"First 5 paths: {[str(p) for p in paths[:5]]}")
    
    return dataset_config

