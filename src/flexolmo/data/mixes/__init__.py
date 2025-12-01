import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np

from olmo_core.data import NumpyDatasetConfig
from olmo_core.data.mixes import DataMixBase
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType

log = logging.getLogger(__name__)

__all__ = ["CustomDataMix"]


class CustomDataMix(DataMixBase):
    """
    An enumeration of data mix names.
    """

    public_mix = "public_mix"

    # domain data
    creative_writing = "creative_writing"
    mj_finemath4plus = "mj_finemath4plus"
    starcoder = "starcoder"
    news = "news"
    pes2o = "pes2o"
    flan = "flan"
    reddit_v0 = "reddit_v0"

    # proxy mixes
    proxy_code = "proxy_code"
    proxy_math = "proxy_math"
    proxy_creative = "proxy_creative"
    proxy_news = "proxy_news"
    proxy_pes2o = "proxy_pes2o"
    proxy_reddit = "proxy_reddit"
    proxy_combined_public_math_code_news = "proxy_combined_public_math_code_news"

    # olmo2 setup
    dolmino_minus_math = "dolmino_minus_math"
    dolmino_minus_math_flan = "dolmino_minus_math_flan"

    # test mixes
    test_mix = "test_mix"
    anneal_test_mix = "anneal_test_mix"
    
    # router training mix
    router_training_mix = "router_training_mix"
    router_training_mix_midtraining = "router_training_mix_midtraining"
    
    # 2x7B router training mixes
    math_general_rt_mix = "math_general_rt_mix"
    code_general_rt_mix = "code_general_rt_mix"
    
    # 4x7B supervised router training mix (with one-hot labels)
    router_training_mix_labeled = "router_training_mix_labeled"
    
    # Evaluation benchmark mix (for debugging - training on test data!)
    eval_benchmark_mix = "eval_benchmark_mix"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        """
        Construct the data mix.

        :param base_dir: Where the mix is stored, e.g. "s3://ai2-llm" or "/weka/oe-training-default/ai2-llm".
        :param tokenizer: The tokenizer identifier.

        :returns: A list of paths/URLs to the tokenized numpy data files in the mix and list
            of corresponding labels.
        """
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        tokenizer_id: str = tokenizer

        paths = []
        labels = []
        with _get_data_mix_path(self) as mix_path:
            with mix_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Handle two formats:
                    # 1. Standard: "label,path/to/file.npy"
                    # 2. Labeled: "label,one_hot_label,path/to/file.npy" (e.g., "mj_finemath4plus,1,0,0,0,path/to/file.npy")
                    parts = line.split(",")
                    if len(parts) == 2:
                        # Standard format
                        label, path = parts
                        path = path.replace("{TOKENIZER}", tokenizer_id)
                        paths.append(f"{base_dir}{path}")
                        labels.append(label)
                    elif len(parts) >= 6:
                        # Labeled format: label,one_hot_0,one_hot_1,one_hot_2,one_hot_3,path/to/file.npy
                        # Parse one-hot label (4 values: expert 0, 1, 2, 3)
                        domain_label = parts[0]
                        one_hot_label = ",".join(parts[1:5])  # "1,0,0,0"
                        path = ",".join(parts[5:])  # Handle paths with commas
                        path = path.replace("{TOKENIZER}", tokenizer_id)
                        paths.append(f"{base_dir}{path}")
                        # Store as "domain_label|one_hot_label" for later parsing
                        labels.append(f"{domain_label}|{one_hot_label}")
                    else:
                        # Try to handle as standard format (fallback)
                        label, path = parts[0], ",".join(parts[1:])
                        path = path.replace("{TOKENIZER}", tokenizer_id)
                        paths.append(f"{base_dir}{path}")
                        labels.append(label)
        return paths, labels


def get_mixture_dataset_config(
    prev_dataset_config: NumpyDatasetConfig,
) -> SourceMixtureDatasetConfig:
    """
    Example usage:

    dataset_config.mix = "public_mix,mj_finemath4plus,pes2o,starcoder"
    dataset_config.mix_base_dir = "/weka/oe-training-default/ai2-llm/"

    dataset_config.source_mixture_config = get_mixture_dataset_config(dataset_config)
    dataset_config.mix = None
    """

    assert prev_dataset_config.mix is not None
    names = prev_dataset_config.mix.split(",")
    base_dir = prev_dataset_config.mix_base_dir

    assert base_dir is not None

    if not base_dir.endswith("/"):
        base_dir = base_dir + "/"

    source_configs: List[SourceMixtureConfig] = []

    for name in names:
        with _get_data_mix_path(name) as mix_path:
            paths = []
            with mix_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    label, path = line.split(",")
                    # This is not needed
                    # path = path.replace("{TOKENIZER}", tokenizer_id)
                    paths.append(f"{base_dir}{path}")

            source_configs.append(
                SourceMixtureConfig(
                    source_name=name,
                    paths=paths,
                    max_repetition_ratio=3,  # needed for smaller datasets like mj_finemath
                    target_ratio=1.0 / len(names),
                )
            )

    # sewonm: here, max_tokens is needed to see how much data to prepare; so, it doesn't have to be precise
    # but it should be large enough to cover the actual duration, and small enough to be efficient.
    # right now, we will set it to 400B since we are likely to use 50B x 8 tokens,
    # but this is hard-coded and should be modified in the future
    #
    # seed and processes not to be hard-coded in the future as well
    assert prev_dataset_config.sequence_length is not None
    return SourceMixtureDatasetConfig(
        source_configs=source_configs,
        max_tokens=5_000_000_000,
        sequence_length=prev_dataset_config.sequence_length,
        seed=2025,
        dtype=NumpyDatasetDType(prev_dataset_config.get_dtype().__name__),
        processes=8,
    )


def _validate_numpy_file(file_path: str, dtype_name: str) -> bool:
    """Validate that a numpy file can be memory-mapped with the expected dtype.
    
    Returns True if file is valid, False if corrupted or inaccessible.
    """
    try:
        # Map dtype name to numpy dtype
        dtype_map = {
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int32": np.int32,
        }
        dtype = dtype_map.get(dtype_name, np.uint16)
        
        # Try to memory-map the file
        # If the file size is not a multiple of dtype size, this will raise ValueError
        mmap = np.memmap(file_path, mode="r", dtype=dtype)
        del mmap  # Close the memmap
        return True
    except (ValueError, OSError, FileNotFoundError) as e:
        log.warning(f"Skipping corrupted/inaccessible file: {file_path} ({e})")
        return False


def get_mixture_dataset_config_by_domain(
    prev_dataset_config: NumpyDatasetConfig,
    validate_files: bool = False,  # Disabled by default to match regular mix handling behavior
) -> SourceMixtureDatasetConfig:
    """
    Create SourceMixtureDatasetConfig where each domain label gets its own source.
    
    This is useful for supervised router training where we need to track which domain
    (starcoder, mj_finemath4plus, etc.) each sequence came from.
    
    Args:
        prev_dataset_config: The dataset config containing mix and mix_base_dir
        validate_files: If True, validate numpy files before including them (default: False).
                       This helps catch corrupted files early, but adds overhead.
                       Set to True if you encounter "Size of available data is not a multiple" errors.
    
    Example usage:
        dataset_config.mix = "router_training_mix"
        dataset_config.mix_base_dir = "/weka/oe-training-default/ai2-llm/"
        dataset_config.source_mixture_config = get_mixture_dataset_config_by_domain(dataset_config)
        dataset_config.mix = None
    """
    assert prev_dataset_config.mix is not None
    mix_name = prev_dataset_config.mix.split(",")[0]  # Take first mix name
    base_dir = prev_dataset_config.mix_base_dir

    assert base_dir is not None

    if not base_dir.endswith("/"):
        base_dir = base_dir + "/"

    # Group paths by domain label
    domain_to_paths: dict[str, list[str]] = {}
    total_files = 0
    skipped_files = 0

    with _get_data_mix_path(mix_name) as mix_path:
        with mix_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Parse: "domain_label,path/to/file.npy"
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                
                domain_label = parts[0]
                path = parts[1]
                full_path = f"{base_dir}{path}"
                total_files += 1
                
                # Validate file if requested (helps catch corrupted files early)
                if validate_files:
                    dtype_name = prev_dataset_config.get_dtype().__name__
                    if not _validate_numpy_file(full_path, dtype_name):
                        skipped_files += 1
                        continue  # Skip corrupted/inaccessible files
                
                if domain_label not in domain_to_paths:
                    domain_to_paths[domain_label] = []
                domain_to_paths[domain_label].append(full_path)
    
    if validate_files and skipped_files > 0:
        log.warning(
            f"File validation: Skipped {skipped_files}/{total_files} corrupted/inaccessible files. "
            f"Proceeding with {total_files - skipped_files} valid files."
        )
    elif validate_files:
        log.info(f"File validation: All {total_files} files are valid.")

    # Create a SourceMixtureConfig for each domain
    source_configs: List[SourceMixtureConfig] = []
    num_domains = len(domain_to_paths)
    
    for domain_label, paths in domain_to_paths.items():
        source_configs.append(
            SourceMixtureConfig(
                source_name=domain_label,  # Use domain label as source name!
                paths=paths,
                max_repetition_ratio=3,
                target_ratio=1.0 / num_domains,
            )
        )

    assert prev_dataset_config.sequence_length is not None
    return SourceMixtureDatasetConfig(
        source_configs=source_configs,
        max_tokens=5_000_000_000,
        sequence_length=prev_dataset_config.sequence_length,
        seed=2025,
        dtype=NumpyDatasetDType(prev_dataset_config.get_dtype().__name__),
        processes=8,
    )


@contextmanager
def _get_data_mix_path(name: str) -> Generator[Path, None, None]:
    import importlib_resources

    try:
        with importlib_resources.as_file(
            importlib_resources.files("flexolmo").joinpath(
                f"data/mixes/{os.path.basename(name)}.txt"
            )
        ) as path:
            yield path
    finally:
        pass
