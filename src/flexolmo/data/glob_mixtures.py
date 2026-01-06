"""
Glob-based data mixtures for training.

This module defines data mixtures using glob patterns that can be used
with olmo-core's SourceMixtureConfig.
"""

import logging
import re
from typing import Dict, List
from google.cloud import storage
from olmo_core.data.source_mixture import SourceMixtureConfig

log = logging.getLogger(__name__)


def expand_gcs_glob(pattern: str) -> List[str]:
    """
    Expand a GCS glob pattern into a list of matching file paths.

    Supports * and ** wildcards:
    - * matches any characters except /
    - ** matches any characters including /

    Args:
        pattern: A GCS path with glob wildcards, e.g. "gs://bucket/path/**/*.npy"

    Returns:
        List of matching GCS paths
    """
    if not pattern.startswith("gs://"):
        raise ValueError(f"Pattern must start with gs://, got: {pattern}")

    # Parse bucket and prefix
    path_without_scheme = pattern[5:]  # Remove "gs://"
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    blob_pattern = parts[1] if len(parts) > 1 else ""

    # Find the prefix before any wildcards
    prefix_parts = []
    pattern_parts = blob_pattern.split("/")
    for part in pattern_parts:
        if "*" in part:
            break
        prefix_parts.append(part)
    prefix = "/".join(prefix_parts)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Convert glob pattern to regex
    # Escape special regex chars, then convert glob wildcards
    regex_pattern = re.escape(blob_pattern)
    regex_pattern = regex_pattern.replace(r"\*\*", ".*")  # ** matches anything
    regex_pattern = regex_pattern.replace(r"\*", "[^/]*")  # * matches non-slash
    regex_pattern = f"^{regex_pattern}$"
    pattern_re = re.compile(regex_pattern)

    # List blobs and filter
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    matching_paths = []
    log.info(f"Expanding glob pattern: {pattern}")
    log.info(f"  Bucket: {bucket_name}, Prefix: {prefix}")

    for blob in bucket.list_blobs(prefix=prefix):
        if pattern_re.match(blob.name):
            matching_paths.append(f"gs://{bucket_name}/{blob.name}")

    log.info(f"  Found {len(matching_paths)} matching files")

    if not matching_paths:
        raise FileNotFoundError(f"Glob pattern '{pattern}' did not match any files")

    return sorted(matching_paths)


def expand_paths_in_config(config: SourceMixtureConfig) -> SourceMixtureConfig:
    """
    Expand any glob patterns in a SourceMixtureConfig's paths.

    Returns a new SourceMixtureConfig with expanded paths.
    """
    expanded_paths = []
    for path in config.paths:
        if "*" in path:
            expanded_paths.extend(expand_gcs_glob(path))
        else:
            expanded_paths.append(path)

    return SourceMixtureConfig(
        source_name=config.source_name,
        target_ratio=config.target_ratio,
        paths=expanded_paths,
        max_repetition_ratio=config.max_repetition_ratio,
        max_source_fraction=config.max_source_fraction,
    )


def get_code_mixture() -> List[SourceMixtureConfig]:
    """Combined code mixture with FIM and Swallowcode."""
    return [
        SourceMixtureConfig(
            source_name="combined_code",
            target_ratio=1.0,
            paths=[
                # Code FIM paths
                "gs://ai2-llm/preprocessed/stack-edu/sample-fim-weighted-pl-edu-score-decon/**/**/*.npy",
                # Swallowcode paths
                "gs://ai2-llm/preprocessed/tokyotech-llm/swallowcode/scor_final_data-decon-sparkle-motion-with-ids-modelnamefilter2/allenai/dolma2-tokenizer/*.npy",
            ],
            max_repetition_ratio=3,
        ),
    ]


def get_math_mixture() -> List[SourceMixtureConfig]:
    """Math mixture with Megamatt and Dolminos2math."""
    return [
        SourceMixtureConfig(
            source_name="megamatt",
            target_ratio=0.01726315213,
            paths=[
                # 3,883,674,937 tokens
                "gs://ai2-llm/preprocessed/megamath_web_pro_max/beaker_rewrites-decon-sparkle-motion-modelnamefilter2/**/allenai/dolma2-tokenizer/*.npy",
            ],
            max_repetition_ratio=3,
        ),
        SourceMixtureConfig(
            source_name="dolminos2math",
            target_ratio=0.18273684787,
            paths=[
                # 5,624,449,531 tokens
                "gs://ai2-llm/preprocessed/tokyotech-llm/swallowmath/beaker_outputs-decon-sparkle-motion-withids-modelnamefilter2/allenai/dolma2-tokenizer/*.npy",
                # 10,687,987,907 tokens
                "gs://ai2-llm/preprocessed/midtraining-reasoning/flat_dolmino_math-decon-sparkle-motion-SCHEMAFIX-modelnamefilter2/allenai/dolma2-tokenizer/*.npy",
                # 850,848,999 tokens
                "gs://ai2-llm/preprocessed/midtraining-reasoning/OpenMathReasoning/OpenMathReasoning-rewrite-full-thoughts/jsonls-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
                # 898,733,958 tokens
                "gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/MIND/data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
                # 240,590,380 tokens
                "gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
            ],
            max_repetition_ratio=3,
        ),
    ]


def get_code_math_mixture() -> List[SourceMixtureConfig]:
    """Combined code and math mixture."""
    mixtures = []

    # Add code with 50% ratio
    code_mix = get_code_mixture()[0]
    code_mix.target_ratio = 0.5
    mixtures.append(code_mix)

    # Add math sources with adjusted ratios (total 50%)
    math_mixtures = get_math_mixture()
    # Scale math ratios to sum to 0.5
    total_math_ratio = sum(m.target_ratio for m in math_mixtures)
    scale_factor = 0.5 / total_math_ratio

    for math_mix in math_mixtures:
        math_mix.target_ratio *= scale_factor
        mixtures.append(math_mix)

    return mixtures


# Raw glob patterns - these need to be expanded before use
_GLOB_MIXTURE_FACTORIES: Dict[str, callable] = {
    "code_glob": get_code_mixture,
    "math_glob": get_math_mixture,
    "code_math_glob": get_code_math_mixture,
}


def get_glob_mixture(name: str) -> List[SourceMixtureConfig]:
    """
    Get a predefined glob mixture by name, with globs expanded to actual file paths.

    Args:
        name: The name of the mixture (e.g., "code_glob", "math_glob")

    Returns:
        List of SourceMixtureConfig objects with expanded paths

    Raises:
        ValueError: If the mixture name is not found
    """
    if name not in _GLOB_MIXTURE_FACTORIES:
        raise ValueError(
            f"Unknown glob mixture: {name}. "
            f"Available mixtures: {list(_GLOB_MIXTURE_FACTORIES.keys())}"
        )

    # Get the raw configs with glob patterns
    raw_configs = _GLOB_MIXTURE_FACTORIES[name]()

    # Expand globs in each config
    expanded_configs = [expand_paths_in_config(config) for config in raw_configs]

    return expanded_configs
