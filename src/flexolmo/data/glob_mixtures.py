"""
Glob-based data mixtures for training.

This module defines data mixtures using glob patterns that can be used
with olmo-core's SourceMixtureConfig.
"""

from typing import Dict, List, Any
from olmo_core.data.source_mixture import SourceMixtureConfig


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
        # Add remaining sources to reach 1.0 target_ratio
        # SourceMixtureConfig(
        #     source_name="general_data",
        #     target_ratio=0.8,  # ~0.8 to reach 1.0 total
        #     paths=["gs://ai2-llm/preprocessed/other-data/**/*.npy"],
        #     max_repetition_ratio=3,
        # ),
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


# Dictionary mapping mixture names to their configurations
GLOB_MIXTURES: Dict[str, List[SourceMixtureConfig]] = {
    "code_glob": get_code_mixture(),
    "math_glob": get_math_mixture(),
    "code_math_glob": get_code_math_mixture(),
}


def get_glob_mixture(name: str) -> List[SourceMixtureConfig]:
    """
    Get a predefined glob mixture by name.
    
    Args:
        name: The name of the mixture (e.g., "code_glob", "math_glob")
        
    Returns:
        List of SourceMixtureConfig objects
        
    Raises:
        ValueError: If the mixture name is not found
    """
    if name not in GLOB_MIXTURES:
        raise ValueError(
            f"Unknown glob mixture: {name}. "
            f"Available mixtures: {list(GLOB_MIXTURES.keys())}"
        )
    return GLOB_MIXTURES[name]