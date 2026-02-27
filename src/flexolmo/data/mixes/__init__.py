import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple

from olmo_core.data import NumpyDatasetConfig
from olmo_core.data.mixes import DataMixBase
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType

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

    # jacob's new data
    olmo3_code = "olmo3_code"
    olmo3_math = "olmo3_math"
    long_context = "long_context"
    olmo3_long_context_debug = "olmo3_long_context_debug"
    olmo3_reasoning = "olmo3_reasoning"
    kevin_med_data = "kevin_med_data"
    
    # glob-based mixtures
    code_glob = "code_glob"
    math_glob = "math_glob"
    code_math_glob = "code_math_glob"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        """
        Construct the data mix.

        :param base_dir: Where the mix is stored, e.g. "s3://ai2-llm" or "/weka/oe-training-default/ai2-llm".
        :param tokenizer: The tokenizer identifier.

        :returns: A list of paths/URLs to the tokenized numpy data files in the mix and list
            of corresponding labels.
        """
        # Glob-based mixtures should be handled via source_mixture_config in common.py,
        # not through this build() method. If we get here with a glob mix, it's an error.
        if self.value.endswith("_glob"):
            raise ValueError(
                f"Glob-based mixture '{self.value}' should be handled via source_mixture_config, "
                "not through CustomDataMix.build(). This is likely a bug in the config setup."
            )

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
                    label, path = line.split(",")
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
