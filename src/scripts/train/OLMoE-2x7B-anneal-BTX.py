"""
Train an OLMo2-7B model with only FFN parameters trainable (BTX). Run this script without any arguments to see usage info.
"""

import logging
import os
import sys

from olmo_core.config import DType
from olmo_core.data import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from rich import print

from flexolmo.internal.common import (
    CommonComponents,
    build_experiment_config,
    get_root_dir,
    is_dry_run,
    print_model_params,
)
from flexolmo.internal.model_utils import *  # noqa
from flexolmo.internal.train_utils import anneal

SEQUENCE_LENGTH = 4096
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_7B(  # type: ignore
        vocab_size=common.tokenizer.padded_vocab_size(),
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
            # Keep dense FFN weights trainable (not MoE experts)
        ],
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    train_module_config = common.train_module
    train_module_config.rank_microbatch_size = 2 * SEQUENCE_LENGTH
    train_module_config.scheduler = CosWithWarmup(warmup_steps=2000)
    train_module_config.state_dict_save_opts = {"flatten_optimizer_state_dict": True}

    assert isinstance(train_module_config.optim, AdamWConfig)
    train_module_config.optim.lr = 9e-4
    return train_module_config


def build_dataset_config(common: CommonComponents) -> NumpyDatasetConfig:
    from flexolmo.data.mixes import CustomDataMix

    dataset_config = common.dataset
    dataset_config.mix = CustomDataMix.public_mix
    return dataset_config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    trainer_config = common.trainer
    # Add any changes to the trainer configuration here
    return trainer_config


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 2:
        print(
            f"Usage: torchrun [OPTS..] {sys.argv[0]} [dry_run] run_name checkpoint [OVERRIDES...]"
        )
        sys.exit(1)

    dry_run = is_dry_run(sys.argv)

    if dry_run:
        _, run_name, checkpoint, *overrides = sys.argv[1:]
    else:
        run_name, checkpoint, *overrides = sys.argv[1:]
        prepare_training_environment()

    try:
        config = build_experiment_config(
            run_name,
            overrides,
            root_dir=get_root_dir(),
            sequence_length=SEQUENCE_LENGTH,
            global_batch_size=1024 * SEQUENCE_LENGTH,
            include_default_evals=True,
            freeze_embeddings=False,
            model_config_builder=build_model_config,
            dataset_config_builder=build_dataset_config,
            trainer_config_builder=build_trainer_config,
            train_module_config_builder=build_train_module_config,
        )
        print(config)
        print_model_params(config)
        if dry_run:
            sys.exit(0)  # Exit early for dry run
        anneal(checkpoint, config)
    finally:
        teardown_training_environment()
