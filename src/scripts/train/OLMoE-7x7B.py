"""
Train an Nx7B OLMo2 model. Run this script without any arguments to see usage info.
"""

import logging
import sys

from olmo_core.config import DType
from olmo_core.data import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import (
    DurationUnit,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)

# from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.train.train_module import (  # TransformerTensorParallelConfig,
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
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
from flexolmo.internal.train_utils import train

SEQUENCE_LENGTH = 4096

log = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmoe_nx7b_with_expert_bias(  # type: ignore
        vocab_size=common.tokenizer.padded_vocab_size(),
        num_experts=7,
        top_k=7,
        lb_loss_weight=0,
        z_loss_weight=0.001,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
            "blocks.*.feed_forward_moe.experts*",  # TODO: comment if you also want to train the expert weights.
        ],
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=1 * 4096,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=6e-4,
            weight_decay=0.1,  # 0
            betas=(0.9, 0.95),
            fused=True,
            #  group_overrides=[
            #      OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            #  ], # swj check
        ),
        compile_model=True,
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=[
                "blocks.*.attention.*",
                "blocks.*.feed_forward_moe.experts.*",
            ],
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.fine_grained,
            num_replicas=8,  # TODO: set this to number of GPUs / num_experts, 32 when using 8 nodes
        ),
        # NOTE: expert parallelism requires either HSDP or tensor parallelism.
        ep_config=TransformerExpertParallelConfig(degree=7),
        # tp_config=TransformerTensorParallelConfig(degree=-1),
        float8_config=Float8Config(
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
            enabled=False,
        ),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),  # TODO: set as needed
    )


def build_dataset_config(common: CommonComponents) -> NumpyDatasetConfig:
    from flexolmo.data.mixes import CustomDataMix

    dataset_config = common.dataset
    dataset_config.mix = CustomDataMix.proxy_combined_public_math_code_news_reddit_pes2o_creative
    return dataset_config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    trainer_config = common.trainer
    # Add any changes to the trainer configuration here
    trainer_config.max_duration.value = 5_000_000_000
    trainer_config.max_duration.unit = DurationUnit("tokens")
    return trainer_config


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} [dry_run] run_name [OVERRIDES...]")
        sys.exit(1)

    dry_run = is_dry_run(sys.argv)

    if dry_run:
        _, run_name, *overrides = sys.argv[1:]
    else:
        run_name, *overrides = sys.argv[1:]
        prepare_training_environment()

    try:
        config = build_experiment_config(
            run_name,
            overrides,
            root_dir=get_root_dir(),
            sequence_length=SEQUENCE_LENGTH,
            global_batch_size=112 * SEQUENCE_LENGTH,
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
        train(config)
    finally:
        teardown_training_environment()
