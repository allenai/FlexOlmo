"""
Train a 4x7B OLMo2 model with supervised router training using eval benchmark oracle labels.

This is a variant of OLMoE-4x7B-supervised-router.py that uses eval_benchmark_mix
instead of router_training_mix, for training on eval benchmark data with oracle labels.

See OLMoE-4x7B-supervised-router.py for full documentation.
This script only differs in the build_dataset_config function.
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

from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
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
from flexolmo.train.train_module.supervised_router import (
    SupervisedRouterTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    """Build model config for 4x7B MoE with router training."""
    return TransformerConfig.olmoe_nx7b(  # type: ignore
        vocab_size=common.tokenizer.padded_vocab_size(),
        num_experts=4,
        top_k=4,  # Use all 4 experts
        lb_loss_weight=0,  # Disable load balancing loss (using supervised loss instead)
        z_loss_weight=0.001,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
            "blocks.*.feed_forward_moe.experts*",  # Freeze experts, only train router
        ],
    )


def build_train_module_config(common: CommonComponents) -> SupervisedRouterTrainModuleConfig:
    """Build training module config with supervised router training."""
    return SupervisedRouterTrainModuleConfig(
        rank_microbatch_size=1 * 4096,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=2e-3,  # Higher LR for router training
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        ),
        compile_model=False,  # Disabled temporarily: forward hooks don't work well with torch.compile
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
            num_replicas=2,  # Set to number of GPUs / num_experts (8 GPUs / 4 experts = 2)
        ),
        # NOTE: expert parallelism requires either HSDP or tensor parallelism.
        ep_config=TransformerExpertParallelConfig(degree=4),
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
        scheduler=CosWithWarmup(warmup_steps=100),
        # Supervised router training specific configs
        router_loss_weight=1.0,  # Weight for supervised router loss
        router_loss_only=False,  # If True, only train router (set LM loss to 0)
    )


def build_dataset_config(common: CommonComponents) -> NumpyDatasetConfig:
    """Build dataset config using eval_benchmark_mix files directly (no source mixture).
    
    For source-based labeling on the small eval_benchmark dataset, we bypass the
    SourceMixtureConfig system and directly specify files with metadata. This avoids
    the path replication issue where source mixture expands 10 files to hundreds of
    path entries, causing metadata length mismatch.
    """
    dataset_config = common.dataset
    
    # Base directory for eval benchmark data
    base_dir = "/weka/oe-training-default/sanjaya/eval_benchmark_data"
    
    # Define the 10 eval benchmark files with their domain labels
    # This matches eval_benchmark_mix.txt
    domain_files = [
        ("mj_finemath_gsm8k", "mj_finemath_gsm8k/part-00-00000.npy"),
        ("mj_finemath_minerva", "mj_finemath_minerva/part-00-00000.npy"),
        ("starcoder_humaneval", "starcoder_humaneval/part-00-00000.npy"),
        ("starcoder_mbpp", "starcoder_mbpp/part-00-00000.npy"),
        ("mmlu", "mmlu/part-00-00000.npy"),
        ("bbh", "bbh/part-00-00000.npy"),
        ("popqa", "popqa/part-00-00000.npy"),
        ("simpleqa", "simpleqa/part-00-00000.npy"),
        ("gpqa", "gpqa/part-00-00000.npy"),
        ("ifeval", "ifeval/part-00-00000.npy"),
    ]
    
    # Build paths and metadata lists
    paths = []
    metadata = []
    for domain_label, rel_path in domain_files:
        full_path = f"{base_dir}/{rel_path}"
        paths.append(full_path)
        metadata.append({"source_name": domain_label})
    
    # Set paths and metadata directly on the dataset config
    dataset_config.paths = paths
    dataset_config.metadata = metadata
    dataset_config.include_instance_metadata = True
    
    # Clear any mix settings (we're using direct paths)
    dataset_config.mix = None
    dataset_config.source_mixture_config = None
    
    log.info(f"Created direct NumpyDatasetConfig with {len(paths)} files")
    log.info(f"Domains: {[d for d, _ in domain_files]}")
    
    return dataset_config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build trainer config for eval benchmark dataset."""
    trainer_config = common.trainer
    # Default to 10M tokens for small eval benchmark dataset
    # Can be overridden via command line: --trainer.max_duration.value=<value>
    # The launch script passes --trainer.max_duration.value=${MAX_TOKENS} which will override this
    trainer_config.max_duration.value = 10_000_000  # 10M tokens default (can be overridden)
    trainer_config.max_duration.unit = DurationUnit("tokens")
    return trainer_config


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} [dry_run] run_name [OVERRIDES...]")
        print("\nExample (eval benchmark with oracle labels):")
        print(f"  torchrun --nproc-per-node=8 {sys.argv[0]} FlexOlmo-4x7B-Supervised-RT-EvalOracle \\")
        print("    --data_loader.expert_labels_dir=/path/to/per_token_labels \\")
        print("    --data_loader.labeled_indices_file=/path/to/labeled_indices.npy \\")
        print("    --dataset.mix_base_dir=/path/to/eval_benchmark_data \\")
        print("    --trainer.load_path=/path/to/checkpoint \\")
        print("    --train_module.router_loss_only=true \\")
        print("    --trainer.save_folder=/path/to/save")
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
            global_batch_size=8 * SEQUENCE_LENGTH,  # 32768 tokens - works with 8 GPUs (1 node)
            include_default_evals=False,  # Disable evals that expect text-based datasets
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

