"""
Train a 4x7B OLMo2 model with SOFT LABEL router training.

This script trains the router using soft labels derived from expert losses via knowledge
distillation, instead of hard argmin labels.

## Soft Label Approach

For each token, given expert losses L = [L_0, L_1, L_2]:
    q(e) = softmax(-β * (L_e - min(L)))

This soft distribution q is the target. The router is trained with KL divergence:
    L_router = KL(q || p_θ) = -Σ_e q(e) log p_θ(e) + const

Where p_θ(e) is the router's predicted distribution.

## Key Differences from Hard Label Training

1. Uses soft target distributions instead of one-hot labels
2. KL divergence / soft cross-entropy instead of hard cross-entropy
3. β (temperature) controls distribution sharpness:
   - β → ∞: approaches hard labels (argmin)
   - β = 1.0: moderate softness (default)
   - β → 0: uniform distribution

## Usage

torchrun --nproc-per-node=8 src/scripts/train/OLMoE-4x7B-soft-label-router.py \\
    FlexOlmo-4x7B-SoftLabel-RT \\
    --trainer.load_path=/path/to/checkpoint \\
    --train_module.expert_labels_dir=/path/to/per_token_labels_sft \\
    --train_module.soft_label_beta=1.0 \\
    --trainer.save_folder=/path/to/save

Expert mapping (3 active experts, Expert 3 is masked/duplicate):
    Expert 0 = Math (mj_finemath)
    Expert 1 = General (dolma, etc.)
    Expert 2 = Code (starcoder)
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
from flexolmo.train.train_module.soft_label_router import (
    SoftLabelRouterTrainModuleConfig,
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
        lb_loss_weight=0,  # Disable load balancing loss (using soft label loss instead)
        z_loss_weight=0.001,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
            "blocks.*.feed_forward_moe.experts*",  # Freeze experts, only train router
        ],
    )


def build_train_module_config(common: CommonComponents) -> SoftLabelRouterTrainModuleConfig:
    """Build training module config with soft label router training."""
    return SoftLabelRouterTrainModuleConfig(
        rank_microbatch_size=1 * 4096,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=2e-3,  # Higher LR for router training
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        ),
        compile_model=False,  # Disabled: forward hooks don't work well with torch.compile
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
        # Soft label router training specific configs
        router_loss_weight=1.0,  # Weight for soft label router loss
        router_loss_only=True,   # Only train router (set LM loss to 0)
        soft_label_beta=1.0,     # Temperature for soft labels (can be swept)
        num_experts=3,           # Number of experts to consider
        # expert_labels_dir is set via command line override
    )


def build_dataset_config(common: CommonComponents) -> NumpyDatasetConfig:
    """Build dataset config using router_training_mix (full 5B token dataset).
    
    Matches OLMoE-4x7B-supervised-router.py pattern exactly.
    """
    from flexolmo.data.mixes import CustomDataMix, get_mixture_dataset_config_by_domain
    from flexolmo.data.build_dataset_with_source_metadata import add_source_name_metadata

    dataset_config = common.dataset
    
    # Use router_training_mix by default, but allow override via --dataset.mix
    # Note: Command line overrides happen AFTER this function, so we check env var
    import os
    mix_override = os.environ.get("FLEXOLMO_DATASET_MIX")
    if mix_override:
        dataset_config.mix = mix_override
        log.info(f"Using mix from FLEXOLMO_DATASET_MIX: {mix_override}")
    elif dataset_config.mix is None:
        dataset_config.mix = CustomDataMix.router_training_mix
        log.info(f"Using default mix: router_training_mix")
    else:
        log.info(f"Using pre-configured mix: {dataset_config.mix}")
    
    # Use get_mixture_dataset_config_by_domain to split by domain labels
    # This ensures each domain (starcoder, mj_finemath4plus, etc.) becomes its own source
    # Disable file validation to avoid metadata count mismatch with add_source_name_metadata
    source_mixture_config = get_mixture_dataset_config_by_domain(dataset_config, validate_files=False)
    dataset_config.source_mixture_config = source_mixture_config
    dataset_config.mix = None  # Clear mix since we're using source_mixture_config
    
    # Add source_name metadata so it appears in batches via batch["metadata"]
    dataset_config = add_source_name_metadata(dataset_config, source_mixture_config)
    
    # Note: include_instance_metadata is already set by add_source_name_metadata()
    # We rely on instance_indices being available in batches (olmo-core default behavior)
    
    return dataset_config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build trainer config."""
    trainer_config = common.trainer
    # Default to 5B tokens (full RT mix)
    trainer_config.max_duration.value = 5_000_000_000
    trainer_config.max_duration.unit = DurationUnit("tokens")
    return trainer_config


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} [dry_run] run_name [OVERRIDES...]")
        print("\nExample:")
        print(f"  torchrun --nproc-per-node=8 {sys.argv[0]} FlexOlmo-4x7B-SoftLabel-RT \\")
        print("    --trainer.load_path=/path/to/checkpoint \\")
        print("    --train_module.expert_labels_dir=/path/to/per_token_labels_sft \\")
        print("    --train_module.soft_label_beta=1.0 \\")
        print("    --trainer.save_folder=/path/to/save")
        print("\nKey parameters:")
        print("  --train_module.soft_label_beta=<float>  Temperature for soft labels (default: 1.0)")
        print("  --train_module.expert_labels_dir=<path> Directory with seq_*.npz files (required)")
        print("  --train_module.router_loss_weight=<float>  Weight for router loss (default: 1.0)")
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
            global_batch_size=8 * SEQUENCE_LENGTH,  # 32768 tokens
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
