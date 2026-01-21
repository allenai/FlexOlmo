"""
Train a 4x7B OLMo2 model with CE-constrained (hard label) router training.

This script implements a hybrid approach that combines standard router training
(learning from LM loss) with cross-entropy regularization towards hard domain-based labels.

## Approach

Total loss:
    L_total = L_LM + L_Z + λ * CE(hard_labels, router_logits)

Where:
    - L_LM: Language modeling (cross-entropy) loss (ACTIVE - key difference)
    - L_Z: Router stability loss
    - λ: Weight for CE regularization (ce_loss_weight)
    - CE: Cross-entropy with hard one-hot labels based on domain

## Hard Labels

Labels are based on data source domain:
    - Math data → Expert 0: [1, 0, 0, 0]
    - Code data → Expert 2: [0, 0, 1, 0]
    - General data → Expert 1: [0, 1, 0, 0]

## Key Differences from Other Methods

### vs. Standard RT (OLMoE-4x7B.py)
    - Adds CE regularization term to guide router towards domain labels
    - Provides hard supervision via one-hot vectors

### vs. Supervised RT (OLMoE-4x7B-supervised-router.py)
    - LM loss is ACTIVE (not disabled via router_loss_only=True)
    - Router learns primarily from task performance
    - CE term provides gentle guidance, not hard constraints

### vs. Soft Label RT (OLMoE-4x7B-soft-label-router.py)
    - LM loss is ACTIVE (soft label RT sets router_loss_only=True)
    - Uses HARD labels instead of soft distributions
    - No need for pre-computed expert losses

## Advantages

- **Simpler**: No need to pre-compute expert losses
- **Cleaner**: Labels come from domain metadata (already in batch)
- **Flexible**: Router can deviate from domain labels when LM loss benefits
- **Stable**: CE regularization prevents drastic routing changes

## Usage

torchrun --nproc-per-node=8 src/scripts/train/OLMoE-4x7B-kl-constrained-router.py \\
    FlexOlmo-4x7B-CE-Constrained-RT \\
    --trainer.load_path=/path/to/checkpoint \\
    --train_module.ce_loss_weight=0.1 \\
    --trainer.save_folder=/path/to/save

Key parameters:
    --train_module.ce_loss_weight=<float>      Weight λ for CE term (default: 0.1)

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
from flexolmo.train.train_module.kl_constrained_router import (
    KLConstrainedRouterTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    """Build model config for 4x7B MoE with CE-constrained router training."""
    return TransformerConfig.olmoe_nx7b(  # type: ignore
        vocab_size=common.tokenizer.padded_vocab_size(),
        num_experts=4,
        top_k=4,  # Use all 4 experts
        lb_loss_weight=0,  # Disable load balancing loss (using CE regularization instead)
        z_loss_weight=0.001,  # Keep z-loss for router stability
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
            "blocks.*.feed_forward_moe.experts*",  # Freeze experts, only train router
        ],
    )


def build_train_module_config(common: CommonComponents) -> KLConstrainedRouterTrainModuleConfig:
    """Build training module config with CE-constrained router training."""
    return KLConstrainedRouterTrainModuleConfig(
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
        # CE-constrained router training specific configs
        ce_loss_weight=0.1,      # Weight λ for CE regularization (can be swept)
        num_experts=4,           # Total number of experts in model
    )


def build_dataset_config(common: CommonComponents) -> NumpyDatasetConfig:
    """Build dataset config using router_training_mix (full 5B token dataset).
    
    Matches OLMoE-4x7B-supervised-router.py pattern exactly.
    """
    from flexolmo.data.mixes import CustomDataMix, get_mixture_dataset_config_by_domain
    from flexolmo.data.build_dataset_with_source_metadata import add_source_name_metadata

    dataset_config = common.dataset
    
    # ALWAYS use router_training_mix for CE-constrained training
    # The checkpoint may have a different mix (e.g., OLMoE-mix-0824) baked in, so we override it
    import os
    mix_override = os.environ.get("FLEXOLMO_DATASET_MIX")
    if mix_override:
        dataset_config.mix = mix_override
        log.info(f"Using mix from FLEXOLMO_DATASET_MIX: {mix_override}")
    else:
        # Always override to router_training_mix (checkpoint may have different default)
        old_mix = dataset_config.mix
        dataset_config.mix = CustomDataMix.router_training_mix
        log.info(f"Overriding mix from '{old_mix}' to 'router_training_mix'")
    
    # Use get_mixture_dataset_config_by_domain to split by domain labels
    # This ensures each domain (starcoder, mj_finemath4plus, etc.) becomes its own source
    # Enable file validation to skip corrupted files (e.g., tulu-3-sft-personas-math-grade has bad files)
    source_mixture_config = get_mixture_dataset_config_by_domain(dataset_config, validate_files=True)
    dataset_config.source_mixture_config = source_mixture_config
    dataset_config.mix = None  # Clear mix since we're using source_mixture_config
    
    # Add source_name metadata so it appears in batches via batch["metadata"]
    dataset_config = add_source_name_metadata(dataset_config, source_mixture_config)
    
    # Note: include_instance_metadata is already set by add_source_name_metadata()
    # We rely on metadata being available in batches with source_name field
    
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
        print(f"  torchrun --nproc-per-node=8 {sys.argv[0]} FlexOlmo-4x7B-CE-Constrained-RT \\")
        print("    --trainer.load_path=/path/to/checkpoint \\")
        print("    --train_module.ce_loss_weight=0.1 \\")
        print("    --trainer.save_folder=/path/to/save")
        print("\nKey parameters:")
        print("  --train_module.ce_loss_weight=<float>      Weight λ for CE term (default: 0.1)")
        print("\nAdvantages over other methods:")
        print("  - No pre-computed expert losses needed")
        print("  - Labels from domain metadata (simpler)")
        print("  - LM loss active (task-driven)")
        print("  - CE regularization (gentle guidance)")
        print("\nCompare with other methods:")
        print("  - Standard RT: No CE term, only LM loss + auxiliary losses")
        print("  - Supervised RT: router_loss_only=true, no LM loss, hard labels")
        print("  - Soft Label RT: router_loss_only=true, no LM loss, soft labels")
        print("  - CE-Constrained RT (this): LM loss + λ*CE with hard labels (hybrid)")
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
            global_batch_size=128 * SEQUENCE_LENGTH,  # Match other training scripts
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
