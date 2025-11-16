import json
import logging
from fnmatch import fnmatch
from typing import Optional, cast

import torch
from olmo_core.distributed.utils import get_local_rank
from olmo_core.io import resource_path
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train.callbacks import (
    CometCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all

from flexolmo.internal.common import ExperimentConfig
from flexolmo.internal.model_utils import *  # noqa
from flexolmo.train.train_module.supervised_router import SupervisedRouterTrainModule
from flexolmo.data.dataset_with_expert_labels import DatasetWithExpertLabels

log = logging.getLogger(__name__)


def get_last_lr(checkpoint: str) -> float:
    # Get step number and max steps to infer where the learning rate left off.
    train_state = torch.load(resource_path(f"{checkpoint}/train", "rank0.pt"), weights_only=False)
    last_pretrain_step: int = train_state["global_step"]
    max_pretrain_steps: int = train_state.get("max_steps", 774861)  # default found in logs
    log.info(f"Last LR from step {last_pretrain_step:,d} of {max_pretrain_steps:,d}")

    # Now infer the learning rate.
    with resource_path(checkpoint, "config.json").open() as f:
        config = json.load(f)

    try:
        # checkpoint trained on v2 codebase
        base_lr = config["train_module"]["optim"]["lr"]
        scheduler_config = config["train_module"]["scheduler"]
    except KeyError:
        # checkpoint trained on v1 codebase
        base_lr = config["optim"]["lr"]
        scheduler_config = config["trainer"]["callbacks"]["lr_scheduler"]["scheduler"]
    assert scheduler_config.pop("_CLASS_").split(".")[-1] == CosWithWarmup.__name__
    scheduler = CosWithWarmup(**scheduler_config)
    last_lr = float(scheduler.get_lr(base_lr, last_pretrain_step, max_pretrain_steps))
    return last_lr


def _train(
    config: ExperimentConfig, *, checkpoint: Optional[str] = None, use_last_lr: bool = False
):
    """
    Train a model with the given configuration.
    If `checkpoint` is provided, it will load the model from the checkpoint and continue training.
    If `use_last_lr` is True, it will start from the last learning rate of the checkpoint.
    """
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    device = get_default_device()

    if use_last_lr:  # anneal
        # For annealing; start from the last learning rate of the checkpoint.
        assert (
            checkpoint is not None
        ), "Checkpoint must be provided when estimating last learning rate."
        starting_lr = get_last_lr(checkpoint)
        log.info(f"Starting LR: {starting_lr}")
        assert isinstance(config.train_module.optim, AdamWConfig)
        config.train_module.optim.lr = starting_lr

    # Build components.
    model = config.model.build(init_device="meta")
    
    # Build dataset first (needed for supervised router training)
    dataset = config.dataset.build()
    
    # Wrap dataset for supervised router training
    # This injects expert labels directly into dataset items (can't be bypassed!)
    is_supervised_router = False
    if hasattr(config.train_module, '__dict__'):
        config_dict = config.train_module.as_dict(exclude_none=True, recurse=False)
        if 'router_loss_weight' in config_dict or 'router_loss_only' in config_dict:
            is_supervised_router = True
            log.info("Wrapping dataset to inject expert labels at item level")
            dataset = DatasetWithExpertLabels(dataset)
            
            # Test that dataset wrapper actually works
            if get_local_rank() == 0:
                try:
                    test_item = dataset[0]
                    log.info(f"Dataset wrapper test - item 0 keys: {list(test_item.keys()) if isinstance(test_item, dict) else 'not a dict'}")
                    if isinstance(test_item, dict) and 'expert_label' in test_item:
                        log.info(f"✅ Dataset wrapper working! expert_label: {test_item['expert_label']}")
                    else:
                        log.error("❌ Dataset wrapper NOT working - expert_label missing from item!")
                except Exception as e:
                    log.error(f"❌ Dataset wrapper test failed: {e}")
    
    # Pass dataset to train module
    train_module_kwargs = {}
    if is_supervised_router:
        train_module_kwargs['dataset'] = dataset
    
    train_module = config.train_module.build(model, device=device, **train_module_kwargs)

    if config.model.freeze_params:
        for name, param in model.named_parameters():
            for pattern in config.model.freeze_params:
                if fnmatch(name, pattern):
                    param.requires_grad = False
                    log.info(f"Param '{name}' will be frozen")
                    break
            else:
                log.info(f"Param '{name}' will be trainable")

    # Build data loader (dataset already built above, possibly wrapped)
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)  # type: ignore[arg-type]
    
    # CRITICAL FIX: Patch collator to preserve expert_label field
    # The default DataCollator only preserves known fields (input_ids, labels, etc.)
    # We need it to also preserve our custom expert_label field
    if is_supervised_router and hasattr(data_loader, 'collator'):
        original_collator = data_loader.collator
        
        def collate_with_expert_labels(items):
            """Collator that preserves expert_label and domain_label fields."""
            # Call original collator for standard fields (input_ids, etc.)
            batch = original_collator(items)
            
            # Preserve expert_label if present in items
            if items and isinstance(items[0], dict):
                if 'expert_label' in items[0]:
                    # Stack expert labels from all items
                    batch['expert_label'] = torch.stack([item['expert_label'] for item in items])
                    log.debug(f"Collator: Preserved expert_label for {len(items)} items")
                
                if 'domain_label' in items[0]:
                    # Keep domain labels as list
                    batch['domain_label'] = [item['domain_label'] for item in items]
                    log.debug(f"Collator: Preserved domain_label for {len(items)} items")
            
            return batch
        
        data_loader.collator = collate_with_expert_labels  # type: ignore[assignment]
        log.info("✅ Patched collator to preserve expert_label and domain_label fields")
    
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    # cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Load checkpoint if specified (either via parameter or via trainer.load_path in config)
    checkpoint_loaded = False
    if checkpoint is not None:  # anneal or finetune
        # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        checkpoint_loaded = True
    elif hasattr(trainer, 'load_path') and trainer.load_path is not None:
        # Checkpoint specified via config (e.g., --trainer.load_path=...)
        # Trainer will load it automatically, but we need to trigger auto-init after
        checkpoint_loaded = True
    
    # NOTE: When using --trainer.load_path, the checkpoint is loaded INSIDE trainer.fit()
    # So we can't auto-initialize here. The initialization must happen in the model's
    # reset_parameters() method instead (which we've already fixed in router.py)
    # Just log that we're aware of the checkpoint
    if checkpoint_loaded:
        log.info(f"Will load checkpoint during training (load_path configured)")

    if get_local_rank() == 0 and checkpoint is not None:
        print("Updated config:")
        print(config)

    # Train.
    trainer.fit()


def train(config: ExperimentConfig):
    """
    Train a model with the given configuration.
    """
    _train(config)


def finetune(checkpoint: str, config: ExperimentConfig):
    """
    Finetune a model from a checkpoint.
    """
    _train(config, checkpoint=checkpoint, use_last_lr=False)


def anneal(checkpoint: str, config: ExperimentConfig):
    """
    Anneal a model from a checkpoint, starting from the last learning rate.
    """
    _train(config, checkpoint=checkpoint, use_last_lr=True)
