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
    
    # Pass dataset to train module for supervised router training
    # The train module will use batch['index'] to lookup metadata
    train_module_kwargs = {}
    if hasattr(config.train_module, '__dict__'):
        config_dict = config.train_module.as_dict(exclude_none=True, recurse=False)
        log.info(f"Checking train_module config for supervised router...")
        log.info(f"  Config type: {type(config.train_module).__name__}")
        log.info(f"  Has router_loss_weight: {'router_loss_weight' in config_dict}")
        log.info(f"  Has router_loss_only: {'router_loss_only' in config_dict}")
        log.info(f"  Config keys: {list(config_dict.keys())[:10]}")
        
        if 'router_loss_weight' in config_dict or 'router_loss_only' in config_dict:
            # CRITICAL: Wrap dataset to include 'index' in items
            # Monkey-patching doesn't work because NumpyFSLDatasetMixture overrides [] operator
            
            class DatasetWithIndex:
                """Wrapper that adds 'index' field to dataset items."""
                def __init__(self, wrapped_dataset):
                    self.wrapped_dataset = wrapped_dataset
                    # Expose important attributes from wrapped dataset
                    self.metadata = getattr(wrapped_dataset, 'metadata', None)
                    
                def __getitem__(self, idx):
                    item = self.wrapped_dataset[idx]
                    
                    # Always create a new dict to ensure index is added
                    if isinstance(item, torch.Tensor):
                        item_dict = {'input_ids': item, 'index': idx}
                        if self.metadata and idx < len(self.metadata):
                            item_dict['metadata'] = self.metadata[idx]
                        return item_dict
                    elif isinstance(item, dict):
                        # Create new dict with index added
                        item_dict = dict(item)
                        item_dict['index'] = idx
                        # Ensure metadata is present if available
                        if 'metadata' not in item_dict and self.metadata and idx < len(self.metadata):
                            item_dict['metadata'] = self.metadata[idx]
                        return item_dict
                    else:
                        # Unknown type - wrap in dict
                        return {'input_ids': item, 'index': idx}
                
                def __len__(self):
                    return len(self.wrapped_dataset)
                
                def __getattr__(self, name):
                    # Delegate all other attributes to wrapped dataset
                    return getattr(self.wrapped_dataset, name)
            
            # Wrap the dataset
            dataset = DatasetWithIndex(dataset)  # type: ignore[assignment]
            log.info("✅ Wrapped dataset to include 'index' field in items")
            
            # Test the wrapper
            log.info(f"🔍 Testing wrapped dataset[0]:")
            try:
                test_item = dataset[0]
                log.info(f"  Item type: {type(test_item)}")
                log.info(f"  Item keys: {list(test_item.keys()) if isinstance(test_item, dict) else 'NOT DICT'}")
                log.info(f"  Has 'index': {'index' in test_item if isinstance(test_item, dict) else False}")
                log.info(f"  Has 'metadata': {'metadata' in test_item if isinstance(test_item, dict) else False}")
            except Exception as e:
                log.warning(f"  Failed to get test item: {e}")
            
            train_module_kwargs['dataset'] = dataset  # type: ignore[dict-item]
            log.info("Passing wrapped dataset to SupervisedRouterTrainModule for batch['index'] → metadata lookup")
    
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

    # Build data loader (dataset already built above, possibly wrapped for router training)
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)  # type: ignore[arg-type]
    
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
