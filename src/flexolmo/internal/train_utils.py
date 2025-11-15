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
    
    # Pass dataset to train module if it's supervised router training
    # This allows the train module to extract expert labels directly from dataset.metadata
    train_module_kwargs = {}
    if hasattr(config.train_module, '__dict__'):
        # Check if this is SupervisedRouterTrainModule
        config_dict = config.train_module.as_dict(exclude_none=True, recurse=False)
        if 'router_loss_weight' in config_dict or 'router_loss_only' in config_dict:
            train_module_kwargs['dataset'] = dataset
            log.info("Passing dataset to SupervisedRouterTrainModule for direct metadata access")
    
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

    # Build data loader (dataset already built above)
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    # cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if checkpoint is not None:  # anneal or finetune
        # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        
        # CRITICAL FIX: Initialize any parameters that weren't in the checkpoint
        # This handles cases where the checkpoint uses a different router type
        # (e.g., standard router vs router with expert_bias)
        log.info("Checking for uninitialized parameters after checkpoint load...")
        uninitialized_params = []
        fixed_params = []
        
        for name, param in model.named_parameters():
            try:
                # Check if parameter is uninitialized
                needs_init = False
                if param.device.type == "meta":
                    uninitialized_params.append(f"{name}: still on meta device")
                    needs_init = True
                elif param.numel() > 0:  # Skip empty tensors
                    storage_size = (
                        param.untyped_storage().nbytes()
                        if hasattr(param, "untyped_storage")
                        else param.storage().nbytes()
                    )
                    expected_size = param.numel() * param.element_size()
                    if storage_size == 0:
                        uninitialized_params.append(f"{name}: zero storage (expected {expected_size} bytes)")
                        needs_init = True
                
                # Initialize if needed (typically expert_bias or other custom parameters)
                if needs_init and param.numel() > 0:
                    # Move to proper device and initialize
                    with torch.no_grad():
                        # Create new tensor with proper storage on device
                        new_param = torch.empty_like(param, device=device)
                        # Initialize with small random values
                        torch.nn.init.trunc_normal_(new_param, std=0.002, a=-3 * 0.002, b=0)
                        # Replace the parameter data
                        param.data = new_param
                        fixed_params.append(name)
                        log.info(f"Initialized parameter: {name} (shape={param.shape}, device={param.device})")
                        
            except Exception as e:
                uninitialized_params.append(f"{name}: error checking/fixing ({e})")
        
        if uninitialized_params:
            log.warning(f"Found {len(uninitialized_params)} uninitialized parameters")
            for param_info in uninitialized_params[:10]:
                log.warning(f"  - {param_info}")
            if len(uninitialized_params) > 10:
                log.warning(f"  ... and {len(uninitialized_params) - 10} more")
        
        if fixed_params:
            log.info(f"✅ Successfully initialized {len(fixed_params)} missing parameters:")
            for param_name in fixed_params:
                log.info(f"  - {param_name}")

        if get_local_rank() == 0:
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
