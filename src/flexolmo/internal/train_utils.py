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
    train_module = config.train_module.build(model, device=device)

    if config.model.freeze_params:
        for name, param in model.named_parameters():
            for pattern in config.model.freeze_params:
                if fnmatch(name, pattern):
                    param.requires_grad = False
                    log.info(f"Param '{name}' will be frozen")
                    break
            else:
                log.info(f"Param '{name}' will be trainable")

    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    # cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if get_local_rank() == 0 and checkpoint is not None:  # anneal or finetune
        # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)

        # if get_local_rank() == 0:
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
