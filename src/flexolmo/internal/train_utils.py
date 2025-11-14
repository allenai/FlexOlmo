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
from flexolmo.data.expert_label_injector import wrap_data_loader_with_expert_labels
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


def _ensure_router_metadata(dataset_config) -> None:
    """
    Make sure the dataset config that feeds supervised router training contains
    the per-instance metadata required for expert label injection.
    """
    from flexolmo.data.build_dataset_with_source_metadata import add_source_name_metadata
    from flexolmo.data.mixes import get_mixture_dataset_config_by_domain

    metadata = getattr(dataset_config, "metadata", None)
    include_instance_metadata = getattr(dataset_config, "include_instance_metadata", False)

    if metadata and include_instance_metadata:
        log.info(
            "Supervised router training: dataset already has metadata (%s entries)",
            len(metadata),
        )
        return

    log.warning(
        "Supervised router training requires source metadata, but dataset config is missing it. "
        "Rebuilding SourceMixtureDatasetConfig and injecting source_name metadata now."
    )

    mix_base_dir = getattr(dataset_config, "mix_base_dir", None)
    if dataset_config.source_mixture_config is None:
        dataset_mix = getattr(dataset_config, "mix", None)
        if dataset_mix is None:
            raise RuntimeError(
                "Router training dataset must define either mix or source_mixture_config "
                "so we can derive source metadata (see SUPERVISED_ROUTER_TRAINING_GUIDE.md)."
            )
        if mix_base_dir is None:
            raise RuntimeError(
                "dataset.mix_base_dir must be set to build router metadata. "
                "Override --dataset.mix_base_dir=<path> when launching training."
            )
        dataset_config.source_mixture_config = get_mixture_dataset_config_by_domain(
            dataset_config,
            validate_files=True,
        )
        dataset_config.mix = None

    add_source_name_metadata(dataset_config, dataset_config.source_mixture_config)


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

    if isinstance(train_module, SupervisedRouterTrainModule):
        _ensure_router_metadata(config.dataset)

    dataset = config.dataset.build()

    if isinstance(train_module, SupervisedRouterTrainModule):
        dataset_metadata = getattr(dataset, "metadata", None)
        if dataset_metadata:
            example_meta = dataset_metadata[0] if isinstance(dataset_metadata, list) else dataset_metadata
            log.info(
                "Supervised router dataset ready: metadata entries=%s, first entry=%s",
                len(dataset_metadata) if isinstance(dataset_metadata, list) else "unknown",
                example_meta,
            )
        else:
            log.warning(
                "Dataset object does not expose metadata after preparation; "
                "ExpertLabelDataLoaderWrapper will enforce metadata presence at batch-time."
            )
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    
    # Wrap data loader with expert label injection if using supervised router training
    if isinstance(train_module, SupervisedRouterTrainModule):
        log.info("Wrapping data loader with expert label injection for supervised router training")
        data_loader = wrap_data_loader_with_expert_labels(
            data_loader,
            use_domain_labels=train_module.use_domain_labels,
            dataset=dataset,
        )  # type: ignore
        if hasattr(data_loader, "strict_metadata"):
            data_loader.strict_metadata = True
            data_loader.max_missing_metadata_batches = 5
    
    trainer = config.trainer.build(train_module, data_loader)  # type: ignore[arg-type]

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    # cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if checkpoint is not None:  # anneal or finetune
        # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)

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
