import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.config import DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_full_tensor, get_local_tensor
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module.transformer import (
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import move_to_device
from torch.distributed.tensor import DTensor, distribute_tensor

log = logging.getLogger(__name__)


def distribute_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not isinstance(source, DTensor):
        return get_full_tensor(target)
    if isinstance(target, DTensor):
        if target.device_mesh == source.device_mesh and target.placements == source.placements:
            return target
        else:
            return target.redistribute(device_mesh=source.device_mesh, placements=source.placements)
    return distribute_tensor(target, device_mesh=source.device_mesh, placements=source.placements)


@dataclass
class FreezeTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    def __init__(self, *args, **kwargs):
        self.freeze_experts = kwargs.pop("freeze_experts", "first_half")
        super().__init__(*args, **kwargs)

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> "FreezeTransformerTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # bp()
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return FreezeTransformerTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class FreezeTransformerTrainModule(TransformerTrainModule):
    """
    Custom transformer train module that zeros out gradients for the first half of expert parameters.
    Inherits from the original TransformerTrainModule.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomTransformerTrainModule by calling the parent class's __init__ method.
        #"""
        # swj change
        # extract freeze_experts from kwargs
        self.freeze_experts = kwargs.pop("freeze_experts", "first_half")
        super().__init__(*args, **kwargs)

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self.model.train()

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        if self.cp_enabled:
            assert self._cp_config is not None
            batch_num_tokens_for_loss = batch_num_tokens_for_loss / self._cp_config.degree

        # Batch losses to record.
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        auxiliary_batch_losses: Dict[str, torch.Tensor] = {}

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Run forward pass, get losses.
                _, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )

                # Get loss to optimize for.
                loss = ce_loss
                if z_loss is not None:
                    loss += z_loss

                # Update total batch CE and Z loss.
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                # print("self.trainer.global_step: ", self.trainer.global_step)
                # Optionally get model auxiliary losses and update the total batch auxiliary losses. , step=self.trainer.global_step
                # step=self.trainer.global_step
                auxiliary_losses = self.model.compute_auxiliary_losses(
                    batch_num_tokens_for_loss, reset=True
                )
                for loss_name, loss_val in auxiliary_losses.items():
                    loss += loss_val
                    loss_val = get_local_tensor(loss_val.detach())
                    if loss_name in auxiliary_batch_losses:
                        auxiliary_batch_losses[loss_name] += loss_val
                    else:
                        auxiliary_batch_losses[loss_name] = loss_val
                del auxiliary_losses

                # Run backward pass.
                loss.backward()

        # for name, param in self.model.named_parameters():
        # if "expert2_bias" in name:
        # from ipdb import set_trace as bp
        # bp()
        # print(f"{name}: {get_full_tensor(param)}")
        # if param.numel() > /0:  # Only record non-empty tensors
        #     self.record_metric(
        #         name,
        #         get_full_tensor(param),
        #         ReduceType.mean,
        #         namespace="train",
        #     )
        # log.info("swj")
        # for name, param in self.model.named_parameters():
        # log.info(f"{name}: requires_grad={param.requires_grad}")
        # bp()
        # swj change
        # fsdp summon
        for name, param in self.model.named_parameters():
            if "experts" in name or "router" in name:
                # bp()
                if self.freeze_experts == "first_half":
                    # print("name: ", name, "shape: ", param.shape)
                    # check whether the param is frozen
                    # print("param.grad: ", param.grad)
                    if param.grad is None:
                        # print(f"{name} grad is None")
                        continue
                    full_grad = get_full_tensor(param.grad)
                    if "experts" in name:
                        # get_full_tensor(param.grad)[
                        #     : get_full_tensor(param.grad).shape[0] // 2, :
                        # ] = 0
                        mask = torch.zeros_like(full_grad, dtype=torch.bool)
                        mask[: full_grad.shape[0] // 2, :] = True
                        local_mask = get_local_tensor(distribute_like(param, mask))
                        # print("target.device_mesh: ", param.device_mesh)
                        get_local_tensor(param.grad).masked_fill_(local_mask, 0.0)
                        # print("mask: ", mask)
                    elif "router" in name:
                        # get_full_tensor(param.grad)[
                        #     : get_full_tensor(param.grad).shape[0] // 2
                        # ] = 0
                        mask = torch.zeros_like(full_grad, dtype=torch.bool)
                        # swj change, free the entire router
                        mask[: full_grad.shape[0] // 2] = 1
                        # mask = torch.ones_like(full_grad, dtype=torch.bool)
                        local_mask = get_local_tensor(distribute_like(param, mask))
                        get_local_tensor(param.grad).masked_fill_(local_mask, 0.0)
                        # print("mask: ", mask)
                        # get_local_tensor(param.grad) = get_local_tensor(param.grad).mul(local_mask)
                # elif self.freeze_experts == "last_half":
                #     if "experts" in name:
                #         get_full_tensor(param.grad)[
                #             get_full_tensor(param.grad).shape[0] // 2 :, :
                #         ] = 0
                #     elif "router" in name:
                #         get_full_tensor(param.grad)[
                #             get_full_tensor(param.grad).shape[0] // 2 :
                #         ] = 0
                else:
                    raise ValueError(f"Invalid freeze_experts value: {self.freeze_experts}")

        del batch  # In case this helps with memory utilization.

        if dry_run:
            self.model.reset_auxiliary_losses()
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
        for loss_name, loss_val in auxiliary_batch_losses.items():
            self.record_metric(
                loss_name,
                loss_val,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            batch_num_tokens_for_loss,
            reset=True,
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )
        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss
