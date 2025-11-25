"""Supervised Router Training Module for MoE models.

Trains router via cross-entropy loss between router logits and ground truth expert labels.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.config import DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_full_tensor, get_local_tensor
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module.transformer import (
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import move_to_device
from torch.distributed.tensor import DTensor

log = logging.getLogger(__name__)


def _unwrap_fsdp_model(model):
    """Unwrap FSDP-wrapped models."""
    if hasattr(model, '_fsdp_wrapped_module'):
        return model._fsdp_wrapped_module
    elif hasattr(model, 'module'):
        return model.module
    elif hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


@dataclass
class SupervisedRouterTrainModuleConfig(TransformerTrainModuleConfig):
    """Configuration for supervised router training."""
    
    router_loss_weight: float = 1.0
    """Weight for the supervised router loss."""
    
    router_loss_only: bool = False
    """If True, only train router (set language modeling loss weight to 0)."""
    
    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
        dataset=None,
        **extra_kwargs,
    ) -> "SupervisedRouterTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return SupervisedRouterTrainModule(
            model=model,
            device=device,
            dataset=dataset,
            **kwargs,
            **extra_kwargs,
        )


class SupervisedRouterTrainModule(TransformerTrainModule):
    """Training module for supervised router training.
    
    Expects batch["expert_labels"] with shape (batch_size, num_experts) where each row
    is a one-hot encoding indicating the target expert for that instance.
    """

    def __init__(
        self,
        router_loss_weight: float = 1.0,
        router_loss_only: bool = False,
        dataset=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.router_loss_weight = router_loss_weight
        self.router_loss_only = router_loss_only
        self.dataset = dataset
        self._patch_routers_for_supervised_loss()
        log.info(f"SupervisedRouterTrainModule initialized (router_loss_only={router_loss_only})")
    
    def _patch_routers_for_supervised_loss(self):
        """Patch router forward methods to compute supervised loss during forward."""
        self._current_expert_labels: Optional[torch.Tensor] = None
        self._router_supervised_losses: List[float] = []
        
        patched_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                if not hasattr(module, 'num_experts') or not hasattr(module, '_latest_router_logits'):
                    log.warning(f"Skipping {name}: isinstance MoERouter but missing router attributes")
                    continue
                    
                original_forward = module.forward
                router_name = name
                
                def make_patched_forward(router, orig_fn, train_module_self, router_name_inner):
                    def patched_forward(x, *, loss_div_factor=None):
                        result = orig_fn(x, loss_div_factor=loss_div_factor)
                        expert_weights, expert_indices, batch_size_per_expert, aux_loss = result
                        
                        expert_labels = train_module_self._current_expert_labels
                        if expert_labels is not None and router.training and torch.is_grad_enabled():
                            router_logits = router._latest_router_logits
                            if router_logits is not None:
                                supervised_loss = train_module_self._compute_supervised_loss_during_forward(
                                    router_logits, expert_labels, loss_div_factor, router.num_experts
                                )
                                if supervised_loss is not None:
                                    scaled_loss = train_module_self.router_loss_weight * supervised_loss
                                    aux_loss = scaled_loss if aux_loss is None else aux_loss + scaled_loss
                                    train_module_self._router_supervised_losses.append(get_local_tensor(scaled_loss.detach()).item())
                                    if not hasattr(train_module_self, '_logged_supervised_loss'):
                                        train_module_self._logged_supervised_loss = set()
                                    if router_name_inner not in train_module_self._logged_supervised_loss and train_module_self.trainer.global_step <= 1:
                                        log.info(f"[Router Training] Computed supervised loss in router '{router_name_inner}': {scaled_loss.item():.6f}")
                                        train_module_self._logged_supervised_loss.add(router_name_inner)
                        
                        return expert_weights, expert_indices, batch_size_per_expert, aux_loss
                    
                    return patched_forward
                
                module.forward = make_patched_forward(module, original_forward, self, router_name)
                patched_count += 1
        
        log.info(f"[Router Training] Patched {patched_count} routers for supervised loss computation")
    
    def _compute_supervised_loss_during_forward(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
        num_experts: int,
    ) -> Optional[torch.Tensor]:
        """Compute supervised loss during forward pass."""
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        if expert_labels.shape[-1] != num_experts:
            return None
        
        expert_labels = expert_labels.to(router_logits.device)
        
        if router_logits.dim() == 3:
            batch_size, seq_len, num_experts = router_logits.shape
            router_logits_flat = router_logits.reshape(-1, num_experts)
        elif router_logits.dim() == 2:
            router_logits_flat = router_logits
            batch_size = expert_labels.shape[0]
            seq_len = router_logits_flat.shape[0] // batch_size
        else:
            return None
        
        expert_indices = expert_labels.argmax(dim=-1).long()
        expert_indices = expert_indices.repeat_interleave(seq_len)
        
        num_tokens = expert_indices.numel()
        if num_tokens == 0:
            return None
        
        router_loss = F.cross_entropy(router_logits_flat, expert_indices, reduction="sum")
        
        if loss_div_factor is not None:
            router_loss = router_loss / (loss_div_factor if isinstance(loss_div_factor, torch.Tensor) else float(loss_div_factor))
        else:
            router_loss = router_loss / num_tokens
        
        return router_loss
    
    def _create_dummy_loss_for_dry_run(self) -> Optional[torch.Tensor]:
        """Create dummy loss from router parameters for dry run."""
        dummy_loss = None
        router_param_found = False
        
        for name, module in self.model.named_modules():
            if hasattr(module, '_latest_router_logits'):
                router_logits = module._latest_router_logits
                if router_logits is not None and isinstance(router_logits, torch.Tensor):
                    try:
                        if isinstance(router_logits, DTensor):
                            router_logits = get_full_tensor(router_logits)
                        dummy_loss = router_logits.sum() * 1e-8
                        if dummy_loss.requires_grad:
                            log.debug(f"Dry-run: Created dummy loss from router logits in '{name}'")
                            return dummy_loss
                    except Exception as e:
                        log.debug(f"Dry-run: Failed to create dummy loss from router logits in '{name}': {e}")
        
        for name, param in self.model.named_parameters():
            if "router" in name.lower():
                router_param_found = True
                if param.requires_grad and param.numel() > 0:
                    try:
                        dummy_loss = param.sum() * 1e-8
                        if dummy_loss.requires_grad:
                            log.debug(f"Dry-run: Created dummy loss from router param '{name}'")
                            return dummy_loss
                    except Exception as e:
                        log.debug(f"Dry-run: Failed to create dummy loss from '{name}': {e}")
        
        if not router_param_found:
            log.warning("Dry-run: No router parameters found in model")
        else:
            log.warning("Dry-run: Found router parameters but none have requires_grad=True")
        return None
    
    def _prepare_batch(self, batch: Dict[str, Any]):  # type: ignore[override]
        """Preserve expert_labels, metadata, and index. Clone tensors to avoid storage issues."""
        preserved = {k: batch.get(k) for k in ['metadata', 'index', 'expert_labels']}
        if 'expert_labels' in preserved and isinstance(preserved['expert_labels'], torch.Tensor):
            preserved['expert_labels'] = move_to_device(
                preserved['expert_labels'].clone(), self.device
            )
        
        input_ids, labels, model_kwargs = super()._prepare_batch(batch)
        
        if isinstance(input_ids, torch.Tensor):
            input_ids = move_to_device(input_ids.clone(), self.device)
        if isinstance(labels, torch.Tensor):
            labels = move_to_device(labels.clone(), self.device)
        
        for key, value in model_kwargs.items():
            if isinstance(value, torch.Tensor):
                model_kwargs[key] = move_to_device(value.clone(), self.device)
        
        model_kwargs.update({k: v for k, v in preserved.items() if v is not None})
        return input_ids, labels, model_kwargs

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """Train on a batch with supervised router loss."""
        self.model.train()

        batch_size = batch["input_ids"].shape[0]
        has_expert_labels = "expert_labels" in batch and batch["expert_labels"] is not None
        
        if not hasattr(self, '_first_batch_logged'):
            self._first_batch_logged = True
            if dry_run:
                log.info(f"Dry-run batch: keys={list(batch.keys())}, has_expert_labels={has_expert_labels} (expected - mock batch)")
            else:
                log.info(f"First batch: keys={list(batch.keys())}, has_expert_labels={has_expert_labels}")
        
        if not has_expert_labels:
            if self.router_loss_only:
                if dry_run:
                    log.info("Dry-run batch: skipping router loss (mock batch has no expert_labels). Real batches will have expert_labels.")
                else:
                    error_msg = (
                        "router_loss_only=True but 'expert_labels' missing from batch. "
                        "\n\nThis usually means one of these issues:"
                        "\n1. Dataset config doesn't have include_instance_metadata=True"
                        "\n2. Dataset config doesn't have metadata set (missing source_name)"
                        "\n3. DataCollator isn't receiving items with metadata"
                        "\n4. source_mixture_config wasn't created correctly"
                    )
                    if 'metadata' not in batch:
                        error_msg += "\n\n❌ Batch has NO 'metadata' field - dataset items don't have metadata!"
                    elif batch.get('metadata') and isinstance(batch['metadata'], list):
                        first_meta = batch['metadata'][0] if batch['metadata'] else None
                        if first_meta and 'source_name' not in first_meta:
                            error_msg += f"\n\n❌ Metadata exists but missing 'source_name' field. Metadata keys: {list(first_meta.keys())}"
                        elif not first_meta:
                            error_msg += "\n\n❌ Metadata list is empty!"
                    raise RuntimeError(error_msg)
            else:
                log.debug("No expert labels found, skipping router loss")

        # Generate labels for language modeling
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record masked instances
        if (instance_mask := batch.get("instance_mask")) is not None:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate tokens for loss
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        if self.cp_enabled:
            assert self._cp_config is not None
            batch_num_tokens_for_loss = batch_num_tokens_for_loss / self._cp_config.degree

        # Batch losses
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        router_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        auxiliary_batch_losses: Dict[str, torch.Tensor] = {}

        # Split into micro-batches
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)
        _micro_batch_refs = list(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(_micro_batch_refs):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                micro_expert_labels = None
                if has_expert_labels and "expert_labels" in model_kwargs:
                    micro_expert_labels = model_kwargs["expert_labels"]
                
                if self.router_loss_only and micro_expert_labels is None:
                    if dry_run:
                        log.debug("Dry-run: skipping router loss computation (no expert_labels in mock batch)")
                    else:
                        raise RuntimeError(f"router_loss_only=True but missing expert_labels for micro_batch {micro_batch_idx}")

                self._current_expert_labels = micro_expert_labels if has_expert_labels else None
                
                model_forward_result = self.model_forward(
                    input_ids, labels=labels, ignore_index=self.label_ignore_index,
                    loss_reduction="sum", z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss, return_logits=False,
                    **model_kwargs,
                )
                
                if isinstance(model_forward_result, tuple):
                    output_dict, ce_loss, z_loss = model_forward_result[:3]  # type: ignore[misc]
                else:
                    raise TypeError(f"Unexpected return type: {type(model_forward_result)}")
                
                self._current_expert_labels = None

                if self.router_loss_only:
                    loss = None
                else:
                    loss = ce_loss
                    if z_loss is not None:
                        loss = loss + z_loss
                
                # Collect supervised loss values for logging/metrics (losses already attached via attach_auxiliary_loss)
                should_log = micro_batch_idx == 0 and (self.trainer.global_step <= 1 or self.trainer.global_step % 100 == 0)
                if self._router_supervised_losses:
                    router_loss_sum = sum(self._router_supervised_losses)
                    router_batch_loss += router_loss_sum
                    if should_log:
                        log.info(f"[Router Training] Collected supervised loss from {len(self._router_supervised_losses)} routers: {router_loss_sum:.6f}")
                    self._router_supervised_losses.clear()
                
                # Collect auxiliary losses (router Z loss, load balancing, supervised router loss, etc.)
                model_for_aux = _unwrap_fsdp_model(self.model)
                if hasattr(model_for_aux, 'compute_auxiliary_losses'):
                    auxiliary_losses = model_for_aux.compute_auxiliary_losses(  # type: ignore[attr-defined]
                        reset=True
                    )
                    
                    if should_log:
                        log.info(f"[Router Training] compute_auxiliary_losses returned {len(auxiliary_losses)} losses: {list(auxiliary_losses.keys())}")
                        for loss_name, loss_val in auxiliary_losses.items():
                            loss_val_local = get_local_tensor(loss_val.detach())
                            log.info(f"  - {loss_name}: {loss_val_local.item():.6f}")
                    
                    for loss_name, loss_val in auxiliary_losses.items():
                        loss_val_local = get_local_tensor(loss_val.detach())
                        
                        # Add auxiliary losses to main loss
                        # For router_loss_only mode, we need these losses (especially supervised router loss)
                        # to flow through backward
                        if loss is None:
                            loss = loss_val
                        else:
                            loss = loss + loss_val
                        
                        if loss_name in auxiliary_batch_losses:
                            auxiliary_batch_losses[loss_name] += loss_val_local
                        else:
                            auxiliary_batch_losses[loss_name] = loss_val_local
                    
                    del auxiliary_losses
                
                # Fallback: if no auxiliary losses were found, create dummy loss for dry run
                if loss is None:
                    if dry_run:
                        dummy_loss = self._create_dummy_loss_for_dry_run()
                        loss = dummy_loss if dummy_loss is not None and dummy_loss.requires_grad else None
                    elif self.router_loss_only:
                        raise RuntimeError(
                            "router_loss_only=True but no router loss found. "
                            "This likely means no auxiliary losses were returned from the model."
                        )
                
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss
                
                if loss is not None and isinstance(loss, torch.Tensor):
                    loss.backward()
                elif micro_batch_idx == 0 and self.trainer.global_step < 1:
                    log.warning(f"[Router Training] Skipping backward: loss is None or not a tensor")

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            model_for_aux = _unwrap_fsdp_model(self.model)
            if hasattr(model_for_aux, 'reset_auxiliary_losses'):
                model_for_aux.reset_auxiliary_losses()  # type: ignore[attr-defined]
            if hasattr(model_for_aux, 'reset_auxiliary_metrics'):
                model_for_aux.reset_auxiliary_metrics()  # type: ignore[attr-defined]
            return

        if not self.router_loss_only:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        
        if isinstance(router_batch_loss, torch.Tensor):
            router_loss_val = router_batch_loss.item()
            if router_loss_val > 0:
                self.record_metric("Router loss", router_batch_loss, ReduceType.mean, namespace="train")
            elif self.trainer.global_step % 100 == 0:
                log.warning(f"[Router Training] router_batch_loss is 0 (step={self.trainer.global_step})")
        
        for loss_name, loss_val in auxiliary_batch_losses.items():
            self.record_metric(loss_name, loss_val, ReduceType.mean, namespace="train")

        model_for_aux = _unwrap_fsdp_model(self.model)
        if hasattr(model_for_aux, 'compute_auxiliary_metrics'):
            metrics = model_for_aux.compute_auxiliary_metrics(reset=True)  # type: ignore[attr-defined]
            for metric_name, (metric_val, reduction) in metrics.items():
                self.record_metric(metric_name, metric_val, reduction, namespace="train")
        
        if isinstance(self.optim, SkipStepOptimizer):
            if self.router_loss_only:
                self.optim.latest_loss = router_batch_loss if isinstance(router_batch_loss, torch.Tensor) else ce_batch_loss
            else:
                self.optim.latest_loss = ce_batch_loss
