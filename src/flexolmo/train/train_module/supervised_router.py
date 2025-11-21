"""
Supervised Router Training Module for MoE models.

Trains router via cross-entropy loss between router logits and ground truth expert labels.
Expert labels are injected into items before collation, preserved by DataCollator, and
flow through Transformer._prepare_inputs() to this module.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
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
from torch.distributed.tensor import DTensor

log = logging.getLogger(__name__)


def _unwrap_fsdp_model(model):
    """Helper to unwrap FSDP-wrapped models."""
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
        dataset=None,  # Allow dataset to be passed for metadata access
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
            dataset=dataset,  # Pass dataset for metadata access
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
        log.info(f"SupervisedRouterTrainModule initialized (router_loss_only={router_loss_only})")
    
    def _prepare_batch(self, batch: Dict[str, Any]):  # type: ignore[override]
        """Preserve expert_labels, metadata, and index for supervised router training."""
        preserved = {k: batch.get(k) for k in ['metadata', 'index', 'expert_labels']}
        input_ids, labels, model_kwargs = super()._prepare_batch(batch)
        model_kwargs.update({k: v for k, v in preserved.items() if v is not None})
        return input_ids, labels, model_kwargs

    def _extract_expert_labels_from_batch(self, batch: Dict[str, Any], batch_size: int) -> Optional[torch.Tensor]:
        """Extract expert_labels tensor from batch (shape: batch_size, num_experts)."""
        if "expert_labels" in batch:
            expert_labels = batch["expert_labels"]
            if isinstance(expert_labels, torch.Tensor):
                # Verify shape matches expected batch_size
                if expert_labels.shape[0] != batch_size:
                    log.warning(
                        f"expert_labels shape {expert_labels.shape[0]} doesn't match "
                        f"batch_size {batch_size}"
                    )
                return expert_labels.to(self.device).contiguous()
        
        # Don't log error here - let train_batch handle logging based on dry_run flag
        return None

    def _compute_router_loss(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor,
        num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised cross-entropy loss for router."""
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        if router_logits.dim() == 3:
            router_logits = router_logits.mean(dim=0)
        elif router_logits.dim() != 2:
            raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")
 
        # Fix orientation if needed
        if router_logits.shape[1] != expert_labels.shape[1] and router_logits.shape[0] == expert_labels.shape[1]:
            router_logits = router_logits.transpose(0, 1)

        router_logits = router_logits.contiguous()
        expert_labels = expert_labels.to(router_logits.device).contiguous()
        
        batch_size = expert_labels.shape[0]
        seq_len = router_logits.shape[0] // batch_size
        
        # Convert one-hot to indices and expand to per-token labels
        expert_indices = expert_labels.argmax(dim=-1).long().repeat_interleave(seq_len)
        
        if expert_indices.numel() != batch_size * seq_len:
            raise RuntimeError(
                f"Shape mismatch: {expert_indices.shape} vs expected {(batch_size * seq_len,)}"
            )
        
        router_loss = F.cross_entropy(router_logits, expert_indices, reduction="sum") / num_tokens
        
        return router_loss

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """Train on a batch with supervised router loss."""
        self.model.train()

        batch_size = batch["input_ids"].shape[0]
        
        if not hasattr(self, '_first_batch_logged'):
            self._first_batch_logged = True
            log.info(f"First batch: keys={list(batch.keys())}, has_expert_labels={'expert_labels' in batch}")
            log.info(f"  Batch has 'metadata': {'metadata' in batch}")
            log.info(f"  Batch has 'index': {'index' in batch}")
            if 'metadata' in batch and batch['metadata']:
                log.info(f"  First metadata entry: {batch['metadata'][0] if isinstance(batch['metadata'], list) else batch['metadata']}")
            elif 'metadata' not in batch:
                if dry_run:
                    log.info(
                        "ℹ️  Dry-run batch doesn't have 'metadata' field (expected - dry-run uses mock batch). "
                        "Real training batches should have metadata if dataset is configured correctly."
                    )
                else:
                    log.error(
                        "❌ CRITICAL: Batch doesn't have 'metadata' field! "
                        "This means dataset items don't have metadata. "
                        "Check that dataset config has include_instance_metadata=True and metadata is set."
                    )
        
        expert_labels = self._extract_expert_labels_from_batch(batch, batch_size)
        
        if expert_labels is None:
            if self.router_loss_only:
                if dry_run:
                    # For dry-run, create dummy expert_labels (default to Expert 1 - General)
                    # This allows the dry-run to test forward/backward pass structure
                    log.warning(
                        "router_loss_only=True but 'expert_labels' missing from dry-run batch. "
                        "Creating dummy expert_labels (Expert 1) for dry-run."
                    )
                    # Try to get num_experts from model blocks or config
                    num_experts = 4  # Default
                    try:
                        # Try to get from first MoE block
                        for block in getattr(self.model, 'blocks', {}).values():
                            if hasattr(block, 'feed_forward_moe') and hasattr(block.feed_forward_moe, 'router'):
                                router = block.feed_forward_moe.router
                                if hasattr(router, 'num_experts'):
                                    num_experts = router.num_experts
                                    break
                                elif hasattr(block.feed_forward_moe, 'num_experts'):
                                    num_experts = block.feed_forward_moe.num_experts
                                    break
                    except Exception:
                        pass  # Use default of 4
                    
                    # Create dummy expert_labels on the correct device and make them contiguous
                    expert_labels = torch.zeros(batch_size, num_experts, dtype=torch.float32, device=self.device)
                    expert_labels[:, 1] = 1.0  # Set Expert 1 (General) as default
                    expert_labels = expert_labels.contiguous()
                else:
                    # Real batch without expert_labels - this means metadata isn't flowing through
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
        
        if expert_labels is not None:
            expert_labels = move_to_device(expert_labels, self.device)
            if not expert_labels.is_contiguous():
                expert_labels = expert_labels.contiguous()

        # Generate labels for language modeling
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record masked instances
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
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

        # Train one micro-batch at a time
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                micro_expert_labels = None
                if expert_labels is not None:
                    micro_batch_size = input_ids.shape[0]
                    start_idx = micro_batch_idx * micro_batch_size
                    micro_expert_labels = expert_labels[start_idx:start_idx + micro_batch_size].contiguous()
                
                if self.router_loss_only and micro_expert_labels is None:
                    raise RuntimeError(f"router_loss_only=True but missing expert_labels for micro_batch {micro_batch_idx}")

                router_loss: Optional[torch.Tensor] = None
                should_capture_router_logits = (micro_expert_labels is not None) or self.router_loss_only
                
                if should_capture_router_logits:
                    router_loss_terms: List[torch.Tensor] = []
                    router_hook_errors: List[str] = []
                    
                    def router_hook(module, input, output):
                        if isinstance(output, tuple) and len(output) >= 1 and isinstance(output[0], torch.Tensor):
                            if micro_expert_labels is None:
                                return
                            try:
                                loss_term = self._compute_router_loss(
                                    output[0], micro_expert_labels, batch_num_tokens_for_loss
                                )
                                router_loss_terms.append(loss_term)
                            except Exception as exc:
                                error_msg = f"Failed to compute router loss for {module}: {exc}"
                                if dry_run:
                                    # During dry run, just log warnings instead of failing
                                    log.debug(f"Dry run: {error_msg}")
                                    router_hook_errors.append(error_msg)
                                else:
                                    log.warning(error_msg)
                    
                    hooks = []
                    router_modules = [(n, m) for n, m in self.model.named_modules() 
                                     if 'router' in n.lower() and hasattr(m, 'forward')]
                    
                    if router_modules:
                        for _, router_module in router_modules:
                            hooks.append(router_module.register_forward_hook(router_hook))
                    else:
                        # Fallback: search blocks for MoE routers
                        for block in getattr(self.model, 'blocks', []):
                            if not isinstance(block, torch.nn.Module):
                                continue
                            for attr in ['feed_forward_moe', 'block_sparse_moe', 'moe', 'feed_forward']:
                                if hasattr(block, attr):
                                    moe_module = getattr(block, attr)
                                    if moe_module is not None and hasattr(moe_module, 'router'):
                                        router = getattr(moe_module, 'router')
                                        if isinstance(router, torch.nn.Module):
                                            hooks.append(router.register_forward_hook(router_hook))
                                            break
                    
                    if not hooks:
                        log.error(f"No router modules found in model (type: {type(self.model)})")
                    else:
                        log.debug(f"Registered {len(hooks)} router hooks")
                    
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
                    
                    for hook in hooks:
                        hook.remove()
                    
                    if router_loss_terms:
                        router_loss = torch.stack(router_loss_terms, dim=0).mean()
                        log.debug(f"Router loss from {len(router_loss_terms)} modules: {router_loss.item():.4f}")
                    elif self.router_loss_only:
                        if dry_run:
                            # During dry run, create a dummy loss if router hooks failed
                            log.debug("Dry run: router hooks failed to compute loss, creating dummy loss for backward pass")
                            router_loss = move_to_device(torch.tensor(1.0, requires_grad=True), self.device)
                        else:
                            raise RuntimeError(
                                "router_loss_only=True but no router loss terms computed. "
                                "Check that router modules are found and output correct shapes."
                            )
                else:
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

                if router_loss is not None:
                    router_batch_loss += get_local_tensor(router_loss.detach())
                
                if self.router_loss_only:
                    if router_loss is None:
                        raise RuntimeError("router_loss_only=True but router_loss is None")
                    loss = router_loss * self.router_loss_weight
                else:
                    loss = ce_loss
                    if z_loss is not None:
                        loss = loss + z_loss
                    if router_loss is not None:
                        loss = loss + (router_loss * self.router_loss_weight)

                # Update batch losses
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                model_for_aux = _unwrap_fsdp_model(self.model)
                if hasattr(model_for_aux, 'compute_auxiliary_losses'):
                    auxiliary_losses = model_for_aux.compute_auxiliary_losses(  # type: ignore[attr-defined]
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

                # Backward pass
                loss.backward()

        del batch

        if dry_run:
            model_for_aux = _unwrap_fsdp_model(self.model)
            if hasattr(model_for_aux, 'reset_auxiliary_losses'):
                model_for_aux.reset_auxiliary_losses()  # type: ignore[attr-defined]
            if hasattr(model_for_aux, 'reset_auxiliary_metrics'):
                model_for_aux.reset_auxiliary_metrics()  # type: ignore[attr-defined]
            return

        # Record metrics
        if not self.router_loss_only:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
        if router_batch_loss.item() > 0:
            self.record_metric(
                "Router loss",
                router_batch_loss,
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

        model_for_aux = _unwrap_fsdp_model(self.model)
        if hasattr(model_for_aux, 'compute_auxiliary_metrics'):
            for metric_name, (metric_val, reduction) in model_for_aux.compute_auxiliary_metrics(  # type: ignore[attr-defined]
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
            self.optim.latest_loss = ce_batch_loss if not self.router_loss_only else router_batch_loss
