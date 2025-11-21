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
        """Preserve expert_labels, metadata, and index for supervised router training.
        
        IMPORTANT: Clone all tensors to avoid storage issues with views from split_batch.
        split_batch() creates views that share storage with the original batch. During
        backward pass, these views can become invalid, causing "storage of size 0" errors.
        """
        preserved = {k: batch.get(k) for k in ['metadata', 'index', 'expert_labels']}
        # Clone expert_labels if it's a tensor and ensure it's on the right device
        if 'expert_labels' in preserved and isinstance(preserved['expert_labels'], torch.Tensor):
            preserved['expert_labels'] = move_to_device(
                preserved['expert_labels'].clone(), self.device
            )
        
        input_ids, labels, model_kwargs = super()._prepare_batch(batch)
        
        # Clone input_ids and labels to ensure they have their own storage
        # (split_batch creates views, which can cause issues during backward pass)
        if isinstance(input_ids, torch.Tensor):
            input_ids = move_to_device(input_ids.clone(), self.device)
        if isinstance(labels, torch.Tensor):
            labels = move_to_device(labels.clone(), self.device)
        
        # Clone any tensors in model_kwargs that might be views from split_batch
        for key, value in model_kwargs.items():
            if isinstance(value, torch.Tensor):
                model_kwargs[key] = move_to_device(value.clone(), self.device)
        
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
        
        # Clone to ensure we have our own storage (operations below may create views)
        router_logits = router_logits.clone()
        
        if router_logits.dim() == 3:
            router_logits = router_logits.mean(dim=0).clone()  # mean can return a view
        elif router_logits.dim() != 2:
            raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")
 
        # Fix orientation if needed (transpose creates a view!)
        if router_logits.shape[1] != expert_labels.shape[1] and router_logits.shape[0] == expert_labels.shape[1]:
            router_logits = router_logits.transpose(0, 1).clone()  # clone after transpose

        router_logits = router_logits.contiguous()
        expert_labels = expert_labels.to(router_logits.device).clone().contiguous()
        
        batch_size = expert_labels.shape[0]
        seq_len = router_logits.shape[0] // batch_size
        
        # Convert one-hot to indices and expand to per-token labels
        # Clone to ensure we have our own storage (repeat_interleave may create views)
        expert_indices = expert_labels.argmax(dim=-1).long().repeat_interleave(seq_len).clone()
        
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
            if dry_run:
                log.info("Dry-run batch: keys={}, has_expert_labels={} (expected - mock batch)".format(
                    list(batch.keys()), 'expert_labels' in batch
                ))
            else:
                log.info(f"First batch: keys={list(batch.keys())}, has_expert_labels={'expert_labels' in batch}")
                log.info(f"  Batch has 'metadata': {'metadata' in batch}")
                log.info(f"  Batch has 'index': {'index' in batch}")
                if 'metadata' in batch and batch['metadata']:
                    log.info(f"  First metadata entry: {batch['metadata'][0] if isinstance(batch['metadata'], list) else batch['metadata']}")
                elif 'metadata' not in batch:
                    log.error(
                        "❌ CRITICAL: Batch doesn't have 'metadata' field! "
                        "This means dataset items don't have metadata. "
                        "Check that dataset config has include_instance_metadata=True and metadata is set."
                    )
        
        # Check if batch has expert_labels for validation purposes
        has_expert_labels = "expert_labels" in batch and batch["expert_labels"] is not None
        
        # Handle dry-run batches (Trainer automatically runs dry run before training)
        # Dry run uses mock batch without metadata/expert_labels - skip router loss for dry run only
        if not has_expert_labels:
            if self.router_loss_only:
                if dry_run:
                    # Dry run batch doesn't have expert_labels (expected) - skip router loss for dry run
                    log.info("Dry-run batch: skipping router loss (mock batch has no expert_labels). Real batches will have expert_labels.")
                else:
                    # Real batch without expert_labels - this is an error
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
        
        # Keep references to prevent garbage collection of view tensors during backward pass
        # split_batch creates views, and these views must remain valid during backward()
        _micro_batch_refs = list(micro_batches)

        # Train one micro-batch at a time
        for micro_batch_idx, micro_batch in enumerate(_micro_batch_refs):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Extract expert_labels from model_kwargs (already cloned and moved to device in _prepare_batch)
                micro_expert_labels = None
                if has_expert_labels and "expert_labels" in model_kwargs:
                    micro_expert_labels = model_kwargs["expert_labels"]
                
                if self.router_loss_only and micro_expert_labels is None:
                    if dry_run:
                        # Dry run batch - skip router loss, will use ce_loss for backward pass
                        log.debug("Dry-run: skipping router loss computation (no expert_labels in mock batch)")
                    else:
                        raise RuntimeError(f"router_loss_only=True but missing expert_labels for micro_batch {micro_batch_idx}")

                router_loss: Optional[torch.Tensor] = None
                # Skip router logits capture if dry_run with router_loss_only but no expert_labels
                should_capture_router_logits = (
                    (micro_expert_labels is not None) or 
                    (self.router_loss_only and not (dry_run and micro_expert_labels is None))
                )
                
                if should_capture_router_logits:
                    router_loss_terms: List[torch.Tensor] = []
                    
                    def router_hook(module, input, output):
                        """Capture router forward output and compute supervised loss."""
                        if micro_expert_labels is None:
                            return
                        
                        try:
                            router_logits: Optional[torch.Tensor] = None

                            pop_logits = getattr(module, "pop_router_logits", None)
                            if callable(pop_logits):
                                router_logits = cast(Optional[torch.Tensor], pop_logits())

                            if router_logits is None and hasattr(module, "get_expert_logits"):
                                # Fallback path (e.g., for modules that don't expose cached logits)
                                if not isinstance(input, tuple) or len(input) == 0:
                                    return
                                x = input[0]
                                if not isinstance(x, torch.Tensor):
                                    return
                                x = x.clone()
                                if hasattr(module, "jitter"):
                                    x = module.jitter(x)
                                router_logits = module.get_expert_logits(x).float()

                            if router_logits is None:
                                log.error(
                                    f"Router {module.__class__.__name__} did not provide logits for supervised loss"
                                )
                                return

                            # Reshape to (batch_size * seq_len, num_experts) for loss computation
                            if router_logits.dim() == 3:
                                batch_size, seq_len, num_experts = router_logits.shape
                                router_logits_flat = router_logits.reshape(-1, num_experts).contiguous()
                            elif router_logits.dim() == 2:
                                router_logits_flat = router_logits.contiguous()
                            else:
                                raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")

                            loss_term = self._compute_router_loss(
                                router_logits_flat, micro_expert_labels, batch_num_tokens_for_loss
                            )
                            router_loss_terms.append(loss_term)
                        except Exception as exc:
                            log.error(f"Failed to compute router loss for {module}: {exc}", exc_info=True)
                    
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
                            # Dry run batch - no router loss terms (expected if no expert_labels)
                            log.debug("Dry-run: no router loss terms computed (expected for mock batch)")
                            router_loss = None
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
                        if dry_run:
                            # Dry run batch without expert_labels - use ce_loss for backward pass test
                            log.debug("Dry-run: router_loss_only=True but no router_loss (expected), using ce_loss for backward pass")
                            loss = ce_loss
                        else:
                            raise RuntimeError("router_loss_only=True but router_loss is None")
                    else:
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

        self.model.post_batch(dry_run=dry_run)
        
        # Don't explicitly delete batch/_micro_batch_refs - let Python GC handle it
        # after the function returns. Explicit deletion can cause "storage of size 0" 
        # errors if the computation graph still references view tensors.

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
            compute_metrics_fn = getattr(model_for_aux, 'compute_auxiliary_metrics')
            metrics = compute_metrics_fn(
                batch_num_tokens_for_loss,
                reset=True,
            )
            for metric_name, (metric_val, reduction) in metrics.items():
                self.record_metric(
                    metric_name,
                    metric_val,
                    reduction,
                    namespace="train",
                )
        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss if not self.router_loss_only else router_batch_loss
