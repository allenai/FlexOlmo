"""
Supervised Router Training Module for MoE models.

Trains router via cross-entropy loss between router logits and ground truth expert labels.
Expert labels are injected into items before collation, preserved by DataCollator, and
flow through Transformer._prepare_inputs() to this module.
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


class SupervisedMoERouter(MoERouter):
    """
    MoE Router that computes supervised cross-entropy loss when expert_labels are provided.
    
    Extends MoERouter to accept expert_labels in forward() and compute supervised loss
    as part of the auxiliary loss.
    """
    
    def __init__(self, *args, router_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_loss_weight = router_loss_weight
        self._supervised_loss: Optional[torch.Tensor] = None
    
    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        expert_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional supervised loss computation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            loss_div_factor: Factor to divide loss by
            expert_labels: Optional expert labels of shape (batch_size, num_experts) for supervised loss
        
        Returns:
            Same as MoERouter.forward(), but aux_loss may include supervised loss.
        """
        # Call parent forward to get standard routing
        expert_weights, expert_indices, batch_size_per_expert, aux_loss = super().forward(
            x, loss_div_factor=loss_div_factor
        )
        
        # Compute supervised loss if expert_labels provided
        if expert_labels is not None and self.training and torch.is_grad_enabled():
            # Get router logits (already computed in parent forward)
            router_logits = self._latest_router_logits
            if router_logits is not None and isinstance(router_logits, torch.Tensor):
                supervised_loss = self._compute_supervised_loss(
                    router_logits, expert_labels, loss_div_factor
                )
                
                if supervised_loss is not None:
                    self._supervised_loss = supervised_loss
                    scaled_supervised_loss = self.router_loss_weight * supervised_loss
                    
                    # Add to auxiliary loss
                    if aux_loss is None:
                        aux_loss = scaled_supervised_loss
                    else:
                        aux_loss = aux_loss + scaled_supervised_loss
        
        return expert_weights, expert_indices, batch_size_per_expert, aux_loss
    
    def _compute_supervised_loss(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Optional[torch.Tensor]:
        """Compute supervised cross-entropy loss for router."""
        # Handle DTensor
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        # Clone to ensure own storage
        router_logits = router_logits.clone().contiguous()
        expert_labels = expert_labels.clone().contiguous()
        
        # Validate expert label dimensions match model
        num_model_experts = router_logits.shape[-1]
        if expert_labels.shape[-1] != num_model_experts:
            log.warning(
                f"Expert label dimension mismatch: labels have {expert_labels.shape[-1]} experts, "
                f"but model has {num_model_experts} experts. Skipping supervised loss."
            )
            return None
        
        # Ensure same device
        expert_labels = expert_labels.to(router_logits.device)
        
        # Reshape router_logits to (batch_size * seq_len, num_experts)
        if router_logits.dim() == 3:
            batch_size, seq_len, num_experts = router_logits.shape
            router_logits_flat = router_logits.reshape(-1, num_experts).contiguous()
        elif router_logits.dim() == 2:
            router_logits_flat = router_logits.contiguous()
            batch_size = expert_labels.shape[0]
            seq_len = router_logits_flat.shape[0] // batch_size
        else:
            log.warning(f"Unexpected router_logits shape: {router_logits.shape}. Skipping supervised loss.")
            return None
        
        # Convert one-hot expert_labels to indices and expand to per-token
        expert_indices = expert_labels.argmax(dim=-1).long().clone()
        expert_indices = expert_indices.repeat_interleave(seq_len).clone()
        
        # Compute cross-entropy loss
        num_tokens = expert_indices.numel()
        if num_tokens == 0:
            return None
        
        router_loss = F.cross_entropy(router_logits_flat, expert_indices, reduction="sum")
        
        # Divide by loss_div_factor if provided
        if loss_div_factor is not None:
            if isinstance(loss_div_factor, torch.Tensor):
                router_loss = router_loss / loss_div_factor
            else:
                router_loss = router_loss / float(loss_div_factor)
        else:
            router_loss = router_loss / num_tokens
        
        return router_loss
    
    def get_supervised_loss(self) -> Optional[torch.Tensor]:
        """Get the most recent supervised loss."""
        loss = self._supervised_loss
        self._supervised_loss = None  # Clear after retrieval
        return loss


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
        
        # Patch routers to compute supervised loss during forward pass
        # This is the cleanest approach - loss computed as part of computation graph
        self._patch_routers_for_supervised_loss()
        
        log.info(f"SupervisedRouterTrainModule initialized (router_loss_only={router_loss_only})")
    
    def _patch_routers_for_supervised_loss(self):
        """Patch router forward methods to compute supervised loss during forward.
        
        Clean solution: Loss computed DURING forward when computation graph is valid.
        No cloning needed - tensors are part of the graph and storage is guaranteed.
        """
        # Store current expert_labels in a thread-local or module-level variable
        # that routers can access during forward
        self._current_expert_labels: Optional[torch.Tensor] = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                original_forward = module.forward
                
                def make_patched_forward(router, orig_fn, train_module_self):
                    def patched_forward(x, *, loss_div_factor=None):
                        # Call original forward
                        result = orig_fn(x, loss_div_factor=loss_div_factor)
                        expert_weights, expert_indices, batch_size_per_expert, aux_loss = result
                        
                        # Compute supervised loss DURING forward if expert_labels available
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
                                    # Minimal log: verify supervised loss computed (once per router, first step only)
                                    if not hasattr(train_module_self, '_logged_supervised_loss'):
                                        train_module_self._logged_supervised_loss = set()
                                    if name not in train_module_self._logged_supervised_loss and train_module_self.trainer.global_step < 1:
                                        log.info(f"[Router Training] Computed supervised loss in '{name}': {scaled_loss.item():.6f}")
                                        train_module_self._logged_supervised_loss.add(name)
                        
                        return expert_weights, expert_indices, batch_size_per_expert, aux_loss
                    
                    return patched_forward
                
                module.forward = make_patched_forward(module, original_forward, self)
                log.debug(f"Patched router {name} for supervised loss")
    
    def _compute_supervised_loss_during_forward(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
        num_experts: int,
    ) -> Optional[torch.Tensor]:
        """Compute supervised loss during forward pass - no cloning needed, part of computation graph.
        
        This is called DURING the forward pass, so all tensors are part of the computation graph
        and storage is guaranteed to be valid. No cloning needed!
        """
        # Handle DTensor - get full tensor but don't clone (we're in forward, graph is valid)
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        # Validate expert label dimensions match model
        if expert_labels.shape[-1] != num_experts:
            return None
        
        # Ensure same device (no clone - just move if needed)
        expert_labels = expert_labels.to(router_logits.device)
        
        # Reshape router_logits to (batch_size * seq_len, num_experts)
        # reshape() creates a view, but that's fine during forward - graph keeps it alive
        if router_logits.dim() == 3:
            batch_size, seq_len, num_experts = router_logits.shape
            router_logits_flat = router_logits.reshape(-1, num_experts)
        elif router_logits.dim() == 2:
            router_logits_flat = router_logits
            batch_size = expert_labels.shape[0]
            seq_len = router_logits_flat.shape[0] // batch_size
        else:
            return None
        
        # Convert one-hot to indices and expand to per-token
        # These operations create new tensors, so no storage issues
        expert_indices = expert_labels.argmax(dim=-1).long()
        expert_indices = expert_indices.repeat_interleave(seq_len)
        
        # Compute cross-entropy loss - this creates a new tensor, part of computation graph
        num_tokens = expert_indices.numel()
        if num_tokens == 0:
            return None
        
        router_loss = F.cross_entropy(router_logits_flat, expert_indices, reduction="sum")
        
        # Divide by loss_div_factor if provided
        if loss_div_factor is not None:
            if isinstance(loss_div_factor, torch.Tensor):
                router_loss = router_loss / loss_div_factor
            else:
                router_loss = router_loss / float(loss_div_factor)
        else:
            router_loss = router_loss / num_tokens
        
        return router_loss
    
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

    def _compute_router_loss_from_stored_logits(
        self,
        expert_labels: torch.Tensor,
        num_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute supervised router loss from logits stored in routers after forward pass.
        
        This approach avoids storage issues by:
        1. Accessing logits AFTER forward pass completes (they're stored in _latest_router_logits)
        2. Cloning immediately to ensure own storage
        3. Computing loss from cloned tensors (which are still part of computation graph)
        """
        router_loss_terms: List[torch.Tensor] = []
        
        # Iterate through all routers and get their stored logits
        for name, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                router_logits = getattr(module, '_latest_router_logits', None)
                if router_logits is not None and isinstance(router_logits, torch.Tensor):
                    try:
                        # CRITICAL: Clone immediately to ensure we have our own storage
                        # This prevents storage invalidation issues during backward pass
                        # clone() preserves gradients, so loss computation will still work correctly
                        router_logits = router_logits.clone().contiguous()
                        expert_labels_cloned = expert_labels.clone().contiguous()
                        
                        loss_term = self._compute_router_loss(
                            router_logits, expert_labels_cloned, num_tokens
                        )
                        router_loss_terms.append(loss_term)
                    except Exception as exc:
                        log.error(f"Failed to compute router loss for {name}: {exc}", exc_info=True)
        
        if router_loss_terms:
            return torch.stack(router_loss_terms, dim=0).mean()
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
 
        # Validate expert label dimensions match model
        num_model_experts = router_logits.shape[-1]
        if expert_labels.shape[-1] != num_model_experts:
            raise ValueError(
                f"Expert label dimension mismatch: labels have {expert_labels.shape[-1]} experts, "
                f"but model has {num_model_experts} experts"
            )

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

                # Set expert_labels so routers can access during forward
                # Routers are patched to compute supervised loss during forward pass
                self._current_expert_labels = micro_expert_labels if has_expert_labels else None
                
                # Forward pass - routers compute supervised loss during forward and add to aux_loss
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
                
                # Clear expert_labels after forward
                self._current_expert_labels = None
                
                # Router loss is included in auxiliary losses via attach_auxiliary_loss
                # We'll extract it from compute_auxiliary_losses below

                # Build total loss (before deleting ce_loss/z_loss)
                if self.router_loss_only:
                    # Router loss is in auxiliary losses - will extract below
                    # Initialize to None, will be set when we extract router_loss_from_aux
                    loss = None
                else:
                    loss = ce_loss
                    if z_loss is not None:
                        loss = loss + z_loss
                
                model_for_aux = _unwrap_fsdp_model(self.model)
                if hasattr(model_for_aux, 'compute_auxiliary_losses'):
                    auxiliary_losses = model_for_aux.compute_auxiliary_losses(  # type: ignore[attr-defined]
                        reset=True
                    )
                    
                    # Extract router loss from auxiliary losses
                    # When router_loss_only=True, all auxiliary losses should be router-related
                    router_loss_from_aux = None
                    
                    for loss_name, loss_val in auxiliary_losses.items():
                        loss_val_local = get_local_tensor(loss_val.detach())
                        
                        # Check if this looks like a router/supervised loss
                        if 'router' in loss_name.lower() or 'supervised' in loss_name.lower():
                            if router_loss_from_aux is None:
                                router_loss_from_aux = loss_val
                                router_batch_loss += loss_val_local
                            else:
                                # Multiple router losses - sum them
                                router_loss_from_aux = router_loss_from_aux + loss_val
                                router_batch_loss += loss_val_local
                        
                        # Collect all auxiliary losses for potential use
                        if not self.router_loss_only:
                            loss += loss_val
                        
                        # Track for logging
                        if loss_name in auxiliary_batch_losses:
                            auxiliary_batch_losses[loss_name] += loss_val_local
                        else:
                            auxiliary_batch_losses[loss_name] = loss_val_local
                    
                    # If router_loss_only=True but no named router loss found, use ALL auxiliary losses
                    # (router_loss_only means only router losses should exist)
                    if self.router_loss_only and router_loss_from_aux is None and len(auxiliary_losses) > 0:
                        router_loss_from_aux = sum(auxiliary_losses.values())
                        router_batch_loss = sum(get_local_tensor(loss_val.detach()) for loss_val in auxiliary_losses.values())
                        # Log: verify extraction (first step or every 100 steps for debugging)
                        if micro_batch_idx == 0 and (self.trainer.global_step < 1 or self.trainer.global_step % 100 == 0):
                            loss_sum = sum(get_local_tensor(loss_val.detach()).item() for loss_val in auxiliary_losses.values())
                            log.info(f"[Router Training] Using sum of {len(auxiliary_losses)} auxiliary losses as router loss: {loss_sum:.6f} (names: {list(auxiliary_losses.keys())})")
                    elif self.router_loss_only and router_loss_from_aux is not None:
                        # Log: verify extraction (first step or every 100 steps for debugging)
                        if micro_batch_idx == 0 and (self.trainer.global_step < 1 or self.trainer.global_step % 100 == 0):
                            loss_val_local = get_local_tensor(router_loss_from_aux.detach())
                            log.info(f"[Router Training] Extracted router loss: {loss_val_local.item():.6f}")
                    elif self.router_loss_only and len(auxiliary_losses) == 0:
                        # Warning: no auxiliary losses found (log occasionally)
                        if micro_batch_idx == 0 and (self.trainer.global_step < 1 or self.trainer.global_step % 100 == 0):
                            log.warning(f"[Router Training] router_loss_only=True but NO auxiliary losses found!")
                    
                    del auxiliary_losses
                    
                    # For router_loss_only mode, use only router loss
                    if self.router_loss_only:
                        if router_loss_from_aux is None:
                            if dry_run:
                                log.debug("Dry-run: router_loss_only=True but no router_loss (expected), creating dummy loss from router params")
                                # For dry run, create a dummy loss from router parameters to test computation graph
                                # This is needed because ce_loss comes from frozen lm_head and doesn't require grad
                                dummy_loss = None
                                router_param_found = False
                                
                                # Try to get router logits from the forward pass first (cleaner approach)
                                for name, module in self.model.named_modules():
                                    if hasattr(module, '_latest_router_logits'):
                                        router_logits = module._latest_router_logits
                                        if router_logits is not None and isinstance(router_logits, torch.Tensor):
                                            # Create a dummy loss from router logits (they're part of computation graph)
                                            try:
                                                if isinstance(router_logits, DTensor):
                                                    router_logits = get_full_tensor(router_logits)
                                                # Create a dummy loss: sum of logits * tiny constant
                                                dummy_loss = router_logits.sum() * 1e-8
                                                if dummy_loss.requires_grad:
                                                    log.debug(f"Dry-run: Created dummy loss from router logits in '{name}'")
                                                    break
                                            except Exception as e:
                                                log.debug(f"Dry-run: Failed to create dummy loss from router logits in '{name}': {e}")
                                
                                # Fallback: try to create dummy loss from router parameters
                                if dummy_loss is None or not dummy_loss.requires_grad:
                                    for name, param in self.model.named_parameters():
                                        if "router" in name.lower():
                                            router_param_found = True
                                            if param.requires_grad and param.numel() > 0:
                                                try:
                                                    # Create a dummy loss that requires grad from router parameters
                                                    dummy_loss = param.sum() * 1e-8
                                                    if dummy_loss.requires_grad:
                                                        log.debug(f"Dry-run: Created dummy loss from router param '{name}' (shape={param.shape})")
                                                        break
                                                except Exception as e:
                                                    log.debug(f"Dry-run: Failed to create dummy loss from '{name}': {e}")
                                                    continue
                                
                                if dummy_loss is not None and dummy_loss.requires_grad:
                                    loss = dummy_loss
                                else:
                                    if not router_param_found:
                                        log.warning("Dry-run: No router parameters found in model (checking for 'router' in name)")
                                    else:
                                        log.warning("Dry-run: Found router parameters but none have requires_grad=True or are empty")
                                    # Fallback: skip backward for dry run if no valid router params found
                                    loss = None
                            else:
                                raise RuntimeError("router_loss_only=True but router_loss not found in auxiliary losses")
                        else:
                            loss = router_loss_from_aux
                
                # Update batch losses for logging (after we've determined final loss)
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss
                
                # Backward pass (skip if loss is None, which can happen during dry run)
                if loss is not None and isinstance(loss, torch.Tensor):
                    loss.backward()
                elif micro_batch_idx == 0 and self.trainer.global_step < 1:
                    log.warning(f"[Router Training] Skipping backward: loss is None or not a tensor")

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
        # Log router batch loss value for debugging (occasionally)
        if isinstance(router_batch_loss, torch.Tensor):
            router_loss_val = router_batch_loss.item()
            if router_loss_val > 0:
                self.record_metric(
                    "Router loss",
                    router_batch_loss,
                    ReduceType.mean,
                    namespace="train",
                )
            elif self.trainer.global_step % 100 == 0:
                # Log warning if router loss is 0 (shouldn't happen when router_loss_only=True)
                log.warning(f"[Router Training] router_batch_loss is 0 (step={self.trainer.global_step})")
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
            metrics = compute_metrics_fn(reset=True)
            for metric_name, (metric_val, reduction) in metrics.items():
                self.record_metric(
                    metric_name,
                    metric_val,
                    reduction,
                    namespace="train",
                )
        if isinstance(self.optim, SkipStepOptimizer):
            # latest_loss expects a Tensor, ensure we have one
            if self.router_loss_only:
                if isinstance(router_batch_loss, torch.Tensor):
                    self.optim.latest_loss = router_batch_loss
                else:
                    # Fallback to ce_batch_loss if router_batch_loss is not a tensor
                    self.optim.latest_loss = ce_batch_loss
            else:
                self.optim.latest_loss = ce_batch_loss
