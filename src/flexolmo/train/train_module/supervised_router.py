"""
Supervised Router Training Module for 4x7B MoE models.

This module implements supervised router training where ground truth expert labels
are provided and used to train the router via cross-entropy loss on router logits.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

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
from torch.distributed.tensor import DTensor, distribute_tensor

from flexolmo.train.train_module.transformer import distribute_like
from flexolmo.data.expert_label_utils import get_expert_label_tensor

log = logging.getLogger(__name__)


@dataclass
class SupervisedRouterTrainModuleConfig(TransformerTrainModuleConfig):
    """Configuration for supervised router training."""
    
    router_loss_weight: float = 1.0
    """Weight for the supervised router loss."""
    
    router_loss_only: bool = False
    """If True, only train router (set language modeling loss weight to 0)."""
    
    use_domain_labels: bool = True
    """If True, extract expert labels from domain labels in batch metadata (fallback)."""
    
    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
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
            **kwargs,
        )


class SupervisedRouterTrainModule(TransformerTrainModule):
    """
    Training module for supervised router training.
    
    This module computes cross-entropy loss between router logits and ground truth
    expert labels. Expert labels are provided via batch["expert_labels"] (injected by
    ExpertLabelDataLoaderWrapper) or batch["domain_labels"] as a fallback.
    """

    def __init__(
        self,
        router_loss_weight: float = 1.0,
        router_loss_only: bool = False,
        use_domain_labels: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.router_loss_weight = router_loss_weight
        self.router_loss_only = router_loss_only
        self.use_domain_labels = use_domain_labels
        
        # Log model structure for debugging
        log.info(f"SupervisedRouterTrainModule initialized")
        log.info(f"  Model type: {type(self.model)}")
        log.info(f"  Has blocks: {hasattr(self.model, 'blocks')}")
        if hasattr(self.model, 'blocks'):
            try:
                blocks_list = list(self.model.blocks) if hasattr(self.model.blocks, '__iter__') else []
                log.info(f"  Number of blocks: {len(blocks_list)}")
                if len(blocks_list) > 0:
                    first_block = blocks_list[0]
                    log.info(f"  First block type: {type(first_block)}")
                    block_attrs = [attr for attr in dir(first_block) if not attr.startswith('_')]
                    moe_related = [attr for attr in block_attrs if 'moe' in attr.lower() or 'expert' in attr.lower() or 'router' in attr.lower() or 'feed_forward' in attr.lower()]
                    log.info(f"  First block MoE-related attributes: {moe_related}")
            except Exception as e:
                log.warning(f"  Could not inspect model blocks: {e}")

    def _extract_expert_labels_from_batch(self, batch: Dict[str, Any], batch_size: int) -> Optional[torch.Tensor]:
        """
        Extract expert labels from batch.
        
        Primary method: batch["expert_labels"] (injected by ExpertLabelDataLoaderWrapper)
        Fallback: batch["domain_labels"] if use_domain_labels is True
        
        Returns:
            Expert labels tensor of shape (batch_size, 4) or None
        """
        # Primary method: Direct expert labels in batch (from ExpertLabelDataLoaderWrapper)
        if "expert_labels" in batch:
            expert_labels = batch["expert_labels"]
            if isinstance(expert_labels, torch.Tensor):
                return expert_labels.to(self.device)
        
        # Fallback: Domain labels that we convert to expert labels
        if self.use_domain_labels and "domain_labels" in batch:
            domain_labels = batch["domain_labels"]
            if isinstance(domain_labels, (list, tuple)):
                expert_labels_list = [get_expert_label_tensor(str(d)) for d in domain_labels]
                return torch.stack(expert_labels_list, dim=0).to(self.device)
        
        return None

    def _compute_router_loss(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor,
        num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised cross-entropy loss for router.
        
        Args:
            router_logits: Router logits of shape (num_moe_layers, batch_size * seq_len, num_experts)
                or (batch_size * seq_len, num_experts) if single layer
            expert_labels: Ground truth expert labels of shape (batch_size, num_experts) (one-hot)
            num_tokens: Number of tokens in the batch (for normalization)
        
        Returns:
            Router loss (scalar tensor)
        """
        # Handle different router_logits shapes
        # CRITICAL: Ensure router_logits is fully materialized before any operations
        # FSDP may return tensors with storage issues that cause problems during backward
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        # Force materialization by adding 0 (preserves gradients, forces storage allocation)
        router_logits = (router_logits + 0.0).contiguous()
        
        if router_logits.dim() == 3:
            # (num_layers, batch_size * seq_len, num_experts)
            # Average across layers for simplicity (or sum, depending on preference)
            # For FSDP, we need to ensure the mean operation creates a properly materialized tensor
            # Use sum and divide instead of mean to avoid potential view issues
            router_logits = router_logits.sum(dim=0) / router_logits.shape[0]  # (batch_size * seq_len, num_experts)
            # Force materialization again after reduction
            router_logits = (router_logits + 0.0).contiguous()
        elif router_logits.dim() != 2:
            raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")
        
        batch_size = expert_labels.shape[0]
        seq_len = router_logits.shape[0] // batch_size
        
        # Ensure expert_labels is on the same device as router_logits and is contiguous
        expert_labels = expert_labels.to(router_logits.device)
        
        # CRITICAL: Materialize expert_labels to ensure proper storage
        # This prevents storage allocation issues during backward pass
        expert_labels = (expert_labels + 0.0).contiguous()
        
        # Expand expert_labels to match sequence length
        # expert_labels: (batch_size, num_experts) -> (batch_size * seq_len, num_experts)
        # Use repeat and ensure contiguous to avoid view issues with FSDP
        expert_labels_expanded = expert_labels.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, num_experts)
        expert_labels_expanded = expert_labels_expanded.reshape(-1, expert_labels.shape[-1])  # (batch_size * seq_len, num_experts)
        
        # CRITICAL: Materialize expanded labels to ensure proper storage
        expert_labels_expanded = (expert_labels_expanded + 0.0).contiguous()

        # ---- Defensive detach / clone with gradient bridge -------------------
        # Some FSDP-sharded bf16 tensors are still views on zero-sized storage.
        # We isolate them by:
        #   1. Detaching so autograd never tries to write into the old storage.
        #   2. clone() + cast → fresh FP32 storage that is safe for CE.
        #   3. Re-enable grad and register a hook that back-propagates the gradient
        #      to the original `router_logits` tensor.

        router_logits_safe = (
            router_logits
            .detach()               # cut graph to problematic view
            .clone()                # fresh dense storage
            .to(torch.float32)      # numerically stable dtype
            .requires_grad_(True)
        )

        # Bridge gradients: when grad w.r.t. safe tensor is produced, send it to
        # the original tensor so router weights still learn.
        def _bridge_grad(grad: torch.Tensor):  # grad has dtype fp32
            # Cast back to original dtype to match router_logits
            router_logits.backward(grad.to(router_logits.dtype))
            # We return None because we manually handled the gradient.
            return None

        router_logits_safe.register_hook(_bridge_grad)
        
        # ----------------------------------------------------------------------
        # Compute cross-entropy loss
        # router_logits: (batch_size * seq_len, num_experts)
        # expert_labels_expanded: (batch_size * seq_len, num_experts)
        # Convert one-hot to class indices for cross_entropy
        expert_indices = expert_labels_expanded.argmax(dim=-1)  # (batch_size * seq_len,)
        
        # Ensure expert_indices is contiguous and on the same device as router_logits_safe
        expert_indices = expert_indices.to(router_logits_safe.device).contiguous()
        
        # For FSDP compatibility, clone expert_indices to ensure proper storage
        # This is safe since expert_indices are target labels and don't need gradients
        expert_indices = expert_indices.clone().detach().long()
        
        router_loss = F.cross_entropy(
            router_logits_safe,
            expert_indices,
            reduction="sum",
        ) / num_tokens
        
        return router_loss

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """Train on a batch with supervised router loss."""
        self.model.train()

        batch_size = batch["input_ids"].shape[0]
        
        # Log batch structure on first call (for debugging)
        if not hasattr(self, '_first_batch_logged'):
            self._first_batch_logged = True
            log.info(f"SupervisedRouterTrainModule.train_batch: First batch received")
            log.info(f"  Batch keys: {list(batch.keys())}")
            log.info(f"  Batch size: {batch_size}")
            log.info(f"  Has expert_labels: {'expert_labels' in batch}")
            log.info(f"  Has metadata: {'metadata' in batch}")
            if 'metadata' in batch:
                metadata = batch['metadata']
                log.info(f"  Metadata type: {type(metadata)}, length: {len(metadata) if isinstance(metadata, (list, tuple)) else 'N/A'}")
                if isinstance(metadata, (list, tuple)) and len(metadata) > 0:
                    log.info(f"  First metadata entry: {metadata[0]}")
        
        # Extract expert labels if present
        expert_labels = self._extract_expert_labels_from_batch(batch, batch_size)
        
        # If router_loss_only=True, we MUST have expert labels
        # If not found, use default general expert labels as fallback
        if expert_labels is None:
            if self.router_loss_only:
                log.warning(
                    f"router_loss_only=True but expert_labels not found in batch. "
                    f"Batch keys: {list(batch.keys())}. "
                    f"Using default general expert labels as fallback."
                )
                # Use general expert (Expert 1: [0, 1, 0, 0]) as default
                from flexolmo.data.expert_label_utils import get_expert_label_tensor
                # Create expert labels on the correct device and ensure they're contiguous
                expert_labels_list = [get_expert_label_tensor("general") for _ in range(batch_size)]
                expert_labels = torch.stack(expert_labels_list, dim=0).to(self.device).contiguous()
            else:
                log.debug("No expert labels found in batch, skipping router loss")
        
        if expert_labels is not None:
            expert_labels = move_to_device(expert_labels, self.device)

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

                # Extract expert labels for this micro-batch if available
                micro_expert_labels = None
                if expert_labels is not None:
                    micro_batch_size = input_ids.shape[0]
                    start_idx = micro_batch_idx * micro_batch_size
                    end_idx = start_idx + micro_batch_size
                    micro_expert_labels = expert_labels[start_idx:end_idx]
                
                # If router_loss_only=True, we must have expert labels (should be set above, but double-check)
                if self.router_loss_only and micro_expert_labels is None:
                    log.warning(
                        f"router_loss_only=True but micro_expert_labels is None for micro_batch {micro_batch_idx}. "
                        f"This should not happen if expert_labels was set above."
                    )
                    # Use default general expert as fallback
                    from flexolmo.data.expert_label_utils import get_expert_label_tensor
                    micro_batch_size = input_ids.shape[0]
                    expert_labels_list = [get_expert_label_tensor("general") for _ in range(micro_batch_size)]
                    micro_expert_labels = torch.stack(expert_labels_list, dim=0).to(self.device).contiguous()

                # Run forward pass with router logits if we have expert labels OR if router_loss_only
                # (we need router logits to compute router loss)
                router_logits = None
                should_capture_router_logits = (micro_expert_labels is not None) or self.router_loss_only
                
                if should_capture_router_logits:
                    # Store router logits using a forward hook
                    router_logits_list = []
                    
                    def router_hook(module, input, output):
                        # The router forward returns: (logits, scores, expert_weights, expert_indices, batch_size_per_expert)
                        if isinstance(output, tuple) and len(output) >= 1:
                            logits = output[0]
                            # Ensure logits are a tensor and part of computation graph
                            if isinstance(logits, torch.Tensor):
                                # Don't clone - we need to keep the computation graph intact for gradients
                                # Just ensure it's contiguous (handled later when stacking)
                                if logits.requires_grad:
                                    router_logits_list.append(logits)
                                else:
                                    # If logits don't require grad, we still need them for loss computation
                                    # but we should log a warning
                                    log.warning(f"Router logits from {module} don't require grad, but capturing anyway")
                                    router_logits_list.append(logits)
                            else:
                                log.warning(f"Router output[0] is not a tensor: {type(logits)}")
                    
                    # Register hooks on all router modules
                    # Handle FSDP-wrapped models - need to access the underlying module
                    hooks = []
                    blocks_checked = 0
                    
                    # Get the actual model (unwrap if FSDP wrapped)
                    # FSDP wraps models, so we need to access the underlying module
                    model_to_inspect = self.model
                    
                    # Try different ways to unwrap FSDP
                    if hasattr(self.model, '_fsdp_wrapped_module'):
                        model_to_inspect = self.model._fsdp_wrapped_module
                    elif hasattr(self.model, 'module'):
                        model_to_inspect = self.model.module
                    elif hasattr(self.model, '_orig_mod'):
                        model_to_inspect = self.model._orig_mod
                    
                    # Use named_modules to find router modules (works with FSDP)
                    # This is more reliable than trying to unwrap manually
                    blocks = []
                    router_modules = []
                    
                    # Search for router modules using named_modules
                    for name, module in self.model.named_modules():
                        # Look for router modules directly
                        if 'router' in name.lower() and hasattr(module, 'forward'):
                            router_modules.append((name, module))
                        # Also collect blocks for fallback
                        if 'blocks' in name and '.' not in name.split('blocks')[0]:  # Top-level blocks
                            try:
                                if isinstance(module, (list, tuple)):
                                    blocks.extend([m for m in module if isinstance(m, torch.nn.Module)])
                                elif hasattr(module, '__iter__') and not isinstance(module, (str, bytes)):
                                    # Type checker doesn't know module is iterable, but we check at runtime
                                    blocks.extend([m for m in module if isinstance(m, torch.nn.Module)])  # type: ignore
                            except Exception:
                                pass
                    
                    # If we found router modules directly, use those
                    if router_modules:
                        log.debug(f"Found {len(router_modules)} router modules via named_modules")
                        for name, router_module in router_modules:
                            hook = router_module.register_forward_hook(router_hook)  # type: ignore
                            hooks.append(hook)
                        blocks_checked = len(router_modules)
                    # Otherwise, try to get blocks and find routers within them
                    elif hasattr(model_to_inspect, 'blocks'):
                        blocks_attr = getattr(model_to_inspect, 'blocks')
                        # If it's a ModuleList or similar, iterate directly
                        if hasattr(blocks_attr, '__iter__') and not isinstance(blocks_attr, (str, bytes)):
                            try:
                                blocks = [m for m in blocks_attr if isinstance(m, torch.nn.Module)]
                            except Exception:
                                blocks = []
                    
                    # If we already registered hooks from named_modules, skip block iteration
                    if not router_modules and blocks:
                        for i, block in enumerate(blocks):
                            blocks_checked += 1
                            
                            # Skip if block is not a module (e.g., string)
                            if not isinstance(block, torch.nn.Module):
                                continue
                            
                            # Try multiple ways to find the MoE router
                            router = None
                            
                            # Method 1: feed_forward_moe.router (standard OLMoE structure)
                            if hasattr(block, "feed_forward_moe"):
                                feed_forward_moe = getattr(block, "feed_forward_moe")
                                if feed_forward_moe is not None and hasattr(feed_forward_moe, "router"):
                                    router = getattr(feed_forward_moe, "router")
                            
                            # Method 2: block_sparse_moe.router (alternative structure)
                            if router is None and hasattr(block, "block_sparse_moe"):
                                block_sparse_moe = getattr(block, "block_sparse_moe")
                                if block_sparse_moe is not None and hasattr(block_sparse_moe, "router"):
                                    router = getattr(block_sparse_moe, "router")
                            
                            # Method 3: moe.router (another alternative)
                            if router is None and hasattr(block, "moe"):
                                moe = getattr(block, "moe")
                                if moe is not None and hasattr(moe, "router"):
                                    router = getattr(moe, "router")
                            
                            # Method 4: feed_forward.router (if feed_forward is MoE)
                            if router is None and hasattr(block, "feed_forward"):
                                feed_forward = getattr(block, "feed_forward")
                                if feed_forward is not None and hasattr(feed_forward, "router"):
                                    router = getattr(feed_forward, "router")
                            
                            if router is not None and isinstance(router, torch.nn.Module):
                                hook = router.register_forward_hook(router_hook)  # type: ignore
                                hooks.append(hook)
                    
                    if len(hooks) == 0:
                        # Log detailed debugging info
                        log.error(
                            f"No router modules found in model blocks after checking {blocks_checked} blocks. "
                            f"Model type: {type(self.model)}, "
                            f"Has blocks: {hasattr(self.model, 'blocks')}, "
                            f"Number of blocks: {len(self.model.blocks) if hasattr(self.model, 'blocks') else 'N/A'}"
                        )
                        # Log attributes of first block for debugging
                        if hasattr(self.model, 'blocks'):
                            blocks_list = list(self.model.blocks) if hasattr(self.model.blocks, '__iter__') else []
                            if len(blocks_list) > 0:
                                first_block = blocks_list[0]
                                block_attrs = [attr for attr in dir(first_block) if not attr.startswith('_')]
                                moe_related = [attr for attr in block_attrs if 'moe' in attr.lower() or 'expert' in attr.lower() or 'router' in attr.lower()]
                                log.error(f"First block attributes (MoE-related): {moe_related}")
                                log.error(f"First block type: {type(first_block)}")
                    else:
                        log.debug(f"Registered {len(hooks)} router hooks on {blocks_checked} blocks")
                    
                    # Forward pass
                    output_dict, ce_loss, z_loss = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    
                    # Stack router logits from all layers
                    if router_logits_list:
                        # Validate router logits before stacking
                        valid_logits = []
                        for i, logits in enumerate(router_logits_list):
                            if logits is None:
                                log.warning(f"Router logits from layer {i} are None, skipping")
                                continue
                            if not isinstance(logits, torch.Tensor):
                                log.warning(f"Router logits from layer {i} are not a tensor (type: {type(logits)}), skipping")
                                continue
                            if logits.numel() == 0:
                                log.warning(f"Router logits from layer {i} are empty, skipping")
                                continue
                            # Ensure logits are part of computation graph (not detached)
                            if not logits.requires_grad:
                                log.warning(f"Router logits from layer {i} don't require grad, but continuing anyway")
                            valid_logits.append(logits)
                        
                        if not valid_logits:
                            log.error("No valid router logits captured after validation")
                            router_logits = None
                        else:
                            try:
                                # Log shapes before stacking
                                log.info(f"Stacking {len(valid_logits)} router logits with shapes: {[l.shape for l in valid_logits]}")
                                
                                # Check devices - all should be on the same device
                                devices = [l.device for l in valid_logits]
                                unique_devices = set(devices)
                                if len(unique_devices) > 1:
                                    log.warning(f"Router logits are on different devices: {unique_devices}. Moving all to {self.device}")
                                    valid_logits = [l.to(self.device) for l in valid_logits]
                                
                                # CRITICAL: Materialize router logits as full tensors (not views/sharded)
                                # FSDP may return sharded tensors or views that cause storage issues during backward
                                # We need to ensure they're fully materialized tensors with proper storage
                                materialized_logits = []
                                for i, logits in enumerate(valid_logits):
                                    try:
                                        # For FSDP/DTensor compatibility, use get_full_tensor to ensure
                                        # we have a properly materialized tensor with correct storage
                                        if isinstance(logits, DTensor):
                                            # If it's a DTensor, get the full tensor
                                            materialized = get_full_tensor(logits)
                                        else:
                                            # For regular tensors, ensure they're materialized
                                            # Use an identity operation (add 0) to force materialization
                                            # while preserving the computation graph and gradients
                                            materialized = (logits + 0.0).contiguous()
                                        
                                        # Ensure it's on the correct device
                                        if materialized.device != self.device:
                                            materialized = materialized.to(self.device)
                                        
                                        materialized_logits.append(materialized)
                                    except Exception as e:
                                        log.warning(f"Error materializing logits from layer {i}: {e}, using original")
                                        # Fallback: ensure contiguous
                                        materialized_logits.append(logits.contiguous() if not logits.is_contiguous() else logits)
                                
                                # Stack router logits
                                router_logits = torch.stack(materialized_logits, dim=0)  # (num_layers, batch*seq_len, num_experts)
                                
                                # Ensure stacked tensor is contiguous and on correct device
                                router_logits = router_logits.contiguous().to(self.device)
                                
                                log.info(f"Stacked router logits shape: {router_logits.shape}, dtype: {router_logits.dtype}, device: {router_logits.device}")
                                log.info(f"Router logits requires_grad: {router_logits.requires_grad}, is_leaf: {router_logits.is_leaf}, is_contiguous: {router_logits.is_contiguous()}")
                            except Exception as e:
                                log.error(f"Error stacking router logits: {e}")
                                log.error(f"Router logits shapes: {[l.shape for l in valid_logits]}")
                                log.error(f"Router logits dtypes: {[l.dtype for l in valid_logits]}")
                                log.error(f"Router logits devices: {[l.device for l in valid_logits]}")
                                router_logits = None
                    else:
                        router_logits = None
                    
                    if router_logits is None and self.router_loss_only:
                        # This is a critical error - we can't train router without router logits
                        raise RuntimeError(
                            f"router_loss_only=True but no router logits captured. "
                            f"This likely means the model is not a MoE model or the router modules weren't found. "
                            f"Checked {blocks_checked} blocks, registered {len(hooks)} hooks. "
                            f"Please verify: "
                            f"1. The checkpoint is from a MoE model (not dense), "
                            f"2. The model config has num_experts > 1, "
                            f"3. The model blocks have MoE layers with router modules."
                        )
                else:
                    # Standard forward pass without router logits
                    output_dict, ce_loss, z_loss = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )

                # Compute supervised router loss if expert labels are provided
                router_loss = None
                if micro_expert_labels is not None and router_logits is not None:
                    router_loss = self._compute_router_loss(
                        router_logits,
                        micro_expert_labels,
                        batch_num_tokens_for_loss,
                    )
                    if router_loss is not None:
                        router_batch_loss += get_local_tensor(router_loss.detach())
                        # CRITICAL: Materialize router_loss to ensure proper storage for backward pass
                        # This prevents "storage of size 0" errors during backward
                        router_loss = (router_loss + 0.0).contiguous()
                
                # Combine losses
                # Critical: loss must always be part of the computation graph (never a leaf tensor)
                # This is essential for torch.compile to work correctly during backward pass
                if self.router_loss_only:
                    # Router-only training: must have router_loss
                    if router_loss is None:
                        raise RuntimeError(
                            f"router_loss_only=True but router_loss is None. "
                            f"expert_labels available: {micro_expert_labels is not None}, "
                            f"router_logits available: {router_logits is not None}"
                        )
                    # router_loss comes from cross_entropy with router_logits, so it's in the graph
                    # Use multiplication (not in-place) to ensure it stays in graph
                    loss = router_loss * self.router_loss_weight
                    # Ensure final loss is materialized
                    loss = (loss + 0.0).contiguous()
                else:
                    # Standard training: start with CE loss (always in graph)
                    loss = ce_loss
                    if z_loss is not None:
                        loss = loss + z_loss
                    # Add router loss if available
                    if router_loss is not None:
                        loss = loss + (router_loss * self.router_loss_weight)

                # Update batch losses
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                # Get auxiliary losses
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

                # Backward pass
                loss.backward()

        del batch

        if dry_run:
            self.model.reset_auxiliary_losses()
            self.model.reset_auxiliary_metrics()
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

        # Additional metrics
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
            self.optim.latest_loss = ce_batch_loss if not self.router_loss_only else router_batch_loss
