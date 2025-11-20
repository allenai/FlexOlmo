"""
Supervised Router Training Module for 4x7B MoE models.

This module implements supervised router training where ground truth expert labels
are provided and used to train the router via cross-entropy loss on router logits.
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
        dataset=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.router_loss_weight = router_loss_weight
        self.router_loss_only = router_loss_only
        self.use_domain_labels = use_domain_labels
        self.dataset = dataset
        
        # Build instance index to source_name mapping from dataset metadata
        self._instance_to_source = {}
        if dataset is not None and hasattr(dataset, 'metadata') and dataset.metadata:
            log.info(f"Building instance → source_name mapping from dataset metadata ({len(dataset.metadata)} entries)")
            for idx, metadata in enumerate(dataset.metadata):
                if isinstance(metadata, dict) and 'source_name' in metadata:
                    self._instance_to_source[idx] = metadata['source_name']
            log.info(f"Built mapping for {len(self._instance_to_source)} instances")
            # Show sample mappings
            sample_sources = list(set(self._instance_to_source.values()))[:10]
            log.info(f"Sample sources in mapping: {sample_sources}")
        else:
            log.warning("No dataset metadata available - will use fallback expert labels")
        
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
        
        # NOTE: olmo-core now has two changes for supervised router training:
        # 1. Dataset.__getitem__() includes 'index' field (commit 71e621140a8331b5621d67444b1fb5930a07acdc)
        # 2. Transformer._prepare_inputs() preserves metadata/index/expert_labels (commit 2484e39...)
        # These ensure index/metadata flow through: Dataset → Collator → Model → Forward pass
        log.info("✅ Using olmo-core with dataset 'index' field and _prepare_inputs() preservation")
    
    def _prepare_batch(self, batch: Dict[str, Any]):  # type: ignore[override]
        """
        Override parent's _prepare_batch to preserve metadata and index fields.
        
        This ensures metadata/index/expert_labels are passed to model_kwargs, which
        are then preserved by olmo-core's Transformer._prepare_inputs() and made available
        throughout the forward pass for supervised router training.
        """
        # Preserve metadata and index BEFORE calling parent
        preserved_metadata = batch.get('metadata')
        preserved_index = batch.get('index')
        preserved_expert_labels = batch.get('expert_labels')
        
        # Call parent to get standard preprocessing
        input_ids, labels, model_kwargs = super()._prepare_batch(batch)
        
        # Add preserved fields back to model_kwargs
        if preserved_metadata is not None:
            model_kwargs['metadata'] = preserved_metadata
        if preserved_index is not None:
            model_kwargs['index'] = preserved_index
        if preserved_expert_labels is not None:
            model_kwargs['expert_labels'] = preserved_expert_labels
        
        return input_ids, labels, model_kwargs

    def _extract_expert_labels_from_batch(self, batch: Dict[str, Any], batch_size: int) -> Optional[torch.Tensor]:
        """
        Extract expert labels from batch.
        
        Methods (in priority order):
        1. batch["expert_labels"] - Direct labels (tensor)
        2. batch["expert_label"] - Per-item labels from dataset wrapper (list of tensors)
        3. batch["domain_labels"] - Domain names to convert
        
        Returns:
            Expert labels tensor of shape (batch_size, 4) or None
        """
        # Method 1: Direct expert labels tensor in batch
        if "expert_labels" in batch:
            expert_labels = batch["expert_labels"]
            if isinstance(expert_labels, torch.Tensor):
                return expert_labels.to(self.device).contiguous()
        
        # Method 2: Use 'index' field (DataCollator preserves this!)
        # This is likely what we should use instead of instance_indices
        if self._instance_to_source and "index" in batch:
            index_data = batch["index"]
            if isinstance(index_data, torch.Tensor):
                indices = index_data.cpu().tolist()
            elif isinstance(index_data, (list, tuple)):
                indices = list(index_data)
            else:
                indices = None
            
            if indices and len(indices) == batch_size:
                expert_labels_list = []
                domain_labels = []
                
                for idx in indices:
                    source_name = self._instance_to_source.get(idx, 'general')
                    expert_label = get_expert_label_tensor(source_name)
                    expert_labels_list.append(expert_label)
                    domain_labels.append(source_name)
                
                if expert_labels_list:
                    log.info(f"✅ Extracted expert labels from batch['index']: {domain_labels[:3]}...")
                    return torch.stack(expert_labels_list, dim=0).to(self.device).contiguous()
        
        # Method 3: Per-item expert labels from dataset wrapper (if collator preserves it)
        # When DatasetWithExpertLabels is used, each item has 'expert_label'
        if "expert_label" in batch:
            expert_label_data = batch["expert_label"]
            # Could be a stacked tensor or a list of tensors
            if isinstance(expert_label_data, torch.Tensor):
                if expert_label_data.shape[0] == batch_size:
                    log.info(f"✅ Extracted expert labels from batch['expert_label'] (tensor)")
                    return expert_label_data.to(self.device).contiguous()
            elif isinstance(expert_label_data, (list, tuple)):
                if len(expert_label_data) == batch_size:
                    expert_labels = torch.stack(expert_label_data, dim=0)
                    log.info(f"✅ Extracted expert labels from batch['expert_label'] (list)")
                    return expert_labels.to(self.device).contiguous()
        
        # Method 3: Domain labels that we convert to expert labels (fallback)
        if self.use_domain_labels and "domain_labels" in batch:
            domain_labels = batch["domain_labels"]
            if isinstance(domain_labels, (list, tuple)):
                expert_labels_list = [get_expert_label_tensor(str(d)) for d in domain_labels]
                return torch.stack(expert_labels_list, dim=0).to(self.device).contiguous()
        
        # Method 4: Domain label (singular) from dataset wrapper
        if "domain_label" in batch:
            domain_label_data = batch["domain_label"]
            if isinstance(domain_label_data, (list, tuple)) and len(domain_label_data) == batch_size:
                expert_labels_list = [get_expert_label_tensor(str(d)) for d in domain_label_data]
                log.debug(f"Extracted expert labels from batch['domain_label']")
                return torch.stack(expert_labels_list, dim=0).to(self.device).contiguous()
        
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
        # Handle different router_logits shapes - SIMPLIFIED (no aggressive materialization)
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        if router_logits.dim() == 3:
            # (num_layers, batch_size * seq_len, num_experts)
            # Average across layers
            router_logits = router_logits.mean(dim=0)  # (batch_size * seq_len, num_experts)
        elif router_logits.dim() != 2:
            raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")
        
        batch_size = expert_labels.shape[0]
        seq_len = router_logits.shape[0] // batch_size
        
        # Move expert_labels to same device and ensure it's contiguous
        expert_labels = expert_labels.to(router_logits.device).contiguous()
        
        # Expand expert_labels to match all tokens in sequence
        # Instead of using `expand` (which can create tensors with unusual strides and
        # trigger "storage size 0" errors after `.contiguous()` in some PyTorch
        # builds), repeat the *class indices* along the sequence dimension. This keeps
        # the underlying storage layout simple and robust across versions.

        # Step 1. Convert one-hot labels to class indices – shape: `(batch_size,)`.
        expert_indices_per_item = expert_labels.argmax(dim=-1).long()

        # Step 2. Repeat each item-level label `seq_len` times so we have one label per
        # token.  The resulting tensor has shape `(batch_size * seq_len,)`.
        expert_indices = expert_indices_per_item.repeat_interleave(seq_len)

        # Sanity check (cheap, only on first call) – make sure the tensor is the
        # expected size to avoid silent shape mismatches later on.
        if expert_indices.numel() != batch_size * seq_len:
            raise RuntimeError(
                f"Unexpected expert_indices size {expert_indices.shape}, expected {(batch_size * seq_len,)}"
            )
        
        # Compute cross-entropy loss - NO cloning, keep in computation graph
        router_loss = F.cross_entropy(
            router_logits,  # Use original logits, not cloned
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
            log.info(f"  Has expert_label: {'expert_label' in batch}")
            log.info(f"  Has domain_label: {'domain_label' in batch}")
            log.info(f"  Has metadata: {'metadata' in batch}")
            log.info(f"  Has instance_indices: {'instance_indices' in batch}")
            log.info(f"  Has index: {'index' in batch}")  # DataCollator preserves this!
            log.info(f"  Have instance_to_source mapping: {len(self._instance_to_source) > 0 if hasattr(self, '_instance_to_source') else False}")
            
            # Show what expert_label looks like if present
            if 'expert_label' in batch:
                expert_label_data = batch['expert_label']
                log.info(f"  expert_label type: {type(expert_label_data)}, shape/len: {expert_label_data.shape if hasattr(expert_label_data, 'shape') else len(expert_label_data) if hasattr(expert_label_data, '__len__') else 'N/A'}")
                if isinstance(expert_label_data, torch.Tensor) and len(expert_label_data) > 0:
                    log.info(f"  First expert_label: {expert_label_data[0]}")
            
            if 'index' in batch:
                index_data = batch['index']
                log.info(f"  Index type: {type(index_data)}, shape/len: {index_data.shape if hasattr(index_data, 'shape') else len(index_data) if hasattr(index_data, '__len__') else 'N/A'}")
                if isinstance(index_data, torch.Tensor) and len(index_data) > 0:
                    sample_indices = index_data.cpu().tolist()[:3]
                    log.info(f"  First 3 indices: {sample_indices}")
                    if hasattr(self, '_instance_to_source') and self._instance_to_source:
                        sample_sources = [self._instance_to_source.get(idx, 'UNKNOWN') for idx in sample_indices]
                        log.info(f"  Their sources: {sample_sources}")
        
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
            # Ensure expert_labels is contiguous to avoid storage issues
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

                # Extract expert labels for this micro-batch if available
                micro_expert_labels = None
                if expert_labels is not None:
                    micro_batch_size = input_ids.shape[0]
                    start_idx = micro_batch_idx * micro_batch_size
                    end_idx = start_idx + micro_batch_size
                    micro_expert_labels = expert_labels[start_idx:end_idx].contiguous()
                
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
                router_loss: Optional[torch.Tensor] = None
                should_capture_router_logits = (micro_expert_labels is not None) or self.router_loss_only
                
                # Flag to skip CE/Z loss computation when only training router
                skip_lm_loss = self.router_loss_only and should_capture_router_logits
                
                if should_capture_router_logits:
                    router_loss_terms: List[torch.Tensor] = []
                    
                    def router_hook(module, input, output):
                        # The router forward returns: (logits, scores, expert_weights, expert_indices, batch_size_per_expert)
                        if isinstance(output, tuple) and len(output) >= 1:
                            logits = output[0]
                            # Ensure logits are a tensor and part of computation graph
                            if isinstance(logits, torch.Tensor):
                                if micro_expert_labels is None:
                                    log.warning("Router hook triggered without expert labels; skipping router loss term")
                                    return
                                
                                try:
                                    loss_term = self._compute_router_loss(
                                        logits,
                                        micro_expert_labels,
                                        batch_num_tokens_for_loss,
                                    )
                                    router_loss_terms.append(loss_term)
                                except Exception as exc:
                                    log.warning(
                                        "Failed to compute router loss for module %s: %s",
                                        module,
                                        exc,
                                    )
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
                    model_forward_result = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )
                    # Handle different return types from model_forward
                    if isinstance(model_forward_result, tuple):
                        if len(model_forward_result) == 3:
                            output_dict, ce_loss, z_loss = model_forward_result  # type: ignore[misc]
                        elif len(model_forward_result) == 4:
                            output_dict, ce_loss, z_loss, _ = model_forward_result  # type: ignore[misc]
                        else:
                            # Fallback: assume first 3 are what we need
                            output_dict, ce_loss, z_loss = model_forward_result[:3]  # type: ignore[misc]
                    else:
                        # Not a tuple - might be LMOutputWithLoss or similar
                        raise TypeError(f"Unexpected return type from model_forward: {type(model_forward_result)}")
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    
                    if router_loss_terms:
                        router_loss = torch.stack(router_loss_terms, dim=0).mean()
                        log.debug(
                            "Computed router loss from %s router modules; sample loss=%s",
                            len(router_loss_terms),
                            router_loss.detach().float().item()
                            if router_loss.requires_grad
                            else router_loss.item(),
                        )
                    elif self.router_loss_only:
                        raise RuntimeError(
                            f"router_loss_only=True but no router loss terms were computed. "
                            f"Checked {blocks_checked} blocks, registered {len(hooks)} hooks. "
                            f"Please verify that the model contains MoE router modules."
                        )
                else:
                    # Standard forward pass without router logits
                    model_forward_result = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )
                    # Handle different return types from model_forward
                    if isinstance(model_forward_result, tuple):
                        if len(model_forward_result) == 3:
                            output_dict, ce_loss, z_loss = model_forward_result  # type: ignore[misc]
                        elif len(model_forward_result) == 4:
                            output_dict, ce_loss, z_loss, _ = model_forward_result  # type: ignore[misc]
                        else:
                            # Fallback: assume first 3 are what we need
                            output_dict, ce_loss, z_loss = model_forward_result[:3]  # type: ignore[misc]
                    else:
                        # Not a tuple - might be LMOutputWithLoss or similar
                        raise TypeError(f"Unexpected return type from model_forward: {type(model_forward_result)}")

                # Accumulate supervised router loss if available (for logging)
                if router_loss is not None:
                    router_batch_loss += get_local_tensor(router_loss.detach())
                
                # Combine losses - KEEP IN COMPUTATION GRAPH (no .contiguous()!)
                if self.router_loss_only:
                    # Router-only training: must have router_loss
                    if router_loss is None:
                        raise RuntimeError(
                            f"router_loss_only=True but router_loss is None. "
                            f"expert_labels available: {micro_expert_labels is not None}"
                        )
                    # Use router loss directly (already in computation graph from F.cross_entropy)
                    loss = router_loss * self.router_loss_weight
                else:
                    # Standard training: start with CE loss
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

                # Get auxiliary losses (handle FSDP-wrapped models)
                # FSDP wrapping may hide methods, so we need to access the underlying module
                model_for_aux = self.model
                if hasattr(self.model, '_fsdp_wrapped_module'):
                    model_for_aux = self.model._fsdp_wrapped_module  # type: ignore[attr-defined]
                elif hasattr(self.model, 'module'):
                    model_for_aux = self.model.module  # type: ignore[attr-defined]
                elif hasattr(self.model, '_orig_mod'):
                    model_for_aux = self.model._orig_mod  # type: ignore[attr-defined]
                
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
            # Handle FSDP-wrapped models
            model_for_aux = self.model
            if hasattr(self.model, '_fsdp_wrapped_module'):
                model_for_aux = self.model._fsdp_wrapped_module  # type: ignore[attr-defined]
            elif hasattr(self.model, 'module'):
                model_for_aux = self.model.module  # type: ignore[attr-defined]
            elif hasattr(self.model, '_orig_mod'):
                model_for_aux = self.model._orig_mod  # type: ignore[attr-defined]
            
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

        # Additional metrics (handle FSDP-wrapped models)
        model_for_aux = self.model
        if hasattr(self.model, '_fsdp_wrapped_module'):
            model_for_aux = self.model._fsdp_wrapped_module  # type: ignore[attr-defined]
        elif hasattr(self.model, 'module'):
            model_for_aux = self.model.module  # type: ignore[attr-defined]
        elif hasattr(self.model, '_orig_mod'):
            model_for_aux = self.model._orig_mod  # type: ignore[attr-defined]
        
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
