"""Soft Label Router Training Module for MoE models.

Trains router via KL divergence between router probabilities and soft target distributions
derived from expert losses.

The soft labels are computed on-the-fly from pre-computed all_expert_losses:
    q(e) = softmax(-β * (L_e - min(L)))

This is knowledge distillation / distribution matching, not hard classification.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import re

import numpy as np
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


def compute_soft_labels(expert_losses: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Convert expert losses to soft target distribution.
    
    Args:
        expert_losses: Per-token expert losses, shape (seq_len, num_experts)
        beta: Temperature parameter (higher = sharper distribution)
    
    Returns:
        Soft target distribution, shape (seq_len, num_experts)
        q(e) = softmax(-β * (L_e - min(L)))
    """
    # Subtract min for numerical stability
    min_losses = expert_losses.min(dim=-1, keepdim=True).values
    normalized_losses = expert_losses - min_losses
    
    # Apply negative and temperature, then softmax
    # Lower loss = higher probability
    soft_labels = F.softmax(-beta * normalized_losses, dim=-1)
    
    return soft_labels


@dataclass
class SoftLabelRouterTrainModuleConfig(TransformerTrainModuleConfig):
    """Configuration for soft label router training."""
    
    router_loss_weight: float = 1.0
    """Weight for the soft label router loss."""
    
    router_loss_only: bool = True
    """If True, only train router (set language modeling loss weight to 0)."""
    
    supervised_layers: Optional[List[int]] = None
    """If set, only supervise routers in these layer indices. E.g., [0] for layer 0 only."""
    
    soft_label_beta: float = 1.0
    """Temperature parameter for soft labels. Higher = sharper (closer to hard labels)."""
    
    expert_labels_dir: str = ""
    """Directory containing per-token expert losses (seq_*.npz files with all_expert_losses)."""
    
    num_experts: int = 3
    """Number of experts to consider for soft labels."""
    
    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
        dataset=None,
        **extra_kwargs,
    ) -> "SoftLabelRouterTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return SoftLabelRouterTrainModule(
            model=model,
            device=device,
            dataset=dataset,
            **kwargs,
            **extra_kwargs,
        )


class SoftLabelRouterTrainModule(TransformerTrainModule):
    """Training module for soft label router training.
    
    Uses KL divergence between router probabilities and soft target distributions
    derived from expert losses:
        L_router = KL(q || p_θ) = -Σ_e q(e) log p_θ(e) + const
    
    Where q(e) = softmax(-β * (L_e - min(L)))
    """

    def __init__(
        self,
        router_loss_weight: float = 1.0,
        router_loss_only: bool = True,
        supervised_layers: Optional[List[int]] = None,
        soft_label_beta: float = 1.0,
        expert_labels_dir: str = "",
        num_experts: int = 3,
        dataset=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.router_loss_weight = router_loss_weight
        self.router_loss_only = router_loss_only
        self.supervised_layers = supervised_layers
        self.soft_label_beta = soft_label_beta
        self.expert_labels_dir = Path(expert_labels_dir) if expert_labels_dir else None
        self.num_experts = num_experts
        self.dataset = dataset
        
        # Cache for loaded expert losses
        self._expert_losses_cache: Dict[int, np.ndarray] = {}
        
        self._patch_routers_for_soft_loss()
        log.info(f"SoftLabelRouterTrainModule initialized:")
        log.info(f"  router_loss_only={router_loss_only}")
        log.info(f"  soft_label_beta={soft_label_beta}")
        log.info(f"  expert_labels_dir={expert_labels_dir}")
        log.info(f"  supervised_layers={supervised_layers}")
    
    def _load_expert_losses(self, seq_idx: int) -> Optional[np.ndarray]:
        """Load all_expert_losses for a given sequence index."""
        if self.expert_labels_dir is None:
            return None
        
        # Check cache first
        if seq_idx in self._expert_losses_cache:
            return self._expert_losses_cache[seq_idx]
        
        # Load from file
        label_file = self.expert_labels_dir / f"seq_{seq_idx:08d}.npz"
        if not label_file.exists():
            log.warning(f"Label file not found: {label_file}")
            return None
        
        try:
            data = np.load(label_file)
            if "all_expert_losses" not in data:
                log.warning(f"all_expert_losses not found in {label_file}")
                return None
            
            all_expert_losses = data["all_expert_losses"]  # (seq_len-1, num_experts)
            
            # Cache it (limit cache size)
            if len(self._expert_losses_cache) > 10000:
                # Clear half the cache when it gets too big
                keys = list(self._expert_losses_cache.keys())[:5000]
                for k in keys:
                    del self._expert_losses_cache[k]
            
            self._expert_losses_cache[seq_idx] = all_expert_losses
            return all_expert_losses
            
        except Exception as e:
            log.warning(f"Error loading {label_file}: {e}")
            return None
    
    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        """Extract layer index from module name like 'blocks.5.feed_forward_moe.router'."""
        match = re.search(r'blocks\.(\d+)\.', module_name)
        if match:
            return int(match.group(1))
        return None
    
    def _should_supervise_layer(self, layer_idx: Optional[int]) -> bool:
        """Check if this layer should be supervised based on supervised_layers config."""
        if self.supervised_layers is None:
            return True  # Supervise all layers if not specified
        if layer_idx is None:
            return False  # Can't determine layer, skip
        return layer_idx in self.supervised_layers
    
    def _patch_routers_for_soft_loss(self):
        """Patch router forward methods to compute soft label loss during forward."""
        self._current_soft_labels: Optional[torch.Tensor] = None
        self._router_soft_losses: List[float] = []
        
        patched_count = 0
        skipped_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, MoERouter):
                if not hasattr(module, 'num_experts') or not hasattr(module, '_latest_router_logits'):
                    log.warning(f"Skipping {name}: isinstance MoERouter but missing router attributes")
                    continue
                
                # Check if this layer should be supervised
                layer_idx = self._extract_layer_index(name)
                if not self._should_supervise_layer(layer_idx):
                    skipped_layers.append(layer_idx)
                    continue
                    
                original_forward = module.forward
                router_name = name
                
                def make_patched_forward(router, orig_fn, train_module_self, router_name_inner):
                    def patched_forward(x, *, loss_div_factor=None):
                        result = orig_fn(x, loss_div_factor=loss_div_factor)
                        expert_weights, expert_indices, batch_size_per_expert, aux_loss = result
                        
                        soft_labels = train_module_self._current_soft_labels
                        if soft_labels is not None and router.training and torch.is_grad_enabled():
                            router_logits = router._latest_router_logits
                            if router_logits is not None:
                                soft_loss = train_module_self._compute_soft_loss_during_forward(
                                    router_logits, soft_labels, loss_div_factor, router.num_experts
                                )
                                if soft_loss is not None:
                                    scaled_loss = train_module_self.router_loss_weight * soft_loss
                                    aux_loss = scaled_loss if aux_loss is None else aux_loss + scaled_loss
                                    train_module_self._router_soft_losses.append(get_local_tensor(scaled_loss.detach()).item())
                                    if not hasattr(train_module_self, '_logged_soft_loss'):
                                        train_module_self._logged_soft_loss = set()
                                    if router_name_inner not in train_module_self._logged_soft_loss and train_module_self.trainer.global_step <= 1:
                                        log.info(f"[Soft Router Training] Computed soft loss in router '{router_name_inner}': {scaled_loss.item():.6f}")
                                        train_module_self._logged_soft_loss.add(router_name_inner)
                        
                        return expert_weights, expert_indices, batch_size_per_expert, aux_loss
                    
                    return patched_forward
                
                module.forward = make_patched_forward(module, original_forward, self, router_name)
                patched_count += 1
        
        if skipped_layers:
            log.info(f"[Soft Router Training] Skipped supervision for layers: {sorted(set(skipped_layers))}")
        log.info(f"[Soft Router Training] Patched {patched_count} routers for soft loss computation")
    
    def _compute_soft_loss_during_forward(
        self,
        router_logits: torch.Tensor,
        soft_labels: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
        num_experts: int,
    ) -> Optional[torch.Tensor]:
        """
        Compute soft label (KL divergence) loss during forward pass.
        
        This is equivalent to soft cross-entropy:
            L = -Σ_e q(e) log p_θ(e)
        
        Where q(e) is the soft target distribution and p_θ(e) is the router's prediction.
        """
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        soft_labels = soft_labels.to(router_logits.device)
        
        # Get shapes
        if router_logits.dim() == 3:
            batch_size, seq_len, num_experts_logits = router_logits.shape
            router_logits_flat = router_logits.reshape(-1, num_experts_logits)  # (B*S, E)
        elif router_logits.dim() == 2:
            router_logits_flat = router_logits
            num_experts_logits = router_logits.shape[-1]
            batch_size = soft_labels.shape[0] if soft_labels.dim() >= 2 else 1
            seq_len = router_logits_flat.shape[0] // batch_size
        else:
            return None
        
        # Flatten soft labels: (B, S-1, E) -> (B*S-1, E)
        # Or if already flat: (B*(S-1), E)
        if soft_labels.dim() == 3:
            soft_labels_flat = soft_labels.reshape(-1, soft_labels.shape[-1])  # (B*(S-1), E)
        elif soft_labels.dim() == 2:
            soft_labels_flat = soft_labels
        else:
            return None
        
        num_logit_tokens = router_logits_flat.shape[0]
        num_label_tokens = soft_labels_flat.shape[0]
        
        # Align lengths (handle seq_len vs seq_len-1 offset)
        if num_label_tokens != num_logit_tokens:
            if not hasattr(self, '_logged_shape_align'):
                self._logged_shape_align = True
                log.warning(f"Shape alignment: labels={num_label_tokens} -> logits={num_logit_tokens}")
            
            if num_label_tokens > num_logit_tokens:
                soft_labels_flat = soft_labels_flat[:num_logit_tokens]
            else:
                # Pad with uniform distribution
                num_experts_labels = soft_labels_flat.shape[-1]
                padding = torch.ones(num_logit_tokens - num_label_tokens, num_experts_labels, 
                                   dtype=soft_labels_flat.dtype, device=soft_labels_flat.device) / num_experts_labels
                soft_labels_flat = torch.cat([soft_labels_flat, padding])
        
        # Ensure soft labels only cover the experts we have logits for
        if soft_labels_flat.shape[-1] > num_experts_logits:
            soft_labels_flat = soft_labels_flat[:, :num_experts_logits]
            # Re-normalize
            soft_labels_flat = soft_labels_flat / soft_labels_flat.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        elif soft_labels_flat.shape[-1] < num_experts_logits:
            # Pad with zeros
            padding = torch.zeros(soft_labels_flat.shape[0], num_experts_logits - soft_labels_flat.shape[-1],
                                dtype=soft_labels_flat.dtype, device=soft_labels_flat.device)
            soft_labels_flat = torch.cat([soft_labels_flat, padding], dim=-1)
        
        num_tokens = soft_labels_flat.shape[0]
        if num_tokens == 0:
            return None
        
        # Compute router log probabilities
        router_log_probs = F.log_softmax(router_logits_flat, dim=-1)  # (N, E)
        
        # Soft cross-entropy: -Σ_e q(e) log p_θ(e)
        # This is equivalent to KL(q || p) + H(q), where H(q) is constant w.r.t. parameters
        soft_ce_loss = -(soft_labels_flat * router_log_probs).sum(dim=-1)  # (N,)
        soft_ce_loss = soft_ce_loss.sum()  # Scalar
        
        if loss_div_factor is not None:
            soft_ce_loss = soft_ce_loss / (loss_div_factor if isinstance(loss_div_factor, torch.Tensor) else float(loss_div_factor))
        else:
            soft_ce_loss = soft_ce_loss / num_tokens
        
        return soft_ce_loss
    
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
    
    def _get_soft_labels_for_batch(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Get soft labels for a batch by loading expert losses and computing soft distribution.
        
        Returns:
            Soft labels tensor of shape (batch_size, seq_len-1, num_experts) or None
        """
        if self.expert_labels_dir is None:
            return None
        
        # Get sequence indices from batch
        indices = batch.get("index")
        if indices is None:
            log.warning("No 'index' in batch, cannot load expert losses")
            return None
        
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        batch_size = len(indices)
        
        # Load expert losses for each sequence
        all_soft_labels = []
        for seq_idx in indices:
            expert_losses = self._load_expert_losses(int(seq_idx))
            if expert_losses is None:
                # Use uniform distribution as fallback
                seq_len_minus_1 = 4095  # Assume 4096 sequence length
                expert_losses = np.ones((seq_len_minus_1, self.num_experts), dtype=np.float32)
            
            # Only use first num_experts experts
            expert_losses = expert_losses[:, :self.num_experts]
            
            # Convert to tensor and compute soft labels
            expert_losses_tensor = torch.from_numpy(expert_losses.astype(np.float32))
            soft_labels = compute_soft_labels(expert_losses_tensor, beta=self.soft_label_beta)
            all_soft_labels.append(soft_labels)
        
        # Stack into batch: (B, S-1, E)
        soft_labels_batch = torch.stack(all_soft_labels, dim=0)
        return soft_labels_batch
    
    def _prepare_batch(self, batch: Dict[str, Any]):  # type: ignore[override]
        """Preserve metadata, index, and compute soft labels. Clone tensors to avoid storage issues."""
        preserved = {k: batch.get(k) for k in ['metadata', 'index']}
        
        # Clone index tensor for soft label loading
        if 'index' in preserved and isinstance(preserved['index'], torch.Tensor):
            preserved['index'] = preserved['index'].clone()
        
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
        """Train on a batch with soft label router loss."""
        self.model.train()

        batch_size = batch["input_ids"].shape[0]
        
        # Get soft labels for this batch (skip during dry_run as mock batch won't have proper indices)
        soft_labels = None
        if not dry_run:
            soft_labels = self._get_soft_labels_for_batch(batch)
        has_soft_labels = soft_labels is not None
        
        if not hasattr(self, '_first_batch_logged'):
            self._first_batch_logged = True
            if dry_run:
                log.info(f"Dry-run batch: keys={list(batch.keys())}, has_soft_labels={has_soft_labels} (expected - mock batch)")
            else:
                log.info(f"First batch: keys={list(batch.keys())}, has_soft_labels={has_soft_labels}")
                if has_soft_labels:
                    log.info(f"  Soft labels shape: {soft_labels.shape}")
                    log.info(f"  Beta (temperature): {self.soft_label_beta}")
        
        if not has_soft_labels:
            if self.router_loss_only:
                if dry_run:
                    log.info("Dry-run batch: skipping router loss (mock batch). Real batches will have soft labels.")
                else:
                    raise RuntimeError(
                        "router_loss_only=True but soft labels could not be computed. "
                        f"Check expert_labels_dir: {self.expert_labels_dir}"
                    )
            else:
                log.debug("No soft labels found, skipping router loss")

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
        
        # Split soft labels to match micro-batches
        soft_labels_list: List[torch.Tensor] = []
        if has_soft_labels and soft_labels is not None:
            soft_labels_list = list(torch.split(soft_labels, self.rank_microbatch_size // seq_len, dim=0))

        for micro_batch_idx, micro_batch in enumerate(_micro_batch_refs):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Get soft labels for this micro-batch
                micro_soft_labels = None
                if has_soft_labels and soft_labels_list and micro_batch_idx < len(soft_labels_list):
                    micro_soft_labels = move_to_device(soft_labels_list[micro_batch_idx], self.device)
                
                if self.router_loss_only and micro_soft_labels is None:
                    if dry_run:
                        log.debug("Dry-run: skipping router loss computation (no soft_labels in mock batch)")
                    else:
                        raise RuntimeError(f"router_loss_only=True but missing soft_labels for micro_batch {micro_batch_idx}")

                self._current_soft_labels = micro_soft_labels
                
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
                
                self._current_soft_labels = None

                if self.router_loss_only:
                    loss = None
                else:
                    loss = ce_loss
                    if z_loss is not None:
                        loss = loss + z_loss
                
                # Collect soft loss values for logging/metrics
                should_log = micro_batch_idx == 0 and (self.trainer.global_step <= 1 or self.trainer.global_step % 100 == 0)
                if self._router_soft_losses:
                    router_loss_sum = sum(self._router_soft_losses)
                    router_batch_loss += router_loss_sum
                    if should_log:
                        log.info(f"[Soft Router Training] Collected soft loss from {len(self._router_soft_losses)} routers: {router_loss_sum:.6f}")
                    self._router_soft_losses.clear()
                
                # For router_loss_only mode, we need a loss that's downstream from the routers to trigger backward.
                if self.router_loss_only and loss is None:
                    if dry_run:
                        dummy_loss = self._create_dummy_loss_for_dry_run()
                        loss = dummy_loss if dummy_loss is not None and dummy_loss.requires_grad else None
                    else:
                        # Use a tiny fraction of the CE loss to trigger backward through the graph
                        if ce_loss is not None and ce_loss.requires_grad:
                            loss = ce_loss * 1e-10
                        else:
                            raise RuntimeError("router_loss_only=True but ce_loss is not available to trigger backward")
                
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss
                
                if loss is not None and isinstance(loss, torch.Tensor):
                    loss.backward()
                elif micro_batch_idx == 0 and self.trainer.global_step < 1:
                    log.warning(f"[Soft Router Training] Skipping backward: loss is None or not a tensor")

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            model_for_aux = _unwrap_fsdp_model(self.model)
            if hasattr(model_for_aux, 'reset_auxiliary_metrics'):
                model_for_aux.reset_auxiliary_metrics()  # type: ignore[attr-defined]
            return

        if not self.router_loss_only:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        
        # Record the soft router loss for tracking in wandb
        if isinstance(router_batch_loss, torch.Tensor):
            router_loss_val = router_batch_loss.item()
            self.record_metric("router loss (soft)", router_batch_loss, ReduceType.mean, namespace="train")
            
            if router_loss_val == 0 and self.trainer.global_step % 100 == 0:
                log.warning(f"[Soft Router Training] router_batch_loss is 0 at step {self.trainer.global_step}")

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
