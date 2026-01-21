"""CE-Constrained (Hard Label) Router Training Module for MoE models.

Combines standard router training (via LM loss) with cross-entropy regularization
towards hard domain-based labels (one-hot vectors).

This is a hybrid approach that allows the router to learn from task performance
while being gently guided by domain labels.

Total loss:
    L_total = L_LM + L_Z + λ * CE(hard_labels, router_logits)
    
Where:
    - L_LM: Language modeling (cross-entropy) loss
    - L_Z: Z-loss (router stability)
    - CE: Cross-entropy between hard one-hot labels and router predictions
    - λ: Weight for CE regularization (ce_loss_weight)
    - hard_labels: Domain-based one-hot vectors (math/code/general)
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


def get_hard_label_for_domain(domain: str, num_experts: int = 4) -> List[float]:
    """
    Get hard one-hot label for a domain.
    
    Args:
        domain: Domain name (e.g., 'math', 'code', 'general')
        num_experts: Total number of experts (default: 4)
    
    Returns:
        One-hot vector as list
    
    Mapping:
        - math/finemath → Expert 0: [1, 0, 0, 0]
        - code/starcoder → Expert 2: [0, 0, 1, 0]
        - general/other → Expert 1: [0, 1, 0, 0]
        - Expert 3: unused (duplicate)
    """
    label = [0.0] * num_experts
    
    domain_lower = domain.lower()
    
    # Math domains
    if any(kw in domain_lower for kw in ['math', 'finemath', 'minerva', 'numina']):
        label[0] = 1.0  # Expert 0
    # Code domains
    elif any(kw in domain_lower for kw in ['code', 'starcoder', 'programming']):
        label[2] = 1.0  # Expert 2
    # General/other domains
    else:
        label[1] = 1.0  # Expert 1
    
    return label


@dataclass
class KLConstrainedRouterTrainModuleConfig(TransformerTrainModuleConfig):
    """Configuration for CE-constrained (hard label) router training."""
    
    ce_loss_weight: float = 0.1
    """Weight (λ) for the cross-entropy regularization term. Lower = more freedom for router."""
    
    num_experts: int = 4
    """Total number of experts in the model."""
    
    supervised_layers: Optional[List[int]] = None
    """If set, only supervise routers in these layer indices. E.g., [0] for layer 0 only."""
    
    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
        dataset=None,
        **extra_kwargs,
    ) -> "KLConstrainedRouterTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return KLConstrainedRouterTrainModule(
            model=model,
            device=device,
            dataset=dataset,
            **kwargs,
            **extra_kwargs,
        )


class KLConstrainedRouterTrainModule(TransformerTrainModule):
    """Training module for CE-constrained (hard label) router training.
    
    Trains the model end-to-end with language modeling loss, while adding
    cross-entropy regularization to guide the router towards domain-based labels.
    
    Key differences:
        - LM loss is ACTIVE (not disabled)
        - CE term acts as soft regularization, not the primary signal
        - Router can deviate from domain labels when it improves LM loss
        - Uses hard one-hot labels from domain metadata (no expert losses needed)
    """

    def __init__(
        self,
        ce_loss_weight: float = 0.1,
        num_experts: int = 4,
        supervised_layers: Optional[List[int]] = None,
        dataset=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ce_loss_weight = ce_loss_weight
        self.num_experts = num_experts
        self.supervised_layers = supervised_layers
        self.dataset = dataset
        
        self._patch_routers_for_ce_loss()
        log.info(f"CEConstrainedRouterTrainModule initialized:")
        log.info(f"  ce_loss_weight={ce_loss_weight} (λ in loss = L_LM + λ*CE)")
        log.info(f"  num_experts={num_experts}")
        log.info(f"  supervised_layers={supervised_layers}")
        log.info(f"  LM loss: ACTIVE (key difference from router_loss_only mode)")
        log.info(f"  Using HARD domain labels (one-hot vectors)")
    
    def _get_domain_from_metadata(self, metadata: Any) -> Optional[str]:
        """Extract domain name from metadata."""
        if metadata is None:
            return None
        
        # metadata could be dict, list, or string
        if isinstance(metadata, dict):
            return metadata.get('source_name') or metadata.get('domain')
        elif isinstance(metadata, (list, tuple)) and len(metadata) > 0:
            return self._get_domain_from_metadata(metadata[0])
        elif isinstance(metadata, str):
            return metadata
        
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
    
    def _patch_routers_for_ce_loss(self):
        """Patch router forward methods to compute CE loss during forward."""
        self._current_hard_labels: Optional[torch.Tensor] = None
        self._router_ce_losses: List[float] = []
        
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
                        
                        hard_labels = train_module_self._current_hard_labels
                        if hard_labels is not None and router.training and torch.is_grad_enabled():
                            router_logits = router._latest_router_logits
                            if router_logits is not None:
                                ce_loss = train_module_self._compute_ce_loss_during_forward(
                                    router_logits, hard_labels, loss_div_factor, router.num_experts
                                )
                                if ce_loss is not None:
                                    scaled_loss = train_module_self.ce_loss_weight * ce_loss
                                    aux_loss = scaled_loss if aux_loss is None else aux_loss + scaled_loss
                                    train_module_self._router_ce_losses.append(get_local_tensor(scaled_loss.detach()).item())
                                    if not hasattr(train_module_self, '_logged_ce_loss'):
                                        train_module_self._logged_ce_loss = set()
                                    if router_name_inner not in train_module_self._logged_ce_loss and train_module_self.trainer.global_step <= 1:
                                        log.info(f"[CE-Constrained RT] Computed CE loss in router '{router_name_inner}': {scaled_loss.item():.6f}")
                                        train_module_self._logged_ce_loss.add(router_name_inner)
                        
                        return expert_weights, expert_indices, batch_size_per_expert, aux_loss
                    
                    return patched_forward
                
                module.forward = make_patched_forward(module, original_forward, self, router_name)
                patched_count += 1
        
        if skipped_layers:
            log.info(f"[CE-Constrained RT] Skipped supervision for layers: {sorted(set(skipped_layers))}")
        log.info(f"[CE-Constrained RT] Patched {patched_count} routers for CE loss computation")
    
    def _compute_ce_loss_during_forward(
        self,
        router_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
        num_experts: int,
    ) -> Optional[torch.Tensor]:
        """
        Compute cross-entropy loss during forward pass using hard one-hot labels.
        
        CE(hard_labels, p_θ) = -Σ_e hard_labels(e) * log p_θ(e)
        
        Where hard_labels(e) is a one-hot vector and p_θ(e) is the router's prediction.
        """
        if isinstance(router_logits, DTensor):
            router_logits = get_full_tensor(router_logits)
        
        hard_labels = hard_labels.to(router_logits.device)
        
        # Get shapes
        if router_logits.dim() == 3:
            batch_size, seq_len, num_experts_logits = router_logits.shape
            router_logits_flat = router_logits.reshape(-1, num_experts_logits)  # (B*S, E)
        elif router_logits.dim() == 2:
            router_logits_flat = router_logits
            num_experts_logits = router_logits.shape[-1]
        else:
            return None
        
        # hard_labels should be (B, E) - one label per sequence
        # Expand to (B*S, E) by repeating for each token
        if hard_labels.dim() == 2:
            batch_size_labels, num_experts_labels = hard_labels.shape
            # Repeat for each token in sequence
            if router_logits.dim() == 3:
                seq_len = router_logits.shape[1]
                hard_labels_expanded = hard_labels.unsqueeze(1).expand(batch_size_labels, seq_len, num_experts_labels)
                hard_labels_flat = hard_labels_expanded.reshape(-1, num_experts_labels)
            else:
                hard_labels_flat = hard_labels
        elif hard_labels.dim() == 1:
            # Class indices format
            hard_labels_flat = hard_labels
        else:
            return None
        
        num_logit_tokens = router_logits_flat.shape[0]
        num_label_items = hard_labels_flat.shape[0]
        
        # Align lengths if needed
        if hard_labels_flat.dim() == 2 and num_label_items != num_logit_tokens:
            if not hasattr(self, '_logged_shape_align_ce'):
                self._logged_shape_align_ce = True
                log.warning(f"CE Shape alignment: labels={num_label_items} -> logits={num_logit_tokens}")
            
            if num_label_items > num_logit_tokens:
                hard_labels_flat = hard_labels_flat[:num_logit_tokens]
            else:
                # This shouldn't happen with proper batch setup
                log.warning(f"Unexpected: more logits than labels in CE loss computation")
                return None
        
        num_tokens = router_logits_flat.shape[0]
        if num_tokens == 0:
            return None
        
        # Compute router log probabilities
        router_log_probs = F.log_softmax(router_logits_flat, dim=-1)  # (N, E)
        
        # Cross-entropy with one-hot labels: -Σ_e hard_labels(e) * log p_θ(e)
        # Since hard_labels is one-hot, this simplifies to -log p_θ(correct_expert)
        if hard_labels_flat.dim() == 2:
            # One-hot format
            ce_loss = -(hard_labels_flat * router_log_probs).sum(dim=-1)  # (N,)
        else:
            # Class indices format
            ce_loss = F.nll_loss(router_log_probs, hard_labels_flat, reduction='none')
        
        ce_loss = ce_loss.sum()  # Scalar
        
        if loss_div_factor is not None:
            ce_loss = ce_loss / (loss_div_factor if isinstance(loss_div_factor, torch.Tensor) else float(loss_div_factor))
        else:
            ce_loss = ce_loss / num_tokens
        
        return ce_loss
    
    def _get_hard_labels_for_batch(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Get hard one-hot labels for a batch based on domain metadata.
        
        Returns:
            Hard labels tensor of shape (batch_size, num_experts)
        """
        metadata = batch.get("metadata")
        if metadata is None:
            log.warning("No 'metadata' in batch, cannot determine domain labels")
            return None
        
        batch_size = batch["input_ids"].shape[0]
        
        # Get hard label for each item in batch
        hard_labels = []
        for i in range(batch_size):
            if isinstance(metadata, (list, tuple)):
                item_metadata = metadata[i] if i < len(metadata) else None
            else:
                item_metadata = metadata
            
            domain = self._get_domain_from_metadata(item_metadata)
            
            if domain is None:
                # Default to general expert if domain unknown
                label = get_hard_label_for_domain("general", self.num_experts)
                if not hasattr(self, '_logged_missing_domain'):
                    self._logged_missing_domain = True
                    log.warning(f"Could not determine domain from metadata, defaulting to 'general' expert")
            else:
                label = get_hard_label_for_domain(domain, self.num_experts)
            
            hard_labels.append(label)
        
        # Convert to tensor: (B, E)
        hard_labels_tensor = torch.tensor(hard_labels, dtype=torch.float32)
        return hard_labels_tensor
    
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
        """Train on a batch with LM loss + CE regularization."""
        self.model.train()

        batch_size = batch["input_ids"].shape[0]
        
        # Get hard labels for this batch (skip during dry_run as mock batch won't have proper metadata)
        hard_labels = None
        if not dry_run:
            hard_labels = self._get_hard_labels_for_batch(batch)
        has_hard_labels = hard_labels is not None
        
        if not hasattr(self, '_first_batch_logged'):
            self._first_batch_logged = True
            if dry_run:
                log.info(f"[CE-Constrained RT] Dry-run batch: keys={list(batch.keys())}, has_hard_labels={has_hard_labels} (expected - mock batch)")
            else:
                log.info(f"[CE-Constrained RT] First batch: keys={list(batch.keys())}, has_hard_labels={has_hard_labels}")
                if has_hard_labels:
                    log.info(f"  Hard labels shape: {hard_labels.shape}")
                    log.info(f"  CE weight (λ): {self.ce_loss_weight}")
                    # Log some example labels
                    if hard_labels.shape[0] > 0:
                        example_label = hard_labels[0].tolist()
                        log.info(f"  Example label: {example_label}")
                log.info(f"  Training mode: LM loss + λ*CE (λ={self.ce_loss_weight})")
                log.info(f"  Using HARD domain-based labels (one-hot vectors)")

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
        lm_ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)  # LM cross-entropy loss
        router_ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)  # Router CE loss (domain guidance)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        auxiliary_batch_losses: Dict[str, torch.Tensor] = {}

        # Split into micro-batches
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)
        _micro_batch_refs = list(micro_batches)
        
        # Split hard labels to match micro-batches (labels are per-sequence, not per-token)
        hard_labels_list: List[torch.Tensor] = []
        if has_hard_labels and hard_labels is not None:
            hard_labels_list = list(torch.split(hard_labels, self.rank_microbatch_size // seq_len, dim=0))

        for micro_batch_idx, micro_batch in enumerate(_micro_batch_refs):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Get hard labels for this micro-batch
                micro_hard_labels = None
                if has_hard_labels and hard_labels_list and micro_batch_idx < len(hard_labels_list):
                    micro_hard_labels = move_to_device(hard_labels_list[micro_batch_idx], self.device)

                self._current_hard_labels = micro_hard_labels
                
                model_forward_result = self.model_forward(
                    input_ids, labels=labels, ignore_index=self.label_ignore_index,
                    loss_reduction="sum", z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss, return_logits=False,
                    **model_kwargs,
                )
                
                if isinstance(model_forward_result, tuple):
                    output_dict, lm_ce_loss, z_loss = model_forward_result[:3]  # type: ignore[misc]
                else:
                    raise TypeError(f"Unexpected return type: {type(model_forward_result)}")
                
                self._current_hard_labels = None

                # LM loss is ALWAYS active (key difference from supervised RT)
                loss = lm_ce_loss
                if z_loss is not None:
                    loss = loss + z_loss
                
                # Collect router CE loss values for logging/metrics
                should_log = micro_batch_idx == 0 and (self.trainer.global_step <= 1 or self.trainer.global_step % 100 == 0)
                if self._router_ce_losses:
                    router_ce_loss_sum = sum(self._router_ce_losses)
                    router_ce_batch_loss += router_ce_loss_sum
                    if should_log:
                        log.info(f"[CE-Constrained RT] Collected router CE loss from {len(self._router_ce_losses)} routers: {router_ce_loss_sum:.6f}")
                    self._router_ce_losses.clear()
                
                # Track LM CE loss separately
                lm_ce_batch_loss += get_local_tensor(lm_ce_loss.detach())
                del lm_ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss
                
                if loss is not None and isinstance(loss, torch.Tensor):
                    loss.backward()
                elif micro_batch_idx == 0 and self.trainer.global_step < 1:
                    log.warning(f"[CE-Constrained RT] Skipping backward: loss is None or not a tensor")

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            model_for_aux = _unwrap_fsdp_model(self.model)
            if hasattr(model_for_aux, 'reset_auxiliary_metrics'):
                model_for_aux.reset_auxiliary_metrics()  # type: ignore[attr-defined]
            return

        # Record all losses - separately track LM loss and router guidance loss
        self.record_ce_loss(lm_ce_batch_loss, ReduceType.mean)  # Standard LM CE loss
        if z_batch_loss is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        
        # Record the router CE loss (domain guidance) for tracking in wandb
        if isinstance(router_ce_batch_loss, torch.Tensor):
            router_ce_loss_val = router_ce_batch_loss.item()
            self.record_metric("router CE loss (domain)", router_ce_batch_loss, ReduceType.mean, namespace="train")
            
            if router_ce_loss_val == 0 and has_hard_labels and self.trainer.global_step % 100 == 0:
                log.warning(f"[CE-Constrained RT] router CE loss is 0 at step {self.trainer.global_step}")

        model_for_aux = _unwrap_fsdp_model(self.model)
        if hasattr(model_for_aux, 'compute_auxiliary_metrics'):
            metrics = model_for_aux.compute_auxiliary_metrics(reset=True)  # type: ignore[attr-defined]
            for metric_name, (metric_val, reduction) in metrics.items():
                self.record_metric(metric_name, metric_val, reduction, namespace="train")
        
        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = lm_ce_batch_loss

