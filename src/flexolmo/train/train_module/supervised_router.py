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
        if router_logits.dim() == 3:
            # (num_layers, batch_size * seq_len, num_experts)
            # Average across layers for simplicity (or sum, depending on preference)
            router_logits = router_logits.mean(dim=0)  # (batch_size * seq_len, num_experts)
        elif router_logits.dim() != 2:
            raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")
        
        batch_size = expert_labels.shape[0]
        seq_len = router_logits.shape[0] // batch_size
        
        # Expand expert_labels to match sequence length
        # expert_labels: (batch_size, num_experts) -> (batch_size * seq_len, num_experts)
        expert_labels_expanded = expert_labels.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, num_experts)
        expert_labels_expanded = expert_labels_expanded.reshape(-1, expert_labels.shape[-1])  # (batch_size * seq_len, num_experts)
        
        # Move to same device as router_logits
        expert_labels_expanded = expert_labels_expanded.to(router_logits.device)
        
        # Compute cross-entropy loss
        # router_logits: (batch_size * seq_len, num_experts)
        # expert_labels: (batch_size * seq_len, num_experts)
        router_loss = F.cross_entropy(
            router_logits,
            expert_labels_expanded.argmax(dim=-1),  # Convert one-hot to class indices
            reduction="sum",
        ) / num_tokens
        
        return router_loss

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """Train on a batch with supervised router loss."""
        self.model.train()

        batch_size = batch["input_ids"].shape[0]
        
        # Extract expert labels if present
        expert_labels = self._extract_expert_labels_from_batch(batch, batch_size)
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

                # Run forward pass with router logits if we have expert labels
                router_logits = None
                if micro_expert_labels is not None:
                    # Store router logits using a forward hook
                    router_logits_list = []
                    
                    def router_hook(module, input, output):
                        # The router forward returns: (logits, scores, expert_weights, expert_indices, batch_size_per_expert)
                        if isinstance(output, tuple) and len(output) >= 1:
                            router_logits_list.append(output[0])  # Extract logits
                    
                    # Register hooks on all router modules
                    hooks = []
                    for block in self.model.blocks:
                        if hasattr(block, "feed_forward_moe") and hasattr(block.feed_forward_moe, "router"):
                            hook = block.feed_forward_moe.router.register_forward_hook(router_hook)
                            hooks.append(hook)
                    
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
                        router_logits = torch.stack(router_logits_list, dim=0)  # (num_layers, batch*seq_len, num_experts)
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
                
                # Combine losses
                loss = torch.tensor(0.0, device=self.device)
                if not self.router_loss_only:
                    loss = ce_loss
                    if z_loss is not None:
                        loss += z_loss
                
                if router_loss is not None:
                    loss += self.router_loss_weight * router_loss

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
