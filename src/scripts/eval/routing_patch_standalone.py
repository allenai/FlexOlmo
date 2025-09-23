"""
Standalone routing tracking patch for MoE models during evaluations.

This module provides a lightweight way to capture router logits during
model inference without requiring external dependencies.
"""

import os
import pickle as pkl
import json
import numpy as np
import torch
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RoutingHook:
    """
    Lightweight hook to capture router logits during model inference.
    """
    
    def __init__(self, model_name: str, task_name: str, output_dir: str = "routing_output"):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        
        # Initialize tracking data
        self.routing_data = []
        self.total_tokens = 0
        
        # Create output directory
        Path(f"{self.output_dir}/{self.model_name}").mkdir(parents=True, exist_ok=True)
        
    def capture_routing(self, router_logits: List[torch.Tensor], input_ids: torch.Tensor, 
                       model_num_experts: int, model_num_experts_per_tok: int):
        """Capture routing information from a single forward pass."""
        if not router_logits:
            return
            
        # Convert to numpy for analysis
        router_logits_np = [logits.detach().cpu().numpy() for logits in router_logits]
        input_ids_np = input_ids.detach().cpu().numpy()
        
        # Determine expert selections (top-k)
        expert_selections = []
        for layer_logits in router_logits_np:
            # Get top-k experts for each token
            top_k_indices = np.argsort(-layer_logits, axis=-1)[:, :model_num_experts_per_tok]
            expert_selections.append(top_k_indices)
        
        # Store the routing data
        routing_entry = {
            "task_name": self.task_name,
            "input_shape": input_ids_np.shape,
            "num_layers": len(router_logits),
            "expert_selections": expert_selections,
            "input_tokens": input_ids_np.flatten().tolist(),
        }
        
        self.routing_data.append(routing_entry)
        self.total_tokens += input_ids_np.size
        
    def save_results(self):
        """Save captured routing data to files in the same format as run_routing_analysis.py."""
        if not self.routing_data:
            logger.warning("No routing data captured")
            return
            
        # Create the same directory structure as the original script
        Path(f"{self.output_dir}/{self.model_name}/expert_counts").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/{self.model_name}/expert_counts_crosslayer").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/{self.model_name}/eid2token").mkdir(parents=True, exist_ok=True)
        
        # Process the routing data to match original format
        layer_counters, crosslayer_counters, eid2token_mappings = self._process_routing_data()
        
        # Save in the exact same format as run_routing_analysis.py
        expert_counts_file = f"{self.output_dir}/{self.model_name}/expert_counts/{self.task_name}.pkl"
        with open(expert_counts_file, "wb") as f:
            pkl.dump([layer_counters[0], layer_counters[7], layer_counters[15]], f)
            
        crosslayer_file = f"{self.output_dir}/{self.model_name}/expert_counts_crosslayer/{self.task_name}.pkl"
        with open(crosslayer_file, "wb") as f:
            pkl.dump([crosslayer_counters[(0, 7)], crosslayer_counters[(7, 15)]], f)
            
        token_file = f"{self.output_dir}/{self.model_name}/eid2token/{self.task_name}.pkl"
        with open(token_file, "wb") as f:
            pkl.dump(eid2token_mappings, f)
            
        logger.info(f"Saved routing analysis for {self.task_name}: {len(self.routing_data)} forward passes, {self.total_tokens} tokens")
        
    def _process_routing_data(self):
        """Process routing data to match the exact format of run_routing_analysis.py."""
        # Initialize data structures exactly like the original script
        layer_counters = defaultdict(Counter)
        crosslayer_counters = defaultdict(Counter)
        eid2token_layer0 = defaultdict(Counter)
        eid2token_layer7 = defaultdict(Counter)
        eid2token_layer15 = defaultdict(Counter)
        
        for entry in self.routing_data:
            expert_selections = entry["expert_selections"]
            input_tokens = entry["input_tokens"]
            
            # Convert to numpy array for processing (same as original)
            exp_ids = np.stack(expert_selections, axis=-1)  # Shape: (batch_size, seq_len, num_layers)
            
            # Extract specific layers (same as original)
            if exp_ids.shape[2] > 0:
                exp_ids_layer0 = exp_ids[:, :, 0]
            if exp_ids.shape[2] > 7:
                exp_ids_layer7 = exp_ids[:, :, 7]
            if exp_ids.shape[2] > 15:
                exp_ids_layer15 = exp_ids[:, :, 15]
            
            # Process token-to-expert mappings (same logic as original)
            for id, token in enumerate(input_tokens):
                if exp_ids.shape[2] > 0 and id < exp_ids_layer0.shape[0]:
                    experts = exp_ids_layer0[id, :]
                    experts_list = experts.flatten().astype(int).tolist()
                    for expert_id in experts_list:
                        eid2token_layer0[expert_id][token] += 1
                        
                if exp_ids.shape[2] > 7 and id < exp_ids_layer7.shape[0]:
                    experts = exp_ids_layer7[id, :]
                    experts_list = experts.flatten().astype(int).tolist()
                    for expert_id in experts_list:
                        eid2token_layer7[expert_id][token] += 1
                        
                if exp_ids.shape[2] > 15 and id < exp_ids_layer15.shape[0]:
                    experts = exp_ids_layer15[id, :]
                    experts_list = experts.flatten().astype(int).tolist()
                    for expert_id in experts_list:
                        eid2token_layer15[expert_id][token] += 1
            
            # Process layer counters (same logic as original)
            for layer in range(exp_ids.shape[2]):
                layer_experts = exp_ids[:, :, layer].flatten().astype(int).tolist()
                exp_counts = Counter(layer_experts)
                layer_counters[layer].update(exp_counts)
            
            # Process cross-layer counters (same logic as original)
            for layer_i in range(exp_ids.shape[2] - 1):
                for layer_j in range(exp_ids.shape[2]):
                    layer_i_experts = exp_ids[:, :, layer_i].flatten().astype(int).tolist()
                    layer_j_experts = exp_ids[:, :, layer_j].flatten().astype(int).tolist()
                    exps_counts = Counter(zip(layer_i_experts, layer_j_experts))
                    crosslayer_counters[(layer_i, layer_j)].update(exps_counts)
        
        # Return in the exact same format as original script
        eid2token_mappings = [eid2token_layer0, eid2token_layer7, eid2token_layer15]
        
        return layer_counters, crosslayer_counters, eid2token_mappings


def add_routing_hook_to_model(model, model_name: str, task_name: str, output_dir: str = "routing_output"):
    """Add routing tracking hook to an existing model."""
    # Create routing hook
    routing_hook = RoutingHook(model_name, task_name, output_dir)
    
    # Store the hook as an attribute
    model._routing_hook = routing_hook
    
    # Save original forward method
    original_forward = model.forward if hasattr(model, 'forward') else None
    
    if original_forward is None:
        logger.warning("Model does not have a forward method, routing tracking not available")
        return model
        
    def routing_aware_forward(*args, **kwargs):
        # Call original forward method
        result = original_forward(*args, **kwargs)
        
        # Try to extract router logits if available
        if hasattr(result, 'router_logits') and result.router_logits is not None:
            # Get input_ids from args
            input_ids = args[0] if args else kwargs.get('input_ids')
            if input_ids is not None:
                # Try to get model configuration
                num_experts = getattr(model.config, 'num_experts', 64)
                num_experts_per_tok = getattr(model.config, 'num_experts_per_tok', 8)
                
                routing_hook.capture_routing(
                    router_logits=result.router_logits,
                    input_ids=input_ids,
                    model_num_experts=num_experts,
                    model_num_experts_per_tok=num_experts_per_tok
                )
        
        return result
    
    # Replace forward method
    model.forward = routing_aware_forward
    
    # Add method to save routing results
    def save_routing_results():
        if hasattr(model, '_routing_hook'):
            model._routing_hook.save_results()
    
    model.save_routing_results = save_routing_results
    
    logger.info(f"Added routing tracking hook to model: {model_name}")
    return model


def setup_routing_tracking(model_path: str, output_dir: str = "routing_output"):
    """Setup routing tracking for a model path."""
    global _routing_hook_instance
    
    # Set environment variable to enable routing tracking
    os.environ["FLEXOLMO_ROUTING_TRACKING"] = "true"
    os.environ["FLEXOLMO_ROUTING_OUTPUT_DIR"] = output_dir
    
    # Initialize the routing hook instance
    model_name = os.path.basename(str(model_path)) if model_path else "unknown_model"
    task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
    _routing_hook_instance = RoutingHook(model_name, task_name, output_dir)
    
    logger.info(f"Routing tracking enabled for model: {model_path}")


def get_routing_output_dir() -> str:
    """Get the routing output directory from environment variables."""
    return os.environ.get("FLEXOLMO_ROUTING_OUTPUT_DIR", "routing_output")


def is_routing_tracking_enabled() -> bool:
    """Check if routing tracking is enabled."""
    return os.environ.get("FLEXOLMO_ROUTING_TRACKING", "false").lower() == "true"


def ensure_routing_hook_initialized():
    """Ensure the routing hook instance is initialized if routing tracking is enabled."""
    global _routing_hook_instance
    
    if is_routing_tracking_enabled() and _routing_hook_instance is None:
        model_name = os.environ.get("FLEXOLMO_MODEL_NAME", "unknown_model")
        task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
        output_dir = get_routing_output_dir()
        _routing_hook_instance = RoutingHook(model_name, task_name, output_dir)
        logger.info(f"Initialized routing hook for model: {model_name}, task: {task_name}")


def patch_hflm_verbose():
    """Patch the HFLM_Verbose class to add routing tracking."""
    try:
        from oe_eval.models.eleuther_huggingface import HFLM_Verbose
        
        # Patch the forward method to capture router logits
        _original_hflm_verbose_forward = HFLM_Verbose.forward

        def _patched_hflm_verbose_forward(self, *args, **kwargs):
            # Call original forward method
            output = _original_hflm_verbose_forward(self, *args, **kwargs)

            # If routing tracking is enabled, capture router logits
            if is_routing_tracking_enabled():
                ensure_routing_hook_initialized()
                if _routing_hook_instance:
                    input_ids = kwargs.get("input_ids") or args[0] if args else None
                    if input_ids is not None and hasattr(output, 'router_logits') and output.router_logits is not None:
                        # Get model attributes safely
                        model = getattr(self, 'model', None)
                        num_experts = getattr(model, 'num_experts', 64) if model else 64
                        num_experts_per_tok = getattr(model, 'num_experts_per_tok', 8) if model else 8
                        
                        _routing_hook_instance.track_batch(
                            input_ids=input_ids,
                            router_logits=output.router_logits,
                            model_num_experts=num_experts,
                            model_num_experts_per_tok=num_experts_per_tok,
                        )
            return output

        HFLM_Verbose.forward = _patched_hflm_verbose_forward
        logger.info("Successfully patched HFLM_Verbose to capture router logits.")
        
    except ImportError:
        logger.warning("Could not import HFLM_Verbose, routing patch not applied")


def setup_routing_for_task(task_name: str):
    """Setup routing tracking for a specific task."""
    os.environ['CURRENT_TASK'] = task_name
    logger.info(f"Routing tracking setup for task: {task_name}")


def save_routing_results_for_task(model):
    """Save routing results for the current task."""
    if not is_routing_tracking_enabled():
        return
        
    # Try to save routing results
    if hasattr(model, 'save_routing_results'):
        model.save_routing_results()
    elif hasattr(model, 'model') and hasattr(model.model, 'save_routing_results'):
        model.model.save_routing_results()
    else:
        logger.warning("Could not find routing results to save")


# Auto-apply the patch when this module is imported
patch_hflm_verbose()
