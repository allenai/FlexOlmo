"""
Simple routing tracking hook for MoE models during evaluations.

This module provides a lightweight way to capture router logits during
model inference without requiring major changes to the evaluation pipeline.
"""

import pickle as pkl
import json
import numpy as np
import torch
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class RoutingHook:
    """
    Lightweight hook to capture router logits during model inference.
    
    This hook can be attached to any model to track expert routing
    during evaluation runs.
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
        """
        Capture routing information from a single forward pass.
        
        Args:
            router_logits: List of router logits from each layer
            input_ids: Input token IDs
            model_num_experts: Number of experts in the model
            model_num_experts_per_tok: Number of experts selected per token
        """
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
        
        logger.debug(f"Captured routing data: {len(expert_selections)} layers, {input_ids_np.size} tokens")
        
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
        # expert_counts: [layer_counters[0], layer_counters[7], layer_counters[15]]
        expert_counts_file = f"{self.output_dir}/{self.model_name}/expert_counts/{self.task_name}.pkl"
        with open(expert_counts_file, "wb") as f:
            pkl.dump([layer_counters[0], layer_counters[7], layer_counters[15]], f)
            
        # expert_counts_crosslayer: [crosslayer_counters[(0, 7)], crosslayer_counters[(7, 15)]]
        crosslayer_file = f"{self.output_dir}/{self.model_name}/expert_counts_crosslayer/{self.task_name}.pkl"
        with open(crosslayer_file, "wb") as f:
            pkl.dump([crosslayer_counters[(0, 7)], crosslayer_counters[(7, 15)]], f)
            
        # eid2token: [eid2token_layer0, eid2token_layer7, eid2token_layer15]
        token_file = f"{self.output_dir}/{self.model_name}/eid2token/{self.task_name}.pkl"
        with open(token_file, "wb") as f:
            pkl.dump(eid2token_mappings, f)
            
        # Create summary statistics
        summary = self._create_summary()
        summary_file = f"{self.output_dir}/{self.model_name}/{self.task_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved routing analysis for {self.task_name}: {len(self.routing_data)} forward passes, {self.total_tokens} tokens")
        logger.info(f"Files saved: {expert_counts_file}, {crosslayer_file}, {token_file}")
        
    def _process_routing_data(self):
        """Process routing data to match the exact format of run_routing_analysis.py."""
        from collections import defaultdict, Counter
        
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
                    # Convert experts array to list of ints
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
                # Convert numpy array to Python list of ints
                layer_experts = exp_ids[:, :, layer].flatten().astype(int).tolist()
                exp_counts = Counter(layer_experts)
                layer_counters[layer].update(exp_counts)
            
            # Process cross-layer counters (same logic as original)
            for layer_i in range(exp_ids.shape[2] - 1):
                for layer_j in range(exp_ids.shape[2]):
                    # Convert numpy arrays to Python lists of ints
                    layer_i_experts = exp_ids[:, :, layer_i].flatten().astype(int).tolist()
                    layer_j_experts = exp_ids[:, :, layer_j].flatten().astype(int).tolist()
                    exps_counts = Counter(zip(layer_i_experts, layer_j_experts))
                    crosslayer_counters[(layer_i, layer_j)].update(exps_counts)
        
        # Return in the exact same format as original script
        eid2token_mappings = [eid2token_layer0, eid2token_layer7, eid2token_layer15]
        
        return layer_counters, crosslayer_counters, eid2token_mappings
        
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary statistics from captured routing data."""
        if not self.routing_data:
            return {"error": "No data captured"}
            
        # Analyze expert usage patterns
        expert_usage = defaultdict(int)
        layer_usage = defaultdict(int)
        
        for entry in self.routing_data:
            num_layers = entry["num_layers"]
            layer_usage[num_layers] += 1
            
            for layer_idx, expert_selections in enumerate(entry["expert_selections"]):
                # expert_selections should be shape (batch_size, seq_len, num_experts_per_tok)
                for batch_idx in range(expert_selections.shape[0]):
                    for token_idx in range(expert_selections.shape[1]):
                        for expert_idx in range(expert_selections.shape[2]):
                            expert_id = expert_selections[batch_idx, token_idx, expert_idx]
                            # Convert numpy scalar to Python int
                            if hasattr(expert_id, 'item'):
                                expert_id = expert_id.item()
                            expert_usage[int(expert_id)] += 1
                        
        # Get most used experts
        most_used_experts = dict(Counter(expert_usage).most_common(10))
        
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "num_forward_passes": len(self.routing_data),
            "total_tokens": self.total_tokens,
            "layer_distribution": dict(layer_usage),
            "most_used_experts": most_used_experts,
            "total_experts_used": len(expert_usage),
        }


def add_routing_hook_to_model(model, model_name: str, task_name: str, output_dir: str = "routing_output"):
    """
    Add routing tracking hook to an existing model.
    
    This function modifies the model's forward method to capture router logits.
    
    Args:
        model: The model to add routing tracking to
        model_name: Name/identifier for the model
        task_name: Name of the current task
        output_dir: Directory to save routing analysis results
        
    Returns:
        The model with routing tracking enabled
    """
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
    """
    Setup routing tracking for a model path.
    
    This function can be called before model loading to enable routing tracking.
    """
    # Set environment variable to enable routing tracking
    os.environ["FLEXOLMO_ROUTING_TRACKING"] = "true"
    os.environ["FLEXOLMO_ROUTING_OUTPUT_DIR"] = output_dir
    
    logger.info(f"Routing tracking enabled for model: {model_path}")


def get_routing_output_dir() -> str:
    """Get the routing output directory from environment variables."""
    return os.environ.get("FLEXOLMO_ROUTING_OUTPUT_DIR", "routing_output")


def is_routing_tracking_enabled() -> bool:
    """Check if routing tracking is enabled."""
    return os.environ.get("FLEXOLMO_ROUTING_TRACKING", "false").lower() == "true"
