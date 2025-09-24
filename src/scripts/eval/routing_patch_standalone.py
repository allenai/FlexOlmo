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
        
        # Calculate router weights (softmax of logits) instead of just selections
        expert_weights = []
        for layer_logits in router_logits_np:
            # Apply softmax to get weights (probabilities)
            # Handle both 2D (seq_len, num_experts) and 3D (batch_size, seq_len, num_experts) shapes
            if layer_logits.ndim == 3:
                # 3D case: (batch_size, seq_len, num_experts) -> (seq_len, num_experts)
                layer_logits = layer_logits[0]  # Take first batch item
            
            weights = np.exp(layer_logits - np.max(layer_logits, axis=-1, keepdims=True))
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            expert_weights.append(weights)
        
        # Also keep the original logits for reference
        router_logits_flat = [logits.flatten() for logits in router_logits_np]
        
        # Store the routing data with weights
        routing_entry = {
            "task_name": self.task_name,
            "input_shape": input_ids_np.shape,
            "num_layers": len(router_logits),
            "expert_weights": expert_weights,  # New: actual router weights
            "router_logits": router_logits_flat,  # Keep logits for reference
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
        # Use the first available layers for backward compatibility
        available_layers = sorted(layer_counters.keys())
        if len(available_layers) >= 3:
            selected_layers = [available_layers[0], available_layers[len(available_layers)//2], available_layers[-1]]
        else:
            selected_layers = available_layers[:3]  # Take up to 3 layers
        
        expert_counts_file = f"{self.output_dir}/{self.model_name}/expert_counts/{self.task_name}.pkl"
        with open(expert_counts_file, "wb") as f:
            pkl.dump([layer_counters.get(i, Counter()) for i in selected_layers], f)
            
        crosslayer_file = f"{self.output_dir}/{self.model_name}/expert_counts_crosslayer/{self.task_name}.pkl"
        with open(crosslayer_file, "wb") as f:
            if len(selected_layers) >= 2:
                crosslayer_data = [crosslayer_counters.get((selected_layers[0], selected_layers[1]), Counter())]
                if len(selected_layers) >= 3:
                    crosslayer_data.append(crosslayer_counters.get((selected_layers[1], selected_layers[2]), Counter()))
            else:
                crosslayer_data = [Counter()]
            pkl.dump(crosslayer_data, f)
            
        token_file = f"{self.output_dir}/{self.model_name}/eid2token/{self.task_name}.pkl"
        with open(token_file, "wb") as f:
            pkl.dump(eid2token_mappings, f)
            
        logger.info(f"Saved routing analysis for {self.task_name}: {len(self.routing_data)} forward passes, {self.total_tokens} tokens")
        
        # For S3 paths, also save to the main evaluation output directory
        # so they get uploaded automatically with the other results
        if self.output_dir.startswith('s3://'):
            # Get the main evaluation output directory from environment
            eval_output_dir = os.environ.get('EVAL_OUTPUT_DIR', '/tmp/eval_output')
            routing_dir = f"{eval_output_dir}/routing_analysis/{self.model_name}"
            
            Path(f"{routing_dir}/expert_counts").mkdir(parents=True, exist_ok=True)
            Path(f"{routing_dir}/expert_counts_crosslayer").mkdir(parents=True, exist_ok=True)
            Path(f"{routing_dir}/eid2token").mkdir(parents=True, exist_ok=True)
            
            # Copy files to evaluation output directory
            import shutil
            eval_expert_counts = f"{routing_dir}/expert_counts/{self.task_name}.pkl"
            eval_crosslayer = f"{routing_dir}/expert_counts_crosslayer/{self.task_name}.pkl"
            eval_token = f"{routing_dir}/eid2token/{self.task_name}.pkl"
            
            shutil.copy2(expert_counts_file, eval_expert_counts)
            shutil.copy2(crosslayer_file, eval_crosslayer)
            shutil.copy2(token_file, eval_token)
            
            logger.info(f"Saved routing files to evaluation output directory: {routing_dir}")
        
    def _process_routing_data(self):
        """Process routing data using weights instead of binary selections."""
        # Initialize data structures for weighted analysis
        layer_counters = defaultdict(Counter)
        crosslayer_counters = defaultdict(Counter)
        eid2token_layer0 = defaultdict(Counter)
        eid2token_layer7 = defaultdict(Counter)
        eid2token_layer15 = defaultdict(Counter)
        
        for entry in self.routing_data:
            expert_weights = entry["expert_weights"]  # Use weights instead of selections
            input_tokens = entry["input_tokens"]
            
            # Convert weights to numpy array for processing
            # expert_weights is a list of arrays, each with shape (batch_size, seq_len, num_experts)
            # Stack along the layer dimension: (batch_size, seq_len, num_experts, num_layers)
            weights_array = np.stack(expert_weights, axis=-1)
            
            # Debug: Print shapes to understand the data structure
            logger.info(f"Weights array shape: {weights_array.shape}")
            logger.info(f"Number of layers: {len(expert_weights)}")
            
            # Extract specific layers - dynamically choose layers based on model depth
            num_layers = len(expert_weights)
            
            # Choose representative layers: first, middle, and last
            layer_indices = [0]  # Always include first layer
            if num_layers > 1:
                layer_indices.append(num_layers // 2)  # Middle layer
            if num_layers > 2:
                layer_indices.append(num_layers - 1)  # Last layer
            
            # Extract weights for selected layers
            selected_layers = {}
            for i, layer_idx in enumerate(layer_indices):
                if layer_idx < num_layers:
                    selected_layers[i] = expert_weights[layer_idx]
            
            # Process token-to-expert mappings using weights
            # weights_layer shape is (seq_len, num_experts) - no batch dimension
            for id, token in enumerate(input_tokens):
                for layer_key, layer_weights in selected_layers.items():
                    if id < layer_weights.shape[0]:
                        # Get weights for this token across all experts
                        token_weights = layer_weights[id, :]  # Shape: (num_experts,)
                        for expert_id, weight in enumerate(token_weights):
                            # Use layer_key as the layer identifier
                            if layer_key == 0:
                                eid2token_layer0[expert_id][token] += weight
                            elif layer_key == 1:
                                eid2token_layer7[expert_id][token] += weight
                            elif layer_key == 2:
                                eid2token_layer15[expert_id][token] += weight
            
            # Process layer counters using weights (sum of weights per expert)
            for layer_idx, layer_weights in enumerate(expert_weights):
                # Sum weights across all tokens for each expert
                # layer_weights shape is (seq_len, num_experts)
                expert_weight_sums = np.sum(layer_weights, axis=0)  # Sum across seq_len, shape: (num_experts,)
                for expert_id, weight_sum in enumerate(expert_weight_sums):
                    layer_counters[layer_idx][expert_id] += weight_sum
            
            # Process cross-layer counters using weights
            # Only analyze cross-layer patterns for selected representative layers
            for i, layer_i in enumerate(layer_indices):
                for j, layer_j in enumerate(layer_indices):
                    if layer_i < num_layers and layer_j < num_layers:
                        weights_i = expert_weights[layer_i]  # Shape: (seq_len, num_experts)
                        weights_j = expert_weights[layer_j]  # Shape: (seq_len, num_experts)
                        
                        # For each expert pair, compute the sum of their weight products
                        for expert_i in range(weights_i.shape[1]):  # num_experts
                            for expert_j in range(weights_j.shape[1]):  # num_experts
                                weight_product = np.sum(weights_i[:, expert_i] * weights_j[:, expert_j])
                                crosslayer_counters[(layer_i, layer_j)][(expert_i, expert_j)] += weight_product
        
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
    
    # Initialize the routing hook instance if not already initialized
    if _routing_hook_instance is None:
        model_name = os.path.basename(str(model_path)) if model_path else "unknown_model"
        task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
        _routing_hook_instance = RoutingHook(model_name, task_name, output_dir)
        logger.info(f"Routing tracking enabled for model: {model_path}")
    else:
        # Update the task name if it's different
        task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
        if _routing_hook_instance.task_name != task_name:
            _routing_hook_instance.task_name = task_name
            logger.info(f"Updated routing tracking task to: {task_name}")


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


# Global routing hook instance
_routing_hook_instance = None

def patch_hflm_verbose():
    """Patch the HFLM_Verbose class to add routing tracking."""
    try:
        logger.info("PATCHING HFLM_Verbose...")
        from oe_eval.models.eleuther_huggingface import HFLM_Verbose  # type: ignore
        
        # Check what methods are available on HFLM_Verbose
        logger.info(f"HFLM_Verbose methods: {[m for m in dir(HFLM_Verbose) if not m.startswith('_')]}")
        
        # Try to patch different methods that might be used for generation
        methods_to_patch = ['generate', 'forward', '__call__']
        
        for method_name in methods_to_patch:
            if hasattr(HFLM_Verbose, method_name):
                logger.info(f"Found method {method_name} on HFLM_Verbose, patching it...")
                original_method = getattr(HFLM_Verbose, method_name)
                
                def create_patched_method(original_method, method_name):
                    def patched_method(self, *args, **kwargs):
                        global _routing_hook_instance
                        # Determine prefill vs decode
                        # Treat any call without past_key_values as prefill
                        is_prefill = ('past_key_values' not in kwargs)

                        # Only force output_router_logits during prefill
                        if is_routing_tracking_enabled() and is_prefill and 'output_router_logits' not in kwargs:
                            kwargs['output_router_logits'] = True
                            
                            # Initialize routing hook if needed
                            if '_routing_hook_instance' not in globals() or _routing_hook_instance is None:
                                model_name = os.environ.get("FLEXOLMO_MODEL_NAME", "unknown_model")
                                task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
                                output_dir = get_routing_output_dir()
                                _routing_hook_instance = RoutingHook(model_name, task_name, output_dir)
                                logger.info(f"Initialized routing hook for model: {model_name}, task: {task_name}")
                        
                        # Call original method
                        output = original_method(self, *args, **kwargs)
                        
                        # If routing tracking is enabled, capture router logits only during prefill
                        if (is_routing_tracking_enabled() and is_prefill):
                            # Ensure routing hook is initialized
                            ensure_routing_hook_initialized()
                            
                            # Try to extract input_ids and router_logits from various sources
                            input_ids = None
                            router_logits = None
                            
                            # Check kwargs for input_ids
                            if 'input_ids' in kwargs:
                                input_ids = kwargs['input_ids']
                            elif args and len(args) > 0:
                                input_ids = args[0]
                            
                            # Check for router_logits in the output
                            if hasattr(output, 'router_logits') and output.router_logits is not None:
                                router_logits = output.router_logits
                            elif isinstance(output, dict) and 'router_logits' in output:
                                router_logits = output['router_logits']
                            elif hasattr(output, 'logits') and not isinstance(output, dict) and hasattr(output.logits, 'router_logits'):
                                router_logits = output.logits.router_logits
                            
                            if input_ids is not None and router_logits is not None and _routing_hook_instance:
                                # Get model attributes safely
                                model = getattr(self, 'model', None)
                                num_experts = getattr(model, 'num_experts', 64) if model else 64
                                num_experts_per_tok = getattr(model, 'num_experts_per_tok', 8) if model else 8
                                
                                _routing_hook_instance.capture_routing(
                                    router_logits=router_logits,
                                    input_ids=input_ids,
                                    model_num_experts=num_experts,
                                    model_num_experts_per_tok=num_experts_per_tok,
                                )
                                logger.info(f"Captured router logits from {method_name} method (prefill)")
                        
                        return output
                    return patched_method
                
                setattr(HFLM_Verbose, method_name, create_patched_method(original_method, method_name))
                logger.info(f"Successfully patched {method_name} method")
        
        # Also try to patch the underlying model's forward method when HFLM_Verbose is instantiated
        original_init = HFLM_Verbose.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Try to patch the underlying model's forward method
            if hasattr(self, 'model') and self.model is not None:
                logger.info("Attempting to patch underlying model's forward method...")
                if hasattr(self.model, 'forward'):
                    original_model_forward = self.model.forward
                    
                    def patched_model_forward(*args, **kwargs):
                        global _routing_hook_instance
                        
                        logger.info(f"PATCHED FORWARD CALLED - routing enabled: {is_routing_tracking_enabled()}")
                        
                        # Determine prefill vs decode
                        # Treat any call without past_key_values as prefill
                        is_prefill = ('past_key_values' not in kwargs)

                        # Only capture routing during prefill (first forward pass)
                        should_capture_routing = (
                            is_routing_tracking_enabled() and 
                            is_prefill and
                            'output_router_logits' not in kwargs
                        )
                        
                        if should_capture_routing:
                            kwargs['output_router_logits'] = True
                            
                            # Initialize routing hook if needed
                            if '_routing_hook_instance' not in globals() or _routing_hook_instance is None:
                                model_name = os.environ.get("FLEXOLMO_MODEL_NAME", "unknown_model")
                                task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
                                output_dir = get_routing_output_dir()
                                _routing_hook_instance = RoutingHook(model_name, task_name, output_dir)
                                logger.info(f"Initialized routing hook for model: {model_name}, task: {task_name}")
                        
                        try:
                            output = original_model_forward(*args, **kwargs)
                        except RuntimeError as e:
                            if "size of tensor" in str(e) and "must match" in str(e):
                                # If we get a tensor size mismatch, retry without forcing output_router_logits
                                logger.warning(f"Tensor size mismatch detected, retrying without output_router_logits: {e}")
                                if 'output_router_logits' in kwargs:
                                    del kwargs['output_router_logits']
                                try:
                                    output = original_model_forward(*args, **kwargs)
                                except Exception as retry_e:
                                    logger.error(f"Retry also failed: {retry_e}")
                                    raise
                            else:
                                raise
                        
                        # If routing tracking is enabled, capture router logits only during prefill
                        if (is_routing_tracking_enabled() and should_capture_routing and is_prefill):
                            logger.info("ATTEMPTING ROUTING CAPTURE")
                            # Ensure routing hook is initialized
                            ensure_routing_hook_initialized()
                            input_ids = kwargs.get("input_ids") or args[0] if args else None
                            
                            logger.info(f"Input IDs shape: {input_ids.shape if input_ids is not None else None}")
                            logger.info(f"Output type: {type(output)}")
                            logger.info(f"Output has router_logits: {hasattr(output, 'router_logits')}")
                            
                            # Check for router_logits in the output
                            router_logits = None
                            if hasattr(output, 'router_logits') and output.router_logits is not None:
                                router_logits = output.router_logits
                                logger.info(f"Found router_logits with length: {len(router_logits)}")
                            elif isinstance(output, dict) and 'router_logits' in output:
                                router_logits = output['router_logits']
                                logger.info("Found router_logits in dict")
                            
                            logger.info(f"Routing capture - input_ids: {input_ids is not None}, router_logits: {router_logits is not None}, hook: {_routing_hook_instance is not None}")
                            
                            if input_ids is not None and router_logits is not None and _routing_hook_instance:
                                try:
                                    num_experts = getattr(self.model, 'num_experts', 64)
                                    num_experts_per_tok = getattr(self.model, 'num_experts_per_tok', 8)
                                    
                                    _routing_hook_instance.capture_routing(
                                        router_logits=router_logits,
                                        input_ids=input_ids,
                                        model_num_experts=num_experts,
                                        model_num_experts_per_tok=num_experts_per_tok,
                                    )
                                    logger.info("SUCCESS: Captured router logits from prefill step")
                                except Exception as e:
                                    logger.warning(f"Failed to capture routing data: {e}")
                                    # Continue without routing capture to avoid breaking the evaluation
                        
                        return output
                    
                    self.model.forward = patched_model_forward
                    logger.info("Successfully patched underlying model's forward method")
        
        HFLM_Verbose.__init__ = patched_init
        logger.info("Successfully patched HFLM_Verbose.__init__ to patch underlying model")
        
        logger.info("Successfully patched HFLM_Verbose for routing tracking")
        
    except ImportError:
        logger.warning("Could not import HFLM_Verbose, routing patch not applied")
    except Exception as e:
        logger.warning(f"Error patching HFLM_Verbose: {e}")


def setup_routing_for_task(task_name: str):
    """Setup routing tracking for a specific task."""
    os.environ['CURRENT_TASK'] = task_name
    logger.info(f"Routing tracking setup for task: {task_name}")


def save_routing_results_for_task(model):
    """Save routing results for the current task."""
    global _routing_hook_instance
    
    if not is_routing_tracking_enabled():
        return
        
    # Save routing results using the global hook instance
    if _routing_hook_instance is not None:
        _routing_hook_instance.save_results()
        logger.info(f"Saved routing results for task: {_routing_hook_instance.task_name}")
    else:
        logger.warning("No routing hook instance found to save results")


# Auto-apply the patch when this module is imported
patch_hflm_verbose()
