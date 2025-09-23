"""
Patch for existing evaluation scripts to add routing tracking.

This module provides a simple way to add routing tracking to existing
evaluation scripts without major modifications.
"""

import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Import our routing hook
from .routing_hook import add_routing_hook_to_model, is_routing_tracking_enabled, get_routing_output_dir


def patch_model_for_routing(model: Any, model_name: str = None, task_name: str = None) -> Any:
    """
    Patch a model to add routing tracking if enabled.
    
    Args:
        model: The model to patch
        model_name: Name of the model (auto-detected if None)
        task_name: Name of the current task (auto-detected if None)
        
    Returns:
        The patched model with routing tracking enabled
    """
    if not is_routing_tracking_enabled():
        return model
        
    # Auto-detect model name if not provided
    if model_name is None:
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            model_name = getattr(model.model.config, '_name_or_path', 'unknown_model')
        else:
            model_name = 'unknown_model'
            
    # Auto-detect task name if not provided
    if task_name is None:
        task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
        
    # Get output directory
    output_dir = get_routing_output_dir()
    
    logger.info(f"Adding routing tracking to model: {model_name}")
    
    # Add routing hook to the underlying model
    if hasattr(model, 'model'):
        model.model = add_routing_hook_to_model(
            model.model, 
            model_name, 
            task_name, 
            output_dir
        )
    else:
        model = add_routing_hook_to_model(
            model, 
            model_name, 
            task_name, 
            output_dir
        )
        
    return model


def setup_routing_for_task(task_name: str):
    """Setup routing tracking for a specific task."""
    os.environ['CURRENT_TASK'] = task_name
    logger.info(f"Routing tracking setup for task: {task_name}")


def save_routing_results_for_task(model: Any):
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


# Monkey patch for the standard model loading
def patch_hflm_verbose():
    """Patch the HFLM_Verbose class to add routing tracking."""
    try:
        from oe_eval.models.eleuther_huggingface import HFLM_Verbose
        
        # Save original __init__ method
        original_init = HFLM_Verbose.__init__
        
        def routing_aware_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Add routing tracking if enabled
            if is_routing_tracking_enabled():
                model_name = kwargs.get('pretrained', 'unknown_model')
                if isinstance(model_name, str):
                    model_name = os.path.basename(model_name)
                    
                task_name = os.environ.get('CURRENT_TASK', 'unknown_task')
                output_dir = get_routing_output_dir()
                
                logger.info(f"Adding routing tracking to HFLM_Verbose: {model_name}")
                
                # Patch the underlying model
                if hasattr(self, 'model'):
                    self.model = add_routing_hook_to_model(
                        self.model, 
                        model_name, 
                        task_name, 
                        output_dir
                    )
                    
        # Replace the __init__ method
        HFLM_Verbose.__init__ = routing_aware_init
        
        logger.info("Successfully patched HFLM_Verbose for routing tracking")
        
    except ImportError:
        logger.warning("Could not import HFLM_Verbose, routing patch not applied")


# Auto-apply the patch when this module is imported
patch_hflm_verbose()
