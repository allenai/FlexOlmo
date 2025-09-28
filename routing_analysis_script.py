#!/usr/bin/env python3
"""
Router Analysis Script for MoE Model Evaluation

This script analyzes expert routing patterns across different evaluation tasks
by running forward passes on completed prefix+completion pairs and extracting
router logits to understand expert specialization.

Usage:
    python routing_analysis_script.py --config config.yaml
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import boto3
from botocore.exceptions import ClientError

# Add FlexOLmo to path for custom model loading
sys.path.append('/weka/oe-training-default/sanjaya/FlexOlmo')

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class S3DataLoader:
    """Handle S3 data loading and caching"""
    
    def __init__(self, s3_bucket: str = "ai2-sewonm"):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.local_cache_dir = Path("./eval_data_cache")
        self.local_cache_dir.mkdir(exist_ok=True)
    
    def download_file_if_needed(self, s3_key: str, local_path: Path) -> bool:
        """Download file from S3 if not already cached locally"""
        if local_path.exists():
            logger.info(f"Using cached file: {local_path}")
            return True
        
        try:
            logger.info(f"Downloading {s3_key} from S3...")
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            logger.info(f"Downloaded to: {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    def load_jsonl_file(self, s3_key: str) -> List[Dict]:
        """Load a JSONL file from S3 (with local caching)"""
        local_path = self.local_cache_dir / f"{s3_key.replace('/', '_')}.jsonl"
        
        if not self.download_file_if_needed(s3_key, local_path):
            return []
        
        data = []
        with open(local_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} samples from {s3_key}")
        return data


class RouterAnalyzer:
    """Analyze router patterns from model forward passes"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        
        # Load model and tokenizer using a direct approach that bypasses transformers auto-detection
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer from the model path (since all tokenizer files are there)
        logger.info(f"Loading tokenizer from model path: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model using a more direct approach
        logger.info("Loading model with custom configuration handling...")
        
        # Try to load the model configuration first
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            logger.info(f"Loaded config with model_type: {config.model_type}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            # Create a minimal config
            from transformers import PretrainedConfig
            config = PretrainedConfig()
            config.model_type = "olmoe2"
        
        # Try to load the model with the config
        try:
            # Method 1: Try with the loaded config
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Successfully loaded model with custom config")
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
            try:
                # Method 2: Try with a different approach
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Successfully loaded model using AutoModel")
            except Exception as e2:
                logger.warning(f"Method 2 failed: {e2}")
                # Method 3: Try with more permissive settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=False,
                    ignore_mismatched_sizes=True
                )
                logger.info("Successfully loaded model with permissive settings")
        self.model.eval()
        
        # Get model info
        self.num_experts = getattr(self.model.config, 'num_experts', 4)
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', 16)
        
        logger.info(f"Model loaded: {self.num_experts} experts, {self.num_layers} layers")
    
    def extract_router_logits(self, text: str, max_length: int = 2048) -> Optional[torch.Tensor]:
        """Extract router logits for a given text"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass with router logits
            with torch.no_grad():
                outputs = self.model(**inputs, output_router_logits=True)
            
            # Extract router logits
            if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                return outputs.router_logits
            
            logger.warning("No router logits found in model output")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting router logits: {e}")
            return None
    
    def analyze_expert_weights(self, router_logits: torch.Tensor) -> Dict[int, np.ndarray]:
        """Analyze expert weights from router logits"""
        expert_weights_by_layer = {}
        
        # router_logits shape: [num_layers, seq_len, num_experts]
        for layer_idx in range(router_logits.shape[0]):
            layer_logits = router_logits[layer_idx]  # [seq_len, num_experts]
            
            # Convert to probabilities (softmax)
            expert_probs = torch.softmax(layer_logits, dim=-1)  # [seq_len, num_experts]
            
            # Average across sequence length
            avg_expert_weights = expert_probs.mean(dim=0).cpu().numpy()  # [num_experts]
            
            expert_weights_by_layer[layer_idx] = avg_expert_weights
        
        return expert_weights_by_layer


class TaskDataProcessor:
    """Process evaluation data to extract prefix+completion pairs"""
    
    def __init__(self, s3_loader: S3DataLoader):
        self.s3_loader = s3_loader
    
    def process_prediction_file(self, s3_key: str) -> List[Tuple[str, str]]:
        """Process prediction file to extract prefix+completion pairs - only for generation tasks"""
        data = self.s3_loader.load_jsonl_file(s3_key)
        
        prefix_completion_pairs = []
        for sample in data:
            # Only process generation tasks, skip MC tasks
            # Check if this is a generation task by looking for 'continuation' field
            if 'continuation' not in sample:
                continue
                
            # Extract model outputs
            model_outputs = sample.get('model_output', [])
            if not model_outputs:
                continue
            
            # Get the first (best) completion
            best_output = model_outputs[0]
            completion = best_output.get('continuation', '')
            
            # For generation tasks, we need to reconstruct the full text
            # The prefix is typically the prompt from the original request
            # For now, we'll use a placeholder - this needs to be matched with requests file
            prefix = ""  # This will be filled from the requests file
            
            if completion:
                prefix_completion_pairs.append((prefix, completion))
        
        logger.info(f"Extracted {len(prefix_completion_pairs)} prefix+completion pairs from {s3_key}")
        return prefix_completion_pairs
    
    def process_request_file(self, s3_key: str) -> List[str]:
        """Process request file to extract prompts (prefixes) - only for generation tasks"""
        data = self.s3_loader.load_jsonl_file(s3_key)
        
        prompts = []
        for sample in data:
            # Only process generation tasks, skip MC tasks
            if sample.get('request_type') == 'generate_until':
                request = sample.get('request', {})
                context = request.get('context', '')
                if context:
                    prompts.append(context)
        
        logger.info(f"Extracted {len(prompts)} prompts from {s3_key}")
        return prompts
    
    def match_prefixes_completions(self, prompts: List[str], completions: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Match prompts with completions by index"""
        matched_pairs = []
        
        min_length = min(len(prompts), len(completions))
        for i in range(min_length):
            prefix = prompts[i]
            completion = completions[i][1]  # completions[i] is (prefix, completion)
            matched_pairs.append((prefix, completion))
        
        logger.info(f"Matched {len(matched_pairs)} prefix+completion pairs")
        return matched_pairs


class RoutingPatternAnalyzer:
    """Main class for analyzing routing patterns across tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_loader = S3DataLoader()
        self.data_processor = TaskDataProcessor(self.s3_loader)
        self.router_analyzer = RouterAnalyzer(
            model_path=config['model_path'],
            device=config.get('device', 'cuda')
        )
        
        self.results = {}
        self.layers_to_analyze = config.get('layers_to_analyze', [0, 7, 15])
        self.batch_size = config.get('batch_size', 8)
        self.max_sequence_length = config.get('max_sequence_length', 2048)
    
    def analyze_task(self, task_name: str, task_config: Dict[str, str]) -> Dict[str, Any]:
        """Analyze routing patterns for a single task - only generation tasks"""
        logger.info(f"Analyzing task: {task_name}")
        
        # Skip MC tasks entirely
        if ':mc' in task_name or 'mc' in task_name.lower():
            logger.info(f"Skipping MC task: {task_name}")
            return {}
        
        # Extract task info
        predictions_key = task_config['predictions_s3_key']
        requests_key = task_config['requests_s3_key']
        
        # Process data
        prompts = self.data_processor.process_request_file(requests_key)
        completion_pairs = self.data_processor.process_prediction_file(predictions_key)
        
        if not prompts or not completion_pairs:
            logger.warning(f"No data found for task {task_name}")
            return {}
        
        # Match prefixes with completions
        prefix_completion_pairs = self.data_processor.match_prefixes_completions(prompts, completion_pairs)
        
        if not prefix_completion_pairs:
            logger.warning(f"No matched pairs found for task {task_name}")
            return {}
        
        # Analyze routing patterns
        task_results = self._analyze_routing_patterns(task_name, prefix_completion_pairs)
        
        return task_results
    
    def _analyze_routing_patterns(self, task_name: str, prefix_completion_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze routing patterns for a list of prefix+completion pairs"""
        logger.info(f"Analyzing {len(prefix_completion_pairs)} samples for {task_name}")
        
        all_expert_weights = []
        
        # Process in batches
        for i in range(0, len(prefix_completion_pairs), self.batch_size):
            batch = prefix_completion_pairs[i:i + self.batch_size]
            
            for prefix, completion in batch:
                # Combine prefix and completion
                full_text = prefix + completion
                
                # Extract router logits
                router_logits = self.router_analyzer.extract_router_logits(
                    full_text, 
                    max_length=self.max_sequence_length
                )
                
                if router_logits is not None:
                    # Analyze expert weights
                    expert_weights = self.router_analyzer.analyze_expert_weights(router_logits)
                    all_expert_weights.append(expert_weights)
        
        if not all_expert_weights:
            logger.warning(f"No routing data captured for {task_name}")
            return {}
        
        # Aggregate results by layer
        task_analysis = {}
        for layer_idx in self.layers_to_analyze:
            if layer_idx >= self.router_analyzer.num_layers:
                continue
            
            # Collect expert weights for this layer across all samples
            layer_weights = []
            for sample_weights in all_expert_weights:
                if layer_idx in sample_weights:
                    layer_weights.append(sample_weights[layer_idx])
            
            if layer_weights:
                # Stack and compute statistics
                layer_weights = np.stack(layer_weights)  # [num_samples, num_experts]
                
                task_analysis[layer_idx] = {
                    'mean_weights': layer_weights.mean(axis=0),  # [num_experts]
                    'std_weights': layer_weights.std(axis=0),    # [num_experts]
                    'expert_rankings': np.argsort(layer_weights.mean(axis=0))[::-1],  # Expert order by weight
                    'num_samples': len(layer_weights)
                }
        
        logger.info(f"Analyzed {len(all_expert_weights)} samples for {task_name}")
        return task_analysis
    
    def analyze_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Analyze routing patterns for all configured tasks"""
        logger.info("Starting analysis for all tasks")
        
        for task_name, task_config in self.config['tasks'].items():
            try:
                task_results = self.analyze_task(task_name, task_config)
                if task_results:
                    self.results[task_name] = task_results
            except Exception as e:
                logger.error(f"Error analyzing task {task_name}: {e}")
        
        logger.info(f"Completed analysis for {len(self.results)} tasks")
        return self.results
    
    def save_results(self, output_path: str):
        """Save analysis results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def create_visualization(self, output_path: str):
        """Create routing pattern visualization"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for plotting
        tasks = list(self.results.keys())
        layers = self.layers_to_analyze
        experts = list(range(self.router_analyzer.num_experts))
        
        # Create figure
        fig, axes = plt.subplots(len(tasks), len(layers), figsize=(4*len(layers), 3*len(tasks)))
        if len(tasks) == 1:
            axes = axes.reshape(1, -1)
        if len(layers) == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot data
        for task_idx, task_name in enumerate(tasks):
            for layer_idx, layer in enumerate(layers):
                ax = axes[task_idx, layer_idx]
                
                if layer in self.results[task_name]:
                    layer_data = self.results[task_name][layer]
                    mean_weights = layer_data['mean_weights']
                    
                    # Convert to percentages
                    percentages = mean_weights * 100
                    
                    # Create bar plot
                    bars = ax.bar(experts, percentages, alpha=0.7)
                    
                    # Color bars
                    for i, bar in enumerate(bars):
                        if i == 0 or i == 3:  # Expert 0 and 3 (typically dominant)
                            bar.set_color('darkblue')
                        elif i == 1 or i == 2:  # Expert 1 and 2 (typically less used)
                            bar.set_color('lightblue')
                    
                    # Add 25% baseline
                    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Uniform (25%)')
                    
                    # Formatting
                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Routing Probability (%)')
                    ax.set_xlabel('Expert ID')
                    ax.set_title(f'{task_name}\nLayer {layer}')
                    ax.grid(True, alpha=0.3)
                    
                    # Set x-axis ticks
                    ax.set_xticks(experts)
                    ax.set_xticklabels([f'Expert {i}' for i in experts])
                    
                    # Add legend only to first subplot
                    if task_idx == 0 and layer_idx == 0:
                        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Analyze MoE router patterns')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--output-dir', default='./routing_analysis_output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = RoutingPatternAnalyzer(config)
    
    # Run analysis
    results = analyzer.analyze_all_tasks()
    
    # Save results
    analyzer.save_results(output_dir / 'routing_analysis_results.json')
    
    # Create visualization
    if args.visualize:
        analyzer.create_visualization(output_dir / 'routing_pattern_analysis.jpg')
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
