#!/usr/bin/env python3
"""
Router Analysis Script for MoE Model Evaluation (Prompt-Only)

Analyzes expert routing patterns by running forward passes on task prompts
and extracting router softmax probabilities at each MoE layer.

Uses prompt-only (no completions) to cleanly measure how the router responds
to different task types without contamination from generated text.

Usage:
    python routing_analysis_prompt_only.py --config config.yaml --visualize
    python routing_analysis_prompt_only.py --config config.yaml --output-dir ./my_output --visualize
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import boto3
from botocore.exceptions import ClientError

sys.path.append('/weka/oe-training-default/sanjaya/FlexOlmo')

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class S3DataLoader:
    """Handle S3 data loading with local caching."""

    def __init__(self, s3_bucket: str = "ai2-sewonm"):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.local_cache_dir = Path("./eval_data_cache")
        self.local_cache_dir.mkdir(exist_ok=True)

    def download_file_if_needed(self, s3_key: str, local_path: Path) -> bool:
        if local_path.exists():
            logger.info(f"Using cached file: {local_path}")
            return True
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {s3_key} from S3...")
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            logger.info(f"Downloaded to: {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False

    def load_jsonl_file(self, s3_key: str) -> List[Dict]:
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
    """Load model and extract router logits from forward passes."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device

        is_local = os.path.isdir(model_path)

        logger.info(f"Loading tokenizer from {model_path} (local={is_local})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=is_local
        )

        logger.info("Loading model...")
        try:
            from transformers import OlmoeForCausalLM
            self.model = OlmoeForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, local_files_only=is_local,
            )
            logger.info("Loaded with OlmoeForCausalLM")
        except Exception as e:
            logger.warning(f"OlmoeForCausalLM failed: {e}, trying AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, local_files_only=is_local,
            )
            logger.info("Loaded with AutoModelForCausalLM")
        self.model.eval()

        self.num_experts = getattr(self.model.config, 'num_experts', 4)
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', 32)

        self.moe_layers = []
        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            if (hasattr(layer, 'block_sparse_moe') and layer.block_sparse_moe is not None
                    or hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts')
                    or hasattr(layer, 'feed_forward_moe') and layer.feed_forward_moe is not None
                    or hasattr(layer, 'moe') and layer.moe is not None):
                self.moe_layers.append(i)

        if not self.moe_layers and self.num_experts > 1:
            if hasattr(self.model.config, 'moe_layers'):
                self.moe_layers = self.model.config.moe_layers
            else:
                self.moe_layers = list(range(self.num_layers))
                logger.warning(f"Could not detect MoE layers, assuming all {self.num_layers} layers are MoE")

        logger.info(f"Model: {self.num_experts} experts, {self.num_layers} layers, MoE at layers: {self.moe_layers}")

    def extract_router_logits(self, text: str, max_length: int = 2048) -> Optional[torch.Tensor]:
        """Run forward pass and return router logits [num_moe_layers, seq_len, num_experts]."""
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=max_length, truncation=True, padding=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_router_logits=True)

            if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                router_logits = outputs.router_logits
                if isinstance(router_logits, tuple):
                    router_logits = torch.stack(router_logits, dim=0)
                if router_logits.dim() == 4:
                    router_logits = router_logits.squeeze(1)
                return router_logits
            else:
                logger.warning("No router logits in model output")
                return None
        except Exception as e:
            logger.error(f"Error extracting router logits: {e}")
            return None

    def get_expert_probs_by_layer(self, router_logits: torch.Tensor) -> Dict[int, np.ndarray]:
        """Convert router logits to per-layer average expert probabilities."""
        result = {}
        for i, layer_idx in enumerate(self.moe_layers):
            if i < router_logits.shape[0]:
                layer_logits = router_logits[i]  # [seq_len, num_experts]
                expert_probs = torch.softmax(layer_logits, dim=-1)  # [seq_len, num_experts]
                result[layer_idx] = expert_probs.mean(dim=0).cpu().numpy()  # [num_experts]
        return result


class TaskDataProcessor:
    """Extract prompts from eval request files."""

    def __init__(self, s3_loader: S3DataLoader):
        self.s3_loader = s3_loader

    def load_prompts_from_s3(self, s3_key: str) -> List[str]:
        """Load prompts from a requests JSONL file on S3."""
        data = self.s3_loader.load_jsonl_file(s3_key)
        prompts = []
        for sample in data:
            if sample.get('request_type') == 'generate_until':
                context = sample.get('request', {}).get('context', '')
                if context:
                    prompts.append(context)
        logger.info(f"Extracted {len(prompts)} prompts from {s3_key}")
        return prompts

    def load_prompts_from_local(self, filepath: str) -> List[str]:
        """Load prompts from a local JSONL file (same format as S3 requests file)."""
        prompts = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if sample.get('request_type') == 'generate_until':
                        context = sample.get('request', {}).get('context', '')
                        if context:
                            prompts.append(context)
                    elif 'context' in sample:
                        prompts.append(sample['context'])
                    elif 'prompt' in sample:
                        prompts.append(sample['prompt'])
                    elif 'text' in sample:
                        prompts.append(sample['text'])
        logger.info(f"Loaded {len(prompts)} prompts from {filepath}")
        return prompts


class RoutingPatternAnalyzer:
    """Analyze and visualize routing patterns across tasks using prompt-only."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_loader = S3DataLoader(config.get('s3_bucket', 'ai2-sewonm'))
        self.data_processor = TaskDataProcessor(self.s3_loader)
        self.router_analyzer = RouterAnalyzer(
            model_path=config['model_path'],
            device=config.get('device', 'cuda')
        )
        self.results = {}
        self.layers_to_analyze = config.get('layers_to_analyze', [1, 15, 31])
        self.max_sequence_length = config.get('max_sequence_length', 2048)
        self.max_samples = config.get('max_samples', 0)

    def _load_prompts(self, task_config: Dict[str, str]) -> List[str]:
        """Load prompts from either S3 or local file."""
        if 'local_path' in task_config:
            return self.data_processor.load_prompts_from_local(task_config['local_path'])
        elif 'requests_s3_key' in task_config:
            return self.data_processor.load_prompts_from_s3(task_config['requests_s3_key'])
        else:
            logger.error("Task config must have either 'local_path' or 'requests_s3_key'")
            return []

    def analyze_task(self, task_name: str, task_config: Dict[str, str]) -> Dict[str, Any]:
        """Analyze routing patterns for a single task using prompts only."""
        logger.info(f"Analyzing task: {task_name}")

        prompts = self._load_prompts(task_config)
        if not prompts:
            logger.warning(f"No prompts found for task {task_name}")
            return {}

        if self.max_samples > 0 and len(prompts) > self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples (from {len(prompts)})")
            prompts = prompts[:self.max_samples]

        all_expert_weights = []
        for i, prompt in enumerate(prompts):
            if (i + 1) % 50 == 0:
                logger.info(f"  [{task_name}] Processed {i + 1}/{len(prompts)} prompts")

            router_logits = self.router_analyzer.extract_router_logits(
                prompt, max_length=self.max_sequence_length
            )
            if router_logits is not None:
                expert_weights = self.router_analyzer.get_expert_probs_by_layer(router_logits)
                all_expert_weights.append(expert_weights)

        if not all_expert_weights:
            logger.warning(f"No routing data captured for {task_name}")
            return {}

        task_analysis = {}
        for layer_idx in self.layers_to_analyze:
            if layer_idx >= self.router_analyzer.num_layers:
                continue
            layer_weights = [
                sw[layer_idx] for sw in all_expert_weights if layer_idx in sw
            ]
            if layer_weights:
                layer_weights = np.stack(layer_weights)  # [num_samples, num_experts]
                task_analysis[layer_idx] = {
                    'mean_weights': layer_weights.mean(axis=0),
                    'std_weights': layer_weights.std(axis=0),
                    'expert_rankings': np.argsort(layer_weights.mean(axis=0))[::-1],
                    'num_samples': len(layer_weights),
                }

        logger.info(f"Analyzed {len(all_expert_weights)} prompts for {task_name}")
        return task_analysis

    def analyze_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        for task_name, task_config in self.config['tasks'].items():
            try:
                task_results = self.analyze_task(task_name, task_config)
                if task_results:
                    self.results[task_name] = task_results
            except Exception as e:
                logger.error(f"Error analyzing task {task_name}: {e}")
                import traceback
                traceback.print_exc()
        logger.info(f"Completed analysis for {len(self.results)} tasks")
        return self.results

    def save_results(self, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        with open(output_file, 'w') as f:
            json.dump(convert_numpy(self.results), f, indent=2)
        logger.info(f"Results saved to {output_file}")

    def create_visualization(self, output_path: str):
        if not self.results:
            logger.warning("No results to visualize")
            return

        tasks = list(self.results.keys())
        layers = self.layers_to_analyze
        num_experts = self.router_analyzer.num_experts
        experts = list(range(num_experts))

        fig, axes = plt.subplots(
            len(tasks), len(layers),
            figsize=(4 * len(layers), 3 * len(tasks)),
            squeeze=False,
        )

        for task_idx, task_name in enumerate(tasks):
            for layer_idx, layer in enumerate(layers):
                ax = axes[task_idx, layer_idx]

                if layer in self.results[task_name]:
                    layer_data = self.results[task_name][layer]
                    mean_weights = layer_data['mean_weights']
                    std_weights = layer_data['std_weights']
                    percentages = mean_weights * 100
                    std_pct = std_weights * 100

                    bars = ax.bar(experts, percentages, yerr=std_pct, capsize=3, alpha=0.8)

                    for i, bar in enumerate(bars):
                        bar.set_color('#1a237e' if i in (0, 3) else '#90caf9')

                    ax.axhline(
                        y=100.0 / num_experts, color='gray',
                        linestyle='--', alpha=0.7, linewidth=1,
                    )

                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Routing Probability (%)')
                    ax.set_xlabel('Expert ID')
                    ax.set_title(f'{task_name}\nLayer {layer}')
                    ax.set_xticks(experts)
                    ax.set_xticklabels([f'Expert {i}' for i in experts])
                else:
                    ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualization saved to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Analyze MoE router patterns (prompt-only)')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--output-dir', default='./routing_analysis_output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    analyzer = RoutingPatternAnalyzer(config)
    analyzer.analyze_all_tasks()
    analyzer.save_results(str(output_dir / 'routing_analysis_results.json'))

    if args.visualize:
        analyzer.create_visualization(str(output_dir / 'routing_pattern_analysis.png'))

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
