# Router Analysis Script for MoE Model Evaluation

This script analyzes expert routing patterns across different evaluation tasks by running forward passes on completed prefix+completion pairs and extracting router logits to understand expert specialization.

## Overview

The script implements a post-generation analysis approach that:
1. Downloads evaluation results from S3 (predictions and requests) - **generation tasks only**
2. Matches prompts (prefixes) with completions
3. Runs forward passes through the model to extract router logits
4. Analyzes expert weight distributions across layers and tasks
5. Generates visualizations similar to your existing routing analysis plots

**Note**: This script only works with generation tasks (not multiple choice tasks) as MC tasks have a different data structure that doesn't provide meaningful routing analysis.

## Files

- `routing_analysis_script.py` - Main analysis script
- `routing_analysis_config.yaml` - Configuration file with task definitions
- `requirements_routing_analysis.txt` - Python dependencies
- `README_routing_analysis.md` - This documentation

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_routing_analysis.txt
   ```

2. **Configure AWS credentials** (if not already done):
   ```bash
   aws configure
   ```

3. **Verify model path** in `routing_analysis_config.yaml`:
   - Ensure the model path is accessible from your GPU server
   - The script expects the model to support `output_router_logits=True`

## Usage

### Basic Analysis
```bash
python routing_analysis_script.py --config routing_analysis_config.yaml
```

### With Visualization
```bash
python routing_analysis_script.py --config routing_analysis_config.yaml --visualize
```

### Custom Output Directory
```bash
python routing_analysis_script.py --config routing_analysis_config.yaml --output-dir ./my_results --visualize
```

## Configuration

Edit `routing_analysis_config.yaml` to:

1. **Update model path** if needed
2. **Modify layers to analyze** (default: [0, 7, 15])
3. **Adjust batch size** based on GPU memory
4. **Add/remove tasks** by updating the tasks section

### Task Configuration

Each task requires:
- `predictions_s3_key`: Path to the predictions JSONL file in S3
- `requests_s3_key`: Path to the requests JSONL file in S3

**Supported Tasks**: Only generation tasks (e.g., `codex_humaneval`, `gsm8k`, `triviaqa`, `minerva_math_*`, etc.)
**Excluded Tasks**: Multiple choice tasks (e.g., `arc_challenge:mc`, `boolq:mc`, etc.) are automatically skipped

## Output

The script generates:

1. **`routing_analysis_results.json`** - Detailed analysis results with:
   - Mean expert weights per layer per task
   - Standard deviations
   - Expert rankings
   - Sample counts

2. **`routing_pattern_analysis.jpg`** - Visualization showing:
   - Expert routing probabilities across layers and tasks
   - 25% baseline for uniform distribution
   - Color coding for dominant experts

## Expected Results

Based on your existing analysis, you should see:
- **Expert 0 and 3** dominating with ~60-75% routing probability in deeper layers
- **Expert 1 and 2** with low usage (~5-10%) in deeper layers
- **Layer 0** showing more balanced distribution
- **Task-specific patterns** revealing expert specialization

## Troubleshooting

### Common Issues

1. **"No router logits found"**
   - Verify model supports `output_router_logits=True`
   - Check model configuration

2. **"No data found for task"**
   - Verify S3 paths in config file
   - Check AWS credentials and permissions

3. **CUDA out of memory**
   - Reduce `batch_size` in config
   - Reduce `max_sequence_length`

4. **Slow processing**
   - Increase `batch_size` if memory allows
   - Process fewer tasks at once

### Debugging

Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance

- **Processing time**: ~1-2 minutes per 1000 samples
- **Memory usage**: ~8-16GB GPU memory (depending on batch size)
- **Storage**: Results are cached locally to avoid re-downloading

## Customization

### Adding New Tasks

1. Add task entry to config:
   ```yaml
   new_task:
     predictions_s3_key: "path/to/predictions.jsonl"
     requests_s3_key: "path/to/requests.jsonl"
   ```

2. Update task labels in visualization if needed

### Modifying Analysis

- **Different layers**: Update `layers_to_analyze` in config
- **Different metrics**: Modify `_analyze_routing_patterns` method
- **Different visualization**: Update `create_visualization` method

## Example Results

The script will produce analysis similar to your existing visualization showing:
- Expert specialization patterns
- Layer-wise routing changes
- Task-specific expert preferences
- Statistical summaries of routing behavior

This approach eliminates the complexity of patching generation methods while providing comprehensive routing analysis across all your evaluation tasks.
