#!/usr/bin/env python3
"""
Test script to verify MoE config generation works correctly.
"""

import sys
import os

# Add the OLMo-core src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OLMo-core', 'src'))

from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.hf.config import get_hf_config

def test_moe_config():
    """Test that we can generate HF config for MoE models."""
    
    # Create a simple MoE config
    config = TransformerConfig.llama_like_moe(
        vocab_size=1000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        num_experts=4,
        top_k=2,
        expert_hidden_size=1024,
        reordered_norm=True,
    )
    
    print("Created MoE config:")
    print(f"  Model type: {config.name}")
    print(f"  Block type: {config.block.name}")
    print(f"  Number of experts: {config.block.feed_forward_moe.num_experts}")
    print(f"  Top-k: {config.block.feed_forward_moe.top_k}")
    
    # Build the model
    model = config.build()
    print(f"\nBuilt model: {type(model).__name__}")
    
    # Try to generate HF config
    try:
        hf_config = get_hf_config(model)
        print(f"\nSuccessfully generated HF config: {type(hf_config).__name__}")
        print(f"  Vocab size: {hf_config.vocab_size}")
        print(f"  Hidden size: {hf_config.hidden_size}")
        print(f"  Num layers: {hf_config.num_hidden_layers}")
        print(f"  Num experts: {getattr(hf_config, 'num_experts', 'Not set')}")
        print(f"  Top-k: {getattr(hf_config, 'top_k', 'Not set')}")
        
        return True
        
    except Exception as e:
        print(f"\nFailed to generate HF config: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing MoE config generation...")
    success = test_moe_config()
    
    if success:
        print("\n✅ Test passed! MoE config generation works.")
    else:
        print("\n❌ Test failed! MoE config generation has issues.")
        sys.exit(1) 