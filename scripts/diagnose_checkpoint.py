#!/usr/bin/env python3
"""
Diagnostic script to check if checkpoint is compatible with supervised router training.

Supports both standard checkpoints (rank0.pt) and distributed checkpoints (.distcp files).
"""

import sys
import torch
from pathlib import Path

def diagnose_checkpoint(checkpoint_path: str):
    """Check if checkpoint has proper router weights."""
    print(f"\n🔍 Diagnosing checkpoint: {checkpoint_path}\n")
    
    checkpoint_dir = Path(checkpoint_path)
    
    # Check what type of checkpoint format this is
    print("📁 Detecting checkpoint format...")
    
    # Check for distributed checkpoint format (.distcp files)
    distcp_files = list(checkpoint_dir.glob("*.distcp"))
    if distcp_files:
        print(f"✓ Found distributed checkpoint format: {len(distcp_files)} shard files")
        print(f"  Files: {', '.join([f.name for f in distcp_files[:5]])}")
        if len(distcp_files) > 5:
            print(f"  ... and {len(distcp_files) - 5} more")
        print("\n⚠️  This is a DISTRIBUTED checkpoint (sharded across multiple files)")
        print("   Cannot inspect individual parameters without loading the full model.")
        print("\n📊 What this means:")
        print("   ✓ Checkpoint exists and appears valid")
        print("   ✓ Parameters are sharded using PyTorch distributed checkpointing")
        print("   ✓ Model state will be reconstructed when loaded with FSDP")
        print("\n❓ Is this checkpoint compatible with supervised router training?")
        print("   → Need to check the MODEL ARCHITECTURE, not the checkpoint format")
        print("   → If this checkpoint came from an MoE model, it should work")
        print("   → If this checkpoint came from a dense model, it won't have router weights")
        print("\n💡 Next steps:")
        print("   1. Verify what model this checkpoint came from")
        print("   2. If it's an MoE model → Should work (the error is likely something else)")
        print("   3. If it's a dense model → Won't work, need an MoE checkpoint")
        
        # Try to find a metadata file
        metadata_files = list(checkpoint_dir.glob("*.metadata")) + list(checkpoint_dir.glob("metadata*"))
        if metadata_files:
            print(f"\n📄 Found metadata files: {[f.name for f in metadata_files]}")
        
        # Check for config file
        config_file = checkpoint_dir / "config.json"
        if config_file.exists():
            print(f"\n📄 Found config file: {config_file}")
            print("   Checking model architecture...")
            try:
                import json
                with open(config_file) as f:
                    config = json.load(f)
                
                # Try to determine if it's MoE
                model_config = config.get("model", {})
                block_config = model_config.get("block", {})
                
                if "feed_forward_moe" in str(block_config):
                    print("   ✅ Config indicates MoE model (feed_forward_moe found)")
                    print("   → This checkpoint SHOULD be compatible")
                elif "num_experts" in str(model_config) or "num_experts" in str(block_config):
                    print("   ✅ Config indicates MoE model (num_experts found)")
                    print("   → This checkpoint SHOULD be compatible")
                else:
                    print("   ⚠️  Could not confirm MoE architecture from config")
                    print("   → Might be dense model or config format differs")
                    
                # Show relevant config
                if "num_experts" in block_config.get("feed_forward_moe", {}):
                    num_experts = block_config["feed_forward_moe"]["num_experts"]
                    print(f"   📊 Number of experts: {num_experts}")
                
            except Exception as e:
                print(f"   ⚠️  Could not parse config: {e}")
        else:
            print("\n❌ No config.json found - cannot verify model architecture")
        
        return True  # Distributed checkpoint exists
    
    # Check for standard checkpoint format (train/rank0.pt)
    try:
        rank0_path = checkpoint_dir / "train" / "rank0.pt"
        if not rank0_path.exists():
            # Try alternative location
            rank0_path = checkpoint_dir / "rank0.pt"
        
        if not rank0_path.exists():
            print(f"❌ Standard checkpoint file not found")
            print(f"   Expected: {checkpoint_dir / 'train' / 'rank0.pt'}")
            print(f"   Or: {checkpoint_dir / 'rank0.pt'}")
            return False
        
        print(f"✓ Found standard checkpoint file: {rank0_path}")
        checkpoint = torch.load(rank0_path, map_location="cpu", weights_only=False)
        
        # Check for model state
        if "model" not in checkpoint:
            print("❌ No 'model' key in checkpoint")
            return False
        
        model_state = checkpoint["model"]
        print(f"✓ Checkpoint has {len(model_state)} parameters")
        
        # Look for router parameters
        router_params = [k for k in model_state.keys() if "router" in k.lower()]
        print(f"\n📊 Router parameters found: {len(router_params)}")
        
        if router_params:
            print("\nRouter parameters:")
            for i, param_name in enumerate(router_params[:10]):  # Show first 10
                param = model_state[param_name]
                if isinstance(param, torch.Tensor):
                    print(f"  {param_name}: shape={param.shape}, dtype={param.dtype}")
                else:
                    print(f"  {param_name}: {type(param)}")
            
            if len(router_params) > 10:
                print(f"  ... and {len(router_params) - 10} more")
                
            # Check if router weights have proper storage
            for param_name in router_params:
                if "weight" in param_name:
                    param = model_state[param_name]
                    if isinstance(param, torch.Tensor):
                        try:
                            storage_size = param.untyped_storage().nbytes() if hasattr(param, "untyped_storage") else param.storage().nbytes()
                            expected_size = param.numel() * param.element_size()
                            if storage_size == 0:
                                print(f"\n⚠️  WARNING: {param_name} has zero storage!")
                                print(f"    Shape: {param.shape}, Expected storage: {expected_size} bytes")
                            elif storage_size < expected_size:
                                print(f"\n⚠️  WARNING: {param_name} has insufficient storage!")
                                print(f"    Shape: {param.shape}, Storage: {storage_size} bytes, Expected: {expected_size} bytes")
                        except Exception as e:
                            print(f"\n⚠️  WARNING: Could not check storage for {param_name}: {e}")
        else:
            print("❌ No router parameters found in checkpoint!")
            print("\nThis checkpoint might be:")
            print("  1. A dense model (no MoE layers)")
            print("  2. Missing router weights")
            print("  3. Using a different naming convention")
            
            # Show a sample of parameter names
            print("\nSample parameter names:")
            for i, param_name in enumerate(list(model_state.keys())[:20]):
                print(f"  {param_name}")
            if len(model_state) > 20:
                print(f"  ... and {len(model_state) - 20} more")
            
            return False
        
        # Check for MoE-related parameters
        moe_params = [k for k in model_state.keys() if "moe" in k.lower() or "expert" in k.lower()]
        print(f"\n📊 MoE/Expert parameters found: {len(moe_params)}")
        
        # Check model architecture
        print("\n📐 Architecture hints:")
        if any("feed_forward_moe" in k for k in model_state.keys()):
            print("  ✓ Found 'feed_forward_moe' - likely MoE model")
        elif any("feed_forward." in k for k in model_state.keys()):
            print("  ⚠️  Found 'feed_forward.' - might be dense model")
        
        # Check for router bias (custom router type)
        expert_bias_params = [k for k in model_state.keys() if "expert_bias" in k]
        if expert_bias_params:
            print(f"  ✓ Found expert_bias parameters: {len(expert_bias_params)}")
            print("    This suggests MoERouterWithExpertBias is used")
        
        print("\n✅ Checkpoint diagnostic complete")
        return len(router_params) > 0
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_checkpoint.py <checkpoint_path>")
        print("\nExample:")
        print("  python scripts/diagnose_checkpoint.py /weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = diagnose_checkpoint(checkpoint_path)
    sys.exit(0 if success else 1)

