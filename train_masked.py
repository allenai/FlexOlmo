import os
import subprocess
import sys

# The first argument is the run name from torchrun, ignore it
if len(sys.argv) > 1:
    run_name = sys.argv[1]
    print(f"Ignoring run name from torchrun: {run_name}")

# Install olmo-core from your commit with masking support
subprocess.run(['pip', 'uninstall', '-y', 'olmo-core'], check=False)
subprocess.run([
    'pip', 'install',
    'git+https://github.com/allenai/OLMo-core.git@f18a9bf44496acc1fa0cad7d8c8b9fb111eff315'
], check=True)

# Run the training script
cmd = [
    'python', 'src/scripts/train/OLMoE-4x7B.py', 'FlexOlmo-4x7B-RT-experts-sft-masked',
    '--trainer.callbacks.profiler.enabled=true',
    '--dataset.mix_base_dir=/weka/oe-training-default/ai2-llm/',
    '--dataset.mix=router_training_mix',
    '--trainer.max_duration.value=5_000_000_000',
    '--trainer.max_duration.unit=tokens',
    '--trainer.load_path=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-codee-xperts-sft-math-mixed',
    '--model.block.feed_forward_moe.router.disabled_experts=[3]',
    '--model.block.feed_forward_moe.router.top_k=3',
    '--train_module.rank_microbatch_size=4096',
    '--train_module.scheduler.warmup_steps=100',
    '--train_module.optim.lr=2e-3',
    '--trainer.save_folder=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7b-flex-base-merged-math-code-RT-experts-sft-math-mixed-masked'
]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd, check=True)