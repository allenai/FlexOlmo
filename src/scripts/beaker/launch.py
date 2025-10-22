"""
Launch runs on Beaker.
"""

import argparse
import logging
import sys
from typing import List, Tuple

from olmo_core.internal.common import (
    BeakerEnvVar,
    get_beaker_username,
)
from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    BeakerWekaBucket,
)
from olmo_core.utils import (
    generate_uuid,
    prepare_cli_environment,
)
from rich import print

from flexolmo.internal.common import get_root_dir

log = logging.getLogger(__name__)


def build_launch_config(
    name: str,
    command: List[str],
    cluster: str,
    overrides: List[str],
    nccl_debug: bool = False,
    cuda_debug: bool = False,
) -> BeakerLaunchConfig:
    root_dir = get_root_dir(cluster)
    weka_buckets: List[BeakerWekaBucket] = []
    if root_dir.startswith("/weka/"):
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    beaker_user = get_beaker_username()

    return BeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=command,
        task_name="train",
        workspace="ai2/OLMo-modular",
        clusters=[cluster],
        weka_buckets=weka_buckets,
        beaker_image="petew/olmo-core-tch260cu124",
        num_nodes=1,
        num_gpus=8,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_vars=[
            BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if nccl_debug else "WARN"),
            BeakerEnvVar(name="CUDA_LAUNCH_BLOCKING", value="1" if cuda_debug else "0"),
        ],
        env_secrets=[
            BeakerEnvSecret(name="GITHUB_TOKEN", secret=f"{beaker_user}_GITHUB_TOKEN"),
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
            # BeakerEnvSecret(name="COMET_API_KEY", secret=f"{beaker_user}_COMET_API_KEY"),
            BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
            BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
            # BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            # BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            # BeakerEnvSecret(name="SLACK_WEBHOOK_URL", secret="SLACK_WEBHOOK_URL"),
        ],
        setup_steps=[
            # Clone private repo.
            "conda install gh --channel conda-forge",
            # assumes that conda is installed, which is true for our beaker images. # TODO: add to image
            'gh repo clone "$REPO_URL" .',
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[dev,beaker,wandb,train]'",  # we don't need eval, and it causes dependency conflicts
            "pip uninstall -y olmo-core && pip install git+https://github.com/allenai/OLMo-core.git@8f8e97a584fb5b8e78145a8f905efbfec0f4e7ec",
            "pip freeze",
            # Move AWS credentials from env to relevant files
            "mkdir -p ~/.aws",
            "printenv AWS_CONFIG > ~/.aws/config",
            "printenv AWS_CREDENTIALS > ~/.aws/credentials",
        ],
    ).merge(overrides)


def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:

    # Find the separator '--' to split launch args from script args
    separator_idx = None
    try:
        separator_idx = sys.argv.index("--")
    except ValueError:
        pass

    if separator_idx is not None:
        launch_args = sys.argv[1:separator_idx]
        script_args = sys.argv[separator_idx + 1 :]
    else:
        launch_args = sys.argv[1:]
        script_args = []

    # Separate launch overrides from other launch args
    filtered_launch_args = []
    launch_overrides = []

    for arg in launch_args:
        if arg.startswith("--launch."):
            launch_overrides.append(arg.replace("--launch.", "--"))
        else:
            filtered_launch_args.append(arg)

    # Create parser for main launch arguments
    parser = argparse.ArgumentParser(
        description="Launch script with support for launch overrides and script command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments

    parser.add_argument(
        "cmd", choices=["dry_run", "launch"], help="Command to execute: dry_run or launch"
    )

    parser.add_argument("cluster", help="Name of the cluster to launch on")

    # Parse the filtered launch arguments
    parsed_args = parser.parse_args(filtered_launch_args)

    return parsed_args, script_args, launch_overrides


def update_command(script_command: List[str], cmd: str) -> List[str]:
    # TODO: Bit hacky
    if cmd == "dry_run":
        return ["python"] + script_command[:1] + ["dry_run"] + script_command[1:]
    elif cmd == "launch":
        return script_command
    else:
        raise ValueError(f"Unknown command: {cmd}. Expected 'dry_run' or 'launch'.")


def main():

    args, script_command, launch_overrides = parse_args()
    positional_script_args = [arg for arg in script_command[1:] if not arg.startswith("--")]

    assert (
        len(positional_script_args) >= 1
    ), "At least one positional argument (run name) is required in the script command (after -- )."
    run_name = positional_script_args[0]

    script_command = update_command(script_command=script_command, cmd=args.cmd)

    prepare_cli_environment()

    config = build_launch_config(
        name=run_name,
        command=script_command,
        cluster=args.cluster,
        overrides=launch_overrides,
        nccl_debug=False,
        cuda_debug=False,
    )

    print(config)

    if args.cmd == "dry_run":
        import subprocess

        subprocess.run(script_command, check=True)

    elif args.cmd == "launch":
        config.launch(follow=True, torchrun=True)


if __name__ == "__main__":

    """
    # Example usage:
    python src/scripts/beaker/launch.py [launch|dry_run] [cluster] \
        --launch.num_nodes=1 \
        --launch.num_gpus=1 \
        --launch.workspace=workspace-name \
        --launch.priority=high -- src/scripts/train/OLMo2-tiny-train.py test-run01 \
            --trainer.callbacks.profiler.enabled=false \
            --train_module.optim.lr=2e-4
    """
    main()
