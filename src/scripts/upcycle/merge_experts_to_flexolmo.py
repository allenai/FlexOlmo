import argparse
import json
import logging

import torch
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_state_dict, load_keys, get_checkpoint_metadata
from olmo_core.nn.moe import MoEConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import prepare_cli_environment

from flexolmo.internal.model_utils import *  # noqa

log = logging.getLogger(__name__)


def build_model_config(num_experts: int = 3) -> TransformerConfig:
    tokenizer = TokenizerConfig.dolma2()
    return TransformerConfig.olmoe_nx7b_with_expert_bias(  # type: ignore
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=num_experts,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
        ],
    )


def load_model_config(config: dict) -> TransformerConfig:

    # our annealed checkpoints were trained on v1, and v2 doesn't have these keys in the config
    dp_config = config["model"].pop("dp_config", None)  # noqa: F841
    compile_k = config["model"].pop("compile", None)  # noqa: F841
    float8_config = config["model"].pop("float8_config", None)  # noqa: F841

    model_dict = config["model"].copy()

    # Handle the router config manually
    if "block" in model_dict and "feed_forward_moe" in model_dict["block"]:
        moe_config = model_dict["block"]["feed_forward_moe"]
        if "router" in moe_config and moe_config["router"].get("name") == "with_expert_bias":
            # Manually create the extended router config
            router_dict = moe_config["router"].copy()
            router_dict.pop("_CLASS_", None)
            moe_config["router"] = ExtendedMoERouterConfig(**router_dict)

    # Remove _CLASS_ entries and use original TransformerConfig
    model_dict.pop("_CLASS_", None)
    return TransformerConfig(**model_dict)


def load_trainer_config(config: dict) -> TrainerConfig:

    lr_scheduler = config["trainer"]["callbacks"].pop("lr_scheduler", None)  # noqa: F841
    grad_clipper = config["trainer"]["callbacks"].pop("grad_clipper", None)  # noqa: F841
    float8_handler = config["trainer"]["callbacks"].pop("float8_handler", None)  # noqa: F841

    rank_microbatch_size = config["trainer"].pop("rank_microbatch_size", None)  # noqa: F841
    load_key_mapping = config["trainer"].pop("load_key_mapping", None)  # noqa: F841
    fused_loss = config["trainer"].pop("fused_loss", None)  # noqa: F841
    compile_loss = config["trainer"].pop("compile_loss", None)  # noqa: F841
    z_loss_multiplier = config["trainer"].pop("z_loss_multiplier", None)  # noqa: F841

    trainer_config = TrainerConfig.from_dict(config["trainer"])
    return trainer_config


def load_state_dict(path: str):
    state_dict = torch.load(path + "/model.pt", map_location="cpu")
    return state_dict


def load_state_dict_distributed(path: str):
    """
    Load a state dictionary from a distributed checkpoint using OLMo-core's distributed checkpoint loading.
    Returns the same type as the original load_state_dict function.
    """
    try:
        # Try OLMo-core distributed checkpoint path first
        ckpt_dir = path + "/model_and_optim"
        metadata = get_checkpoint_metadata(ckpt_dir)
        model_keys = [
            key[len("model.") :]
            for key in metadata.state_dict_metadata.keys()
            if key.startswith("model.")
        ]
        loaded_values = list(load_keys(ckpt_dir, [f"model.{k}" for k in model_keys]))
        return {k: v for k, v in zip(model_keys, loaded_values)}
    except Exception:
        # Fall back to regular torch.load
        state_dict = torch.load(path + "/model.pt", map_location="cpu")
        return state_dict


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Merge dense unsharded models into a MoE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-m", "--models", nargs="+", default=[])
    parser.add_argument(
        "-t", "--target", type=str, default=None, help="Target path to save the merged model"
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    prepare_cli_environment()

    args = parse_args()

    expert_paths = args.models
    target_path = args.target

    moe_to_expert_mapping = {
        "feed_forward_moe.experts.mlp.w1": "feed_forward_moe.experts.mlp.w1",
        "feed_forward_moe.experts.mlp.w2": "feed_forward_moe.experts.mlp.w2",
        "feed_forward_moe.experts.mlp.w3": "feed_forward_moe.experts.mlp.w3",
        "feed_forward_moe.router.weight": "feed_forward_moe.router.weight",
        "attention.q_norm.weight": "attention.q_norm.weight",
        "attention.k_norm.weight": "attention.k_norm.weight",
        "attention_norm.weight": "attention_norm.weight",
        "attention.w_q.weight": "attention.w_q.weight",
        "attention.w_k.weight": "attention.w_k.weight",
        "attention.w_v.weight": "attention.w_v.weight",
        "attention.w_out.weight": "attention.w_out.weight",
        "feed_forward_norm.weight": "feed_forward_norm.weight",
        "lm_head.norm.weight": "lm_head.norm.weight",
        "lm_head.w_out.weight": "lm_head.w_out.weight",
        "embeddings.weight": "embeddings.weight",
    }

    # load the MoE model config
    model_config = build_model_config(len(expert_paths))
    log.info(model_config)

    assert isinstance(model_config.block.feed_forward_moe, MoEConfig)
    assert model_config.block.feed_forward_moe.num_experts == len(
        expert_paths
    ), "Number of experts should match the number of dense models"

    log.info("Loading the MoE model on cpu")
    model = model_config.build(init_device="cpu")
    log.info("Model loaded on cpu")
    moe_state_dict = model.state_dict()

    merged_config_dict = {}
    for expert, path in enumerate(expert_paths):
        log.info(f"Loading model from {path} as expert {expert}")
        with open(path + "/config.json") as f:
            config = json.load(f)
        if expert == 0:
            merged_config_dict = config

        expert_state_dict = load_state_dict_distributed(path)
        # bp()
        log.info(f"Expert model config {load_model_config(config)}")
        log.info(f"Expert {expert} model loaded")

        # copy over the keys in the dense state_dict to final_state_dict
        for key in list(moe_state_dict.keys()):
            if any(pattern in key for pattern in list(moe_to_expert_mapping.keys())):
                dense_key = None
                for pattern in moe_to_expert_mapping:
                    if pattern in key:
                        dense_key = key.replace(pattern, moe_to_expert_mapping[pattern])
                        break
                if dense_key is None:
                    log.warning(f"No dense key mapping for '{key}', skipping")
                    continue
                if dense_key not in expert_state_dict:
                    sample_keys = list(expert_state_dict.keys())[:25]
                    raise KeyError(
                        f"Missing '{dense_key}' in expert checkpoint at {path}. Sample keys: {sample_keys}"
                    )
                log.info(f"Copying key {dense_key} to {key} in MoE model")
                if "expert" in key:
                    # bp()
                    dim = int(expert_state_dict[dense_key].shape[0] / 2)
                    if expert == 0:
                        # get the first half of the dense weights
                        moe_state_dict[key][dim * (expert) : dim * (expert + 1), :] = (
                            expert_state_dict[dense_key][:dim, :]
                        )
                    else:
                        # get the second half of the dense weights
                        # check if expert is actually frozen for the first part
                        assert torch.equal(
                            moe_state_dict[key][dim * (0) : dim * (0 + 1), :],
                            expert_state_dict[dense_key][:dim, :],
                        ), f"First part of the dense weights are not frozen: {key}"
                        moe_state_dict[key][dim * (expert) : dim * (expert + 1), :] = (
                            expert_state_dict[dense_key][dim:, :]
                        )
                elif "router" in key:
                    dim = int(expert_state_dict[dense_key].shape[0] / 2)
                    if expert == 0:
                        moe_state_dict[key][dim * (expert) : dim * (expert + 1)] = (
                            expert_state_dict[dense_key][:dim]
                        )
                    else:
                        # bp()
                        assert torch.equal(
                            moe_state_dict[key][dim * (0) : dim * (0 + 1)],
                            expert_state_dict[dense_key][:dim],
                        ), f"First part of the dense weights are not frozen: {key}"
                        moe_state_dict[key][dim * (expert) : dim * (expert + 1)] = (
                            expert_state_dict[dense_key][dim:]
                        )
                else:
                    # # option 1: check if the frozen weights are the same
                    if expert > 0:
                        assert torch.equal(
                            moe_state_dict[key], expert_state_dict[dense_key]
                        ), f"Key {key} is different"  # check if the frozen weights are the same
                    else:
                        moe_state_dict[key] = expert_state_dict[dense_key]
            else:
                log.info(f"Key {key} not found in dense model")
                # raise Exception("Key not found")
                # # check if they are the same
                # if key in expert_state_dict:
                #     if not torch.equal(moe_state_dict[key], expert_state_dict[key]):
                #         log.info(f"{key} is different")
                # else:
                #     log.warning(f"{key} equivalent not found in dense model")
        del expert_state_dict
    # save the final_state_dict for the MoE in a format that the olmo_core trainer likes
    save_state_dict(target_path, {"model": moe_state_dict})
    log.info(f"Model saved to {target_path}")

    merged_config_dict["model"] = model_config.as_dict()
    merged_config_dict["run_name"] = (
        f"merged_{model_config.block.feed_forward_moe.num_experts}_experts"
    )

    with open(target_path + "/config.json", "w") as f:
        json.dump(merged_config_dict, f)
    log.info(f"Config saved to {target_path}/config.json")
    log.info("Done")
