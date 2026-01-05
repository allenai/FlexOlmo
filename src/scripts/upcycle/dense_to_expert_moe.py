import argparse
import json
import logging

import numpy as np
import torch
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_state_dict, load_keys, get_checkpoint_metadata
from olmo_core.nn.moe import MoEConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import prepare_cli_environment

from flexolmo.internal.model_utils import *  # noqa

log = logging.getLogger(__name__)


def build_model_config(num_experts: int = 2) -> TransformerConfig:
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
    # Handle both cases:
    # 1. Config is already a model config (e.g., expert 1/3)
    # 2. Config is a full training config with nested model config (e.g., expert 0)
    
    log.info(f"Config keys: {list(config.keys())}")
    
    if "model" in config:
        # Case 2: Full training config with nested model config
        model_config_dict = config["model"].copy()
        log.info(f"Using nested model config. Model config keys: {list(model_config_dict.keys())}")
    else:
        # Case 1: Config is already the model config
        model_config_dict = config.copy()
        log.info(f"Using direct model config. Config keys: {list(model_config_dict.keys())}")

    # our annealed checkpoints were trained on v1, and v2 doesn't have these keys in the config
    dp_config = model_config_dict.pop("dp_config", None)  # noqa: F841
    compile_k = model_config_dict.pop("compile", None)  # noqa: F841
    float8_config = model_config_dict.pop("float8_config", None)  # noqa: F841

    # Fix vocab_size mismatch - use the model's vocab_size if available
    if "dataset" in config and "tokenizer" in config["dataset"]:
        tokenizer_vocab_size = config["dataset"]["tokenizer"].get("vocab_size")
        if tokenizer_vocab_size and model_config_dict.get("vocab_size") != tokenizer_vocab_size:
            log.warning(f"Fixing vocab_size mismatch: model={model_config_dict.get('vocab_size')}, tokenizer={tokenizer_vocab_size}")
            model_config_dict["vocab_size"] = model_config_dict.get("vocab_size")

    log.info(f"Model config dict after cleanup: {list(model_config_dict.keys())}")
    log.info(f"Block config: {model_config_dict.get('block', 'NOT FOUND')}")
    
    if "block" not in model_config_dict:
        raise ValueError(f"No 'block' key found in model config. Available keys: {list(model_config_dict.keys())}")
    
    # Ensure the block config has the correct _CLASS_ field
    if "_CLASS_" not in model_config_dict["block"]:
        model_config_dict["block"]["_CLASS_"] = "olmo_core.nn.transformer.TransformerBlockConfig"
    
    # Clean up any invalid fields that might cause issues
    invalid_fields = ["init_std"]  # This field might not be supported in current version
    for field in invalid_fields:
        if field in model_config_dict:
            log.warning(f"Removing unsupported field: {field}")
            model_config_dict.pop(field, None)

    try:
        model_config = TransformerConfig.from_dict(model_config_dict)
        return model_config
    except Exception as e:
        log.error(f"Failed to create TransformerConfig from dict: {e}")
        log.error(f"Model config dict: {model_config_dict}")
        # Try to identify the specific problematic field
        if "vocab_size" in str(e):
            log.error("Vocab size mismatch detected. Check tokenizer vs model vocab_size.")
        raise


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


def cosine_similarity(a, b):
    return torch.sum(a * b) / (torch.norm(a) * torch.norm(b))


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Merge dense unsharded models into a MoE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-m", "--models", nargs="+", default=[])
    parser.add_argument(
        "-t", "--target", type=str, default=None, help="Target path to save the merged model"
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        nargs="+",
        default=[],
        help="Paths to the embeddings, to optionally seed the router (should follow the same order as models)",
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    prepare_cli_environment()

    args = parse_args()

    moe_to_dense_mapping = {
        "feed_forward_moe.experts.mlp.w1": "feed_forward.w1.weight",
        "feed_forward_moe.experts.mlp.w2": "feed_forward.w2.weight",
        "feed_forward_moe.experts.mlp.w3": "feed_forward.w3.weight",
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

    dense_paths = args.models
    target_path = args.target
    embeddings = args.embeddings

    concat_embed = None

    if len(embeddings) > 0:

        embeds = []
        for embed_path in embeddings:
            log.info(f"Loading embedding from {embed_path}")
            embeds.append(np.load(embed_path))

        # concatenate the embeddings
        concat_embed = np.concatenate(embeds, axis=0)
        # make it a tensor
        concat_embed = torch.from_numpy(concat_embed).float()

        log.info(f"Considering {embeddings[0]} as public embeddings")

        for i, embed_path in enumerate(embeddings[1:]):
            print(
                f"Cosine similarity between public and {embed_path}: {cosine_similarity(concat_embed[:4096], concat_embed[(i+1)*4096:(i+2)*4096])}"
            )

    # load the MoE model config
    model_config = build_model_config(len(dense_paths))
    log.info(model_config)

    assert isinstance(model_config.block.feed_forward_moe, MoEConfig)
    assert model_config.block.feed_forward_moe.num_experts == len(
        dense_paths
    ), "Number of experts should match the number of dense models"

    log.info("Loading the MoE model on cpu")
    model = model_config.build(init_device="cpu")
    log.info("Model loaded on cpu")
    moe_state_dict = model.state_dict()

    # Load config from first expert only (expert == 0)
    first_config = None
    for expert, path in enumerate(dense_paths):
        log.info(f"Loading dense model from {path} as expert {expert}")
        
        # Only load config for first expert, reuse for others
        if expert == 0:
            with open(path + "/config.json") as f:
                first_config = json.load(f)
            log.info(f"Dense model config {load_model_config(first_config)}")
        else:
            log.info(f"Using config from first expert for expert {expert}")

        dense_state_dict = load_state_dict_distributed(path)
        log.info(f"Expert {expert} dense model loaded")

        # copy over the keys in the dense state_dict to final_state_dict
        for key in list(moe_state_dict.keys()):
            if any(pattern in key for pattern in list(moe_to_dense_mapping.keys())):
                dense_key = None
                for pattern in moe_to_dense_mapping:
                    if pattern in key:
                        dense_key = key.replace(pattern, moe_to_dense_mapping[pattern])
                        break
                if dense_key is None:
                    log.warning(f"No dense key mapping for '{key}', skipping")
                    continue
                if dense_key not in dense_state_dict:
                    sample_keys = list(dense_state_dict.keys())[:25]
                    raise KeyError(
                        f"Missing '{dense_key}' in dense checkpoint at {path}. Sample keys: {sample_keys}"
                    )
                log.info(f"Copying key {dense_key} to {key} in MoE model")
                if "expert" in key or "router" in key:
                    dim = dense_state_dict[dense_key].shape[1]
                    log.info(f"Collapsing {key} to 2D. Transposing {dense_key} for {key}")
                    moe_state_dict[key][dim * (expert) : dim * (expert + 1), :] = dense_state_dict[
                        dense_key
                    ].transpose(0, 1)
                else:
                    # Handle frozen weights (embeddings, attention, etc.)
                    if expert > 0:
                        # Check if the frozen weights are the same
                        if torch.equal(moe_state_dict[key], dense_state_dict[dense_key]):
                            log.info(f"Key {key} is identical across experts")
                        else:
                            # Different frozen weights - this can happen with mixed expert types (SFT vs base models)
                            log.warning(f"Key {key} is different between experts - this is expected for mixed expert types")
                            log.warning(f"Expert {expert} has different {key} than expert 0")
                            # For mixed expert types, we need to decide how to handle this
                            # Option 1: Use the first expert's weights (current behavior)
                            # Option 2: Take the mean of all expert weights
                            # Option 3: Use expert-specific weights for each expert
                            
                            # For now, we'll use the first expert's weights and log a warning
                            log.warning(f"Using expert 0's {key} for all experts")
                    else:
                        moe_state_dict[key] = dense_state_dict[dense_key]
            else:
                # check if they are the same
                if key in dense_state_dict:
                    if not torch.equal(moe_state_dict[key], dense_state_dict[key]):
                        log.info(f"{key} is different")
                elif "router.weight" in key:
                    if concat_embed is not None:
                        log.warning(f"{key} copy domain embed")
                        moe_state_dict[key] = concat_embed
                    else:
                        log.warning("No embeddings provided, skipping router weight copy")
                elif "expert_bias" in key:
                    log.warning(f"{key}: {moe_state_dict[key]}")
                else:
                    log.warning(f"{key} equivalent not found in dense model")

    # save the final_state_dict for the MoE in a format that the olmo_core trainer likes
    save_state_dict(target_path, {"model": moe_state_dict}, save_overwrite=True)
    
    # Create the unsharded directory before saving
    import os
    unsharded_path = target_path + "-unsharded"
    os.makedirs(unsharded_path, exist_ok=True)
    torch.save(moe_state_dict, unsharded_path + "/model.pt")

    log.info(f"Model saved to {target_path}")
    log.info("Done")