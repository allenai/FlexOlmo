from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
)

from flexolmo.nn.moe.router import ExtendedMoERouterConfig, ExtendedMoERouterType


def olmo2_tiny(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
    return cls.llama_like(
        d_model=32,
        hidden_size_multiplier=1.5,
        n_layers=kwargs.pop("n_layers", 1),
        n_heads=kwargs.pop("n_heads", 1),
        vocab_size=vocab_size,
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


TransformerConfig.olmo2_tiny = classmethod(olmo2_tiny)  # type: ignore


def olmoe_nx7b(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
    # Possibly more OOM due to imbalance with dropless=True
    dropless = kwargs.pop("dropless", False)
    return cls.llama_like_moe(
        d_model=kwargs.pop("d_model", 4096),
        n_layers=kwargs.pop("n_layers", 32),
        n_heads=kwargs.pop("n_heads", 32),
        num_experts=kwargs.pop("num_experts", 2),
        top_k=kwargs.pop("top_k", 1),
        expert_hidden_size=kwargs.pop("expert_hidden_size", 11008),
        vocab_size=vocab_size,
        dropless=dropless,
        capacity_factor=None if dropless else 1.2,  # adjust as needed
        lb_loss_weight=kwargs.pop("lb_loss_weight", 0.01),
        z_loss_weight=kwargs.pop("z_loss_weight", 0.001),
        reordered_norm=kwargs.pop("reordered_norm", True),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=kwargs.pop("layer_norm_eps", 1e-6),
        **kwargs,
    )


TransformerConfig.olmoe_nx7b = classmethod(olmoe_nx7b)  # type: ignore


def olmoe_nx7b_with_expert_bias(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
    model_config = cls.olmoe_nx7b(vocab_size=vocab_size, **kwargs)
    # Overwrite the feed_forward_moe config to use expert bias
    model_config.block.feed_forward_moe.router = ExtendedMoERouterConfig(
        name=ExtendedMoERouterType.with_expert_bias, top_k=1  # type: ignore
    )
    return model_config


TransformerConfig.olmoe_nx7b_with_expert_bias = classmethod(olmoe_nx7b_with_expert_bias)  # type: ignore
