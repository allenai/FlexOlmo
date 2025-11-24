import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    # NumpyDatasetType,
    TokenizerConfig,
    VSLCurriculumConfig,
    VSLCurriculumType,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.common import (
    get_beaker_username,
    get_work_dir,
)
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

from flexolmo.data.mixes import CustomDataMix, get_mixture_dataset_config
from flexolmo.eval.evaluator_callback import DownstreamEvaluatorUpdatedCallbackConfig

log = logging.getLogger(__name__)


@dataclass
class CommonComponents(Config):
    run_name: str
    save_folder: str
    tokenizer: TokenizerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    callbacks: Dict[str, Callback]
    trainer: TrainerConfig


@dataclass
class ExperimentConfig(Config):
    run_name: str
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def get_root_dir(cluster: str = "") -> str:
    root_dir: str = "/weka/oe-training-default/ai2-llm"
    if "google" in cluster:
        root_dir = "gs://ai2-llm"
    elif "local" in cluster:
        root_dir = "gs://ai2-llm"
    return root_dir


def get_save_dir(root_dir: str, run_name: str) -> str:
    if (beaker_username := get_beaker_username()) is not None:
        return f"{root_dir}/checkpoints/{beaker_username.lower()}/{run_name}"
    else:
        return f"{root_dir}/checkpoints/{run_name}"


def build_common_components(
    run_name: str,
    *,
    root_dir: str,
    global_batch_size: int,
    sequence_length: int = 4096,
    include_default_evals: bool = True,
    freeze_embeddings: bool = False,
) -> CommonComponents:

    tokenizer_config = TokenizerConfig.dolma2()

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        min_sequence_length=min(256, sequence_length),
        max_sequence_length=max(8192, sequence_length),
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False
        ),
        work_dir=get_work_dir(root_dir),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=4
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * sequence_length,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=AdamWConfig(
            lr=9e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=(
                []
                if freeze_embeddings
                else [OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))]
            ),
            fused=True,
        ),
        state_dict_save_opts={"flatten_optimizer_state_dict": True},
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        # float8_config=Float8Config(enabled=False),
        float8_config=Float8Config(
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
            enabled=False,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )

    callbacks: Dict[str, Callback] = {
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=run_name, enabled=False),
        "checkpointer": CheckpointerCallback(
            save_interval=10_000, ephemeral_save_interval=250, save_async=True
        ),
        "comet": CometCallback(
            name=run_name,
            workspace="ai2",
            project="OLMo-modular",
            enabled=False,
            cancel_check_interval=10,
        ),
        "wandb": WandBCallback(
            name=run_name,
            entity="ai2-llm",
            project="OLMo-modular",
            enabled=True,
            cancel_check_interval=10,
        ),
    }
    beaker_user = get_beaker_username()
    if beaker_user is not None:
        callbacks["beaker"] = BeakerCallback()

    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()

    # if include_default_evals:
    #     callbacks["lm_evaluator"] = LMEvaluatorCallbackConfig(
    #         eval_dataset=NumpyDatasetConfig.from_data_mix(
    #             DataMix.v3_small_ppl_validation,
    #             name=NumpyDatasetType.padded_fsl,
    #             mix_base_dir=root_dir,
    #             sequence_length=dataset_config.effective_sequence_length,
    #             tokenizer=tokenizer_config,
    #             work_dir=get_work_dir(root_dir),
    #         ),
    #         eval_interval=1000,
    #     )

    #     tasks = [
    #         "piqa",
    #         "hellaswag",
    #         "winogrande",
    #         "openbook_qa",
    #         "boolq",
    #         "sciq",
    #         "xsum",
    #         "wildbench_math",
    #         "wildbench_reasoning",
    #         "wildbench_coding_debugging",
    #         "wildbench_creative_writing",
    #         "mmlu_stem_val_rc_5shot",
    #         "mmlu_humanities_val_rc_5shot",
    #         "mmlu_social_sciences_val_rc_5shot",
    #         "mmlu_other_val_rc_5shot",
    #     ]

    #     callbacks["downstream_evaluator"] = DownstreamEvaluatorUpdatedCallbackConfig(
    #         tasks=[task for task in tasks[:2] if "_mc" not in task and "_var" not in task],
    #         tokenizer=tokenizer_config,
    #         eval_interval=1_000,
    #     )

    trainer_config = TrainerConfig(
        save_folder=get_save_dir(root_dir, run_name),
        save_overwrite=True,
        metrics_collect_interval=10,
        cancel_check_interval=1,
        max_duration=Duration.tokens(int(50e9)),
    )

    return CommonComponents(
        run_name=run_name,
        save_folder=get_save_dir(root_dir, run_name),
        tokenizer=tokenizer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        callbacks=callbacks,
        trainer=trainer_config,
    )


def build_experiment_config(
    run_name: str,
    overrides: List[str],
    *,
    root_dir: str,
    sequence_length: int,
    global_batch_size: int,
    include_default_evals: bool = True,
    freeze_embeddings: bool = False,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    train_module_config_builder: Callable[[CommonComponents], TransformerTrainModuleConfig],
    trainer_config_builder: Optional[Callable[[CommonComponents], TrainerConfig]] = None,
    dataset_config_builder: Optional[Callable[[CommonComponents], NumpyDatasetConfig]] = None,
    data_loader_config_builder: Optional[
        Callable[[CommonComponents], NumpyDataLoaderConfig]
    ] = None,
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
):

    common = build_common_components(
        run_name=run_name,
        root_dir=root_dir,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        include_default_evals=include_default_evals,
        freeze_embeddings=freeze_embeddings,
    )

    model = model_config_builder(common)

    trainer = (
        trainer_config_builder(common) if trainer_config_builder is not None else common.trainer
    )
    for name, cb in common.callbacks.items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    dataset = (
        dataset_config_builder(common) if dataset_config_builder is not None else common.dataset
    )
    data_loader = (
        data_loader_config_builder(common)
        if data_loader_config_builder is not None
        else common.data_loader
    )

    train_module = (
        train_module_config_builder(common)
        if train_module_config_builder is not None
        else common.train_module
    )

    if model.freeze_params is not None and any(
        ["embeddings." in param for param in model.freeze_params]
    ):
        log.warning("Embeddings are frozen, updating train_module optim group overrides.")
        freeze_embeddings = True
        assert isinstance(train_module.optim, AdamWConfig)
        train_module.optim.group_overrides = (
            []
            if freeze_embeddings
            else [OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))]
        )

    config = ExperimentConfig(
        run_name=run_name,
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
        trainer=trainer,
    )

    if finalize_config is not None:
        finalize_config(config)

    for idx, key in enumerate(overrides):
        if "dataset.mix" in key and "dataset.mix_base_dir" not in key:
            override_datamix = key.split("=")[1]
            override_datamix_idx = idx
            break
    else:
        override_datamix = None
        override_datamix_idx = None

    if override_datamix is not None:
        config.dataset.mix = override_datamix
        if "," in override_datamix:
            config.dataset.source_mixture_config = get_mixture_dataset_config(config.dataset)
            config.dataset.mix = None
        else:
            config.dataset.source_mixture_config = None

        assert override_datamix_idx is not None
        overrides.pop(override_datamix_idx)

    config = config.merge(overrides)

    if config.dataset.mix is not None:
        try:
            config.dataset.mix = CustomDataMix(config.dataset.mix)
        except ValueError:
            config.dataset.mix = DataMix(config.dataset.mix)

    return config


def print_model_params(config: ExperimentConfig):
    print(
        "\n"
        f"[b blue]Total parameters:[/]         {config.model.num_params:,d} ({config.model.num_active_params:,d} active)\n"
        f"[b blue]Non-embedding parameters:[/] {config.model.num_non_embedding_params:,d} ({config.model.num_active_non_embedding_params:,d} active)"
    )


def is_dry_run(arglist: List[str]) -> bool:
    return arglist[1] == "--dry_run" or arglist[1] == "dry_run"
