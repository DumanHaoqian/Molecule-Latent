import os
import argparse
import re
from datetime import datetime

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import AutoTokenizer

from data_provider.stage1_dm import Stage1DM
from trainer.stage1 import Stage1Trainer
from utils.configuration_mol_llama import MolLLaMAConfig


def _load_configs(train_cfg_path=None, data_cfg_path=None):
    train_cfg_path = train_cfg_path or os.path.join("configs", "stage1", "train_config.yaml")
    data_cfg_path = data_cfg_path or os.path.join("configs", "stage1", "data_config.yaml")
    train_config = OmegaConf.load(train_cfg_path)
    data_config = OmegaConf.load(data_cfg_path)
    return train_config, data_config


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_plain(obj):
    return OmegaConf.to_container(obj, resolve=True) if OmegaConf.is_config(obj) else obj


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, None)
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_trainer_runtime(train_config):
    accelerator = getattr(train_config, "accelerator", "gpu")
    devices = getattr(train_config, "devices", 1)
    strategy = getattr(train_config, "strategy_name", "auto")
    num_nodes = int(getattr(train_config, "num_nodes", 1))
    sync_batchnorm = bool(getattr(train_config, "sync_batchnorm", False))

    use_accelerate_launch = bool(getattr(train_config, "use_accelerate_launch", False))
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", -1)
    rank = _env_int("RANK", 0)
    launched_multi_process = world_size > 1 and local_rank >= 0

    if use_accelerate_launch or launched_multi_process:
        strategy = str(getattr(train_config, "accelerate_strategy", "ddp"))
        devices = int(getattr(train_config, "accelerate_devices_per_process", 1))
        num_nodes = int(getattr(train_config, "accelerate_num_nodes", 1))
        print(
            "[Stage1] accelerate-compatible runtime: "
            f"world_size={world_size}, rank={rank}, local_rank={local_rank}, "
            f"strategy={strategy}, devices_per_process={devices}, num_nodes={num_nodes}"
        )
    else:
        print(f"[Stage1] default runtime: strategy={strategy}, devices={devices}, num_nodes={num_nodes}")

    return accelerator, devices, strategy, num_nodes, sync_batchnorm


def _build_tokenizer(llm_model_name):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    mol_ids = tokenizer("<mol>", add_special_tokens=False).input_ids
    if len(mol_ids) != 1:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
        mol_ids = tokenizer("<mol>", add_special_tokens=False).input_ids
    tokenizer.mol_token_id = mol_ids[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _sanitize_name(name: str, max_len: int = 64) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip())
    s = s.strip("-.")
    if len(s) == 0:
        s = "stage1"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", default=os.path.join("configs", "stage1", "train_config.yaml"))
    parser.add_argument("--data_config", default=os.path.join("configs", "stage1", "data_config.yaml"))
    parser.add_argument("--run_name", default="", help="Optional run name override for W&B and checkpoint naming.")
    parser.add_argument(
        "--init_stage1_ckpt",
        default="",
        help="Optional Stage-I checkpoint to warm-start after loading stage0 backbone.",
    )
    args = parser.parse_args()

    train_config, data_config = _load_configs(args.train_config, args.data_config)
    run_name_override = str(args.run_name or "").strip()
    if len(run_name_override) > 0:
        train_config.wandb_run_name = run_name_override
        print(f"[Stage1] run name override: {run_name_override}")
    pl.seed_everything(int(getattr(train_config, "seed", 42)), workers=True)

    model_cfg = _cfg_get(train_config, "model", {})
    stage1_cfg = _cfg_get(train_config, "stage1", {})
    data_cfg = _cfg_get(data_config, "data", {})
    loss_weights = _cfg_get(train_config, "loss_weights", {})

    stage0_checkpoint_path = str(_cfg_get(model_cfg, "stage0_checkpoint_path", ""))
    init_stage1_ckpt = str(args.init_stage1_ckpt or "").strip()
    llm_model_name = str(_cfg_get(model_cfg, "llm_model", "meta-llama/Llama-3.1-8B-Instruct"))

    model_config = MolLLaMAConfig()
    model_config.llm_config.llm_model = llm_model_name
    model_config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]
    tokenizer = _build_tokenizer(model_config.llm_config.llm_model)
    llama_version = "llama3" if "Llama-3" in model_config.llm_config.llm_model else "llama2"

    model = Stage1Trainer(vocab_size=len(tokenizer), model_config=model_config, train_config=train_config, tokenizer=tokenizer)
    if stage0_checkpoint_path:
        print(f"[Stage1] stage0 checkpoint path: {stage0_checkpoint_path}")
        if not os.path.exists(stage0_checkpoint_path):
            raise FileNotFoundError(f"[Stage1] stage0 checkpoint path not found: {stage0_checkpoint_path}")
        if os.path.isdir(stage0_checkpoint_path):
            model.load_from_hf_dir(stage0_checkpoint_path)
        else:
            model.load_from_ckpt(stage0_checkpoint_path)
    if init_stage1_ckpt:
        print(f"[Stage1] stage1 init checkpoint path: {init_stage1_ckpt}")
        if not os.path.exists(init_stage1_ckpt):
            raise FileNotFoundError(f"[Stage1] stage1 init checkpoint path not found: {init_stage1_ckpt}")
        model.load_from_ckpt(init_stage1_ckpt)
    unimol_dictionary = getattr(model.mol_llama.encoder, "unimol_dictionary", None)

    downstream_paths = _cfg_get(data_cfg, "downstream_paths", [])
    if isinstance(downstream_paths, str):
        downstream_paths = [downstream_paths]
    datamodule = Stage1DM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        num_workers=int(_cfg_get(data_cfg, "num_workers", 0)),
        batch_size=int(_cfg_get(data_cfg, "batch_size", 1)),
        unimol_dictionary=unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        text_max_len=int(_cfg_get(data_cfg, "text_max_len", 512)),
        max_latent_slots=int(_cfg_get(stage1_cfg, "max_latent_slots", getattr(train_config, "max_latent_slots", 4))),
        latent_slot_text_max_len=int(getattr(train_config, "latent_slot_text_max_len", 48)),
        stage1_mixed_training=bool(_cfg_get(stage1_cfg, "use_mixed_stage1_training", getattr(train_config, "stage1_mixed_training", True))),
        latent_world_modeling_path=str(_cfg_get(data_cfg, "latent_path", "")),
        conversation_sft_path=str(_cfg_get(data_cfg, "conversation_path", "")),
        moledit_path=str(_cfg_get(data_cfg, "moledit_path", "")),
        downstream_tasks_paths=downstream_paths,
        fallback_raw_paths=_to_plain(_cfg_get(data_cfg, "fallback_raw_paths", {})),
        source_sampling_weights=_to_plain(_cfg_get(data_cfg, "source_sampling_weights", getattr(train_config, "source_sampling_weights", {}))),
        use_task_tokens=bool(_cfg_get(stage1_cfg, "use_task_tokens", getattr(train_config, "use_task_tokens", True))),
        regression_targets=list(_cfg_get(stage1_cfg, "regression_targets", getattr(train_config, "wm_regression_targets", []))),
        classification_targets=list(_cfg_get(stage1_cfg, "classification_targets", getattr(train_config, "wm_classification_targets", []))),
        eval_downstream_csv_paths=list(_cfg_get(data_cfg, "eval_downstream_csv_paths", [])),
        eval_sample_per_dataset=int(_cfg_get(data_cfg, "eval_sample_per_dataset", 200)),
        eval_stratified_sampling=bool(_cfg_get(data_cfg, "eval_stratified_sampling", True)),
        eval_sample_per_class=int(_cfg_get(data_cfg, "eval_sample_per_class", 100)),
        eval_seed=int(_cfg_get(data_cfg, "eval_seed", 42)),
        eval_moleculeqa_test_path=str(_cfg_get(data_cfg, "eval_moleculeqa_test_path", "")),
        eval_moleculeqa_test_mol_path=str(_cfg_get(data_cfg, "eval_moleculeqa_test_mol_path", "")),
        eval_moleculeqa_sample_size=int(_cfg_get(data_cfg, "eval_moleculeqa_sample_size", 1000)),
        eval_pampa_path=str(_cfg_get(data_cfg, "eval_pampa_path", "")),
        eval_pampa_sample_size=int(_cfg_get(data_cfg, "eval_pampa_sample_size", 1000)),
        eval_moledit_test_paths=list(_cfg_get(data_cfg, "eval_moledit_test_paths", [])),
        eval_moledit_sample_per_task=int(_cfg_get(data_cfg, "eval_moledit_sample_per_task", 0)),
        enabled_sources=list(_cfg_get(data_cfg, "enabled_sources", ["pubchem", "conversation"])),
        eval_from_train_holdout=bool(_cfg_get(data_cfg, "eval_from_train_holdout", False)),
        source_train_fraction=float(_cfg_get(data_cfg, "source_train_fraction", 0.8)),
        total_data_fraction=float(_cfg_get(data_cfg, "total_data_fraction", _cfg_get(data_cfg, "train_subset_fraction", 1.0))),
        total_data_fraction_by_source=_to_plain(
            _cfg_get(data_cfg, "total_data_fraction_by_source", _cfg_get(data_cfg, "train_subset_fraction_by_source", {}))
        ),
        split_seed=int(_cfg_get(data_cfg, "split_seed", _cfg_get(data_cfg, "train_subset_seed", 42))),
        train_subset_fraction=float(_cfg_get(data_cfg, "train_subset_fraction", 1.0)),
        train_subset_fraction_by_source=_to_plain(_cfg_get(data_cfg, "train_subset_fraction_by_source", {})),
        train_subset_seed=int(_cfg_get(data_cfg, "train_subset_seed", 42)),
        seed=int(getattr(train_config, "seed", 42)),
    )

    has_eval_data = bool(_cfg_get(data_cfg, "eval_from_train_holdout", False))
    has_eval_data = has_eval_data or len(list(_cfg_get(data_cfg, "eval_moledit_test_paths", []))) > 0
    has_eval_data = has_eval_data or len(list(_cfg_get(data_cfg, "eval_downstream_csv_paths", []))) > 0
    has_eval_data = has_eval_data or len(str(_cfg_get(data_cfg, "eval_moleculeqa_test_path", "")).strip()) > 0
    has_eval_data = has_eval_data or len(str(_cfg_get(data_cfg, "eval_pampa_path", "")).strip()) > 0

    lr_cb = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir="lightning_logs", name="stage1")
    loggers = [csv_logger]
    run_name_raw = str(getattr(train_config, "wandb_run_name", "stage1-unified-mix"))
    run_name_tag = _sanitize_name(run_name_raw, max_len=48)
    run_id_tag = ""
    if bool(getattr(train_config, "use_wandb", True)):
        wandb_mode = str(getattr(train_config, "wandb_mode", "online"))
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        os.environ.setdefault("WANDB_DIR", os.path.abspath("wandb"))
        wandb_logger = WandbLogger(
            project=str(getattr(train_config, "wandb_project", "mol-modeling-stageI")),
            name=run_name_raw,
            save_dir=os.path.abspath("wandb"),
            log_model=False,
        )
        try:
            run_id = str(getattr(wandb_logger.experiment, "id", "") or "").strip()
            if len(run_id) > 0:
                run_id_tag = _sanitize_name(run_id, max_len=12)
        except Exception:
            run_id_tag = ""
        wandb_logger.log_hyperparams(
            {
                "train_config": OmegaConf.to_container(train_config, resolve=True),
                "data_config": OmegaConf.to_container(data_config, resolve=True),
                "stage0_checkpoint_path": stage0_checkpoint_path,
                "loss_weights": loss_weights,
            }
        )
        loggers.append(wandb_logger)
        print(f"W&B logger enabled (project=mol-modeling-stageI, mode={wandb_mode}).")

    run_time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_tag_parts = [run_time_tag, run_name_tag]
    if len(run_id_tag) > 0:
        dir_tag_parts.append(run_id_tag)
    ckpt_subdir = "-".join(dir_tag_parts)
    ckpt_dir = os.path.join("checkpoints", "stage1", ckpt_subdir)
    print(f"[Stage1] checkpoint dir: {ckpt_dir}")

    if has_eval_data:
        best_ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{run_name_tag}-step{{step:08d}}",
            monitor="val/score",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
    else:
        print("[Stage1] no validation/test dataloader configured; disable val loop and monitor-less checkpointing.")
        best_ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{run_name_tag}-step{{step:08d}}",
            save_top_k=0,
            save_last=True,
        )

    trainer_accelerator, trainer_devices, trainer_strategy, trainer_num_nodes, trainer_sync_bn = _resolve_trainer_runtime(
        train_config
    )
    # Run evaluation every 200 optimizer steps.
    val_check_interval = 200
    print(f"[Stage1] evaluation interval: every {val_check_interval} steps")

    trainer = pl.Trainer(
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        strategy=trainer_strategy,
        num_nodes=trainer_num_nodes,
        sync_batchnorm=trainer_sync_bn,
        precision=train_config.precision,
        max_epochs=int(train_config.max_epochs),
        check_val_every_n_epoch=int(train_config.check_val_every_n_epoch),
        val_check_interval=val_check_interval,
        limit_val_batches=0.0 if not has_eval_data else 1.0,
        num_sanity_val_steps=0 if not has_eval_data else 2,
        accumulate_grad_batches=int(train_config.accumulate_grad_batches),
        callbacks=[best_ckpt_cb, lr_cb],
        logger=loggers,
        default_root_dir=".",
        log_every_n_steps=1,
    )
    if has_eval_data and bool(_cfg_get(stage1_cfg, "eval_before_training", True)):
        print("[Stage1] running pre-finetuning validation at global_step=0 ...")
        model.use_base_forward_for_validation = bool(_cfg_get(stage1_cfg, "eval_before_training_use_base_forward", True))
        trainer.validate(model=model, datamodule=datamodule, verbose=True)
        model.use_base_forward_for_validation = False
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
