import os
import argparse
import re
from datetime import datetime

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import AutoTokenizer

from data_provider.stage2_grpo_dm import Stage2GRPODM
from trainer.stage2_grpo import Stage2GRPOTrainer
from utils.configuration_mol_llama import MolLLaMAConfig


def _load_configs(train_cfg_path=None, data_cfg_path=None):
    train_cfg_path = train_cfg_path or os.path.join("configs", "stage2", "train_config.yaml")
    data_cfg_path = data_cfg_path or os.path.join("configs", "stage2", "data_config.yaml")
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

    # Important: when Lightning launches DDP subprocesses itself, WORLD_SIZE/LOCAL_RANK
    # are already set in child processes. Do not override devices there, otherwise
    # `devices_per_process=1` can conflict with local_rank>0 and crash.
    if use_accelerate_launch:
        strategy = str(getattr(train_config, "accelerate_strategy", "ddp"))
        devices = int(getattr(train_config, "accelerate_devices_per_process", 1))
        num_nodes = int(getattr(train_config, "accelerate_num_nodes", 1))
        print(
            "[Stage2-GRPO] accelerate-compatible runtime: "
            f"world_size={world_size}, rank={rank}, local_rank={local_rank}, "
            f"strategy={strategy}, devices_per_process={devices}, num_nodes={num_nodes}"
        )
    else:
        print(f"[Stage2-GRPO] default runtime: strategy={strategy}, devices={devices}, num_nodes={num_nodes}")

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
        s = "stage2-grpo"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", default=os.path.join("configs", "stage2", "train_config.yaml"))
    parser.add_argument("--data_config", default=os.path.join("configs", "stage2", "data_config.yaml"))
    parser.add_argument("--run_name", default="")
    args = parser.parse_args()

    train_config, data_config = _load_configs(args.train_config, args.data_config)
    if str(args.run_name or "").strip():
        train_config.wandb_run_name = str(args.run_name).strip()

    pl.seed_everything(int(getattr(train_config, "seed", 42)), workers=True)

    model_cfg = _cfg_get(train_config, "model", {})
    stage2_cfg = _cfg_get(train_config, "stage2", {})
    data_cfg = _cfg_get(data_config, "data", {})
    eval_moledit_test_paths = _cfg_get(data_cfg, "eval_moledit_test_paths", [])
    if isinstance(eval_moledit_test_paths, str):
        eval_moledit_test_paths = [eval_moledit_test_paths]
    else:
        eval_moledit_test_paths = list(eval_moledit_test_paths or [])

    llm_model_name = str(_cfg_get(model_cfg, "llm_model", "meta-llama/Llama-3.1-8B-Instruct"))
    model_config = MolLLaMAConfig()
    model_config.llm_config.llm_model = llm_model_name
    model_config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]

    tokenizer = _build_tokenizer(llm_model_name)
    llama_version = "llama3" if "Llama-3" in llm_model_name else "llama2"

    model = Stage2GRPOTrainer(vocab_size=len(tokenizer), model_config=model_config, train_config=train_config, tokenizer=tokenizer)

    init_from_hf = str(_cfg_get(model_cfg, "stage0_checkpoint_path", ""))
    init_from_stage1 = str(_cfg_get(model_cfg, "stage1_checkpoint_path", ""))
    init_from_stage2 = str(_cfg_get(model_cfg, "stage2_checkpoint_path", ""))
    if init_from_hf:
        model.load_from_hf_dir(init_from_hf)
    if init_from_stage1:
        model.load_from_ckpt(init_from_stage1)
    if init_from_stage2:
        model.load_from_ckpt(init_from_stage2)

    unimol_dictionary = getattr(model.mol_llama.encoder, "unimol_dictionary", None)
    datamodule = Stage2GRPODM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        num_workers=int(_cfg_get(data_cfg, "num_workers", 0)),
        batch_size=int(_cfg_get(data_cfg, "batch_size", 1)),
        unimol_dictionary=unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        text_max_len=int(_cfg_get(data_cfg, "text_max_len", 512)),
        max_latent_slots=int(_cfg_get(stage2_cfg, "max_latent_slots", 4)),
        latent_slot_text_max_len=int(_cfg_get(data_cfg, "latent_slot_text_max_len", 48)),
        latent_world_modeling_path=str(_cfg_get(data_cfg, "latent_path", "")),
        conversation_sft_path=str(_cfg_get(data_cfg, "conversation_path", "")),
        moledit_path=str(_cfg_get(data_cfg, "moledit_path", "")),
        stage2_path=str(_cfg_get(data_cfg, "stage2_path", "")),
        eval_moledit_test_paths=eval_moledit_test_paths,
        eval_moledit_sample_per_task=int(_cfg_get(data_cfg, "eval_moledit_sample_per_task", 0)),
        moledit_val_path=str(_cfg_get(data_cfg, "moledit_val_path", "")),
        fallback_raw_paths=_to_plain(_cfg_get(data_cfg, "fallback_raw_paths", {})),
        enabled_sources=list(_cfg_get(data_cfg, "enabled_sources", ["pubchem", "conversation", "stage2"])),
        replay_sources=list(_cfg_get(data_cfg, "replay_sources", ["pubchem", "conversation", "moledit"])),
        rl_sources=list(_cfg_get(data_cfg, "rl_sources", ["stage2"])),
        replay_ratio=float(_cfg_get(data_cfg, "replay_ratio", _cfg_get(data_cfg, "sft_ratio", 0.1))),
        rl_ratio=float(_cfg_get(data_cfg, "rl_ratio", 0.9)),
        replay_source_weights=_to_plain(_cfg_get(data_cfg, "replay_source_weights", {})),
        rl_source_weights=_to_plain(_cfg_get(data_cfg, "rl_source_weights", {})),
        total_data_fraction=float(_cfg_get(data_cfg, "total_data_fraction", 1.0)),
        total_data_fraction_by_source=_to_plain(_cfg_get(data_cfg, "total_data_fraction_by_source", {})),
        stage2_data_fraction=float(_cfg_get(data_cfg, "stage2_data_fraction", 1.0)),
        split_seed=int(_cfg_get(data_cfg, "split_seed", 42)),
        stage2_subtasks=list(_cfg_get(data_cfg, "stage2_subtasks", [])),
        val_subtasks=list(_cfg_get(data_cfg, "val_subtasks", [])),
        regression_targets=list(_cfg_get(stage2_cfg, "regression_targets", [])),
        classification_targets=list(_cfg_get(stage2_cfg, "classification_targets", [])),
        steps_per_epoch=int(_cfg_get(data_cfg, "steps_per_epoch", 0)),
    )

    lr_cb = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir="lightning_logs", name="stage2_grpo")
    loggers = [csv_logger]

    run_name_raw = str(getattr(train_config, "wandb_run_name", "stage2-grpo"))
    run_name_tag = _sanitize_name(run_name_raw, max_len=48)
    run_id_tag = ""
    if bool(getattr(train_config, "use_wandb", True)):
        wandb_mode = str(getattr(train_config, "wandb_mode", "online"))
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        os.environ.setdefault("WANDB_DIR", os.path.abspath("wandb"))
        wandb_logger = WandbLogger(
            project=str(getattr(train_config, "wandb_project", "latent-stage2-grpo")),
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
                "llm_model": llm_model_name,
            }
        )
        loggers.append(wandb_logger)

    run_time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_tag_parts = [run_time_tag, run_name_tag]
    if len(run_id_tag) > 0:
        dir_tag_parts.append(run_id_tag)
    ckpt_dir = os.path.join("checkpoints", "stage2_grpo", "-".join(dir_tag_parts))

    has_eval_data = len(eval_moledit_test_paths) > 0
    has_eval_data = has_eval_data or bool(str(_cfg_get(data_cfg, "moledit_val_path", "")).strip())
    # Avoid run interruption when validation metrics are absent at checkpoint time.
    # Default to a stable training metric; can be overridden in config.
    monitor_name = str(_cfg_get(train_config, "checkpoint_monitor", "train/loss_total_epoch"))
    monitor_mode = str(_cfg_get(train_config, "checkpoint_monitor_mode", "min"))
    if has_eval_data and monitor_name == "auto":
        monitor_name = "val/score"
        monitor_mode = "max"
    if not monitor_name:
        monitor_name = "train/loss_total_epoch"
        monitor_mode = "min"
    print(f"[Stage2-GRPO] checkpoint monitor: {monitor_name} ({monitor_mode})")
    best_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name_tag}-step{{step:08d}}",
        monitor=monitor_name,
        mode=monitor_mode,
        save_top_k=1,
        save_last=True,
    )

    trainer_accelerator, trainer_devices, trainer_strategy, trainer_num_nodes, trainer_sync_bn = _resolve_trainer_runtime(train_config)
    val_check_interval = int(getattr(train_config, "val_check_interval_steps", 200))
    print(f"[Stage2-GRPO] evaluation interval: every {val_check_interval} steps")

    trainer = pl.Trainer(
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        strategy=trainer_strategy,
        num_nodes=trainer_num_nodes,
        sync_batchnorm=trainer_sync_bn,
        # Stage2GRPODM uses a custom BatchSampler for replay/RL mixing.
        # Let the datamodule handle sampling; do not let Lightning inject a distributed sampler.
        use_distributed_sampler=False,
        precision=train_config.precision,
        max_epochs=int(train_config.max_epochs),
        check_val_every_n_epoch=int(getattr(train_config, "check_val_every_n_epoch", 1)),
        val_check_interval=val_check_interval,
        limit_val_batches=0.0 if not has_eval_data else 1.0,
        num_sanity_val_steps=0 if not has_eval_data else 2,
        accumulate_grad_batches=int(train_config.accumulate_grad_batches),
        callbacks=[best_ckpt_cb, lr_cb],
        logger=loggers,
        default_root_dir=".",
        log_every_n_steps=1,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
