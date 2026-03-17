import os
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", default=os.path.join("configs", "stage1", "train_config.yaml"))
    parser.add_argument("--data_config", default=os.path.join("configs", "stage1", "data_config.yaml"))
    args = parser.parse_args()

    train_config, data_config = _load_configs(args.train_config, args.data_config)
    pl.seed_everything(int(getattr(train_config, "seed", 42)), workers=True)

    model_cfg = _cfg_get(train_config, "model", {})
    stage1_cfg = _cfg_get(train_config, "stage1", {})
    data_cfg = _cfg_get(data_config, "data", {})
    loss_weights = _cfg_get(train_config, "loss_weights", {})

    stage0_checkpoint_path = str(_cfg_get(model_cfg, "stage0_checkpoint_path", ""))
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
    unimol_dictionary = getattr(model.mol_llama.encoder, "unimol_dictionary", None)

    downstream_paths = _cfg_get(data_cfg, "downstream_paths", [])
    if isinstance(downstream_paths, str):
        downstream_paths = [downstream_paths]
    datamodule = Stage1DM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        num_workers=int(_cfg_get(data_cfg, "num_workers", data_config.num_workers)),
        batch_size=int(_cfg_get(data_cfg, "batch_size", data_config.batch_size)),
        unimol_dictionary=unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        text_max_len=int(_cfg_get(data_cfg, "text_max_len", data_config.text_max_len)),
        max_latent_slots=int(_cfg_get(stage1_cfg, "max_latent_slots", getattr(train_config, "max_latent_slots", 6))),
        latent_slot_text_max_len=int(getattr(train_config, "latent_slot_text_max_len", 48)),
        stage1_mixed_training=bool(_cfg_get(stage1_cfg, "use_mixed_stage1_training", getattr(train_config, "stage1_mixed_training", True))),
        latent_world_modeling_path=str(_cfg_get(data_cfg, "latent_path", getattr(data_config, "latent_world_modeling_path", ""))),
        conversation_sft_path=str(_cfg_get(data_cfg, "conversation_path", getattr(data_config, "conversation_sft_path", ""))),
        downstream_tasks_paths=downstream_paths or [str(getattr(data_config, "downstream_tasks_path", ""))],
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
        train_subset_fraction=float(_cfg_get(data_cfg, "train_subset_fraction", 1.0)),
        train_subset_fraction_by_source=_to_plain(_cfg_get(data_cfg, "train_subset_fraction_by_source", {})),
        train_subset_seed=int(_cfg_get(data_cfg, "train_subset_seed", 42)),
        seed=int(getattr(train_config, "seed", 42)),
    )

    best_ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", "stage1"),
        filename="best-step{step:08d}-score{val_score:.4f}",
        monitor="val/score",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir="lightning_logs", name="stage1")
    loggers = [csv_logger]
    if bool(getattr(train_config, "use_wandb", True)):
        wandb_mode = str(getattr(train_config, "wandb_mode", "online"))
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        os.environ.setdefault("WANDB_DIR", os.path.abspath("wandb"))
        wandb_logger = WandbLogger(
            project=str(getattr(train_config, "wandb_project", "mol-modeling-stageI")),
            name=str(getattr(train_config, "wandb_run_name", "stage1-unified-mix")),
            save_dir=os.path.abspath("wandb"),
            log_model=False,
        )
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

    trainer = pl.Trainer(
        accelerator=train_config.accelerator,
        devices=train_config.devices,
        strategy=train_config.strategy_name,
        precision=train_config.precision,
        max_epochs=int(train_config.max_epochs),
        check_val_every_n_epoch=int(train_config.check_val_every_n_epoch),
        val_check_interval=int(_cfg_get(stage1_cfg, "eval_every_n_steps", 200)),
        accumulate_grad_batches=int(train_config.accumulate_grad_batches),
        callbacks=[best_ckpt_cb, lr_cb],
        logger=loggers,
        default_root_dir=".",
        log_every_n_steps=1,
    )
    if bool(_cfg_get(stage1_cfg, "eval_before_training", True)):
        print("[Stage1] running pre-finetuning validation at global_step=0 ...")
        model.use_base_forward_for_validation = bool(_cfg_get(stage1_cfg, "eval_before_training_use_base_forward", True))
        trainer.validate(model=model, datamodule=datamodule, verbose=True)
        model.use_base_forward_for_validation = False
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
