import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import AutoTokenizer

from data_provider.stage1_dm import Stage1DM
from trainer.stage1 import Stage1Trainer
from utils.configuration_mol_llama import MolLLaMAConfig


def _load_configs():
    train_cfg_path = os.path.join("configs", "stage1", "train_config.yaml")
    data_cfg_path = os.path.join("configs", "stage1", "data_config.yaml")
    train_config = OmegaConf.load(train_cfg_path)
    data_config = OmegaConf.load(data_cfg_path)
    return train_config, data_config


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
    train_config, data_config = _load_configs()
    pl.seed_everything(int(getattr(train_config, "seed", 42)), workers=True)

    model_config = MolLLaMAConfig()
    model_config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]
    tokenizer = _build_tokenizer(model_config.llm_config.llm_model)
    llama_version = "llama3" if "Llama-3" in model_config.llm_config.llm_model else "llama2"

    model = Stage1Trainer(vocab_size=len(tokenizer), model_config=model_config, train_config=train_config)
    unimol_dictionary = getattr(model.mol_llama.encoder, "unimol_dictionary", None)

    datamodule = Stage1DM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        num_workers=int(data_config.num_workers),
        batch_size=int(data_config.batch_size),
        unimol_dictionary=unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        text_max_len=int(data_config.text_max_len),
        max_latent_slots=int(getattr(train_config, "max_latent_slots", 6)),
        latent_slot_text_max_len=int(getattr(train_config, "latent_slot_text_max_len", 48)),
        stage1_mixed_training=bool(getattr(train_config, "stage1_mixed_training", True)),
        latent_world_modeling_path=str(getattr(data_config, "latent_world_modeling_path", "")),
        conversation_sft_path=str(getattr(data_config, "conversation_sft_path", "")),
        downstream_tasks_path=str(getattr(data_config, "downstream_tasks_path", "")),
        source_sampling_weights=OmegaConf.to_container(getattr(train_config, "source_sampling_weights", {}), resolve=True),
        seed=int(getattr(train_config, "seed", 42)),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", "stage1"),
        filename="{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=int(train_config.save_every_n_epochs),
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
        accumulate_grad_batches=int(train_config.accumulate_grad_batches),
        callbacks=[ckpt_cb, lr_cb],
        logger=loggers,
        default_root_dir=".",
        log_every_n_steps=1,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
