import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import AutoTokenizer

from data_provider.stage2_dm import Stage2DM
from trainer.stage2 import Stage2Trainer
from utils.configuration_mol_llama import MolLLaMAConfig


def _load_configs():
    train_cfg_path = os.path.join("configs", "stage2", "train_config.yaml")
    data_cfg_path = os.path.join("configs", "stage2", "data_config.yaml")
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
    if hasattr(train_config, "llm_model") and train_config.llm_model:
        model_config.llm_config.llm_model = train_config.llm_model
    model_config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]

    tokenizer = _build_tokenizer(model_config.llm_config.llm_model)
    llama_version = "llama3" if "Llama-3" in model_config.llm_config.llm_model else "llama2"

    model = Stage2Trainer(vocab_size=len(tokenizer), model_config=model_config, train_config=train_config)
    unimol_dictionary = getattr(model.mol_llama.encoder, "unimol_dictionary", None)

    datamodule = Stage2DM(
        root=str(getattr(data_config, "root", "")),
        num_workers=int(data_config.num_workers),
        batch_size=int(data_config.batch_size),
        tokenizer=tokenizer,
        llama_version=llama_version,
        unimol_dictionary=unimol_dictionary,
        encoder_types=model_config.graph_encoder_config.encoder_types,
        data_types=list(getattr(data_config, "data_types", [])),
        train_json_path=getattr(data_config, "train_json_path", None),
    )

    stage2_path = getattr(train_config, "stage2_path", "")
    stage1_path = getattr(train_config, "stage1_path", "")
    if stage2_path and os.path.exists(stage2_path):
        if os.path.isdir(stage2_path):
            model.load_from_hf_dir(stage2_path)
        else:
            model.load_from_stage2_ckpt(stage2_path)
    elif stage1_path and os.path.exists(stage1_path):
        model.load_from_stage1_ckpt(stage1_path)

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", "stage2"),
        filename="{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=int(train_config.save_every_n_epochs),
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir="lightning_logs", name="stage2")

    loggers = [csv_logger]
    if bool(getattr(train_config, "use_wandb", False)):
        # Use offline mode by default to avoid blocking on missing API key.
        wandb_mode = str(getattr(train_config, "wandb_mode", "offline"))
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        os.environ.setdefault("WANDB_DIR", os.path.abspath("wandb"))
        try:
            wandb_logger = WandbLogger(
                project=str(getattr(train_config, "wandb_project", "mol-llama-latent")),
                name=str(getattr(train_config, "wandb_run_name", "stage2-latent-reasoning")),
                save_dir=os.path.abspath("wandb"),
                log_model=False,
            )
            # Log the full config once for reproducibility.
            wandb_logger.log_hyperparams(
                {
                    "train_config": OmegaConf.to_container(train_config, resolve=True),
                    "data_config": OmegaConf.to_container(data_config, resolve=True),
                }
            )
            loggers.append(wandb_logger)
            print(f"W&B logger enabled (mode={wandb_mode}).")
        except Exception as e:
            print(f"Failed to initialize W&B logger, fallback to CSV only: {e}")

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
