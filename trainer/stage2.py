from typing import Any, Dict

from torch import optim
import pytorch_lightning as pl

from models.mol_llama import MolLLaMA
from trainer.optims import LinearWarmupCosineLRScheduler


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class Stage2Trainer(pl.LightningModule):
    def __init__(self, vocab_size, model_config, train_config):
        super().__init__()
        self.train_config = train_config
        if train_config.precision == 'bf16-mixed':
            torch_dtype = "bfloat16"
        elif train_config.precision == '16':
            torch_dtype = "float16"
        elif train_config.precision == '32':
            torch_dtype = "float32"


        self.mol_llama = MolLLaMA(
            config=model_config,
            vocab_size=vocab_size,
            torch_dtype = torch_dtype,
            enable_flash = train_config.enable_flash,
            use_latent_reasoning=getattr(train_config, "use_latent_reasoning", False),
            num_latent_steps=getattr(train_config, "num_latent_steps", 4),
            lambda_latent=getattr(train_config, "lambda_latent", 1.0),
            lambda_lm=getattr(train_config, "lambda_lm", 1.0),
            lambda_cls=getattr(train_config, "lambda_cls", 0.5),
        )


    def load_from_stage1_ckpt(self, ckpt_path):
        self.mol_llama.load_from_stage1_ckpt(ckpt_path)        

    def load_from_stage2_ckpt(self, ckpt_path):
        # Load full stage2 weights with strict=False compatibility so
        # newly added latent-reasoning heads can be initialized safely.
        self.mol_llama.load_from_ckpt(ckpt_path)

    def load_from_hf_dir(self, hf_dir):
        self.mol_llama.load_from_hf_dir(hf_dir)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch              
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        output = self.mol_llama(graph_batch, text_batch, other_infos=other_infos)
        loss = {'loss': output['loss']}

        # Unified train metrics for terminal progress bar + CSV/W&B loggers.
        self.log("train/loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        if "loss_lm" in output:
            self.log("train/loss_lm", output["loss_lm"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        if "loss_latent" in output:
            self.log("train/loss_latent", output["loss_latent"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        if "loss_cls" in output:
            self.log("train/loss_cls", output["loss_cls"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        return loss['loss']