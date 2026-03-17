import contextlib
import csv
import os
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from typing import Any, Dict

from models.mol_llama import MolLLaMA
from trainer.optims import LinearWarmupCosineLRScheduler

def precision2dtype(precision):
    if precision == '16':
        return torch.float16
    elif precision == '32':
        return torch.float32
    elif precision.find('bf16') >= 0:
        return torch.bfloat16
    else:
        raise NotImplementedError()


class Stage1Trainer(pl.LightningModule):
    def __init__(self, vocab_size, model_config, train_config):
        super().__init__()
        self.train_config = train_config
        stage1_cfg = getattr(train_config, "stage1", {})
        if train_config.precision == 'bf16-mixed':
            torch_dtype = "bfloat16"
        elif train_config.precision == '16':
            torch_dtype = "float16"
        else:
            torch_dtype = "float32"

        self.mol_llama = MolLLaMA(
            config=model_config,
            vocab_size=vocab_size,
            torch_dtype=torch_dtype,
            enable_flash=getattr(train_config, "enable_flash", False),
            use_latent_reasoning=False,
            stage1_max_latent_slots=int(getattr(stage1_cfg, "max_latent_slots", getattr(train_config, "max_latent_slots", 6))),
            stage1_wm_reg_keys=list(getattr(stage1_cfg, "regression_targets", getattr(train_config, "wm_regression_targets", []))),
            stage1_wm_cls_keys=list(getattr(stage1_cfg, "classification_targets", getattr(train_config, "wm_classification_targets", []))),
        )
        self.lambda_latent = float(getattr(train_config, "loss_weights", {}).get("latent", 1.0))
        self.lambda_wm = float(getattr(train_config, "loss_weights", {}).get("wm", 1.0))
        self.lambda_lm = float(getattr(train_config, "loss_weights", {}).get("lm", 0.1))
        self.lambda_conv_lm = float(getattr(train_config, "loss_weights", {}).get("conv_lm", 1.0))
        self.lambda_down_lm = float(getattr(train_config, "loss_weights", {}).get("downstream_lm", 1.0))
        self.source_counter = defaultdict(int)
        self.task_to_id = {"latent_world_modeling": 0, "conversation": 1, "downstream": 2}
        self.source_to_id = {"PubChemLatent": 0, "ComprehensiveConversation": 1, "DownstreamTasks": 2}
        self.stage1_cfg = getattr(train_config, "stage1", {})
        self.test_results_csv_path = str(
            getattr(
                self.stage1_cfg,
                "test_results_csv_path",
                "/home/haoqian/Data/Molecule/Latent/test.csv",
            )
        )
        self.val_pos_token_id = None
        self.val_neg_token_id = None
        self.use_base_forward_for_validation = False
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

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
        # Reduce checkpoint size for frequent eval/checkpoint cycles.
        # Keep behavior aligned with Stage2Trainer: drop optimizer states and
        # save only trainable parameters.
        checkpoint.pop("optimizer_states", None)
        to_be_removed = []
        for key in list(checkpoint.get("state_dict", {}).keys()):
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except Exception:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint["state_dict"].pop(key, None)

    def load_from_hf_dir(self, hf_dir):
        self.mol_llama.load_from_hf_dir(hf_dir)

    def load_from_ckpt(self, ckpt_path):
        self.mol_llama.load_from_ckpt(ckpt_path)

    def _compute_latent_loss(self, latent_states, slot_targets, slot_mask):
        # cosine alignment on valid latent slots
        if slot_mask.sum() == 0:
            return torch.tensor(0.0, device=latent_states.device)
        z = F.normalize(latent_states, dim=-1)
        t = F.normalize(slot_targets.to(z.dtype), dim=-1)
        cos = (z * t).sum(dim=-1)  # [B, K]
        valid = slot_mask.to(cos.dtype)
        loss = ((1.0 - cos) * valid).sum() / valid.sum().clamp(min=1.0)
        return loss

    def _compute_wm_loss(self, outputs, batch):
        reg_preds = outputs["wm_preds"]["regression"]
        cls_logits = outputs["wm_preds"]["classification_logits"]
        reg_targets = batch["property_regression_targets"].to(reg_preds.device)
        reg_mask = batch["property_regression_mask"].to(reg_preds.device)
        cls_targets = batch["property_classification_targets"].to(cls_logits.device)
        cls_mask = batch["property_classification_mask"].to(cls_logits.device)

        reg_loss = torch.tensor(0.0, device=reg_preds.device)
        if reg_mask.any():
            diff = (reg_preds - reg_targets).pow(2)
            reg_loss = (diff * reg_mask.to(diff.dtype)).sum() / reg_mask.to(diff.dtype).sum().clamp(min=1.0)

        cls_loss = torch.tensor(0.0, device=cls_logits.device)
        if cls_mask.any():
            # multi-target BCE for simple first runnable version
            bce = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
            cls_loss = (bce * cls_mask.to(bce.dtype)).sum() / cls_mask.to(bce.dtype).sum().clamp(min=1.0)

        return reg_loss + cls_loss

    def training_step(self, batch, batch_idx):
        task_type = batch["task_type"]
        source_name = batch["source_dataset"][0] if len(batch["source_dataset"]) > 0 else "unknown"
        self.source_counter[source_name] += 1
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch["input_ids"].size(0)
        outputs = self.mol_llama.forward_stage1(batch)

        lm_loss = outputs["lm_loss"] if outputs["lm_loss"] is not None else torch.tensor(0.0, device=self.device)
        latent_loss = torch.tensor(0.0, device=self.device)
        wm_loss = torch.tensor(0.0, device=self.device)

        if task_type == "latent_world_modeling":
            latent_loss = self._compute_latent_loss(
                outputs["latent_states"],
                outputs["slot_targets"],
                batch["latent_slot_mask"].to(outputs["latent_states"].device),
            )
            wm_loss = self._compute_wm_loss(outputs, batch)
            loss = self.lambda_latent * latent_loss + self.lambda_wm * wm_loss + self.lambda_lm * lm_loss
        elif task_type == "conversation":
            loss = self.lambda_conv_lm * lm_loss
        elif task_type == "downstream":
            loss = self.lambda_down_lm * lm_loss
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        valid_slot_mean = float(batch["latent_slot_mask"].float().sum(dim=1).mean().item())
        prop_cnt = batch["property_regression_mask"].float().sum(dim=1) + batch["property_classification_mask"].float().sum(dim=1)
        valid_prop_mean = float(prop_cnt.mean().item())

        self.log("train/loss_total", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_lm", lm_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_latent", latent_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_wm", wm_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        # Task-classified losses for W&B grouping.
        self.log(f"train/{task_type}/loss_total", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/{task_type}/loss_lm", lm_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/{task_type}/loss_latent", latent_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/{task_type}/loss_wm", wm_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/task_type_id", float(self.task_to_id.get(task_type, -1)), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/source_name_id", float(self.source_to_id.get(source_name, -1)), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/task_type/{task_type}", 1.0, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/source_name/{source_name}", 1.0, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/valid_latent_slots_mean", valid_slot_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/valid_property_targets_mean", valid_prop_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)

        if self.global_step % 100 == 0:
            print(
                f"[Stage1] step={self.global_step} task={task_type} source={source_name} "
                f"loss={float(loss):.4f} lm={float(lm_loss):.4f} "
                f"latent={float(latent_loss):.4f} wm={float(wm_loss):.4f} "
                f"slot_mean={valid_slot_mean:.2f} prop_mean={valid_prop_mean:.2f}"
            )
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.use_base_forward_for_validation:
            outputs = self.mol_llama.forward_stage1_base(batch)
        else:
            outputs = self.mol_llama.forward_stage1(batch)
        lm_loss = outputs["lm_loss"] if outputs["lm_loss"] is not None else torch.tensor(0.0, device=self.device)
        score = -lm_loss
        batch_size = batch["input_ids"].size(0)
        source_name = batch["source_dataset"][0] if len(batch["source_dataset"]) > 0 else "unknown"

        # Downstream test accuracy (binary): use the first supervised token position,
        # and compare P(Yes-token) vs P(No-token).
        acc = torch.tensor(0.0, device=self.device)
        if "task_label_mask" in batch and batch["task_label_mask"].any():
            logits = outputs["logits"]  # [B, T, V]
            labels = batch["labels"].to(logits.device)  # [B, T]
            task_label = batch["task_label"].to(logits.device)
            task_label_mask = batch["task_label_mask"].to(logits.device)

            # Shift for causal LM alignment.
            logit_shift = logits[:, :-1, :]  # predict labels[:, 1:]
            label_shift = labels[:, 1:]
            valid_shift = label_shift != -100

            first_pos = torch.full((batch_size,), -1, dtype=torch.long, device=logits.device)
            for bi in range(batch_size):
                pos = torch.nonzero(valid_shift[bi], as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                first_pos[bi] = int(pos[0].item())

            # Infer yes/no token ids from labeled samples (cached across val steps).
            for bi in range(batch_size):
                if not bool(task_label_mask[bi]) or int(first_pos[bi].item()) < 0:
                    continue
                gold_tok = int(label_shift[bi, first_pos[bi]].item())
                y = int(task_label[bi].item())
                if y == 1 and self.val_pos_token_id is None:
                    self.val_pos_token_id = gold_tok
                if y == 0 and self.val_neg_token_id is None:
                    self.val_neg_token_id = gold_tok

            if self.val_pos_token_id is not None and self.val_neg_token_id is not None:
                used_idx = []
                used_pos = []
                used_label = []
                for bi in range(batch_size):
                    if not bool(task_label_mask[bi]) or int(first_pos[bi].item()) < 0:
                        continue
                    used_idx.append(bi)
                    used_pos.append(int(first_pos[bi].item()))
                    used_label.append(int(task_label[bi].item()))
                if len(used_idx) > 0:
                    idx_t = torch.tensor(used_idx, device=logits.device, dtype=torch.long)
                    pos_t = torch.tensor(used_pos, device=logits.device, dtype=torch.long)
                    y_t = torch.tensor(used_label, device=logits.device, dtype=torch.long)
                    step_logits = logit_shift[idx_t, pos_t, :]  # [N, V]
                    logit_pos = step_logits[:, int(self.val_pos_token_id)]
                    logit_neg = step_logits[:, int(self.val_neg_token_id)]
                    pred_label = (logit_pos > logit_neg).long()
                    acc = (pred_label == y_t).float().mean()

        self.log("val/loss", lm_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/score", score, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"val/loss_{source_name}", lm_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"val/score_{source_name}", score, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"val/acc_{source_name}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        return {"val_loss": lm_loss.detach(), "val_score": score.detach(), "val_acc": acc.detach()}

    def on_validation_epoch_start(self):
        self.val_pos_token_id = None
        self.val_neg_token_id = None

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        if not self.trainer.is_global_zero:
            return

        metrics = self.trainer.callback_metrics

        def _to_float(key, default=0.0):
            v = metrics.get(key, default)
            if hasattr(v, "detach"):
                v = v.detach()
            if hasattr(v, "item"):
                v = v.item()
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        row = {
            "global_step": int(self.global_step),
            "epoch": int(self.current_epoch),
            "val_loss": _to_float("val/loss", 0.0),
            "val_score": _to_float("val/score", 0.0),
            "val_acc": _to_float("val/acc", 0.0),
        }
        # Add dynamic per-dataset val metrics (e.g. BACE/BBBP/HIV/Clintox/Delaney/LIPO/Tox21).
        for k in list(metrics.keys()):
            if not isinstance(k, str):
                continue
            if k.startswith("val/score_") or k.startswith("val/acc_") or k.startswith("val/loss_"):
                row[k.replace("/", "_")] = _to_float(k, 0.0)

        csv_path = self.test_results_csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        fieldnames = list(row.keys())
        existing_rows = []
        if file_exists:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                for old in old_fields:
                    if old not in fieldnames:
                        fieldnames.append(old)
                for new_k in fieldnames:
                    if new_k not in old_fields:
                        old_fields.append(new_k)
                for r in reader:
                    for k2 in fieldnames:
                        if k2 not in r:
                            r[k2] = ""
                    existing_rows.append(r)
        for k2 in fieldnames:
            if k2 not in row:
                row[k2] = ""
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(row)
        print(f"[Stage1Eval] wrote test metrics to {csv_path}: {row}")

    def on_train_epoch_end(self):
        total = sum(self.source_counter.values())
        if total > 0:
            print(f"[Stage1] epoch={self.current_epoch} source_sample_counter={dict(self.source_counter)}")
            for source, cnt in self.source_counter.items():
                self.log(
                    f"train/source_count_{source}",
                    float(cnt),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
        self.source_counter.clear()