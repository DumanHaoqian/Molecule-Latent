import contextlib
import csv
import os
import re
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
    def __init__(self, vocab_size, model_config, train_config, tokenizer=None):
        super().__init__()
        self.train_config = train_config
        self.tokenizer = tokenizer
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
        self._val_metric_buckets = {}
        self._val_metric_counts = {}
        self._reg_key_to_idx = {}
        self._fixed_binary_datasets = {"BACE", "BBBP", "HIV", "CLINTOX", "TOX21"}
        self._binary_pos_token_ids = []
        self._binary_neg_token_ids = []

        # WM regression normalization and molecular_weight curriculum.
        # Default stds roughly match common chemistry ranges; can be overridden in config.
        self.regression_stds = {
            "molecular_weight": 500.0,
            "logp": 5.0,
            "tpsa": 150.0,
            "hbd": 5.0,
            "hba": 10.0,
            "num_rings": 6.0,
            "aromatic_ring_count": 4.0,
            "qed": 1.0,
        }
        cfg_stds = getattr(stage1_cfg, "regression_target_stds", None)
        if isinstance(cfg_stds, dict):
            for k, v in cfg_stds.items():
                try:
                    self.regression_stds[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass

        self.mw_warmup_start = float(getattr(stage1_cfg, "mw_warmup_start", 5.0))
        self.mw_warmup_end = float(getattr(stage1_cfg, "mw_warmup_end", 1.0))
        self.mw_warmup_steps = int(getattr(stage1_cfg, "mw_warmup_steps", 2000))
        self.non_mw_reg_scale = float(getattr(stage1_cfg, "non_mw_reg_scale", 0.5))

        self._init_binary_label_token_ids()
    
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
        reg_keys = outputs["wm_preds"].get("regression_keys", [])
        reg_targets = batch["property_regression_targets"].to(reg_preds.device)
        reg_mask = batch["property_regression_mask"].to(reg_preds.device)
        cls_targets = batch["property_classification_targets"].to(cls_logits.device)
        cls_mask = batch["property_classification_mask"].to(cls_logits.device)

        reg_loss = torch.tensor(0.0, device=reg_preds.device)
        if reg_mask.any():
            # Normalize each regression target by dataset-scale stds to reduce
            # magnitude mismatch (especially molecular_weight).
            std_vec = torch.ones((reg_preds.shape[1],), device=reg_preds.device, dtype=reg_preds.dtype)
            for i, key in enumerate(reg_keys):
                std_vec[i] = float(self.regression_stds.get(key, 1.0))
            norm_preds = reg_preds / std_vec.unsqueeze(0)
            norm_targets = reg_targets / std_vec.unsqueeze(0)

            diff = (norm_preds - norm_targets).pow(2)

            # Dynamic MW emphasis: large in early phase, decays later.
            mw_scale = self.mw_warmup_end
            if self.mw_warmup_steps > 0:
                p = min(1.0, float(self.global_step) / float(self.mw_warmup_steps))
                mw_scale = self.mw_warmup_start + (self.mw_warmup_end - self.mw_warmup_start) * p

            dim_w = torch.full((reg_preds.shape[1],), self.non_mw_reg_scale, device=reg_preds.device, dtype=reg_preds.dtype)
            if "molecular_weight" in reg_keys:
                mw_idx = reg_keys.index("molecular_weight")
                dim_w[mw_idx] = mw_scale
            weighted = diff * dim_w.unsqueeze(0)
            reg_loss = (weighted * reg_mask.to(weighted.dtype)).sum() / reg_mask.to(weighted.dtype).sum().clamp(min=1.0)

        cls_loss = torch.tensor(0.0, device=cls_logits.device)
        if cls_mask.any():
            # multi-target BCE for simple first runnable version
            bce = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
            cls_loss = (bce * cls_mask.to(bce.dtype)).sum() / cls_mask.to(bce.dtype).sum().clamp(min=1.0)

        return reg_loss + cls_loss

    def _decode_pred_texts(self, logits, labels):
        if self.tokenizer is None:
            return ["" for _ in range(labels.shape[0])]
        pred_shift = logits[:, :-1, :].argmax(dim=-1)
        label_shift = labels[:, 1:]
        texts = []
        for bi in range(pred_shift.shape[0]):
            valid = label_shift[bi] != -100
            ids = pred_shift[bi][valid]
            if ids.numel() == 0:
                texts.append("")
            else:
                texts.append(self.tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip())
        return texts

    def _parse_binary_pred(self, text, task_name):
        t = (text or "").strip().lower()
        ds = str(task_name or "").upper()
        if ds == "PAMPA":
            if "high permeability" in t:
                return 1
            if "low-to-moderate" in t or "low to moderate" in t or "low permeability" in t:
                return 0
        if "answer: yes" in t or re.search(r"\byes\b", t):
            return 1
        if "answer: no" in t or re.search(r"\bno\b", t):
            return 0
        return None

    def _parse_mcq_pred(self, text):
        t = text or ""
        m = re.search(r"Answer\s*:\s*([ABCD])", t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([ABCD])\b", t.strip(), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    def _parse_float_pred(self, text):
        t = text or ""
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
        if not m:
            return None
        try:
            return float(m.group(0))
        except (TypeError, ValueError):
            return None

    def _init_binary_label_token_ids(self):
        if self.tokenizer is None:
            return
        pos_forms = ["Yes", " yes", "Answer: Yes", "yes", " yes."]
        neg_forms = ["No", " no", "Answer: No", "no", " no."]

        def _collect(forms):
            ids = set()
            for s in forms:
                tok = self.tokenizer(s, add_special_tokens=False).input_ids
                if len(tok) > 0:
                    ids.add(int(tok[-1]))
            return sorted(ids)

        self._binary_pos_token_ids = _collect(pos_forms)
        self._binary_neg_token_ids = _collect(neg_forms)

    def _binary_fixed_token_predict(self, step_logits_vec):
        if len(self._binary_pos_token_ids) == 0 or len(self._binary_neg_token_ids) == 0:
            return None, None
        pos_scores = step_logits_vec[self._binary_pos_token_ids]
        neg_scores = step_logits_vec[self._binary_neg_token_ids]
        pos = float(torch.max(pos_scores).item())
        neg = float(torch.max(neg_scores).item())
        score = pos - neg
        pred = 1 if score > 0 else 0
        return pred, score

    def _binary_auc(self, y_true, y_score):
        # Rank-based AUC with tie handling.
        if len(y_true) == 0:
            return None
        y = torch.tensor(y_true, dtype=torch.float32)
        s = torch.tensor(y_score, dtype=torch.float32)
        n_pos = int((y == 1).sum().item())
        n_neg = int((y == 0).sum().item())
        if n_pos == 0 or n_neg == 0:
            return None
        order = torch.argsort(s)
        ranks = torch.zeros_like(s)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and float(s[order[j]]) == float(s[order[i]]):
                j += 1
            avg_rank = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg_rank
            i = j
        sum_pos_ranks = float(ranks[y == 1].sum().item())
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

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
        logits = outputs["logits"]
        labels = batch["labels"].to(logits.device)
        pred_texts = self._decode_pred_texts(logits, labels)
        gold_texts = batch.get("text_targets", [""] * batch_size)
        task_kind_list = batch.get("task_kind", [None] * batch_size)
        task_name_list = batch.get("task_name", [source_name] * batch_size)
        task_label = batch.get("task_label", torch.full((batch_size,), -1, dtype=torch.long)).tolist()
        task_label_text = batch.get("task_label_text", [None] * batch_size)
        task_regression_value = batch.get("task_regression_value", torch.zeros((batch_size,), dtype=torch.float32)).tolist()
        task_regression_mask = batch.get("task_regression_mask", torch.zeros((batch_size,), dtype=torch.bool)).tolist()

        logit_shift = logits[:, :-1, :]
        label_shift = labels[:, 1:]
        valid_shift = label_shift != -100
        first_pos = torch.full((batch_size,), -1, dtype=torch.long, device=logits.device)
        for bi in range(batch_size):
            pos = torch.nonzero(valid_shift[bi], as_tuple=False).flatten()
            if pos.numel() > 0:
                first_pos[bi] = int(pos[0].item())

        for i in range(batch_size):
            ds = str(task_name_list[i] or source_name)
            ds_upper = ds.upper()
            kind = str(task_kind_list[i] or "")
            self._val_metric_buckets.setdefault(ds, {"binary_y": [], "binary_pred": [], "binary_score": [], "mcq_y": [], "mcq_pred": [], "reg_y": [], "reg_pred": []})
            pred_txt = pred_texts[i]
            gold_txt = gold_texts[i] if i < len(gold_texts) else ""

            if kind == "binary_classification":
                y = int(task_label[i]) if i < len(task_label) else -1
                if y in (0, 1):
                    p, s = None, None
                    if ds_upper in self._fixed_binary_datasets and int(first_pos[i].item()) >= 0:
                        step_logits_vec = logit_shift[i, int(first_pos[i].item()), :]
                        p, s = self._binary_fixed_token_predict(step_logits_vec)
                    if p is None:
                        p = self._parse_binary_pred(pred_txt, ds)
                        if p is not None:
                            s = float(p)
                    if p is not None and s is not None:
                        self._val_metric_buckets[ds]["binary_y"].append(y)
                        self._val_metric_buckets[ds]["binary_pred"].append(int(p))
                        self._val_metric_buckets[ds]["binary_score"].append(float(s))
            elif kind == "mcq4":
                y = str(task_label_text[i] or "").upper()
                if y in ("A", "B", "C", "D"):
                    p = self._parse_mcq_pred(pred_txt)
                    if p in ("A", "B", "C", "D"):
                        self._val_metric_buckets[ds]["mcq_y"].append(y)
                        self._val_metric_buckets[ds]["mcq_pred"].append(p)
            elif kind == "regression":
                has_reg = bool(task_regression_mask[i]) if i < len(task_regression_mask) else False
                if has_reg:
                    y = float(task_regression_value[i])
                    p = self._parse_float_pred(pred_txt)
                    if p is not None:
                        self._val_metric_buckets[ds]["reg_y"].append(y)
                        self._val_metric_buckets[ds]["reg_pred"].append(float(p))

        self.log("val/loss", lm_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/score", score, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"val/loss_{source_name}", lm_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"val/score_{source_name}", score, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        return {"val_loss": lm_loss.detach(), "val_score": score.detach()}

    def on_validation_epoch_start(self):
        self.val_pos_token_id = None
        self.val_neg_token_id = None
        self._val_metric_buckets = {}

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
        }
        # Aggregate metrics by task family.
        all_bin_y, all_bin_p, all_bin_s = [], [], []
        all_mcq_y, all_mcq_p = [], []
        all_reg_y, all_reg_p = [], []
        for ds, bucket in self._val_metric_buckets.items():
            by, bp, bs = bucket["binary_y"], bucket["binary_pred"], bucket["binary_score"]
            my, mp = bucket["mcq_y"], bucket["mcq_pred"]
            ry, rp = bucket["reg_y"], bucket["reg_pred"]
            if len(by) > 0:
                acc = float(sum(int(a == b) for a, b in zip(bp, by)) / len(by))
                auc = self._binary_auc(by, bs)
                self.log(f"val/acc_{ds}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                row[f"val_acc_{ds}"] = acc
                if auc is not None:
                    self.log(f"val/auc_{ds}", auc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                    row[f"val_auc_{ds}"] = auc
                all_bin_y.extend(by); all_bin_p.extend(bp); all_bin_s.extend(bs)
            if len(my) > 0:
                acc_mcq = float(sum(int(a == b) for a, b in zip(mp, my)) / len(my))
                self.log(f"val/mcq_acc_{ds}", acc_mcq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                row[f"val_mcq_acc_{ds}"] = acc_mcq
                all_mcq_y.extend(my); all_mcq_p.extend(mp)
            if len(ry) > 0:
                err = [abs(a - b) for a, b in zip(rp, ry)]
                mae = float(sum(err) / len(err))
                mse = float(sum((a - b) ** 2 for a, b in zip(rp, ry)) / len(ry))
                rmse = mse ** 0.5
                self.log(f"val/mae_{ds}", mae, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f"val/rmse_{ds}", rmse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                row[f"val_mae_{ds}"] = mae
                row[f"val_rmse_{ds}"] = rmse
                all_reg_y.extend(ry); all_reg_p.extend(rp)

        if len(all_bin_y) > 0:
            acc = float(sum(int(a == b) for a, b in zip(all_bin_p, all_bin_y)) / len(all_bin_y))
            auc = self._binary_auc(all_bin_y, all_bin_s)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            row["val_acc"] = acc
            if auc is not None:
                self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                row["val_auc"] = auc
        if len(all_mcq_y) > 0:
            acc_mcq = float(sum(int(a == b) for a, b in zip(all_mcq_p, all_mcq_y)) / len(all_mcq_y))
            self.log("val/mcq_acc", acc_mcq, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            row["val_mcq_acc"] = acc_mcq
        if len(all_reg_y) > 0:
            err = [abs(a - b) for a, b in zip(all_reg_p, all_reg_y)]
            mae = float(sum(err) / len(err))
            mse = float(sum((a - b) ** 2 for a, b in zip(all_reg_p, all_reg_y)) / len(all_reg_y))
            rmse = mse ** 0.5
            self.log("val/mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            self.log("val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            row["val_mae"] = mae
            row["val_rmse"] = rmse

        # Add dynamic per-dataset val metrics (e.g. BACE/BBBP/HIV/Clintox/Delaney/LIPO/Tox21).
        for k in list(metrics.keys()):
            if not isinstance(k, str):
                continue
            if k.startswith("val/score_") or k.startswith("val/loss_"):
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