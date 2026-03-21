import contextlib
import csv
import os
import re
from collections import defaultdict
from types import SimpleNamespace

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
            stage1_max_latent_slots=int(getattr(stage1_cfg, "max_latent_slots", getattr(train_config, "max_latent_slots", 4))),
            stage1_wm_reg_keys=list(getattr(stage1_cfg, "regression_targets", getattr(train_config, "wm_regression_targets", []))),
            stage1_wm_cls_keys=list(getattr(stage1_cfg, "classification_targets", getattr(train_config, "wm_classification_targets", []))),
        )
        self.lambda_latent = float(getattr(train_config, "loss_weights", {}).get("latent", 1.0))
        self.lambda_lm = float(getattr(train_config, "loss_weights", {}).get("lm", 0.1))
        self.source_counter = defaultdict(int)
        self.task_to_id = {"latent_world_modeling": 0, "conversation": 1}
        self.source_to_id = {"PubChemLatent": 0, "ComprehensiveConversation": 1, "MolEditLatent": 2}
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
        self.latent_viz_every_n_steps = int(getattr(stage1_cfg, "latent_viz_every_n_steps", 500))
        self.latent_viz_max_samples = int(getattr(stage1_cfg, "latent_viz_max_samples", 8))
        self.latent_viz_out_dir = str(
            getattr(stage1_cfg, "latent_viz_out_dir", "/home/haoqian/Data/Molecule/Latent/latent_viz")
        )

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

    def _canonicalize_smiles(self, smiles):
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return None
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    def _extract_predicted_smiles_from_text(self, text):
        if not isinstance(text, str):
            return None
        m = re.search(r"<answer>\s*([^<]+?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            can = self._canonicalize_smiles(m.group(1).strip())
            if can is not None:
                return can
        toks = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+", text)
        toks = sorted(set(toks), key=len, reverse=True)
        for tok in toks:
            can = self._canonicalize_smiles(tok)
            if can is not None:
                return can
        return None

    def _count_group_from_smiles(self, smiles, group):
        GROUP_TO_SMARTS = {
            "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
            "benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
            "hydroxyl": "[OX2H]",
            "aldehyde": "[CX3H1](=O)[#6]",
            "ketone": "[#6][CX3](=O)[#6]",
            "carboxyl": "[CX3](=O)[OX2H1]",
            "ester": "[#6][CX3](=O)[OX2H0][#6]",
            "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "amide": "[NX3][CX3](=[OX1])[#6]",
            "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
            "halo": "[F,Cl,Br,I]",
            "thiol": "[#16X2H]",
            "thioether": "[SX2][CX4]",
            "disulfide": "[#16X2H0][#16X2H0]",
            "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
            "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
            "sulfide": "[#16X2H0]",
            "nitrile": "[NX1]#[CX2]",
            "borane": "[BX3]",
        }
        alias = {"benzene ring": "benzene_ring", "benzene-ring": "benzene_ring"}
        g = str(group or "").strip().lower().replace(".", "")
        g = alias.get(g, g).replace(" ", "_")
        if g not in GROUP_TO_SMARTS:
            return None
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            patt = Chem.MolFromSmarts(GROUP_TO_SMARTS[g])
            if patt is None:
                return None
            matches = mol.GetSubstructMatches(patt)
            if g == "sulfide":
                disulfide = Chem.MolFromSmarts(GROUP_TO_SMARTS["disulfide"])
                disulfide_matches = mol.GetSubstructMatches(disulfide)
                return max(0, len(matches) - len(disulfide_matches))
            return len(matches)
        except Exception:
            return None

    def _check_moledit_correct(self, subtask, src, pred, add_group, remove_group):
        if subtask == "add":
            s0 = self._count_group_from_smiles(src, add_group)
            s1 = self._count_group_from_smiles(pred, add_group)
            return s0 is not None and s1 is not None and s1 == s0 + 1
        if subtask == "delete":
            s0 = self._count_group_from_smiles(src, remove_group)
            s1 = self._count_group_from_smiles(pred, remove_group)
            return s0 is not None and s1 is not None and s1 == s0 - 1
        if subtask == "sub":
            a0 = self._count_group_from_smiles(src, add_group)
            a1 = self._count_group_from_smiles(pred, add_group)
            r0 = self._count_group_from_smiles(src, remove_group)
            r1 = self._count_group_from_smiles(pred, remove_group)
            return (
                a0 is not None and a1 is not None and r0 is not None and r1 is not None
                and a1 == a0 + 1 and r1 == r0 - 1
            )
        return False

    def _project_points_2d(self, points):
        # points: torch.Tensor [N, D]
        if points.size(0) <= 2:
            out = torch.zeros((points.size(0), 2), dtype=torch.float32, device=points.device)
            if points.size(0) == 2:
                out[1, 0] = 1.0
            return out
        # Prefer UMAP when available; fallback to PCA via torch.
        try:
            import umap  # type: ignore

            emb = umap.UMAP(n_components=2, random_state=42).fit_transform(points.detach().cpu().float().numpy())
            return torch.tensor(emb, dtype=torch.float32, device=points.device)
        except Exception:
            pts = points.detach().float()
            centered = pts - pts.mean(dim=0, keepdim=True)
            q = min(2, centered.size(1))
            u, s, _ = torch.pca_lowrank(centered, q=q)
            proj = centered @ _[:, :2]
            if proj.size(1) < 2:
                pad = torch.zeros((proj.size(0), 2 - proj.size(1)), device=proj.device, dtype=proj.dtype)
                proj = torch.cat([proj, pad], dim=1)
            return proj

    @torch.no_grad()
    def _save_latent_visualization(self, outputs, batch):
        if self.latent_viz_every_n_steps <= 0:
            return
        if (self.global_step % self.latent_viz_every_n_steps) != 0:
            return
        if not self.trainer.is_global_zero:
            return
        try:
            import numpy as np
            import matplotlib.pyplot as plt
        except Exception:
            return

        latent_all = outputs.get("latent_states_all", None)
        latent_states = outputs.get("latent_states", None)
        slot_targets = outputs.get("slot_targets", None)
        slot_mask = batch.get("latent_slot_mask", None)
        if latent_all is None or latent_states is None or slot_targets is None or slot_mask is None:
            return

        bsz = int(latent_all.shape[0])
        if bsz <= 0:
            return
        keep_n = max(1, min(self.latent_viz_max_samples, bsz))
        latent_all = latent_all[:keep_n].detach().float().cpu()
        latent_states = latent_states[:keep_n].detach().float().cpu()
        slot_targets = slot_targets[:keep_n].detach().float().cpu()
        slot_mask = slot_mask[:keep_n].detach().cpu()

        sample_ids = list(batch.get("sample_ids", []))[:keep_n]
        source_datasets = list(batch.get("source_dataset", []))[:keep_n]
        task_type = str(batch.get("task_type", "unknown"))

        out_dir = os.path.join(self.latent_viz_out_dir, f"step_{int(self.global_step):08d}")
        os.makedirs(out_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(out_dir, "latent_dump.npz"),
            latent_states_all=latent_all.numpy(),
            latent_states=latent_states.numpy(),
            slot_targets=slot_targets.numpy(),
            slot_mask=slot_mask.numpy(),
            sample_ids=np.array(sample_ids, dtype=object),
            source_dataset=np.array(source_datasets, dtype=object),
            task_type=np.array([task_type], dtype=object),
        )

        # 1) UMAP/PCA plot by latent step using latent_states_all.
        b, k, d = latent_all.shape
        flat = latent_all.reshape(b * k, d)
        coords = self._project_points_2d(flat).cpu().numpy()
        step_ids = np.tile(np.arange(k), b)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap("viridis", k)
        for s in range(k):
            idx = step_ids == s
            ax.scatter(coords[idx, 0], coords[idx, 1], s=14, alpha=0.8, color=cmap(s), label=f"step_{s}")
        ax.set_title(f"Latent trajectory by step (global_step={int(self.global_step)})")
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "umap_latent_by_step.png"), dpi=180)
        plt.close(fig)

        # 2) Heatmap for each sample: cosine(latent_states, slot_targets)
        summary_rows = []
        for i in range(keep_n):
            valid_idx = torch.nonzero(slot_mask[i], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            z = F.normalize(latent_states[i, valid_idx, :], dim=-1)
            t = F.normalize(slot_targets[i, valid_idx, :], dim=-1)
            sim = z @ t.transpose(0, 1)  # [S, S]
            sim_np = sim.numpy()

            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            im = ax.imshow(sim_np, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
            ax.set_title(f"Alignment sample_{i} (S={sim_np.shape[0]})")
            ax.set_xlabel("slot target j")
            ax.set_ylabel("latent token i")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"heatmap_alignment_sample_{i}.png"), dpi=180)
            plt.close(fig)

            diag = sim.diag()
            summary_rows.append(
                {
                    "global_step": int(self.global_step),
                    "sample_index": i,
                    "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                    "source_dataset": source_datasets[i] if i < len(source_datasets) else "",
                    "task_type": task_type,
                    "num_valid_slots": int(valid_idx.numel()),
                    "diag_cos_mean": float(diag.mean().item()),
                    "diag_cos_min": float(diag.min().item()),
                    "diag_cos_max": float(diag.max().item()),
                    "all_cos_mean": float(sim.mean().item()),
                }
            )

        csv_path = os.path.join(out_dir, "summary_alignment.csv")
        if len(summary_rows) > 0:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
        print(f"[LatentViz] saved visualization artifacts to {out_dir}")

    @torch.no_grad()
    def _generate_moledit_pred_text(self, batch, sample_idx):
        input_ids = batch["input_ids"][sample_idx]
        attn = batch["attention_mask"][sample_idx]
        labels = batch["labels"][sample_idx]
        mol_flag = batch["mol_token_flag"][sample_idx]
        valid_labels = torch.nonzero((labels != -100) & (attn > 0), as_tuple=False).flatten()
        if valid_labels.numel() == 0:
            return ""
        first_label_pos = int(valid_labels[0].item())
        if first_label_pos <= 0:
            return ""

        prompt_ids = input_ids[:first_label_pos].unsqueeze(0)
        prompt_attn = torch.ones_like(prompt_ids, dtype=attn.dtype, device=attn.device)
        prompt_mol_flag = mol_flag[:first_label_pos].unsqueeze(0)
        prompt_batch = SimpleNamespace(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            mol_token_flag=prompt_mol_flag,
        )

        smiles = batch["smiles"][sample_idx] if sample_idx < len(batch["smiles"]) else None
        if isinstance(self.tokenizer.name_or_path, str) and "Llama-3" in self.tokenizer.name_or_path:
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else:
            eos_token_id = self.tokenizer.eos_token_id

        outputs = self.mol_llama.generate_with_smiles(
            smiles_list=[smiles] if isinstance(smiles, str) and len(smiles) > 0 else [],
            text_batch=prompt_batch,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            min_new_tokens=8,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return text

    def training_step(self, batch, batch_idx):
        task_type = batch["task_type"]
        if task_type not in {"latent_world_modeling", "conversation"}:
            raise ValueError(
                f"Stage-I training only supports latent_world_modeling/conversation tasks, got task_type={task_type}"
            )
        source_name = batch["source_dataset"][0] if len(batch["source_dataset"]) > 0 else "unknown"
        self.source_counter[source_name] += 1
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch["input_ids"].size(0)
        outputs = self.mol_llama.forward_stage1(batch)
        self._save_latent_visualization(outputs, batch)

        lm_loss = outputs["lm_loss"] if outputs["lm_loss"] is not None else torch.tensor(0.0, device=self.device)
        latent_loss = self._compute_latent_loss(
            outputs["latent_states"],
            outputs["slot_targets"],
            batch["latent_slot_mask"].to(outputs["latent_states"].device),
        )
        loss = self.lambda_lm * lm_loss + self.lambda_latent * latent_loss

        valid_slot_mean = float(batch["latent_slot_mask"].float().sum(dim=1).mean().item())
        prop_cnt = batch["property_regression_mask"].float().sum(dim=1) + batch["property_classification_mask"].float().sum(dim=1)
        valid_prop_mean = float(prop_cnt.mean().item())

        self.log("train/loss_total", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_lm", lm_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_latent", latent_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        # Task-classified losses for W&B grouping.
        self.log(f"train/{task_type}/loss_total", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/{task_type}/loss_lm", lm_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log(f"train/{task_type}/loss_latent", latent_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
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
                f"latent={float(latent_loss):.4f} "
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
            self._val_metric_buckets.setdefault(
                ds,
                {
                    "binary_y": [], "binary_pred": [], "binary_score": [],
                    "mcq_y": [], "mcq_pred": [],
                    "reg_y": [], "reg_pred": [],
                    "moledit_total": 0, "moledit_valid": 0, "moledit_exact": 0, "moledit_correct": 0,
                },
            )
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
            elif isinstance(kind, str) and kind.startswith("mol_edit_"):
                subtask = kind.replace("mol_edit_", "")
                meta_info = batch.get("meta_info", [{} for _ in range(batch_size)])
                meta = meta_info[i] if i < len(meta_info) and isinstance(meta_info[i], dict) else {}
                src = self._canonicalize_smiles(meta.get("source_smiles") or batch["smiles"][i])
                tgt = self._canonicalize_smiles(meta.get("target_smiles") or gold_txt)
                gen_txt = self._generate_moledit_pred_text(batch, i)
                pred_smiles = self._extract_predicted_smiles_from_text(gen_txt)
                is_valid = pred_smiles is not None
                is_exact = bool(is_valid and tgt is not None and pred_smiles == tgt)
                is_correct = False
                if is_valid and src is not None:
                    is_correct = self._check_moledit_correct(
                        subtask=subtask,
                        src=src,
                        pred=pred_smiles,
                        add_group=meta.get("added_group"),
                        remove_group=meta.get("removed_group"),
                    )
                self._val_metric_buckets[ds]["moledit_total"] += 1
                self._val_metric_buckets[ds]["moledit_valid"] += int(is_valid)
                self._val_metric_buckets[ds]["moledit_exact"] += int(is_exact)
                self._val_metric_buckets[ds]["moledit_correct"] += int(is_correct)

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
        all_me_total, all_me_valid, all_me_exact, all_me_correct = 0, 0, 0, 0
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
            me_total = int(bucket.get("moledit_total", 0))
            if me_total > 0:
                me_valid = int(bucket.get("moledit_valid", 0))
                me_exact = int(bucket.get("moledit_exact", 0))
                me_correct = int(bucket.get("moledit_correct", 0))
                valid_rate = float(me_valid / me_total)
                exact_rate = float(me_exact / me_total)
                correct_rate = float(me_correct / me_total)
                self.log(f"val/moledit_valid_rate_{ds}", valid_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f"val/moledit_exact_match_rate_{ds}", exact_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                self.log(f"val/moledit_correct_rate_{ds}", correct_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                row[f"val_moledit_valid_rate_{ds}"] = valid_rate
                row[f"val_moledit_exact_match_rate_{ds}"] = exact_rate
                row[f"val_moledit_correct_rate_{ds}"] = correct_rate
                all_me_total += me_total
                all_me_valid += me_valid
                all_me_exact += me_exact
                all_me_correct += me_correct

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
        if all_me_total > 0:
            me_valid_rate = float(all_me_valid / all_me_total)
            me_exact_rate = float(all_me_exact / all_me_total)
            me_correct_rate = float(all_me_correct / all_me_total)
            self.log("val/moledit_valid_rate", me_valid_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            self.log("val/moledit_exact_match_rate", me_exact_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            self.log("val/moledit_correct_rate", me_correct_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            row["val_moledit_valid_rate"] = me_valid_rate
            row["val_moledit_exact_match_rate"] = me_exact_rate
            row["val_moledit_correct_rate"] = me_correct_rate

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