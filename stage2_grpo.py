import contextlib
import json
import math
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import optim

from models.mol_llama import MolLLaMA
from trainer.optims import LinearWarmupCosineLRScheduler


def precision2dtype(precision: str):
    if precision == "16":
        return torch.float16
    if precision == "32":
        return torch.float32
    if "bf16" in str(precision):
        return torch.bfloat16
    raise NotImplementedError(f"Unsupported precision: {precision}")


@dataclass
class RewardBreakdown:
    total: float
    format_reward: float
    validity_reward: float
    task_reward: float
    faithfulness_reward: float
    text_match_reward: float


class Stage2GRPOTrainer(pl.LightningModule):
    """
    Stage-II GRPO trainer for Molecule-Latent.

    Design choices intentionally follow the current repository structure:
    - reuse MolLLaMA._build_prefill_inputs_embeds / _rollout_latent_tokens
    - optimize only text completion logprobs conditioned on replayed latent states
    - keep a small Stage-I replay path to stabilize latent reasoning
    - freeze encoder / latent interface by default, train LoRA adapters on the LLM
    """

    def __init__(self, vocab_size, model_config, train_config, tokenizer=None):
        super().__init__()
        self.train_config = train_config
        self.tokenizer = tokenizer

        stage2_cfg = getattr(train_config, "stage2", SimpleNamespace())
        torch_dtype = precision2dtype(train_config.precision)
        torch_dtype_name = {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
        }[torch_dtype]

        self.mol_llama = MolLLaMA(
            config=model_config,
            vocab_size=vocab_size,
            torch_dtype=torch_dtype_name,
            enable_flash=getattr(train_config, "enable_flash", False),
            use_latent_reasoning=True,
            num_latent_steps=int(getattr(stage2_cfg, "num_latent_steps", 4)),
            lambda_latent=float(getattr(stage2_cfg, "lambda_latent", 1.0)),
            lambda_lm=float(getattr(stage2_cfg, "lambda_lm", 1.0)),
            lambda_cls=float(getattr(stage2_cfg, "lambda_cls", 0.5)),
            stage1_max_latent_slots=int(getattr(stage2_cfg, "max_latent_slots", getattr(stage2_cfg, "num_latent_steps", 4))),
            stage1_wm_reg_keys=list(getattr(stage2_cfg, "regression_targets", [])),
            stage1_wm_cls_keys=list(getattr(stage2_cfg, "classification_targets", [])),
        )

        self.num_grpo_generations = int(getattr(stage2_cfg, "num_grpo_generations", 4))
        self.num_latent_steps = int(getattr(stage2_cfg, "num_latent_steps", 4))
        self.grpo_clip_eps = float(getattr(stage2_cfg, "grpo_clip_eps", 0.2))
        self.generation_max_new_tokens = int(getattr(stage2_cfg, "generation_max_new_tokens", 128))
        self.generation_min_new_tokens = int(getattr(stage2_cfg, "generation_min_new_tokens", 1))
        self.generation_top_p = float(getattr(stage2_cfg, "generation_top_p", 0.95))
        self.generation_temperature = float(getattr(stage2_cfg, "generation_temperature", 1.0))
        self.replay_lm_weight = float(getattr(stage2_cfg, "replay_lm_weight", 0.2))
        self.replay_latent_weight = float(getattr(stage2_cfg, "replay_latent_weight", 1.0))
        self.reward_format_weight = float(getattr(stage2_cfg, "reward_format_weight", 0.5))
        self.reward_validity_weight = float(getattr(stage2_cfg, "reward_validity_weight", 2.0))
        self.reward_task_weight = float(getattr(stage2_cfg, "reward_task_weight", 3.0))
        self.reward_faithfulness_weight = float(getattr(stage2_cfg, "reward_faithfulness_weight", 1.0))
        self.reward_text_match_weight = float(getattr(stage2_cfg, "reward_text_match_weight", 0.5))
        self.similarity_floor = float(getattr(stage2_cfg, "similarity_floor", 0.15))
        self.use_length_guard = bool(getattr(stage2_cfg, "use_length_guard", True))
        self.max_smiles_length = int(getattr(stage2_cfg, "max_smiles_length", 256))
        self.replay_source_names = set(getattr(stage2_cfg, "replay_source_names", ["PubChemLatent", "ComprehensiveConversation"]))
        self.rl_source_names = set(getattr(stage2_cfg, "rl_source_names", ["MolEditLatent", "DownstreamTasks"]))

        self.stage1_regression_stds = {
            "molecular_weight": 500.0,
            "logp": 5.0,
            "tpsa": 150.0,
            "hbd": 5.0,
            "hba": 10.0,
            "num_rings": 6.0,
            "aromatic_ring_count": 4.0,
            "qed": 1.0,
        }

        self._freeze_for_stage2(stage2_cfg)

    # ---------------------------------------------------------------------
    # Loading / freezing
    # ---------------------------------------------------------------------
    def _set_requires_grad_by_prefix(self, prefixes, flag: bool):
        prefixes = tuple(prefixes)
        for name, param in self.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad = flag

    def _freeze_for_stage2(self, stage2_cfg):
        # Freeze everything first, then selectively re-enable LoRA and optionally a few heads.
        for _, param in self.named_parameters():
            param.requires_grad = False

        train_lora_only = bool(getattr(stage2_cfg, "train_lora_only", True))
        allow_latent_core = bool(getattr(stage2_cfg, "allow_latent_core_update", False))
        allow_stage1_bridge = bool(getattr(stage2_cfg, "allow_stage1_bridge_update", False))

        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        if not train_lora_only:
            self._set_requires_grad_by_prefix([
                "mol_llama.llm_proj",
                "mol_llama.latent_proj",
                "mol_llama.latent_norm",
                "mol_llama.stage1_hidden_to_latent",
                "mol_llama.stage1_slot_target_proj",
            ], True)

        if allow_latent_core:
            self._set_requires_grad_by_prefix([
                "mol_llama.latent_proj",
                "mol_llama.latent_norm",
            ], True)

        if allow_stage1_bridge:
            self._set_requires_grad_by_prefix([
                "mol_llama.stage1_hidden_to_latent",
                "mol_llama.stage1_slot_target_proj",
            ], True)

        # Always keep encoder frozen in Stage-II GRPO by default.
        self._set_requires_grad_by_prefix([
            "mol_llama.encoder",
            "mol_llama.llm_proj",
            "mol_llama.subregion_proj",
            "mol_llama.stage1_wm_head",
            "mol_llama.latent_cls_head",
        ], False)

        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"[Stage2GRPOTrainer] trainable params: {len(trainable)} tensors")
        for n in trainable[:20]:
            print(f"  - {n}")

    def load_from_stage1_ckpt(self, ckpt_path):
        self.mol_llama.load_from_ckpt(ckpt_path)

    def load_from_stage2_ckpt(self, ckpt_path):
        self.mol_llama.load_from_ckpt(ckpt_path)

    def load_from_hf_dir(self, hf_dir):
        self.mol_llama.load_from_hf_dir(hf_dir)

    # ---------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------
    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == "linear_warmup_cosine_lr":
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                self.train_config.max_epochs,
                self.train_config.min_lr,
                self.train_config.init_lr,
                warmup_steps,
                self.train_config.warmup_lr,
            )
        elif self.train_config.scheduler == "None":
            self.scheduler = None
        else:
            raise NotImplementedError(self.train_config.scheduler)
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop("optimizer_states", None)
        to_remove = []
        for key in list(checkpoint.get("state_dict", {}).keys()):
            try:
                if not self.get_parameter(key).requires_grad:
                    to_remove.append(key)
            except Exception:
                to_remove.append(key)
        for key in to_remove:
            checkpoint["state_dict"].pop(key, None)

    # ---------------------------------------------------------------------
    # Helpers for prompt / latent replay
    # ---------------------------------------------------------------------
    def _as_text_batch(self, input_ids, attention_mask, mol_token_flag, labels=None):
        tb = SimpleNamespace()
        tb.input_ids = input_ids
        tb.attention_mask = attention_mask
        tb.mol_token_flag = mol_token_flag
        tb.labels = labels
        return tb

    def _compute_latent_loss(self, latent_states, slot_targets, slot_mask):
        if slot_mask.sum() == 0:
            return torch.tensor(0.0, device=latent_states.device)
        z = F.normalize(latent_states, dim=-1)
        t = F.normalize(slot_targets.to(z.dtype), dim=-1)
        cos = (z * t).sum(dim=-1)
        valid = slot_mask.to(cos.dtype)
        return ((1.0 - cos) * valid).sum() / valid.sum().clamp(min=1.0)

    def _build_prompt_embeds(self, batch):
        text_batch = self._as_text_batch(
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            batch["prompt_mol_token_flag"],
            None,
        )
        return self.mol_llama._build_prefill_inputs_embeds(batch["graph_batch"], text_batch)

    def _extract_valid_prompt_embeds(self, sample_embeds, sample_attention):
        valid_positions = torch.nonzero(sample_attention > 0, as_tuple=False).flatten()
        assert valid_positions.numel() > 0, "Empty prompt after padding removal."
        start = valid_positions[0].item()
        end = valid_positions[-1].item() + 1
        return sample_embeds[start:end]

    def _truncate_generated_ids(self, sequences: torch.Tensor) -> torch.Tensor:
        # For generate(inputs_embeds=...), HF can return either only generated ids or a
        # sequence including a synthetic prefix. Keeping the tail is the safest option.
        if sequences.size(1) > self.generation_max_new_tokens:
            sequences = sequences[:, -self.generation_max_new_tokens :]
        return sequences

    def _sample_completion_group(self, prompt_embeds: torch.Tensor):
        latent_seq = self.mol_llama._rollout_latent_tokens(prompt_embeds, num_steps=self.num_latent_steps)
        cond_embeds = torch.cat([prompt_embeds, latent_seq], dim=0)
        cond_embeds = cond_embeds.unsqueeze(0).expand(self.num_grpo_generations, -1, -1).contiguous()
        cond_attn = torch.ones(
            (self.num_grpo_generations, cond_embeds.shape[1]),
            device=cond_embeds.device,
            dtype=torch.long,
        )
        outputs = self.mol_llama.llm.generate(
            inputs_embeds=cond_embeds,
            attention_mask=cond_attn,
            do_sample=True,
            top_p=self.generation_top_p,
            temperature=self.generation_temperature,
            max_new_tokens=self.generation_max_new_tokens,
            min_new_tokens=self.generation_min_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        completion_ids = self._truncate_generated_ids(outputs)
        old_logprobs = self._compute_completion_logprobs(prompt_embeds, latent_seq, completion_ids).detach()
        texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return latent_seq.detach(), completion_ids.detach(), old_logprobs, texts

    def _compute_completion_logprobs(self, prompt_embeds: torch.Tensor, latent_seq: torch.Tensor, completion_ids: torch.Tensor):
        if completion_ids.ndim == 1:
            completion_ids = completion_ids.unsqueeze(0)
        bsz, tgt_len = completion_ids.shape
        token_embeds = self.mol_llama.llm.get_input_embeddings()(completion_ids)
        prompt_part = prompt_embeds.unsqueeze(0).expand(bsz, -1, -1)
        latent_part = latent_seq.unsqueeze(0).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prompt_part, latent_part, token_embeds], dim=1)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        completion_mask = (completion_ids != pad_id).long()
        if completion_mask.sum() == 0:
            completion_mask = torch.ones_like(completion_ids, dtype=torch.long)
        prefix_mask = torch.ones(
            (bsz, prompt_part.shape[1] + latent_part.shape[1]),
            device=completion_ids.device,
            dtype=torch.long,
        )
        attention_mask = torch.cat([prefix_mask, completion_mask], dim=1)

        labels = completion_ids.clone()
        labels[completion_mask == 0] = -100
        ignore_prefix = torch.full(
            (bsz, prompt_part.shape[1] + latent_part.shape[1]),
            -100,
            device=completion_ids.device,
            dtype=labels.dtype,
        )
        labels = torch.cat([ignore_prefix, labels], dim=1)

        out = self.mol_llama.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        seq_logprobs = (token_logprobs * valid.to(token_logprobs.dtype)).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return seq_logprobs

    # ---------------------------------------------------------------------
    # Reward helpers
    # ---------------------------------------------------------------------
    def _extract_answer_text(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) == 0:
            return text
        m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"Answer\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return text

    def _extract_smiles(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if len(text) == 0:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for key in ["molecule", "smiles", "answer", "edited_smiles"]:
                    value = obj.get(key)
                    if isinstance(value, str) and len(value.strip()) > 0:
                        return value.strip()
        except Exception:
            pass

        answer = self._extract_answer_text(text)
        if len(answer) == 0:
            return None
        candidates = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+", answer)
        candidates = [c for c in candidates if 1 <= len(c) <= self.max_smiles_length]
        if not candidates:
            return None
        # Longest token is usually the actual SMILES rather than a label word.
        return max(candidates, key=len)

    def _canonicalize_smiles(self, smiles: Optional[str]) -> Optional[str]:
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    def _valid_smiles(self, smiles: Optional[str]) -> bool:
        return self._canonicalize_smiles(smiles) is not None

    def _tanimoto(self, smiles_a: Optional[str], smiles_b: Optional[str]) -> float:
        can_a = self._canonicalize_smiles(smiles_a)
        can_b = self._canonicalize_smiles(smiles_b)
        if can_a is None or can_b is None:
            return 0.0
        mol_a = Chem.MolFromSmiles(can_a)
        mol_b = Chem.MolFromSmiles(can_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    def _parse_binary_answer(self, text: str, dataset_name: str = "") -> Optional[int]:
        t = (text or "").strip().lower()
        ds = str(dataset_name or "").upper()
        if ds == "PAMPA":
            if "high permeability" in t:
                return 1
            if "low permeability" in t or "low-to-moderate" in t or "low to moderate" in t:
                return 0
        if re.search(r"\byes\b", t):
            return 1
        if re.search(r"\bno\b", t):
            return 0
        if re.search(r"\btrue\b", t):
            return 1
        if re.search(r"\bfalse\b", t):
            return 0
        return None

    def _parse_float_answer(self, text: str) -> Optional[float]:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _score_task_reward(self, batch: Dict[str, Any], sample_idx: int, answer_text: str, pred_smiles: Optional[str]) -> float:
        task_kind = None
        task_name = None
        if "task_kind" in batch and sample_idx < len(batch["task_kind"]):
            task_kind = batch["task_kind"][sample_idx]
        if "task_name" in batch and sample_idx < len(batch["task_name"]):
            task_name = batch["task_name"][sample_idx]

        # 1) Binary downstream tasks.
        if task_kind == "binary_classification":
            label_mask = bool(batch["task_label_mask"][sample_idx].item()) if "task_label_mask" in batch else False
            if label_mask:
                gold = int(batch["task_label"][sample_idx].item())
                pred = self._parse_binary_answer(answer_text, dataset_name=str(task_name or ""))
                if pred is not None:
                    return 1.0 if int(pred) == gold else 0.0

        # 2) Regression tasks.
        if task_kind == "regression":
            reg_mask = bool(batch["task_regression_mask"][sample_idx].item()) if "task_regression_mask" in batch else False
            if reg_mask:
                gold = float(batch["task_regression_value"][sample_idx].item())
                pred = self._parse_float_answer(answer_text)
                if pred is not None:
                    scale = max(1.0, abs(gold))
                    return math.exp(-abs(pred - gold) / scale)

        # 3) Exact/canonical SMILES if gold text itself looks like a molecule.
        if "text_targets" in batch and sample_idx < len(batch["text_targets"]):
            gold_text = batch["text_targets"][sample_idx]
            gold_smiles = self._extract_smiles(str(gold_text))
            gold_can = self._canonicalize_smiles(gold_smiles)
            pred_can = self._canonicalize_smiles(pred_smiles)
            if gold_can is not None and pred_can is not None:
                return 1.0 if gold_can == pred_can else 0.0

        # 4) Fallback text-match reward as task reward when structured labels are absent.
        if "text_targets" in batch and sample_idx < len(batch["text_targets"]):
            gold_text = str(batch["text_targets"][sample_idx] or "").strip().lower()
            pred_text = self._extract_answer_text(answer_text).strip().lower()
            if len(gold_text) > 0 and len(pred_text) > 0:
                return 1.0 if gold_text == pred_text else 0.0
        return 0.0

    def _score_text_match_reward(self, batch: Dict[str, Any], sample_idx: int, answer_text: str) -> float:
        if "text_targets" not in batch or sample_idx >= len(batch["text_targets"]):
            return 0.0
        gold_text = str(batch["text_targets"][sample_idx] or "").strip().lower()
        pred_text = self._extract_answer_text(answer_text).strip().lower()
        if len(gold_text) == 0 or len(pred_text) == 0:
            return 0.0
        if gold_text == pred_text:
            return 1.0
        if gold_text in pred_text or pred_text in gold_text:
            return 0.5
        return 0.0

    def _score_faithfulness_reward(self, source_smiles: str, pred_smiles: Optional[str]) -> float:
        sim = self._tanimoto(source_smiles, pred_smiles)
        if sim < self.similarity_floor:
            return 0.0
        return sim

    def _compute_reward_breakdown(self, batch: Dict[str, Any], sample_idx: int, generated_text: str) -> RewardBreakdown:
        answer_text = self._extract_answer_text(generated_text)
        pred_smiles = self._extract_smiles(answer_text)
        source_smiles = batch["smiles"][sample_idx] if sample_idx < len(batch["smiles"]) else ""

        format_reward = 1.0 if len(answer_text.strip()) > 0 else 0.0
        validity_reward = 1.0 if self._valid_smiles(pred_smiles) else 0.0
        task_reward = self._score_task_reward(batch, sample_idx, answer_text, pred_smiles)
        faithfulness_reward = self._score_faithfulness_reward(source_smiles, pred_smiles)
        text_match_reward = self._score_text_match_reward(batch, sample_idx, answer_text)

        total = (
            self.reward_format_weight * format_reward
            + self.reward_validity_weight * validity_reward
            + self.reward_task_weight * task_reward
            + self.reward_faithfulness_weight * faithfulness_reward
            + self.reward_text_match_weight * text_match_reward
        )
        return RewardBreakdown(
            total=float(total),
            format_reward=float(format_reward),
            validity_reward=float(validity_reward),
            task_reward=float(task_reward),
            faithfulness_reward=float(faithfulness_reward),
            text_match_reward=float(text_match_reward),
        )

    # ---------------------------------------------------------------------
    # Training step modes
    # ---------------------------------------------------------------------
    def _training_step_replay(self, batch):
        outputs = self.mol_llama.forward_stage1(batch)
        lm_loss = outputs["lm_loss"] if outputs["lm_loss"] is not None else torch.tensor(0.0, device=self.device)
        latent_loss = self._compute_latent_loss(
            outputs["latent_states"],
            outputs["slot_targets"],
            batch["latent_slot_mask"].to(outputs["latent_states"].device),
        )
        loss = self.replay_lm_weight * lm_loss + self.replay_latent_weight * latent_loss
        return loss, {
            "replay/loss": loss.detach(),
            "replay/loss_lm": lm_loss.detach(),
            "replay/loss_latent": latent_loss.detach(),
        }

    def _training_step_grpo(self, batch):
        prompt_embeds_batch = self._build_prompt_embeds(batch)
        sample_losses = []
        total_rewards = []
        format_rewards = []
        validity_rewards = []
        task_rewards = []
        faithfulness_rewards = []
        text_match_rewards = []

        for bi in range(prompt_embeds_batch.size(0)):
            prompt_embeds = self._extract_valid_prompt_embeds(prompt_embeds_batch[bi], batch["prompt_attention_mask"][bi])
            latent_seq, completion_ids, old_logprobs, texts = self._sample_completion_group(prompt_embeds)

            reward_list = [self._compute_reward_breakdown(batch, bi, txt) for txt in texts]
            rewards = torch.tensor([rb.total for rb in reward_list], device=prompt_embeds.device, dtype=torch.float32)
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-6)
            current_logprobs = self._compute_completion_logprobs(prompt_embeds, latent_seq, completion_ids)
            ratio = torch.exp(current_logprobs - old_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps) * advantages
            sample_loss = -torch.min(unclipped, clipped).mean()
            sample_losses.append(sample_loss)

            total_rewards.extend([rb.total for rb in reward_list])
            format_rewards.extend([rb.format_reward for rb in reward_list])
            validity_rewards.extend([rb.validity_reward for rb in reward_list])
            task_rewards.extend([rb.task_reward for rb in reward_list])
            faithfulness_rewards.extend([rb.faithfulness_reward for rb in reward_list])
            text_match_rewards.extend([rb.text_match_reward for rb in reward_list])

        loss = torch.stack(sample_losses).mean() if sample_losses else torch.tensor(0.0, device=self.device)
        metrics = {
            "grpo/loss": loss.detach(),
            "grpo/reward_total": torch.tensor(total_rewards, device=self.device).mean() if total_rewards else torch.tensor(0.0, device=self.device),
            "grpo/reward_format": torch.tensor(format_rewards, device=self.device).mean() if format_rewards else torch.tensor(0.0, device=self.device),
            "grpo/reward_validity": torch.tensor(validity_rewards, device=self.device).mean() if validity_rewards else torch.tensor(0.0, device=self.device),
            "grpo/reward_task": torch.tensor(task_rewards, device=self.device).mean() if task_rewards else torch.tensor(0.0, device=self.device),
            "grpo/reward_faithfulness": torch.tensor(faithfulness_rewards, device=self.device).mean() if faithfulness_rewards else torch.tensor(0.0, device=self.device),
            "grpo/reward_text_match": torch.tensor(text_match_rewards, device=self.device).mean() if text_match_rewards else torch.tensor(0.0, device=self.device),
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch["input_ids"].size(0)
        source_name = batch["source_dataset"][0] if len(batch["source_dataset"]) > 0 else "unknown"
        mode = batch.get("train_mode", "rl")
        if mode == "replay":
            loss, metrics = self._training_step_replay(batch)
        else:
            loss, metrics = self._training_step_grpo(batch)

        self.log("train/loss_total", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/mode_is_replay", 1.0 if mode == "replay" else 0.0, on_step=True, on_epoch=False, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train/source_name", float(hash(source_name) % 1000), on_step=True, on_epoch=False, logger=False, batch_size=batch_size, sync_dist=False)
        for key, value in metrics.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size, sync_dist=True)

        if self.global_step % 50 == 0:
            print(f"[Stage2-GRPO] step={self.global_step} source={source_name} mode={mode} loss={float(loss):.4f}")
        return loss
