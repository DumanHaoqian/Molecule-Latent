import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from trainer.stage1 import Stage1Trainer


ELEMENT_SYMBOLS = {
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "B"
}

FG_SMARTS = {
    "amine": "[NX3;H2,H1,H0;!$(N=*)]",
    "primary_amine": "[NX3;H2;!$(N=*)]",
    "secondary_amine": "[NX3;H1;!$(N=*)]",
    "tertiary_amine": "[NX3;H0;!$(N=*)]",
    "amide": "[NX3][CX3](=O)[#6]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "simple_ester": "[CX3](=O)[OX2][#6]",
    "lactam": "[NX3;R][CX3;R](=O)",
    "lactone": "[OX2;R][CX3;R](=O)",
    "carbamate": "[NX3][CX3](=O)[OX2]",
    "alcohol": "[OX2H][#6;!$(C=O)]",
    "phenol": "[OX2H]-c1ccccc1",
    "alkyl_ether": "[OD2]([#6])[#6]",
    "aryl_ether": "[OD2]([#6])c",
    "cyclic_ether": "[OD2;R]([#6])[#6]",
    "nitrile": "[CX2]#N",
    "sulfonamide": "S(=O)(=O)N",
    "phosphate": "P(=O)(O)(O)O",
    "phosphonate": "P(=O)(O)(O)[#6]",
    "epoxide": "C1OC1",
    "benzene": "c1ccccc1",
    "aryl": "a",
    "CF3": "[CX4](F)(F)F",
    "tert_butyl": "C(C)(C)C",
    "quaternary_ammonium": "[NX4+]",
    "ammonium": "[NH4+,NH3+,NH2+,NH+]",
}


@dataclass
class RolloutCache:
    prefix_embeds: torch.Tensor
    prefix_mask: torch.Tensor
    latent_seq: torch.Tensor
    generated_ids: torch.Tensor
    sampled_texts: List[str]
    old_logprobs: torch.Tensor


class _PromptBatch:
    pass


def _extract_answer_text(text: str) -> str:
    if text is None:
        return ""
    txt = str(text).strip()
    m = re.search(r"<answer>(.*?)</answer>", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    if "Answer:" in txt:
        return txt.split("Answer:", 1)[1].strip()
    return txt


def _find_valid_smiles(text: str) -> Optional[str]:
    if text is None:
        return None
    answer = _extract_answer_text(text)
    candidates = [answer, text]
    for cand in candidates:
        toks = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/.]+", cand)
        toks = sorted(set(toks), key=len, reverse=True)
        for tok in toks:
            try:
                mol = Chem.MolFromSmiles(tok)
                if mol is not None:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                continue
    return None


def _tanimoto(smiles_a: Optional[str], smiles_b: Optional[str]) -> float:
    if not smiles_a or not smiles_b:
        return 0.0
    try:
        ma = Chem.MolFromSmiles(smiles_a)
        mb = Chem.MolFromSmiles(smiles_b)
        if ma is None or mb is None:
            return 0.0
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fa, fb))
    except Exception:
        return 0.0


def _atomic_counts(mol: Chem.Mol) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        out[sym] = out.get(sym, 0) + 1
    return out


def _bond_profile(mol: Chem.Mol) -> Dict[str, int]:
    out = {"single": 0, "double": 0, "triple": 0, "aromatic": 0, "total": 0}
    for bond in mol.GetBonds():
        out["total"] += 1
        bt = bond.GetBondType()
        if bond.GetIsAromatic():
            out["aromatic"] += 1
        elif bt == Chem.rdchem.BondType.SINGLE:
            out["single"] += 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            out["double"] += 1
        elif bt == Chem.rdchem.BondType.TRIPLE:
            out["triple"] += 1
    return out


def _functional_group_score(mol: Chem.Mol, required_groups: List[str]) -> float:
    if mol is None:
        return 0.0
    if not required_groups:
        return 0.0
    hits = 0
    total = 0
    for name in required_groups:
        patt = FG_SMARTS.get(str(name), None)
        if patt is None:
            continue
        total += 1
        smarts = Chem.MolFromSmarts(patt)
        if smarts is not None and mol.HasSubstructMatch(smarts):
            hits += 1
    if total == 0:
        return 0.0
    return float(hits / total)


def _scaled_exactish_reward(pred: Dict[str, int], target: Dict[str, int]) -> float:
    if not target:
        return 0.0
    vals = []
    for k, v in target.items():
        gt = int(v)
        pd = int(pred.get(k, 0))
        vals.append(max(0.0, 1.0 - abs(pd - gt) / max(1, gt)))
    return float(sum(vals) / max(1, len(vals)))


class Stage2GRPOTrainer(Stage1Trainer):
    def __init__(self, vocab_size, model_config, train_config, tokenizer=None):
        super().__init__(vocab_size=vocab_size, model_config=model_config, train_config=train_config, tokenizer=tokenizer)
        stage2_cfg = getattr(train_config, "stage2", {})

        self.num_grpo_generations = int(getattr(stage2_cfg, "num_grpo_generations", 4))
        self.num_latent_steps = int(
            getattr(
                stage2_cfg,
                "num_latent_steps",
                getattr(stage2_cfg, "max_latent_slots", getattr(self.mol_llama, "stage1_max_latent_slots", 4)),
            )
        )
        self.grpo_clip_eps = float(getattr(stage2_cfg, "grpo_clip_eps", 0.2))
        self.grpo_kl_beta = float(getattr(stage2_cfg, "grpo_kl_beta", 0.01))

        self.generation_max_new_tokens = int(getattr(stage2_cfg, "generation_max_new_tokens", 128))
        self.generation_min_new_tokens = int(getattr(stage2_cfg, "generation_min_new_tokens", 1))
        self.generation_top_p = float(getattr(stage2_cfg, "generation_top_p", 0.95))
        self.generation_temperature = float(getattr(stage2_cfg, "generation_temperature", 1.0))
        self.do_sample = bool(getattr(stage2_cfg, "do_sample", True))

        self.reward_format_weight = float(getattr(stage2_cfg, "reward_format_weight", 0.5))
        self.reward_validity_weight = float(getattr(stage2_cfg, "reward_validity_weight", 2.0))
        self.reward_task_weight = float(getattr(stage2_cfg, "reward_task_weight", 3.0))
        self.reward_faithfulness_weight = float(getattr(stage2_cfg, "reward_faithfulness_weight", 1.0))
        self.reward_text_match_weight = float(getattr(stage2_cfg, "reward_text_match_weight", 0.5))
        self.similarity_floor = float(getattr(stage2_cfg, "similarity_floor", 0.15))
        self.use_length_guard = bool(getattr(stage2_cfg, "use_length_guard", True))
        self.max_smiles_length = int(getattr(stage2_cfg, "max_smiles_length", 256))

        self.replay_lm_weight = float(getattr(stage2_cfg, "replay_lm_weight", 0.2))
        self.replay_latent_weight = float(getattr(stage2_cfg, "replay_latent_weight", 1.0))

        self.train_lora_only = bool(getattr(stage2_cfg, "train_lora_only", True))
        self.allow_latent_core_update = bool(getattr(stage2_cfg, "allow_latent_core_update", False))
        self.allow_stage1_bridge_update = bool(getattr(stage2_cfg, "allow_stage1_bridge_update", False))

        if self.train_lora_only:
            self._freeze_for_stage2()

    def _freeze_for_stage2(self):
        for name, p in self.named_parameters():
            p.requires_grad = False

        for name, p in self.named_parameters():
            if "lora_" in name.lower():
                p.requires_grad = True

        if self.allow_latent_core_update:
            for name, p in self.named_parameters():
                if any(k in name for k in ["mol_llama.latent_proj", "mol_llama.latent_norm"]):
                    p.requires_grad = True

        if self.allow_stage1_bridge_update:
            for name, p in self.named_parameters():
                if any(k in name for k in ["mol_llama.stage1_hidden_to_latent", "mol_llama.stage1_slot_target_proj"]):
                    p.requires_grad = True

    def _build_prompt_batch(self, batch: Dict[str, Any]) -> _PromptBatch:
        text_batch = _PromptBatch()
        text_batch.input_ids = batch["prompt_input_ids"]
        text_batch.attention_mask = batch["prompt_attention_mask"]
        text_batch.mol_token_flag = batch["prompt_mol_token_flag"]
        return text_batch

    def _compute_generation_logprobs_from_scores(self, scores, generated_ids):
        if len(scores) == 0:
            return torch.zeros(generated_ids.size(0), 0, device=generated_ids.device)
        step_logprobs = []
        for t, logits_t in enumerate(scores):
            logp_t = F.log_softmax(logits_t, dim=-1)
            ids_t = generated_ids[:, t].unsqueeze(-1)
            token_logp = logp_t.gather(-1, ids_t).squeeze(-1)
            step_logprobs.append(token_logp)
        return torch.stack(step_logprobs, dim=1)

    @torch.no_grad()
    def _rollout_and_sample(self, batch: Dict[str, Any]) -> RolloutCache:
        prompt_batch = self._build_prompt_batch(batch)
        prompt_embeds = self.mol_llama._build_prefill_inputs_embeds(batch["graph_batch"], prompt_batch)
        prompt_mask = batch["prompt_attention_mask"]

        latent_list = []
        for bi in range(prompt_embeds.size(0)):
            prefix_embeds = self.mol_llama._extract_valid_prefix_embeds(prompt_embeds[bi], prompt_mask[bi])
            latent_seq = self.mol_llama._rollout_latent_tokens(prefix_embeds, num_steps=self.num_latent_steps)
            latent_list.append(latent_seq)
        latent_seq = torch.stack(latent_list, dim=0)

        exp_prefix, exp_mask, exp_latent = [], [], []
        for bi in range(prompt_embeds.size(0)):
            for _ in range(self.num_grpo_generations):
                exp_prefix.append(prompt_embeds[bi])
                exp_mask.append(prompt_mask[bi])
                exp_latent.append(latent_seq[bi])
        exp_prefix = torch.stack(exp_prefix, dim=0)
        exp_mask = torch.stack(exp_mask, dim=0)
        exp_latent = torch.stack(exp_latent, dim=0)

        latent_mask = torch.ones(exp_latent.size(0), exp_latent.size(1), device=exp_latent.device, dtype=exp_mask.dtype)
        full_inputs = torch.cat([exp_prefix, exp_latent], dim=1)
        full_mask = torch.cat([exp_mask, latent_mask], dim=1)

        gen_out = self.mol_llama.llm.generate(
            inputs_embeds=full_inputs,
            attention_mask=full_mask,
            max_new_tokens=self.generation_max_new_tokens,
            min_new_tokens=self.generation_min_new_tokens,
            do_sample=self.do_sample,
            temperature=self.generation_temperature,
            top_p=self.generation_top_p,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        sequences = gen_out.sequences
        gen_len = len(gen_out.scores)
        if sequences.size(1) > gen_len:
            generated_ids = sequences[:, -gen_len:]
        else:
            generated_ids = sequences
        old_logprobs = self._compute_generation_logprobs_from_scores(gen_out.scores, generated_ids)
        sampled_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return RolloutCache(
            prefix_embeds=exp_prefix,
            prefix_mask=exp_mask,
            latent_seq=exp_latent,
            generated_ids=generated_ids,
            sampled_texts=sampled_texts,
            old_logprobs=old_logprobs,
        )

    def _build_replay_inputs(self, cache: RolloutCache):
        answer_ids = cache.generated_ids
        answer_embeds = self.mol_llama.llm.get_input_embeddings()(answer_ids)
        inputs_embeds = torch.cat([cache.prefix_embeds, cache.latent_seq, answer_embeds], dim=1)
        latent_mask = torch.ones(cache.latent_seq.size(0), cache.latent_seq.size(1), device=cache.latent_seq.device, dtype=cache.prefix_mask.dtype)
        answer_mask = (answer_ids != self.tokenizer.pad_token_id).long()
        attention_mask = torch.cat([cache.prefix_mask, latent_mask, answer_mask], dim=1)
        labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, device=inputs_embeds.device, dtype=torch.long)
        prompt_len = cache.prefix_embeds.size(1) + cache.latent_seq.size(1)
        labels[:, prompt_len:] = answer_ids
        return inputs_embeds, attention_mask, labels

    def _compute_current_token_logprobs(self, cache: RolloutCache):
        inputs_embeds, attention_mask, labels = self._build_replay_inputs(cache)
        outputs = self.mol_llama.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~valid, 0)
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid.float()
        return token_logp, valid.float()

    def _score_format_reward(self, text: str) -> float:
        txt = str(text).strip()
        if not txt:
            return 0.0
        if "<answer>" in txt and "</answer>" in txt:
            return 1.0
        if "Answer:" in txt:
            return 1.0
        return 0.2

    def _score_text_match_reward(self, pred_smiles: Optional[str], target_smiles: Optional[str]) -> float:
        if not pred_smiles or not target_smiles:
            return 0.0
        try:
            pm = Chem.MolFromSmiles(pred_smiles)
            tm = Chem.MolFromSmiles(target_smiles)
            if pm is None or tm is None:
                return 0.0
            return 1.0 if Chem.MolToSmiles(pm, canonical=True) == Chem.MolToSmiles(tm, canonical=True) else 0.0
        except Exception:
            return 0.0

    def _score_task_reward(self, pred_smiles: Optional[str], task_spec: Dict[str, Any], subtask: Optional[str]) -> float:
        if not pred_smiles:
            return 0.0
        mol = Chem.MolFromSmiles(pred_smiles)
        if mol is None:
            return 0.0

        subtask_low = str(subtask or task_spec.get("subtask") or "").strip().lower()
        if subtask_low == "atomnum":
            target = task_spec.get("element_counts", {}) or {}
            pred = _atomic_counts(mol)
            return _scaled_exactish_reward(pred, target)
        if subtask_low == "bondnum":
            target = task_spec.get("bond_constraints", {}) or {}
            pred = _bond_profile(mol)
            return _scaled_exactish_reward(pred, target)
        if subtask_low == "functionalgroup":
            required = list(task_spec.get("functional_groups", []) or [])
            return _functional_group_score(mol, required)

        # fallback mixed task scoring
        score = 0.0
        if task_spec.get("element_counts"):
            score += _scaled_exactish_reward(_atomic_counts(mol), task_spec.get("element_counts", {}))
        if task_spec.get("bond_constraints"):
            score += _scaled_exactish_reward(_bond_profile(mol), task_spec.get("bond_constraints", {}))
        if task_spec.get("functional_groups"):
            score += _functional_group_score(mol, list(task_spec.get("functional_groups", [])))
        return min(1.0, score / max(1, sum(bool(task_spec.get(k)) for k in ["element_counts", "bond_constraints", "functional_groups"])))

    def _compute_rewards(self, cache: RolloutCache, batch: Dict[str, Any]):
        bsz = batch["input_ids"].size(0)
        rewards = []
        for i, text in enumerate(cache.sampled_texts):
            src_idx = i // self.num_grpo_generations
            pred_smiles = _find_valid_smiles(text)
            format_r = self._score_format_reward(text)
            valid_r = 1.0 if pred_smiles is not None else 0.0
            subtask = batch.get("subtask_list", [None] * bsz)[src_idx]
            task_spec = batch.get("task_spec_list", [{}] * bsz)[src_idx] or {}
            task_r = self._score_task_reward(pred_smiles, task_spec, subtask)

            target_smiles = batch.get("target_smiles_list", [None] * bsz)[src_idx]
            text_match_r = self._score_text_match_reward(pred_smiles, target_smiles)

            source_smiles = batch.get("source_smiles_list", [None] * bsz)[src_idx]
            faith_r = _tanimoto(pred_smiles, source_smiles) if source_smiles else 0.0
            if faith_r < self.similarity_floor:
                faith_r = 0.0

            if self.use_length_guard and pred_smiles is not None and len(pred_smiles) > self.max_smiles_length:
                valid_r = 0.0
                task_r *= 0.5

            reward = (
                self.reward_format_weight * format_r
                + self.reward_validity_weight * valid_r
                + self.reward_task_weight * task_r
                + self.reward_faithfulness_weight * faith_r
                + self.reward_text_match_weight * text_match_r
            )
            rewards.append(reward)

        rew = torch.tensor(rewards, device=self.device, dtype=torch.float32).view(bsz, self.num_grpo_generations)
        mean = rew.mean(dim=1, keepdim=True)
        std = rew.std(dim=1, keepdim=True).clamp(min=1e-6)
        adv = (rew - mean) / std
        return rew, adv

    def _grpo_step(self, batch: Dict[str, Any]):
        cache = self._rollout_and_sample(batch)
        rewards, advantages = self._compute_rewards(cache, batch)
        current_logprobs, valid_mask = self._compute_current_token_logprobs(cache)
        old_logprobs = cache.old_logprobs

        min_len = min(current_logprobs.size(1), old_logprobs.size(1))
        current_logprobs = current_logprobs[:, :min_len]
        old_logprobs = old_logprobs[:, :min_len]
        valid_mask = valid_mask[:, :min_len]

        flat_adv = advantages.reshape(-1).unsqueeze(1).expand_as(current_logprobs)
        ratio = torch.exp(current_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps)
        obj1 = ratio * flat_adv
        obj2 = clipped_ratio * flat_adv
        policy_obj = torch.minimum(obj1, obj2) * valid_mask
        policy_loss = -policy_obj.sum() / valid_mask.sum().clamp(min=1.0)
        approx_kl = ((old_logprobs - current_logprobs) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        total_loss = policy_loss + self.grpo_kl_beta * approx_kl
        return total_loss, {
            "train/rl_loss": total_loss.detach(),
            "train/policy_loss": policy_loss.detach(),
            "train/approx_kl": approx_kl.detach(),
            "train/reward_mean": rewards.mean().detach(),
            "train/reward_max": rewards.max().detach(),
        }

    def training_step(self, batch, batch_idx):
        train_mode = batch.get("train_mode", "rl")
        if train_mode == "replay":
            return super().training_step(batch, batch_idx)

        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        loss, metrics = self._grpo_step(batch)
        batch_size = batch["input_ids"].size(0)
        self.log("train/loss_total", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=("reward" in k or "loss" in k), logger=True, batch_size=batch_size, sync_dist=True)
        if self.global_step % 100 == 0:
            sources = batch.get("source_dataset", [])
            subtasks = batch.get("subtask_list", [])
            print(f"[Stage2-GRPO] step={self.global_step} source={sources[0] if sources else 'unknown'} subtask={subtasks[0] if subtasks else 'unknown'} loss={float(loss):.4f}")
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Reuse Stage-I validation on the provided moledit validation set.
        return super().validation_step(batch, batch_idx)
