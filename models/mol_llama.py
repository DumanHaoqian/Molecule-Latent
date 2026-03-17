"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.cuda.amp import autocast as autocast
from peft import get_peft_model, LoraConfig, TaskType

from utils.configuration_mol_llama import MolLLaMAConfig
from models.mol_llama_encoder import MolLLaMAEncoder
from transformers import LlamaForCausalLM, PreTrainedModel

from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from torch_geometric.data import Data, Batch
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from data_provider.collaters import Mol3DCollater
import numpy as np

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MolLLaMAPreTrainedModel(PreTrainedModel):
    config_class = MolLLaMAConfig
    base_model_prefix = 'mllm'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"encoder.graph_encoder",
        r"llm."
    ]


class Stage1WorldModelHead(nn.Module):
    def __init__(self, hidden_size, regression_keys, classification_keys):
        super().__init__()
        self.regression_keys = list(regression_keys)
        self.classification_keys = list(classification_keys)
        self.reg_head = nn.Linear(hidden_size, len(self.regression_keys))
        self.cls_head = nn.Linear(hidden_size, len(self.classification_keys))

    def forward(self, pooled_latent):
        return {
            "regression": self.reg_head(pooled_latent),
            "classification_logits": self.cls_head(pooled_latent),
            "regression_keys": self.regression_keys,
            "classification_keys": self.classification_keys,
        }

class MolLLaMA(MolLLaMAPreTrainedModel):
    def __init__(
        self,
        config: MolLLaMAConfig,
        vocab_size=None,
        torch_dtype="bfloat16",
        enable_flash=True,
        use_latent_reasoning=False,
        num_latent_steps=4,
        lambda_latent=1.0,
        lambda_lm=1.0,
        lambda_cls=0.5,
        stage1_max_latent_slots=6,
        stage1_wm_reg_keys=None,
        stage1_wm_cls_keys=None,
    ):
        super().__init__(config)

        ## Intialize encoder
        self.encoder = MolLLaMAEncoder(
            graph_encoder_config = config.graph_encoder_config,
            blending_module_config = config.blending_module_config,
            qformer_config = config.qformer_config,
        )
        self.postprocess_encoder()

        ## Initialize LLM
        if torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        if enable_flash:
            self.llm = LlamaForCausalLM.from_pretrained(config.llm_config.llm_model, torch_dtype=torch_dtype, 
                                                            attn_implementation="flash_attention_2")
        else:
            self.llm = LlamaForCausalLM.from_pretrained(config.llm_config.llm_model, torch_dtype=torch_dtype)
        self.llm.resize_token_embeddings(vocab_size)
        
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                    inference_mode=False,
                                    r=config.llm_config.lora_config.r,
                                    lora_alpha=config.llm_config.lora_config.lora_alpha,
                                    lora_dropout=config.llm_config.lora_config.lora_dropout,
                                    target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        self.peft_config = peft_config
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        self.llm_proj = nn.Linear(self.encoder.Qformer.config.hidden_size, 
                                    self.llm.config.hidden_size)

        # Optional decoder-side latent reasoning path.
        self.use_latent_reasoning = use_latent_reasoning
        self.num_latent_steps = num_latent_steps
        self.lambda_latent = lambda_latent
        self.lambda_lm = lambda_lm
        self.lambda_cls = lambda_cls
        self.latent_temperature = 0.1
        self.latent_proj = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        self.latent_norm = nn.LayerNorm(self.llm.config.hidden_size)
        self.subregion_proj = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        self.latent_cls_head = nn.Linear(self.llm.config.hidden_size, 2)

        # Stage-I unified training heads (lightweight and optional).
        # stage1_max_latent_slots means max number of subgraph-aligned latent slots.
        # Stage-I latent reasoning uses 1 extra global latent token.
        self.stage1_max_latent_slots = int(stage1_max_latent_slots)
        self.stage1_slot_queries = nn.Parameter(
            torch.randn(1, self.stage1_max_latent_slots, self.llm.config.hidden_size) * 0.02
        )
        self.stage1_hidden_to_latent = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        self.stage1_slot_target_proj = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        self.stage1_wm_reg_keys = list(stage1_wm_reg_keys or ["molecular_weight", "logp", "tpsa", "hbd", "hba", "num_rings", "aromatic_ring_count", "qed"])
        self.stage1_wm_cls_keys = list(stage1_wm_cls_keys or ["ro5_pass", "ro5_violation_count"])
        self.stage1_wm_head = Stage1WorldModelHead(
            hidden_size=self.llm.config.hidden_size,
            regression_keys=self.stage1_wm_reg_keys,
            classification_keys=self.stage1_wm_cls_keys,
        )

    def postprocess_encoder(self):
        self.encoder.Qformer.cls = None
        self.encoder.Qformer.bert.embeddings.word_embeddings = None
        self.encoder.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.encoder.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.encoder.graph_proj = None
        self.encoder.text_proj = None
        self.encoder.gtm_head = None

    def _build_prefill_inputs_embeds(self, graph_batch, text_batch):
        _, _, query_output = self.encoder.graph_forward(graph_batch)      
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]
        # Robust molecule-token injection:
        # Some conversation history may also contain "<mol>", which can make
        # tokenized mol-placeholder counts different from num_query_tokens.
        # We align per-sample and only fill the first min(count, num_query_tokens)
        # positions to avoid shape mismatch at runtime.
        mol_flag = text_batch.mol_token_flag
        bsz, qnum, _ = query_output.shape
        for bi in range(bsz):
            pos = torch.nonzero(mol_flag[bi], as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            use_n = min(pos.numel(), qnum)
            inputs_embeds[bi, pos[:use_n], :] = query_output[bi, :use_n, :].to(inputs_embeds.dtype)
        return inputs_embeds

    def _sample_segments(self, sample_embeds, sample_labels, sample_attention):
        valid_positions = torch.nonzero(sample_attention > 0, as_tuple=False).flatten()
        assert valid_positions.numel() > 0, "Empty sequence after padding removal."
        start = valid_positions[0].item()
        end = valid_positions[-1].item() + 1

        valid_labels = sample_labels[start:end]
        label_positions = torch.nonzero(valid_labels != -100, as_tuple=False).flatten()
        if label_positions.numel() > 0:
            first_label_idx = start + label_positions[0].item()
        else:
            first_label_idx = end

        # Keep at least one prefix token to start decoder-side latent rollout.
        if first_label_idx <= start:
            first_label_idx = min(start + 1, end)

        prefix_embeds = sample_embeds[start:first_label_idx]
        suffix_embeds = sample_embeds[first_label_idx:end]
        suffix_labels = sample_labels[first_label_idx:end]
        return prefix_embeds, suffix_embeds, suffix_labels

    def _rollout_latent_tokens(self, prefix_embeds, num_steps=None):
        if num_steps is None:
            num_steps = self.num_latent_steps
        prefix_attn = torch.ones((1, prefix_embeds.shape[0]), device=prefix_embeds.device, dtype=torch.long)
        prefill_out = self.llm(
            inputs_embeds=prefix_embeds.unsqueeze(0),
            attention_mask=prefix_attn,
            return_dict=True,
            use_cache=True,
            output_hidden_states=True,
        )
        if hasattr(prefill_out, "last_hidden_state") and prefill_out.last_hidden_state is not None:
            current_hidden = prefill_out.last_hidden_state[:, -1, :]
        else:
            current_hidden = prefill_out.hidden_states[-1][:, -1, :]
        past_key_values = prefill_out.past_key_values

        latent_tokens = []
        rolling_attn = prefix_attn
        for _ in range(num_steps):
            proj_in = current_hidden.to(self.latent_proj.weight.dtype)
            latent_token = self.latent_norm(self.latent_proj(proj_in)).to(current_hidden.dtype)  # [1, D]
            latent_tokens.append(latent_token.squeeze(0))
            rolling_attn = torch.cat(
                [rolling_attn, torch.ones((1, 1), device=rolling_attn.device, dtype=rolling_attn.dtype)], dim=1
            )
            step_out = self.llm(
                inputs_embeds=latent_token.unsqueeze(1),
                attention_mask=rolling_attn,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
            )
            if hasattr(step_out, "last_hidden_state") and step_out.last_hidden_state is not None:
                current_hidden = step_out.last_hidden_state[:, -1, :]
            else:
                current_hidden = step_out.hidden_states[-1][:, -1, :]
            past_key_values = step_out.past_key_values

        return torch.stack(latent_tokens, dim=0)  # [K, D]

    def _extract_valid_prefix_embeds(self, sample_embeds, sample_attention):
        valid_positions = torch.nonzero(sample_attention > 0, as_tuple=False).flatten()
        assert valid_positions.numel() > 0, "Empty sequence after padding removal."
        start = valid_positions[0].item()
        end = valid_positions[-1].item() + 1
        return sample_embeds[start:end]

    def _encode_subregion_smiles(self, candidate_smiles):
        if not isinstance(candidate_smiles, list) or len(candidate_smiles) == 0:
            return None
        normalized_smiles = []
        for item in candidate_smiles:
            if isinstance(item, dict):
                smi = item.get("smiles", "")
            else:
                smi = item
            if isinstance(smi, str) and len(smi) > 0:
                normalized_smiles.append(smi)
        if len(normalized_smiles) == 0:
            return None
        try:
            if hasattr(self.encoder, "unimol_dictionary"):
                graph_batch = get_mol_graphs(normalized_smiles, self.encoder.unimol_dictionary, self.device)
            else:
                mol_graphs = []
                for smiles in normalized_smiles:
                    graph = smiles2graph(smiles)
                    mol_graphs.append(Data(x=graph["node_feat"], edge_index=graph["edge_index"], edge_attr=graph["edge_feat"]))
                graph_batch = {"moleculestm": Batch.from_data_list(mol_graphs).to(self.device)}
        except Exception:
            return None
        if graph_batch is None:
            return None
        _, _, query_output = self.encoder.graph_forward(graph_batch)
        subregion_feats = self.llm_proj(query_output.last_hidden_state).mean(dim=1)  # [N, D]
        subregion_feats = self.subregion_proj(subregion_feats)
        subregion_feats = F.normalize(subregion_feats, dim=-1)
        return subregion_feats

    def forward(self, graph_batch, text_batch, other_infos=None):
        if not self.use_latent_reasoning:
            inputs_embeds = self._build_prefill_inputs_embeds(graph_batch, text_batch)
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=text_batch.attention_mask,
                return_dict=True,
                labels=text_batch.labels,
                use_cache=False,
            )
            return outputs

        # Decoder-side latent reasoning path (prefill -> latent rollout -> text decoding).
        inputs_embeds = self._build_prefill_inputs_embeds(graph_batch, text_batch)
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        lm_losses, latent_losses, cls_losses = [], [], []
        for i in range(batch_size):
            prefix_embeds, suffix_embeds, suffix_labels = self._sample_segments(
                inputs_embeds[i], text_batch.labels[i], text_batch.attention_mask[i]
            )
            latent_tokens = self._rollout_latent_tokens(prefix_embeds)  # [K, D]

            # Build [prefix, latent, suffix] so text loss is computed after latent stage.
            merged_embeds = torch.cat([prefix_embeds, latent_tokens, suffix_embeds], dim=0)
            merged_attention = torch.ones((1, merged_embeds.shape[0]), device=device, dtype=torch.long)
            latent_label_pad = torch.full(
                (prefix_embeds.shape[0] + self.num_latent_steps,), -100, device=device, dtype=torch.long
            )
            merged_labels = torch.cat([latent_label_pad, suffix_labels], dim=0)
            if suffix_labels.numel() > 0:
                lm_out = self.llm(
                    inputs_embeds=merged_embeds.unsqueeze(0),
                    attention_mask=merged_attention,
                    labels=merged_labels.unsqueeze(0),
                    return_dict=True,
                    use_cache=False,
                )
                lm_losses.append(lm_out.loss)

            if other_infos is None:
                continue

            # Latent alignment loss against candidate subregion features.
            candidate_subgraphs = other_infos.get("candidate_subgraphs", [None] * batch_size)[i]
            latent_targets = other_infos.get("latent_targets", [None] * batch_size)[i]
            candidate_feats = self._encode_subregion_smiles(candidate_subgraphs)
            if candidate_feats is not None and isinstance(latent_targets, list) and len(latent_targets) > 0:
                z_norm = F.normalize(latent_tokens, dim=-1)
                step_losses = []
                num_steps = min(self.num_latent_steps, len(latent_targets))
                for step_idx in range(num_steps):
                    target_item = latent_targets[step_idx]
                    if isinstance(target_item, dict):
                        target_idx = int(target_item.get("candidate_id", -1))
                    else:
                        target_idx = int(target_item)
                    if target_idx < 0 or target_idx >= candidate_feats.shape[0]:
                        continue
                    sim_logits = (z_norm[step_idx:step_idx + 1] @ candidate_feats.T) / self.latent_temperature
                    target = torch.tensor([target_idx], device=device, dtype=torch.long)
                    step_losses.append(F.cross_entropy(sim_logits, target))
                if len(step_losses) > 0:
                    latent_losses.append(torch.stack(step_losses).mean())

            # Optional label classification from pooled latent tokens.
            class_label = other_infos.get("class_label", [None] * batch_size)[i]
            if class_label is not None:
                try:
                    class_label = int(class_label)
                    if class_label in (0, 1):
                        cls_logits = self.latent_cls_head(latent_tokens.mean(dim=0, keepdim=True))
                        cls_target = torch.tensor([class_label], device=device, dtype=torch.long)
                        cls_losses.append(F.cross_entropy(cls_logits, cls_target))
                except (TypeError, ValueError):
                    pass

        loss_lm = torch.stack(lm_losses).mean() if len(lm_losses) > 0 else torch.tensor(0.0, device=device)
        loss_latent = torch.stack(latent_losses).mean() if len(latent_losses) > 0 else torch.tensor(0.0, device=device)
        loss_cls = torch.stack(cls_losses).mean() if len(cls_losses) > 0 else torch.tensor(0.0, device=device)
        total_loss = self.lambda_lm * loss_lm + self.lambda_latent * loss_latent + self.lambda_cls * loss_cls

        return {
            "loss": total_loss,
            "loss_lm": loss_lm.detach(),
            "loss_latent": loss_latent.detach(),
            "loss_cls": loss_cls.detach(),
        }

    def forward_stage1(self, batch):
        """
        Stage-I unified forward.
        Returns a dict with lm_loss / latent_states / wm_preds and required
        tensors for trainer-side task-aware loss routing.
        """
        graph_batch = batch["graph_batch"]
        # Build a minimal text batch object to reuse existing multimodal injection.
        class _TextBatch:
            pass
        text_batch = _TextBatch()
        text_batch.input_ids = batch["input_ids"]
        text_batch.attention_mask = batch["attention_mask"]
        text_batch.labels = batch["labels"]
        text_batch.mol_token_flag = batch["mol_token_flag"]

        inputs_embeds = self._build_prefill_inputs_embeds(graph_batch, text_batch)
        llm_out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=text_batch.attention_mask,
            labels=text_batch.labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # Stage-I latent reasoning rollout:
        # hidden -> latent token embedding -> feed back as next token embedding for K steps.
        # K = 1 (global latent token) + max_subgraph_slots.
        max_subgraph_slots = batch["latent_slot_input_ids"].shape[1]
        total_latent_steps = 1 + max_subgraph_slots
        latent_tokens_all = []
        for bi in range(inputs_embeds.shape[0]):
            prefix_embeds = self._extract_valid_prefix_embeds(inputs_embeds[bi], text_batch.attention_mask[bi])
            # [K_total, H], first token is global latent token.
            latent_seq = self._rollout_latent_tokens(prefix_embeds, num_steps=total_latent_steps)
            latent_tokens_all.append(latent_seq)
        latent_tokens_all = torch.stack(latent_tokens_all, dim=0)  # [B, K_total, H]
        global_latent_state = latent_tokens_all[:, 0, :]  # [B, H]
        latent_states = latent_tokens_all[:, 1:, :]       # [B, K_subgraph, H]
        latent_states = self.stage1_hidden_to_latent(latent_states)

        # World modeling uses the global latent token.
        wm_preds = self.stage1_wm_head(global_latent_state)

        # Slot target embedding:
        # Prefer graph-based subgraph feature encoding from slot smiles (closer to
        # "latent token <-> subgraph feature" objective), fallback to descriptor text.
        model_device = next(self.parameters()).device
        slot_mask_batch = batch["latent_slot_mask"].to(model_device)  # [B, K]
        slot_smiles_2d = batch.get("latent_slot_smiles", None)
        bsz, ksz = slot_mask_batch.shape
        slot_targets = torch.zeros(
            (bsz, ksz, self.llm.config.hidden_size),
            device=model_device,
            dtype=inputs_embeds.dtype,
        )
        encoded_any = False
        if isinstance(slot_smiles_2d, list):
            flat_smiles = []
            flat_pos = []
            for bi in range(bsz):
                for si in range(ksz):
                    if not bool(slot_mask_batch[bi, si]):
                        continue
                    smi = ""
                    try:
                        smi = slot_smiles_2d[bi][si]
                    except Exception:
                        smi = ""
                    if isinstance(smi, str) and len(smi) > 0:
                        flat_smiles.append(smi)
                        flat_pos.append((bi, si))
            if len(flat_smiles) > 0:
                try:
                    dictionary = getattr(self.encoder, "unimol_dictionary", None)
                    slot_graph_batch = get_mol_graphs(flat_smiles, dictionary, model_device)
                    _, _, slot_query_out = self.encoder.graph_forward(slot_graph_batch)
                    slot_feats = self.llm_proj(slot_query_out.last_hidden_state).mean(dim=1)  # [N, H]
                    slot_feats = self.stage1_slot_target_proj(slot_feats)
                    for idx, (bi, si) in enumerate(flat_pos):
                        slot_targets[bi, si, :] = slot_feats[idx].to(slot_targets.dtype)
                    encoded_any = True
                except Exception:
                    encoded_any = False

        if not encoded_any:
            slot_ids = batch["latent_slot_input_ids"].to(model_device)  # [B, K, L]
            slot_attn = batch["latent_slot_attention_mask"].to(model_device)  # [B, K, L]
            _, _, slen = slot_ids.shape
            slot_emb = self.llm.get_input_embeddings()(slot_ids.view(bsz * ksz, slen))
            slot_mask = slot_attn.view(bsz * ksz, slen).unsqueeze(-1).to(slot_emb.dtype)
            slot_den = slot_mask.sum(dim=1).clamp(min=1.0)
            slot_pooled = (slot_emb * slot_mask).sum(dim=1) / slot_den
            slot_targets = self.stage1_slot_target_proj(slot_pooled).view(bsz, ksz, -1)

        return {
            "logits": llm_out.logits,
            "lm_loss": llm_out.loss,
            "global_latent_state": global_latent_state,
            "latent_states_all": latent_tokens_all,
            "latent_states": latent_states,
            "slot_targets": slot_targets,
            "wm_preds": wm_preds,
        }

    @torch.no_grad()
    def generate(
        self,
        graph_batch,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=1024,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        if self.use_latent_reasoning:
            if num_return_sequences != 1:
                raise ValueError("Latent reasoning generation currently supports num_return_sequences=1.")
            if graph_batch is not None:
                inputs_embeds = self._build_prefill_inputs_embeds(graph_batch, text_batch)
            else:
                inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids)

            generated = []
            batch_size = inputs_embeds.shape[0]
            for i in range(batch_size):
                sample_attn = text_batch.attention_mask[i]
                sample_valid = torch.nonzero(sample_attn > 0, as_tuple=False).flatten()
                assert sample_valid.numel() > 0, "Empty prompt for generation."
                start = sample_valid[0].item()
                end = sample_valid[-1].item() + 1
                prefix_embeds = inputs_embeds[i, start:end, :]
                latent_tokens = self._rollout_latent_tokens(prefix_embeds)

                gen_input_embeds = torch.cat([prefix_embeds, latent_tokens], dim=0).unsqueeze(0)
                gen_attention = torch.ones((1, gen_input_embeds.shape[1]), device=gen_input_embeds.device, dtype=torch.long)
                out = self.llm.generate(
                    inputs_embeds=gen_input_embeds,
                    attention_mask=gen_attention,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                )
                generated.append(out.squeeze(0))
            return nn.utils.rnn.pad_sequence(generated, batch_first=True, padding_value=pad_token_id or 0)

        if graph_batch is not None:
            _, _, query_output = self.encoder.graph_forward(graph_batch)
            query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

            inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]
            
            inputs_embeds[text_batch.mol_token_flag] = \
                query_output.flatten(0, 1).to(inputs_embeds.dtype) # [batch_size, max_len, dim]
        else:
            inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=text_batch.attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
        )

        return outputs

    @torch.no_grad()
    def generate_with_smiles(
        self,
        smiles_list,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=1024,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        if smiles_list is not []:
            graph_batch = get_mol_graphs(smiles_list, self.encoder.unimol_dictionary, self.device)
        else:
            graph_batch = None
        outputs = self.generate(
            graph_batch=graph_batch,
            text_batch=text_batch,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
        return outputs
    
    def load_from_ckpt(self, ckpt_path):
        print(f"Loading from checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = {k[10:]:v for k,v in ckpt['state_dict'].items() if k.startswith("mol_llama.")}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"
        for k in missing_keys:
            if 'position_ids' in k: continue
            assert k.startswith("encoder.graph_encoder.") or \
                    k.startswith("llm.")
        
    
    def load_from_stage1_ckpt(self, ckpt_path):
        print(f"Loading from stage1 checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = {k[8:]:v for k,v in ckpt['state_dict'].items() if k.startswith("encoder.")}
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
        
        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"
        for k in missing_keys:
            assert k.startswith("graph_encoder.")

    def load_from_hf_dir(self, hf_dir):
        print(f"Loading from HuggingFace directory: {hf_dir}")
        safetensors_path = os.path.join(hf_dir, "model.safetensors")
        bin_path = os.path.join(hf_dir, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found under {hf_dir}"
            )

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        # Allow newly added latent modules to be randomly initialized.
        allowed_missing_prefixes = (
            "latent_proj.",
            "latent_norm.",
            "subregion_proj.",
            "latent_cls_head.",
            "stage1_",
        )
        filtered_missing = [
            k for k in missing_keys
            if not k.startswith("llm.")
            and "position_ids" not in k
            and not k.startswith("encoder.graph_encoder.")
            and not k.startswith(allowed_missing_prefixes)
        ]
        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"
        if len(filtered_missing) > 0:
            raise RuntimeError(f"unexpected missing keys: {filtered_missing[:20]}")
        
def gen_3d_conformation_from_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=8, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=8)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        return None, None
    if mol.GetNumConformers() == 0:
        return None, None

    if num_atoms != mol.GetNumAtoms():
        return None, None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = np.array(mol.GetConformer().GetPositions())
    return atoms, coordinates


def gen_3d_conformation_from_openbabel(smiles):
    mol = pybel.readstring('smi', smiles)
    mol.make3D(forcefield='mmff94', steps=300)
    mol.OBMol.DeleteHydrogens()

    atomic_nums = [atom.atomicnum for atom in mol.atoms]
    pt = Chem.GetPeriodicTable()
    atoms = [pt.GetElementSymbol(atomic_num) for atomic_num in atomic_nums]
    coordinates = np.array([atom.coords for atom in mol.atoms])
    return atoms, coordinates


def gen_3d_conformation_from_libraries(smiles):
    atoms, coordinates = gen_3d_conformation_from_rdkit(smiles)
    if atoms is None or coordinates is None:
        atoms, coordinates = gen_3d_conformation_from_openbabel(smiles)

    return atoms, coordinates


def get_mol_graphs(smiles_list, dictionary, device):
    data_graphs = defaultdict(list)
    for idx, smiles in enumerate(tqdm(smiles_list, desc='Processing Molecules...')):
        atoms, coordinates = gen_3d_conformation_from_libraries(smiles)

        if atoms is None or coordinates is None:
            print(f"Invalid SMILES for {idx}-th SMILES: {smiles}")
            continue

        data_graphs['unimol'].append(
            get_unimol_data(atoms, coordinates, dictionary, remove_Hs=True))

        graph = smiles2graph(smiles)
        data_graphs['moleculestm'].append(Data(x=graph['node_feat'], 
                                        edge_index=graph['edge_index'], 
                                        edge_attr=graph['edge_feat']))

    d3_collater = Mol3DCollater(dictionary.pad())
    graph_batch = {}
    graph_batch['unimol'] = d3_collater(data_graphs['unimol'])
    graph_batch['moleculestm'] = Batch.from_data_list(data_graphs['moleculestm'])

    for key in graph_batch.keys():
        if key == 'unimol':
            for key_ in graph_batch[key].keys():
                graph_batch[key][key_] = graph_batch[key][key_].to(device)
        elif key == 'moleculestm':
            graph_batch[key] = graph_batch[key].to(device)
        
    return graph_batch

def get_mol_graphs_from_data(mol_data_list, dictionary, device):
    data_graphs = defaultdict(list)
    for idx, mol_data in enumerate(tqdm(mol_data_list, desc='Processing Molecules...')):
        data_graphs['unimol'].append(
            get_unimol_data(mol_data['atoms'], np.array(mol_data['coordinates']), dictionary, remove_Hs=True))

        graph = smiles2graph(mol_data['smiles'])
        data_graphs['moleculestm'].append(Data(x=graph['node_feat'], 
                                        edge_index=graph['edge_index'], 
                                        edge_attr=graph['edge_feat']))

    d3_collater = Mol3DCollater(dictionary.pad())
    graph_batch = {}
    graph_batch['unimol'] = d3_collater(data_graphs['unimol'])
    graph_batch['moleculestm'] = Batch.from_data_list(data_graphs['moleculestm'])

    for key in graph_batch.keys():
        if key == 'unimol':
            for key_ in graph_batch[key].keys():
                graph_batch[key][key_] = graph_batch[key][key_].to(device)
        elif key == 'moleculestm':
            graph_batch[key] = graph_batch[key].to(device)
        
    return graph_batch