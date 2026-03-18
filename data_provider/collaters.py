
from utils import data_utils
from collections import defaultdict

import torch
from torch_geometric.data import Batch

from data_provider.tokenization_utils import batch_tokenize_messages_list

class Mol3DCollater:        
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, samples):
        atom_vec = [_['src_tokens'] for _ in samples]
        edge_type = [_['src_edge_type'] for _ in samples]
        dist = [_['src_distance'] for _ in samples]
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]


        return {'src_tokens': padded_atom_vec,
            'src_edge_type': padded_edge_type,
            'src_distance': padded_dist,
        }


class Stage1UnifiedCollater:
    def __init__(
        self,
        tokenizer,
        llama_version,
        pad_idx,
        encoder_types,
        max_latent_slots=6,
        latent_slot_text_max_len=48,
        text_max_len=512,
        regression_targets=None,
        classification_targets=None,
    ):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.encoder_types = encoder_types
        self.max_latent_slots = max_latent_slots
        self.latent_slot_text_max_len = latent_slot_text_max_len
        self.text_max_len = text_max_len
        self.reg_keys = list(regression_targets or [
            "molecular_weight", "logp", "tpsa", "hbd", "hba", "num_rings", "aromatic_ring_count", "qed"
        ])
        self.cls_keys = list(classification_targets or ["ro5_pass", "ro5_violation_count"])
        if "unimol" in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)

    def _slot_descriptor(self, slot):
        value = slot.get("value") if isinstance(slot, dict) else {}
        if not isinstance(value, dict):
            value = {}
        return (
            f"name={value.get('name', '')}; "
            f"type={value.get('type', '')}; "
            f"smiles={value.get('smiles', '')}; "
            f"subtype={value.get('motif_subtype', '')}"
        ).strip()

    def __call__(self, batch):
        data_graphs, messages_list, other_infos = zip(*batch)
        graph_batch = {}
        if "unimol" in self.encoder_types:
            data_unimol = []
            for data in data_graphs:
                data_unimol.extend(data["unimol"])
            graph_batch["unimol"] = self.d3_collater(data_unimol)
        if "moleculestm" in self.encoder_types:
            data_moleculestm = []
            for data in data_graphs:
                data_moleculestm.extend(data["moleculestm"])
            graph_batch["moleculestm"] = Batch.from_data_list(data_moleculestm)

        text_batch = batch_tokenize_messages_list(messages_list, self.tokenizer, self.llama_version, padding_side="left")
        if self.text_max_len is not None and self.text_max_len > 0 and text_batch.input_ids.shape[1] > self.text_max_len:
            text_batch.input_ids = text_batch.input_ids[:, -self.text_max_len:]
            text_batch.attention_mask = text_batch.attention_mask[:, -self.text_max_len:]
            text_batch.labels = text_batch.labels[:, -self.text_max_len:]
            text_batch.mol_token_flag = text_batch.mol_token_flag[:, -self.text_max_len:]

        batch_size = len(other_infos)
        task_types = [info.get("task_type", "unknown") for info in other_infos]
        source_dataset = [info.get("source_dataset", "unknown") for info in other_infos]
        smiles = [info.get("smiles", "") for info in other_infos]
        sample_ids = [info.get("sample_id", None) for info in other_infos]
        # Train sampler should usually keep task-type pure; validation may mix tasks.
        task_type = task_types[0] if len(set(task_types)) == 1 else "mixed"

        # latent slot descriptors -> tokenized targets for latent alignment
        latent_slot_mask = torch.zeros((batch_size, self.max_latent_slots), dtype=torch.bool)
        flat_slot_desc = []
        latent_slot_smiles = [["" for _ in range(self.max_latent_slots)] for _ in range(batch_size)]
        for bi, info in enumerate(other_infos):
            slots = info.get("latent_slots", [])
            if not isinstance(slots, list):
                slots = []
            for si in range(self.max_latent_slots):
                if si < len(slots):
                    latent_slot_mask[bi, si] = True
                    flat_slot_desc.append(self._slot_descriptor(slots[si]))
                    value = slots[si].get("value", {}) if isinstance(slots[si], dict) else {}
                    if isinstance(value, dict):
                        smi = value.get("smiles", "")
                        latent_slot_smiles[bi][si] = smi if isinstance(smi, str) else ""
                else:
                    flat_slot_desc.append("")
        slot_tokens = self.tokenizer(
            flat_slot_desc,
            truncation=True,
            padding="max_length",
            max_length=self.latent_slot_text_max_len,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        latent_slot_input_ids = slot_tokens.input_ids.view(batch_size, self.max_latent_slots, -1)
        latent_slot_attention_mask = slot_tokens.attention_mask.view(batch_size, self.max_latent_slots, -1)

        property_regression_targets = torch.zeros((batch_size, len(self.reg_keys)), dtype=torch.float32)
        property_regression_mask = torch.zeros((batch_size, len(self.reg_keys)), dtype=torch.bool)
        property_classification_targets = torch.zeros((batch_size, len(self.cls_keys)), dtype=torch.float32)
        property_classification_mask = torch.zeros((batch_size, len(self.cls_keys)), dtype=torch.bool)
        for bi, info in enumerate(other_infos):
            reg = info.get("property_regression", {}) or {}
            cls = info.get("property_classification", {}) or {}
            for ki, key in enumerate(self.reg_keys):
                if key in reg and reg[key] is not None:
                    property_regression_targets[bi, ki] = float(reg[key])
                    property_regression_mask[bi, ki] = True
            for ki, key in enumerate(self.cls_keys):
                if key in cls and cls[key] is not None:
                    val = cls[key]
                    if isinstance(val, bool):
                        val = 1.0 if val else 0.0
                    property_classification_targets[bi, ki] = float(val)
                    property_classification_mask[bi, ki] = True

        text_targets = [messages[-1]["content"] if len(messages) > 0 else "" for messages in messages_list]
        has_latent_target = latent_slot_mask.any(dim=1)
        has_property_target = property_regression_mask.any(dim=1) | property_classification_mask.any(dim=1)
        task_label = torch.full((batch_size,), -1, dtype=torch.long)
        task_label_mask = torch.zeros((batch_size,), dtype=torch.bool)
        task_kind = []
        task_label_text = []
        task_regression_value = torch.zeros((batch_size,), dtype=torch.float32)
        task_regression_mask = torch.zeros((batch_size,), dtype=torch.bool)
        task_name = []
        for bi, info in enumerate(other_infos):
            task_obj = info.get("task", {}) or {}
            task_name.append(task_obj.get("task_name"))
            task_kind.append(task_obj.get("task_kind"))
            task_label_text.append(task_obj.get("label_text"))
            label = task_obj.get("label")
            try:
                if label is not None:
                    label_i = int(label)
                    if label_i in (0, 1):
                        task_label[bi] = label_i
                        task_label_mask[bi] = True
            except (TypeError, ValueError):
                pass
            value = task_obj.get("value")
            try:
                if value is not None:
                    task_regression_value[bi] = float(value)
                    task_regression_mask[bi] = True
            except (TypeError, ValueError):
                pass

        return {
            "task_type": task_type,
            "source_dataset": source_dataset,
            "sample_ids": sample_ids,
            "input_ids": text_batch.input_ids,
            "attention_mask": text_batch.attention_mask,
            "labels": text_batch.labels,
            "mol_token_flag": text_batch.mol_token_flag,
            "graph_batch": graph_batch,
            "smiles": smiles,
            "latent_slots": [info.get("latent_slots", []) for info in other_infos],
            "latent_slot_mask": latent_slot_mask,
            "latent_slot_input_ids": latent_slot_input_ids,
            "latent_slot_attention_mask": latent_slot_attention_mask,
            "latent_slot_smiles": latent_slot_smiles,
            "property_regression_targets": property_regression_targets,
            "property_regression_mask": property_regression_mask,
            "property_classification_targets": property_classification_targets,
            "property_classification_mask": property_classification_mask,
            "property_regression_keys": self.reg_keys,
            "property_classification_keys": self.cls_keys,
            "text_targets": text_targets,
            "meta_info": [info.get("meta", {}) for info in other_infos],
            "has_latent_target": has_latent_target,
            "has_property_target": has_property_target,
            "task_label": task_label,
            "task_label_mask": task_label_mask,
            "task_name": task_name,
            "task_kind": task_kind,
            "task_label_text": task_label_text,
            "task_regression_value": task_regression_value,
            "task_regression_mask": task_regression_mask,
        }