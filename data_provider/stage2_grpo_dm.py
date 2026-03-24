from typing import Dict, List

import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Subset

from data_provider.collaters import Stage1UnifiedCollater
from data_provider.stage1_dataset import UnifiedStage1Dataset
from data_provider.tokenization_utils import batch_tokenize_messages_list


class WeightedSourceBatchSampler(BatchSampler):
    def __init__(
        self,
        source_lengths: Dict[str, int],
        source_offsets: Dict[str, int],
        source_weights: Dict[str, float],
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 42,
    ):
        self.source_lengths = source_lengths
        self.source_offsets = source_offsets
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

        names = list(source_lengths.keys())
        weights = torch.tensor([max(0.0, float(source_weights.get(k, 0.0))) for k in names], dtype=torch.float32)
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        self.source_names = names
        self.source_probs = (weights / weights.sum()).tolist()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        local_indices = {}
        local_ptr = {}
        for source in self.source_names:
            perm = torch.randperm(self.source_lengths[source], generator=g).tolist()
            local_indices[source] = perm
            local_ptr[source] = 0

        for _ in range(self.steps_per_epoch):
            sampled = torch.multinomial(torch.tensor(self.source_probs), 1, generator=g).item()
            source = self.source_names[sampled]
            batch = []
            for _ in range(self.batch_size):
                if local_ptr[source] >= len(local_indices[source]):
                    local_indices[source] = torch.randperm(self.source_lengths[source], generator=g).tolist()
                    local_ptr[source] = 0
                idx_in_source = local_indices[source][local_ptr[source]]
                local_ptr[source] += 1
                batch.append(self.source_offsets[source] + idx_in_source)
            yield batch


class Stage2GRPOCollater:
    """Build both teacher-forced full batches and prompt-only batches for GRPO."""

    def __init__(
        self,
        tokenizer,
        llama_version,
        pad_idx,
        encoder_types,
        max_latent_slots=4,
        latent_slot_text_max_len=48,
        text_max_len=512,
        regression_targets=None,
        classification_targets=None,
        replay_source_names=None,
    ):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.text_max_len = text_max_len
        self.replay_source_names = set(replay_source_names or [])
        self.inner = Stage1UnifiedCollater(
            tokenizer=tokenizer,
            llama_version=llama_version,
            pad_idx=pad_idx,
            encoder_types=encoder_types,
            max_latent_slots=max_latent_slots,
            latent_slot_text_max_len=latent_slot_text_max_len,
            text_max_len=text_max_len,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
        )

    def __call__(self, batch):
        full_batch = self.inner(batch)
        _, messages_list, _ = zip(*batch)
        prompt_messages = []
        for messages in messages_list:
            if len(messages) >= 2:
                prompt_messages.append(messages[:-1])
            else:
                prompt_messages.append(list(messages))
        prompt_batch = batch_tokenize_messages_list(prompt_messages, self.tokenizer, self.llama_version, padding_side="left")
        if self.text_max_len is not None and self.text_max_len > 0 and prompt_batch.input_ids.shape[1] > self.text_max_len:
            prompt_batch.input_ids = prompt_batch.input_ids[:, -self.text_max_len:]
            prompt_batch.attention_mask = prompt_batch.attention_mask[:, -self.text_max_len:]
            prompt_batch.labels = prompt_batch.labels[:, -self.text_max_len:]
            prompt_batch.mol_token_flag = prompt_batch.mol_token_flag[:, -self.text_max_len:]

        source_name = full_batch["source_dataset"][0] if len(full_batch["source_dataset"]) > 0 else "unknown"
        train_mode = "replay" if source_name in self.replay_source_names else "rl"
        full_batch.update(
            {
                "prompt_input_ids": prompt_batch.input_ids,
                "prompt_attention_mask": prompt_batch.attention_mask,
                "prompt_mol_token_flag": prompt_batch.mol_token_flag,
                "messages_list": list(messages_list),
                "train_mode": train_mode,
            }
        )
        return full_batch


class Stage2GRPODM(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        llama_version,
        num_workers: int = 0,
        batch_size: int = 2,
        unimol_dictionary=None,
        encoder_types=None,
        text_max_len: int = 512,
        max_latent_slots: int = 4,
        latent_slot_text_max_len: int = 48,
        latent_world_modeling_path: str = "",
        conversation_sft_path: str = "",
        moledit_path: str = "",
        downstream_tasks_paths=None,
        fallback_raw_paths=None,
        source_sampling_weights=None,
        enabled_sources=None,
        replay_sources=None,
        total_data_fraction: float = 1.0,
        total_data_fraction_by_source=None,
        split_seed: int = 42,
        regression_targets=None,
        classification_targets=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.num_workers = int(num_workers)
        self.batch_size = int(batch_size)
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types or ["unimol", "moleculestm"]
        self.text_max_len = int(text_max_len)
        self.max_latent_slots = int(max_latent_slots)
        self.latent_slot_text_max_len = int(latent_slot_text_max_len)
        self.latent_world_modeling_path = latent_world_modeling_path
        self.conversation_sft_path = conversation_sft_path
        self.moledit_path = moledit_path
        if isinstance(downstream_tasks_paths, str):
            downstream_tasks_paths = [downstream_tasks_paths]
        self.downstream_tasks_paths = list(downstream_tasks_paths or [])
        self.fallback_raw_paths = fallback_raw_paths or {}
        self.source_sampling_weights = dict(source_sampling_weights or {})
        self.enabled_sources = list(enabled_sources or ["pubchem", "conversation", "moledit"])
        self.replay_sources = set(replay_sources or ["pubchem", "conversation"])
        self.total_data_fraction = float(total_data_fraction)
        self.total_data_fraction_by_source = dict(total_data_fraction_by_source or {})
        self.split_seed = int(split_seed)
        self.regression_targets = list(regression_targets or [])
        self.classification_targets = list(classification_targets or [])

        self.train_dataset = None
        self.train_batch_sampler = None

    def _resolve_source_paths(self):
        resolved = {"pubchem": [], "conversation": [], "moledit": [], "downstream": []}
        if self.latent_world_modeling_path:
            resolved["pubchem"] = [self.latent_world_modeling_path]
        if self.conversation_sft_path:
            resolved["conversation"] = [self.conversation_sft_path]
        if self.moledit_path:
            resolved["moledit"] = [self.moledit_path]
        if self.downstream_tasks_paths:
            resolved["downstream"] = [p for p in self.downstream_tasks_paths if p]

        fallback_latent = self.fallback_raw_paths.get("latent_path", "")
        fallback_conversation = self.fallback_raw_paths.get("conversation_path", "")
        fallback_moledit = self.fallback_raw_paths.get("moledit_path", "")
        fallback_downstream = self.fallback_raw_paths.get("downstream_paths", [])
        if isinstance(fallback_downstream, str):
            fallback_downstream = [fallback_downstream]

        if (not resolved["pubchem"]) or (not all(os.path.exists(p) for p in resolved["pubchem"])):
            if fallback_latent:
                resolved["pubchem"] = [fallback_latent]
        if (not resolved["conversation"]) or (not all(os.path.exists(p) for p in resolved["conversation"])):
            if fallback_conversation:
                resolved["conversation"] = [fallback_conversation]
        if (not resolved["moledit"]) or (not all(os.path.exists(p) for p in resolved["moledit"])):
            if fallback_moledit:
                resolved["moledit"] = [fallback_moledit]
        if (not resolved["downstream"]) or (not all(os.path.exists(p) for p in resolved["downstream"])):
            resolved["downstream"] = [p for p in fallback_downstream if p]

        for source, paths in resolved.items():
            if source not in self.enabled_sources:
                continue
            if len(paths) == 0:
                raise FileNotFoundError(f"[Stage2GRPODM] no path configured for source '{source}'")
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"[Stage2GRPODM] missing files for source '{source}': {missing}")
        return resolved

    def setup(self, stage=None):
        resolved_paths = self._resolve_source_paths()
        datasets = {}
        if "pubchem" in self.enabled_sources and resolved_paths["pubchem"]:
            datasets["pubchem"] = UnifiedStage1Dataset(
                resolved_paths["pubchem"],
                source_name="PubChemLatent",
                task_type="latent_world_modeling",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=True,
            )
        if "conversation" in self.enabled_sources and resolved_paths["conversation"]:
            datasets["conversation"] = UnifiedStage1Dataset(
                resolved_paths["conversation"],
                source_name="ComprehensiveConversation",
                task_type="conversation",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=True,
            )
        if "moledit" in self.enabled_sources and resolved_paths["moledit"]:
            datasets["moledit"] = UnifiedStage1Dataset(
                resolved_paths["moledit"],
                source_name="MolEditLatent",
                task_type="latent_world_modeling",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=True,
            )
        if "downstream" in self.enabled_sources and resolved_paths["downstream"]:
            datasets["downstream"] = UnifiedStage1Dataset(
                resolved_paths["downstream"],
                source_name="DownstreamTasks",
                task_type="downstream",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=True,
            )
        if len(datasets) == 0:
            raise ValueError("No Stage-II dataset source was enabled.")

        g = torch.Generator()
        g.manual_seed(self.split_seed)
        reduced = {}
        for source, ds in datasets.items():
            n_total = len(ds)
            frac = float(self.total_data_fraction_by_source.get(source, self.total_data_fraction))
            frac = max(0.0, min(1.0, frac))
            if frac >= 1.0:
                reduced[source] = ds
                continue
            selected_n = max(1, int(n_total * frac))
            perm = torch.randperm(n_total, generator=g).tolist()[:selected_n]
            reduced[source] = Subset(ds, perm)
            print(f"[Stage2GRPODM] source={source} selected={selected_n}/{n_total}")

        ordered_sources = list(reduced.keys())
        concat_parts = [reduced[s] for s in ordered_sources]
        self.train_dataset = ConcatDataset(concat_parts)

        source_lengths = {s: len(reduced[s]) for s in ordered_sources}
        source_offsets = {}
        offset = 0
        for s in ordered_sources:
            source_offsets[s] = offset
            offset += len(reduced[s])

        steps_per_epoch = max(1, sum(source_lengths.values()) // max(1, self.batch_size))
        self.train_batch_sampler = WeightedSourceBatchSampler(
            source_lengths=source_lengths,
            source_offsets=source_offsets,
            source_weights=self.source_sampling_weights,
            batch_size=self.batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=self.split_seed,
        )

    def train_dataloader(self):
        collate_fn = Stage2GRPOCollater(
            tokenizer=self.tokenizer,
            llama_version=self.llama_version,
            pad_idx=self.unimol_dictionary.pad(),
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            latent_slot_text_max_len=self.latent_slot_text_max_len,
            text_max_len=self.text_max_len,
            regression_targets=self.regression_targets,
            classification_targets=self.classification_targets,
            replay_source_names={
                name for name in [
                    "PubChemLatent" if "pubchem" in self.replay_sources else None,
                    "ComprehensiveConversation" if "conversation" in self.replay_sources else None,
                    "MolEditLatent" if "moledit" in self.replay_sources else None,
                    "DownstreamTasks" if "downstream" in self.replay_sources else None,
                ] if name is not None
            },
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )
