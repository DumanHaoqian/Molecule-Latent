from typing import Dict, List

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Sampler

from data_provider.collaters import Stage1UnifiedCollater
from data_provider.stage1_dataset import UnifiedStage1Dataset


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


class Stage1DM(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        llama_version,
        num_workers: int = 0,
        batch_size: int = 8,
        unimol_dictionary=None,
        encoder_types=None,
        text_max_len: int = 512,
        max_latent_slots: int = 6,
        latent_slot_text_max_len: int = 48,
        stage1_mixed_training: bool = True,
        latent_world_modeling_path: str = "",
        conversation_sft_path: str = "",
        downstream_tasks_path: str = "",
        source_sampling_weights=None,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types
        self.text_max_len = text_max_len
        self.max_latent_slots = max_latent_slots
        self.latent_slot_text_max_len = latent_slot_text_max_len
        self.stage1_mixed_training = stage1_mixed_training
        self.seed = seed

        self.latent_world_modeling_path = latent_world_modeling_path
        self.conversation_sft_path = conversation_sft_path
        self.downstream_tasks_path = downstream_tasks_path
        self.source_sampling_weights = source_sampling_weights or {
            "pubchem": 0.8,
            "conversation": 0.1,
            "downstream": 0.1,
        }
        self.source_sample_counter = {"pubchem": 0, "conversation": 0, "downstream": 0}

        self.train_dataset = None
        self.train_batch_sampler = None

    def setup(self, stage=None):
        if not self.stage1_mixed_training:
            raise ValueError("Legacy stage1 pretrain dataset is disabled. Please enable stage1_mixed_training.")

        datasets = {}
        if self.latent_world_modeling_path:
            datasets["pubchem"] = UnifiedStage1Dataset(
                self.latent_world_modeling_path,
                source_name="PubChemLatent",
                task_type="latent_world_modeling",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
            )
        if self.conversation_sft_path:
            datasets["conversation"] = UnifiedStage1Dataset(
                self.conversation_sft_path,
                source_name="ComprehensiveConversation",
                task_type="conversation",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
            )
        if self.downstream_tasks_path:
            datasets["downstream"] = UnifiedStage1Dataset(
                self.downstream_tasks_path,
                source_name="DownstreamTasks",
                task_type="downstream",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
            )
        if len(datasets) == 0:
            raise ValueError("No stage1 unified dataset path provided.")

        self._source_datasets = datasets
        self._source_names = list(datasets.keys())
        concat_parts = [datasets[k] for k in self._source_names]
        self.train_dataset = ConcatDataset(concat_parts)

        source_lengths = {k: len(v) for k, v in datasets.items()}
        source_offsets, cur = {}, 0
        for k in self._source_names:
            source_offsets[k] = cur
            cur += source_lengths[k]
        approx_total_batches = max(1, sum(source_lengths.values()) // max(1, self.batch_size))
        self.train_batch_sampler = WeightedSourceBatchSampler(
            source_lengths=source_lengths,
            source_offsets=source_offsets,
            source_weights=self.source_sampling_weights,
            batch_size=self.batch_size,
            steps_per_epoch=approx_total_batches,
            seed=self.seed,
        )
        print(f"[Stage1DM] source lengths: {source_lengths}")
        print(f"[Stage1DM] source weights: {self.source_sampling_weights}")
        print(f"[Stage1DM] steps_per_epoch: {approx_total_batches}")

    def train_dataloader(self):
        collate = Stage1UnifiedCollater(
            tokenizer=self.tokenizer,
            llama_version=self.llama_version,
            pad_idx=self.unimol_dictionary.pad(),
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            latent_slot_text_max_len=self.latent_slot_text_max_len,
            text_max_len=self.text_max_len,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate,
        )