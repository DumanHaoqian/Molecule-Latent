from typing import Dict

import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Subset

from data_provider.collaters import Stage1UnifiedCollater
from data_provider.stage1_dataset import (
    UnifiedStage1Dataset,
    build_downstream_eval_samples_from_csv,
    build_moleculeqa_eval_samples,
    build_pampa_eval_samples,
)


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
        downstream_tasks_paths=None,
        fallback_raw_paths=None,
        source_sampling_weights=None,
        use_task_tokens: bool = True,
        regression_targets=None,
        classification_targets=None,
        eval_downstream_csv_paths=None,
        eval_sample_per_dataset: int = 200,
        eval_stratified_sampling: bool = True,
        eval_sample_per_class: int = 100,
        eval_seed: int = 42,
        eval_moleculeqa_test_path: str = "",
        eval_moleculeqa_test_mol_path: str = "",
        eval_moleculeqa_sample_size: int = 1000,
        eval_pampa_path: str = "",
        eval_pampa_sample_size: int = 1000,
        enabled_sources=None,
        eval_from_train_holdout: bool = False,
        train_subset_fraction: float = 1.0,
        train_subset_fraction_by_source=None,
        train_subset_seed: int = 42,
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
        self.use_task_tokens = use_task_tokens
        self.regression_targets = list(regression_targets or [])
        self.classification_targets = list(classification_targets or [])
        if isinstance(eval_downstream_csv_paths, str):
            eval_downstream_csv_paths = [eval_downstream_csv_paths]
        self.eval_downstream_csv_paths = list(eval_downstream_csv_paths or [])
        self.eval_sample_per_dataset = int(eval_sample_per_dataset)
        self.eval_stratified_sampling = bool(eval_stratified_sampling)
        self.eval_sample_per_class = int(eval_sample_per_class)
        self.eval_seed = int(eval_seed)
        self.eval_moleculeqa_test_path = eval_moleculeqa_test_path
        self.eval_moleculeqa_test_mol_path = eval_moleculeqa_test_mol_path
        self.eval_moleculeqa_sample_size = int(eval_moleculeqa_sample_size)
        self.eval_pampa_path = eval_pampa_path
        self.eval_pampa_sample_size = int(eval_pampa_sample_size)
        self.enabled_sources = list(enabled_sources or ["pubchem", "conversation", "downstream"])
        self.eval_from_train_holdout = bool(eval_from_train_holdout)
        self.train_subset_fraction = float(train_subset_fraction)
        self.train_subset_fraction_by_source = dict(train_subset_fraction_by_source or {})
        self.train_subset_seed = int(train_subset_seed)

        self.latent_world_modeling_path = latent_world_modeling_path
        self.conversation_sft_path = conversation_sft_path
        if isinstance(downstream_tasks_paths, str):
            downstream_tasks_paths = [downstream_tasks_paths]
        self.downstream_tasks_paths = list(downstream_tasks_paths or [])
        self.fallback_raw_paths = fallback_raw_paths or {}
        self.source_sampling_weights = source_sampling_weights or {
            "pubchem": 0.8,
            "conversation": 0.1,
            "downstream": 0.1,
        }
        self.source_sample_counter = {"pubchem": 0, "conversation": 0, "downstream": 0}

        self.train_dataset = None
        self.train_batch_sampler = None
        self.val_dataset = None

    def _resolve_source_paths(self):
        resolved = {"pubchem": [], "conversation": [], "downstream": []}
        # Primary unified paths
        if self.latent_world_modeling_path:
            resolved["pubchem"] = [self.latent_world_modeling_path]
        if self.conversation_sft_path:
            resolved["conversation"] = [self.conversation_sft_path]
        if self.downstream_tasks_paths:
            resolved["downstream"] = [p for p in self.downstream_tasks_paths if p]

        # Fallback raw paths
        fallback_latent = self.fallback_raw_paths.get("latent_path", "")
        fallback_conversation = self.fallback_raw_paths.get("conversation_path", "")
        fallback_downstream = self.fallback_raw_paths.get("downstream_paths", [])
        if isinstance(fallback_downstream, str):
            fallback_downstream = [fallback_downstream]

        if (not resolved["pubchem"]) or (not all(os.path.exists(p) for p in resolved["pubchem"])):
            if fallback_latent:
                resolved["pubchem"] = [fallback_latent]
        if (not resolved["conversation"]) or (not all(os.path.exists(p) for p in resolved["conversation"])):
            if fallback_conversation:
                resolved["conversation"] = [fallback_conversation]
        if (not resolved["downstream"]) or (not all(os.path.exists(p) for p in resolved["downstream"])):
            resolved["downstream"] = [p for p in fallback_downstream if p]

        for source, paths in resolved.items():
            if source not in self.enabled_sources:
                continue
            if len(paths) == 0:
                raise FileNotFoundError(f"[Stage1DM] no path configured for source '{source}'.")
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"[Stage1DM] missing files for source '{source}': {missing}")
        return resolved

    def setup(self, stage=None):
        if not self.stage1_mixed_training:
            raise ValueError("Legacy stage1 pretrain dataset is disabled. Please enable stage1_mixed_training.")

        resolved_paths = self._resolve_source_paths()
        print(f"[Stage1DM] active paths (pubchem): {resolved_paths['pubchem']}")
        print(f"[Stage1DM] active paths (conversation): {resolved_paths['conversation']}")
        print(f"[Stage1DM] active paths (downstream): {resolved_paths['downstream']}")

        datasets = {}
        if "pubchem" in self.enabled_sources and resolved_paths["pubchem"]:
            datasets["pubchem"] = UnifiedStage1Dataset(
                resolved_paths["pubchem"],
                source_name="PubChemLatent",
                task_type="latent_world_modeling",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=self.use_task_tokens,
            )
        if "conversation" in self.enabled_sources and resolved_paths["conversation"]:
            datasets["conversation"] = UnifiedStage1Dataset(
                resolved_paths["conversation"],
                source_name="ComprehensiveConversation",
                task_type="conversation",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=self.use_task_tokens,
            )
        if "downstream" in self.enabled_sources and resolved_paths["downstream"]:
            datasets["downstream"] = UnifiedStage1Dataset(
                resolved_paths["downstream"],
                source_name="DownstreamTasks",
                task_type="downstream",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=self.use_task_tokens,
            )
        if len(datasets) == 0:
            raise ValueError("No stage1 unified dataset path provided.")

        # Optional train subset control: keep only a fraction of each source.
        # Useful for fast ablations (e.g. 50% training data).
        holdout_samples = []
        if self.train_subset_fraction < 1.0 or len(self.train_subset_fraction_by_source) > 0:
            print(
                f"[Stage1DM] applying train subset: global_fraction={self.train_subset_fraction}, "
                f"per_source={self.train_subset_fraction_by_source}, seed={self.train_subset_seed}"
            )
            g = torch.Generator()
            g.manual_seed(self.train_subset_seed)
            reduced = {}
            for source, ds in datasets.items():
                frac = float(self.train_subset_fraction_by_source.get(source, self.train_subset_fraction))
                frac = max(0.0, min(1.0, frac))
                n_total = len(ds)
                if frac >= 1.0:
                    reduced[source] = ds
                    print(f"[Stage1DM] subset source={source}: keep {n_total}/{n_total} (1.0)")
                    continue
                if n_total <= 1 or frac <= 0.0:
                    keep_n = 1
                else:
                    keep_n = max(1, int(n_total * frac))
                perm = torch.randperm(n_total, generator=g).tolist()
                keep_indices = perm[:keep_n]
                holdout_indices = perm[keep_n:]
                reduced[source] = Subset(ds, keep_indices)
                print(f"[Stage1DM] subset source={source}: keep {keep_n}/{n_total} ({frac:.3f})")
                if self.eval_from_train_holdout and len(holdout_indices) > 0 and hasattr(ds, "samples"):
                    ds_samples = getattr(ds, "samples", [])
                    for hi in holdout_indices:
                        if 0 <= hi < len(ds_samples):
                            holdout_samples.append(ds_samples[hi])
            datasets = reduced

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

        if self.eval_from_train_holdout and len(holdout_samples) > 0:
            print(f"[Stage1DM] building val set from train holdout samples: {len(holdout_samples)}")
            self.val_dataset = UnifiedStage1Dataset(
                data_paths=[],
                source_name="Stage1HoldoutEval",
                task_type="downstream",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=self.use_task_tokens,
                preloaded_samples=holdout_samples,
            )
            print(f"[Stage1DM] holdout eval size: {len(self.val_dataset)}")
        elif len(self.eval_downstream_csv_paths) > 0:
            eval_samples = build_downstream_eval_samples_from_csv(
                self.eval_downstream_csv_paths,
                sample_per_dataset=self.eval_sample_per_dataset,
                stratified_sampling=self.eval_stratified_sampling,
                sample_per_class=self.eval_sample_per_class,
                seed=self.eval_seed,
            )
            if isinstance(self.eval_moleculeqa_test_path, str) and len(self.eval_moleculeqa_test_path) > 0:
                eval_samples.extend(
                    build_moleculeqa_eval_samples(
                        self.eval_moleculeqa_test_path,
                        test_mol_json_path=self.eval_moleculeqa_test_mol_path,
                        sample_size=self.eval_moleculeqa_sample_size,
                        seed=self.eval_seed,
                    )
                )
            if isinstance(self.eval_pampa_path, str) and len(self.eval_pampa_path) > 0:
                eval_samples.extend(
                    build_pampa_eval_samples(
                        self.eval_pampa_path,
                        sample_size=self.eval_pampa_sample_size,
                        seed=self.eval_seed,
                    )
                )
            self.val_dataset = UnifiedStage1Dataset(
                data_paths=[],
                source_name="DownstreamEval",
                task_type="downstream",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=self.use_task_tokens,
                preloaded_samples=eval_samples,
            )
            print(f"[Stage1DM] downstream eval size: {len(self.val_dataset)}")

    def train_dataloader(self):
        collate = Stage1UnifiedCollater(
            tokenizer=self.tokenizer,
            llama_version=self.llama_version,
            pad_idx=self.unimol_dictionary.pad(),
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            latent_slot_text_max_len=self.latent_slot_text_max_len,
            text_max_len=self.text_max_len,
            regression_targets=self.regression_targets,
            classification_targets=self.classification_targets,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        collate = Stage1UnifiedCollater(
            tokenizer=self.tokenizer,
            llama_version=self.llama_version,
            pad_idx=self.unimol_dictionary.pad(),
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            latent_slot_text_max_len=self.latent_slot_text_max_len,
            text_max_len=self.text_max_len,
            regression_targets=self.regression_targets,
            classification_targets=self.classification_targets,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collate,
        )