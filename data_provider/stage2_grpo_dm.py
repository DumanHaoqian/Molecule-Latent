import math
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset, Subset

from data_provider.collaters import Stage1UnifiedCollater
from data_provider.stage1_dataset import UnifiedStage1Dataset, build_moledit_eval_samples_from_json
from data_provider.tokenization_utils import batch_tokenize_messages_list


class SourceFilteredSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def _safe_lower(x) -> str:
    return str(x).strip().lower() if x is not None else ""


def _extract_subtask_from_sample(sample: Dict) -> str:
    if not isinstance(sample, dict):
        return ""
    meta = sample.get("meta") or {}
    if isinstance(meta, dict) and meta.get("subtask"):
        return _safe_lower(meta.get("subtask"))
    targets = sample.get("targets") or {}
    task = targets.get("task") or {}
    if isinstance(task, dict):
        task_spec = task.get("task_spec") or {}
        if isinstance(task_spec, dict) and task_spec.get("subtask"):
            return _safe_lower(task_spec.get("subtask"))
        if task.get("task_kind"):
            return _safe_lower(task.get("task_kind"))
    if sample.get("subtask"):
        return _safe_lower(sample.get("subtask"))
    return ""


def _filter_dataset_by_subtasks(dataset: Dataset, subtasks: Sequence[str]) -> Dataset:
    wanted = {_safe_lower(x) for x in subtasks if str(x).strip()}
    if not wanted:
        return dataset
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, list):
        return dataset
    keep = []
    for i, sample in enumerate(samples):
        st = _extract_subtask_from_sample(sample)
        if st in wanted:
            keep.append(i)
    return SourceFilteredSubset(dataset, keep)


def _fraction_subset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    fraction = float(fraction)
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    if n <= 0:
        return dataset
    keep_n = max(1, int(round(n * max(0.0, fraction))))
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    indices = indices[:keep_n]
    return SourceFilteredSubset(dataset, indices)


class MixedGroupBatchSampler(BatchSampler):
    """
    Sample replay vs rl groups according to replay_ratio / rl_ratio,
    then sample a concrete source within that group using source weights.
    Every batch comes from a single source to keep metadata homogeneous.
    """

    def __init__(
        self,
        source_lengths: Dict[str, int],
        source_offsets: Dict[str, int],
        batch_size: int,
        steps_per_epoch: int,
        replay_sources: Sequence[str],
        rl_sources: Sequence[str],
        replay_ratio: float,
        rl_ratio: float,
        replay_source_weights: Optional[Dict[str, float]] = None,
        rl_source_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.source_lengths = {k: int(v) for k, v in source_lengths.items() if int(v) > 0}
        self.source_offsets = dict(source_offsets)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        self.replay_sources = [s for s in replay_sources if s in self.source_lengths]
        self.rl_sources = [s for s in rl_sources if s in self.source_lengths]
        if len(self.replay_sources) == 0 and len(self.rl_sources) == 0:
            raise RuntimeError("No non-empty sources are available for Stage-II sampling.")
        if len(self.rl_sources) == 0:
            self.rl_sources = list(self.replay_sources)
        if len(self.replay_sources) == 0:
            self.replay_sources = list(self.rl_sources)

        self.replay_ratio = max(0.0, float(replay_ratio))
        self.rl_ratio = max(0.0, float(rl_ratio))
        if self.replay_ratio + self.rl_ratio <= 0:
            self.replay_ratio, self.rl_ratio = 0.2, 0.8

        self.replay_source_weights = dict(replay_source_weights or {})
        self.rl_source_weights = dict(rl_source_weights or {})
        self.world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        self.rank = max(0, int(os.environ.get("RANK", "0")))

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _sample_source(self, rng: random.Random, group: str) -> str:
        names = self.replay_sources if group == "replay" else self.rl_sources
        weights_dict = self.replay_source_weights if group == "replay" else self.rl_source_weights
        weights = [max(0.0, float(weights_dict.get(s, 1.0))) for s in names]
        if sum(weights) <= 0:
            weights = [1.0 for _ in names]
        total = sum(weights)
        p = rng.random() * total
        c = 0.0
        for s, w in zip(names, weights):
            c += w
            if p <= c:
                return s
        return names[-1]

    def __iter__(self):
        # Different ranks use different RNG streams to avoid identical batches in DDP
        # when `use_distributed_sampler=False`.
        rng = random.Random(self.seed + self.epoch + 100003 * self.rank)
        per_source_order = {}
        per_source_ptr = {}
        for source, n in self.source_lengths.items():
            order = list(range(n))
            rng.shuffle(order)
            per_source_order[source] = order
            per_source_ptr[source] = 0

        total_group = self.replay_ratio + self.rl_ratio
        replay_p = self.replay_ratio / total_group

        for _ in range(self.steps_per_epoch):
            group = "replay" if rng.random() < replay_p else "rl"
            source = self._sample_source(rng, group)
            batch = []
            for _ in range(self.batch_size):
                if per_source_ptr[source] >= len(per_source_order[source]):
                    order = list(range(self.source_lengths[source]))
                    rng.shuffle(order)
                    per_source_order[source] = order
                    per_source_ptr[source] = 0
                local_idx = per_source_order[source][per_source_ptr[source]]
                per_source_ptr[source] += 1
                batch.append(self.source_offsets[source] + local_idx)
            yield batch


class Stage2GRPOCollater:
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
        _, messages_list, other_infos = zip(*batch)

        prompt_messages = []
        for messages in messages_list:
            if len(messages) >= 2:
                prompt_messages.append(messages[:-1])
            else:
                prompt_messages.append(list(messages))

        prompt_batch = batch_tokenize_messages_list(
            prompt_messages,
            self.tokenizer,
            self.llama_version,
            padding_side="left",
        )
        if self.text_max_len is not None and self.text_max_len > 0 and prompt_batch.input_ids.shape[1] > self.text_max_len:
            prompt_batch.input_ids = prompt_batch.input_ids[:, -self.text_max_len:]
            prompt_batch.attention_mask = prompt_batch.attention_mask[:, -self.text_max_len:]
            prompt_batch.labels = prompt_batch.labels[:, -self.text_max_len:]
            prompt_batch.mol_token_flag = prompt_batch.mol_token_flag[:, -self.text_max_len:]

        source_name = full_batch["source_dataset"][0] if len(full_batch.get("source_dataset", [])) > 0 else "unknown"
        train_mode = "replay" if source_name in self.replay_source_names else "rl"

        subtask_list = []
        task_spec_list = []
        meta_list = []
        source_smiles = []
        target_smiles = []
        added_group = []
        removed_group = []
        edit_type = []
        for oi in other_infos:
            meta = oi.get("meta", {}) if isinstance(oi, dict) else {}
            task_obj = oi.get("task", {}) if isinstance(oi, dict) else {}
            task_spec = {}
            if isinstance(task_obj, dict):
                task_spec = task_obj.get("task_spec", {}) or {}
            if (not task_spec) and isinstance(meta, dict):
                task_spec = meta.get("task_spec", {}) or {}
            subtask = None
            if isinstance(meta, dict):
                subtask = meta.get("subtask")
            if not subtask and isinstance(task_spec, dict):
                subtask = task_spec.get("subtask")
            subtask_list.append(subtask)
            task_spec_list.append(task_spec if isinstance(task_spec, dict) else {})
            meta_list.append(meta if isinstance(meta, dict) else {})
            source_smiles.append(meta.get("source_smiles") if isinstance(meta, dict) else None)
            target_smiles.append(meta.get("target_smiles") if isinstance(meta, dict) else None)
            added_group.append(meta.get("added_group") if isinstance(meta, dict) else None)
            removed_group.append(meta.get("removed_group") if isinstance(meta, dict) else None)
            edit_type.append(meta.get("edit_type") if isinstance(meta, dict) else None)

        full_batch.update(
            {
                "prompt_input_ids": prompt_batch.input_ids,
                "prompt_attention_mask": prompt_batch.attention_mask,
                "prompt_mol_token_flag": prompt_batch.mol_token_flag,
                "messages_list": list(messages_list),
                "train_mode": train_mode,
                "subtask_list": subtask_list,
                "task_spec_list": task_spec_list,
                "meta_list": meta_list,
                "source_smiles_list": source_smiles,
                "target_smiles_list": target_smiles,
                "added_group_list": added_group,
                "removed_group_list": removed_group,
                "edit_type_list": edit_type,
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
        stage2_path: str = "",
        eval_moledit_test_paths=None,
        eval_moledit_sample_per_task: int = 0,
        moledit_val_path: str = "",
        fallback_raw_paths=None,
        enabled_sources=None,
        replay_sources=None,
        rl_sources=None,
        replay_ratio: float = 0.2,
        rl_ratio: float = 0.8,
        replay_source_weights=None,
        rl_source_weights=None,
        total_data_fraction: float = 1.0,
        total_data_fraction_by_source=None,
        stage2_data_fraction: Optional[float] = None,
        split_seed: int = 42,
        stage2_subtasks=None,
        val_subtasks=None,
        regression_targets=None,
        classification_targets=None,
        steps_per_epoch: int = 0,
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
        self.stage2_path = stage2_path
        if isinstance(eval_moledit_test_paths, str):
            eval_moledit_test_paths = [eval_moledit_test_paths]
        self.eval_moledit_test_paths = list(eval_moledit_test_paths or [])
        self.eval_moledit_sample_per_task = int(eval_moledit_sample_per_task)
        self.moledit_val_path = moledit_val_path
        self.fallback_raw_paths = fallback_raw_paths or {}

        self.enabled_sources = list(enabled_sources or ["pubchem", "conversation", "moledit", "stage2"])
        self.replay_sources = list(replay_sources or ["pubchem", "conversation", "moledit"])
        self.rl_sources = list(rl_sources or ["stage2"])
        self.replay_ratio = float(replay_ratio)
        self.rl_ratio = float(rl_ratio)
        self.replay_source_weights = dict(replay_source_weights or {})
        self.rl_source_weights = dict(rl_source_weights or {})

        self.total_data_fraction = float(total_data_fraction)
        self.total_data_fraction_by_source = dict(total_data_fraction_by_source or {})
        if stage2_data_fraction is not None:
            self.total_data_fraction_by_source["stage2"] = float(stage2_data_fraction)
        self.split_seed = int(split_seed)
        self.stage2_subtasks = list(stage2_subtasks or [])
        self.val_subtasks = list(val_subtasks or [])
        self.regression_targets = list(regression_targets or [])
        self.classification_targets = list(classification_targets or [])
        self.steps_per_epoch = int(steps_per_epoch)

        self.train_dataset = None
        self.val_dataset = None
        self.train_batch_sampler = None

    def _resolve_paths(self):
        resolved = {
            "pubchem": self.latent_world_modeling_path,
            "conversation": self.conversation_sft_path,
            "moledit": self.moledit_path,
            "stage2": self.stage2_path,
            "val_moledit": self.moledit_val_path,
        }
        for key, fallback_key in [
            ("pubchem", "latent_path"),
            ("conversation", "conversation_path"),
            ("moledit", "moledit_path"),
            ("stage2", "stage2_path"),
            ("val_moledit", "moledit_val_path"),
        ]:
            if (not resolved[key]) and self.fallback_raw_paths.get(fallback_key):
                resolved[key] = self.fallback_raw_paths.get(fallback_key)
        return resolved

    def _build_unified_dataset(self, path: str, source_name: str) -> UnifiedStage1Dataset:
        return UnifiedStage1Dataset(
            [path],
            source_name=source_name,
            task_type="latent_world_modeling",
            unimol_dictionary=self.unimol_dictionary,
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            use_task_tokens=True,
        )

    def setup(self, stage=None):
        resolved = self._resolve_paths()
        datasets = {}

        if "pubchem" in self.enabled_sources and resolved.get("pubchem"):
            ds = self._build_unified_dataset(resolved["pubchem"], "PubChemLatent")
            frac = float(self.total_data_fraction_by_source.get("pubchem", self.total_data_fraction))
            ds = _fraction_subset(ds, frac, self.split_seed + 11)
            datasets["pubchem"] = ds

        if "conversation" in self.enabled_sources and resolved.get("conversation"):
            ds = self._build_unified_dataset(resolved["conversation"], "ComprehensiveConversation")
            frac = float(self.total_data_fraction_by_source.get("conversation", self.total_data_fraction))
            ds = _fraction_subset(ds, frac, self.split_seed + 23)
            datasets["conversation"] = ds

        if "moledit" in self.enabled_sources and resolved.get("moledit"):
            ds = self._build_unified_dataset(resolved["moledit"], "MolEditLatent")
            frac = float(self.total_data_fraction_by_source.get("moledit", self.total_data_fraction))
            ds = _fraction_subset(ds, frac, self.split_seed + 31)
            datasets["moledit"] = ds

        if "stage2" in self.enabled_sources and resolved.get("stage2"):
            ds = self._build_unified_dataset(resolved["stage2"], "OpenMolInsStage2")
            ds = _filter_dataset_by_subtasks(ds, self.stage2_subtasks)
            frac = float(self.total_data_fraction_by_source.get("stage2", self.total_data_fraction))
            ds = _fraction_subset(ds, frac, self.split_seed + 37)
            datasets["stage2"] = ds

        if len(datasets) == 0:
            raise RuntimeError("Stage2GRPODM found no training dataset. Please check data paths and enabled_sources.")

        concat_parts = []
        source_offsets = {}
        source_lengths = {}
        cursor = 0
        for name, ds in datasets.items():
            concat_parts.append(ds)
            source_offsets[name] = cursor
            source_lengths[name] = len(ds)
            cursor += len(ds)
        self.train_dataset = ConcatDataset(concat_parts)

        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch <= 0:
            total_len = sum(source_lengths.values())
            steps_per_epoch = max(1, math.ceil(total_len / max(1, self.batch_size)))

        self.train_batch_sampler = MixedGroupBatchSampler(
            source_lengths=source_lengths,
            source_offsets=source_offsets,
            batch_size=self.batch_size,
            steps_per_epoch=steps_per_epoch,
            replay_sources=self.replay_sources,
            rl_sources=self.rl_sources,
            replay_ratio=self.replay_ratio,
            rl_ratio=self.rl_ratio,
            replay_source_weights=self.replay_source_weights,
            rl_source_weights=self.rl_source_weights,
            seed=self.split_seed,
        )

        self.val_dataset = None
        if len(self.eval_moledit_test_paths) > 0:
            eval_samples = build_moledit_eval_samples_from_json(
                self.eval_moledit_test_paths,
                sample_per_task=self.eval_moledit_sample_per_task,
                seed=self.split_seed,
            )
            self.val_dataset = UnifiedStage1Dataset(
                data_paths=[],
                source_name="MolEditEval",
                task_type="downstream",
                unimol_dictionary=self.unimol_dictionary,
                encoder_types=self.encoder_types,
                max_latent_slots=self.max_latent_slots,
                use_task_tokens=True,
                preloaded_samples=eval_samples,
            )
        elif resolved.get("val_moledit") and os.path.exists(resolved["val_moledit"]):
            val_ds = self._build_unified_dataset(resolved["val_moledit"], "MolEditVal")
            val_ds = _filter_dataset_by_subtasks(val_ds, self.val_subtasks)
            self.val_dataset = val_ds

    def train_dataloader(self):
        collater = Stage2GRPOCollater(
            tokenizer=self.tokenizer,
            llama_version=self.llama_version,
            pad_idx=self.tokenizer.pad_token_id,
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            latent_slot_text_max_len=self.latent_slot_text_max_len,
            text_max_len=self.text_max_len,
            regression_targets=self.regression_targets,
            classification_targets=self.classification_targets,
            replay_source_names=["PubChemLatent", "ComprehensiveConversation", "MolEditLatent"],
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collater,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        collater = Stage2GRPOCollater(
            tokenizer=self.tokenizer,
            llama_version=self.llama_version,
            pad_idx=self.tokenizer.pad_token_id,
            encoder_types=self.encoder_types,
            max_latent_slots=self.max_latent_slots,
            latent_slot_text_max_len=self.latent_slot_text_max_len,
            text_max_len=self.text_max_len,
            regression_targets=self.regression_targets,
            classification_targets=self.classification_targets,
            replay_source_names=["MolEditVal"],
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collater,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
