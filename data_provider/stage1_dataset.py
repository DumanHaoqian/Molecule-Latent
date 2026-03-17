import json
import os
from collections import defaultdict

import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_provider.mol_dataset import get_unimol_data, smiles2graph


def _gen_3d_conformation_from_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        num_atoms = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol, numConfs=1, numThreads=8, pruneRmsThresh=1, maxAttempts=2000, useRandomCoords=False
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=8)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0 or mol.GetNumAtoms() != num_atoms:
            return None, None
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = np.array(mol.GetConformer().GetPositions())
        return atoms, coordinates
    except Exception:
        return None, None


def _gen_3d_conformation_from_openbabel(smiles):
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D(forcefield="mmff94", steps=300)
        mol.OBMol.DeleteHydrogens()
        pt = Chem.GetPeriodicTable()
        atoms = [pt.GetElementSymbol(atom.atomicnum) for atom in mol.atoms]
        coordinates = np.array([atom.coords for atom in mol.atoms])
        return atoms, coordinates
    except Exception:
        return None, None


def _gen_3d_conformation(smiles):
    atoms, coordinates = _gen_3d_conformation_from_rdkit(smiles)
    if atoms is None or coordinates is None:
        atoms, coordinates = _gen_3d_conformation_from_openbabel(smiles)
    return atoms, coordinates


class UnifiedStage1Dataset(Dataset):
    def __init__(
        self,
        data_paths,
        source_name,
        task_type,
        unimol_dictionary,
        encoder_types,
        max_latent_slots=6,
        max_atoms=512,
        use_task_tokens=True,
    ):
        super().__init__()
        self.source_name = source_name
        self.task_type = task_type
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types
        self.max_latent_slots = max_latent_slots
        self.max_atoms = max_atoms
        self.use_task_tokens = use_task_tokens
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
        self.default_system = "You are a helpful assistant specializing in chemistry and biology."
        self.samples = []
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = [p for p in (data_paths or []) if isinstance(p, str) and len(p) > 0]
        self._load_all_samples()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"UnifiedStage1Dataset(source={self.source_name}, task={self.task_type}) loaded 0 samples from {self.data_paths}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_all_samples(self):
        for path in self.data_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".json":
                self._load_json_file(path)
            else:
                self._load_jsonl_file(path)

    def _load_jsonl_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sample = self._normalize_sample(obj)
                if sample is not None:
                    self.samples.append(sample)

    def _load_json_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = data.get("data", [])
        else:
            iterable = []
        for obj in iterable:
            sample = self._normalize_sample(obj)
            if sample is not None:
                self.samples.append(sample)

    def _normalize_sample(self, obj):
        if not isinstance(obj, dict):
            return None
        if "task_type" in obj and "targets" in obj and "molecule" in obj:
            return obj
        if self.task_type == "latent_world_modeling":
            return self._normalize_latent_raw(obj)
        if self.task_type == "conversation":
            return self._normalize_conversation_raw(obj)
        if self.task_type == "downstream":
            return self._normalize_downstream_raw(obj)
        return None

    def _normalize_latent_raw(self, obj):
        features = obj.get("molecule_features") or {}
        candidate_subgraphs = obj.get("candidate_subgraphs") or []
        latent_slots = []
        for idx, slot in enumerate(candidate_subgraphs):
            if not isinstance(slot, dict):
                continue
            latent_slots.append(
                {
                    "slot_name": f"slot_{idx + 1}",
                    "target_type": "subgraph",
                    "value": {
                        "name": slot.get("name"),
                        "type": slot.get("type"),
                        "smiles": slot.get("smiles"),
                        "atom_ids": slot.get("atom_ids"),
                        "motif_subtype": slot.get("motif_subtype"),
                    },
                }
            )
            if len(latent_slots) >= self.max_latent_slots:
                break
        return {
            "sample_id": obj.get("id"),
            "source_dataset": "PubChemLatent",
            "split": obj.get("split", "train"),
            "task_type": "latent_world_modeling",
            "stage_tags": ["stage1"],
            "molecule": {
                "smiles": obj.get("smiles"),
                "canonical_smiles": obj.get("smiles"),
                "iupac_name": obj.get("iupac_name"),
            },
            "input": {
                "system_prompt": self.default_system,
                "instruction": "Describe the molecular structure, key subgraphs, and physicochemical properties of the given molecule.",
                "conversation_history": [],
            },
            "targets": {
                "text": obj.get("enriched_description", ""),
                "latent": {"slots": latent_slots},
                "properties": {
                    "regression": {
                        "molecular_weight": features.get("molecular_weight"),
                        "logp": features.get("logp"),
                        "tpsa": features.get("tpsa"),
                        "hbd": features.get("hbd"),
                        "hba": features.get("hba"),
                        "num_rings": features.get("num_rings"),
                        "aromatic_ring_count": features.get("aromatic_ring_count"),
                        "qed": features.get("qed"),
                    },
                    "classification": {
                        "ro5_pass": features.get("ro5_pass"),
                        "ro5_violation_count": features.get("ro5_violation_count"),
                    },
                },
                "task": {"task_name": None, "task_kind": None, "label": None, "label_text": None},
            },
            "quality": {
                "quality_score": ((obj.get("qa_report") or {}).get("total_score")),
                "quality_label": ((obj.get("qa_report") or {}).get("quality_label")),
            },
            "meta": {
                "has_latent_target": len(latent_slots) > 0,
                "has_text_target": bool(obj.get("enriched_description")),
                "has_property_target": True,
                "has_task_label": False,
            },
        }

    def _normalize_conversation_raw(self, obj):
        convs = obj.get("conversations") or []
        history = []
        instruction = obj.get("instruction", "")
        target_text = obj.get("output", "")
        if isinstance(convs, list) and len(convs) > 0:
            turns = [t for t in convs if isinstance(t, dict) and isinstance(t.get("user"), str) and isinstance(t.get("assistant"), str)]
            if len(turns) > 0:
                instruction = turns[-1]["user"]
                target_text = turns[-1]["assistant"]
                for t in turns[:-1]:
                    history.append({"role": "user", "content": t["user"]})
                    history.append({"role": "assistant", "content": t["assistant"]})
        return {
            "sample_id": obj.get("id") or obj.get("cid"),
            "source_dataset": "ComprehensiveConversation",
            "split": obj.get("split", "train"),
            "task_type": "conversation",
            "stage_tags": ["stage1", "stage2"],
            "molecule": {
                "smiles": obj.get("smiles"),
                "canonical_smiles": obj.get("smiles"),
                "iupac_name": obj.get("iupac_name"),
            },
            "input": {
                "system_prompt": obj.get("system") or self.default_system,
                "instruction": instruction,
                "conversation_history": history,
            },
            "targets": {
                "text": target_text,
                "latent": {"slots": []},
                "properties": {"regression": {}, "classification": {}},
                "task": {"task_name": None, "task_kind": None, "label": None, "label_text": None},
            },
            "quality": {"quality_score": None, "quality_label": "unknown"},
            "meta": {
                "has_latent_target": False,
                "has_text_target": bool(target_text),
                "has_property_target": False,
                "has_task_label": False,
            },
        }

    def _normalize_downstream_raw(self, obj):
        features = obj.get("molecule_features") or {}
        label = obj.get("label")
        label_text = None
        try:
            if label is not None:
                label_i = int(label)
                if label_i in (0, 1):
                    label_text = "Yes" if label_i == 1 else "No"
                    label = label_i
        except (TypeError, ValueError):
            pass
        return {
            "sample_id": obj.get("id"),
            "source_dataset": obj.get("dataset", "DownstreamTasks"),
            "split": obj.get("split", "train"),
            "task_type": "downstream",
            "stage_tags": ["stage1", "stage3"],
            "molecule": {
                "smiles": obj.get("smiles"),
                "canonical_smiles": obj.get("smiles"),
                "iupac_name": obj.get("iupac_name"),
            },
            "input": {
                "system_prompt": self.default_system,
                "instruction": "Analyze the molecule and answer the downstream task.",
                "conversation_history": [],
            },
            "targets": {
                "text": obj.get("enriched_description") or label_text or "",
                "latent": {"slots": []},
                "properties": {
                    "regression": {
                        "molecular_weight": features.get("molecular_weight"),
                        "logp": features.get("logp"),
                        "tpsa": features.get("tpsa"),
                        "hbd": features.get("hbd"),
                        "hba": features.get("hba"),
                        "num_rings": features.get("num_rings"),
                        "aromatic_ring_count": features.get("aromatic_ring_count"),
                        "qed": features.get("qed"),
                    },
                    "classification": {
                        "ro5_pass": features.get("ro5_pass"),
                        "ro5_violation_count": features.get("ro5_violation_count"),
                    },
                },
                "task": {
                    "task_name": obj.get("dataset"),
                    "task_kind": "binary_classification",
                    "label": label,
                    "label_text": label_text,
                },
            },
            "quality": {"quality_score": ((obj.get("qa_report") or {}).get("total_score")), "quality_label": ((obj.get("qa_report") or {}).get("quality_label"))},
            "meta": {
                "has_latent_target": False,
                "has_text_target": bool(obj.get("enriched_description")) or label_text is not None,
                "has_property_target": True,
                "has_task_label": label in (0, 1),
            },
        }

    def _build_graphs(self, smiles):
        data_graph = defaultdict(list)
        if "unimol" in self.encoder_types:
            atoms, coordinates = _gen_3d_conformation(smiles)
            if atoms is None or coordinates is None:
                raise ValueError(f"Cannot generate 3D conformation for SMILES: {smiles}")
            data_graph["unimol"].append(
                get_unimol_data(np.array(atoms), np.array(coordinates), self.unimol_dictionary, self.max_atoms, remove_Hs=True)
            )
        if "moleculestm" in self.encoder_types:
            graph = smiles2graph(smiles)
            data_graph["moleculestm"].append(
                Data(x=graph["node_feat"], edge_index=graph["edge_index"], edge_attr=graph["edge_feat"])
            )
        return data_graph

    def _task_token(self, sample):
        if not self.use_task_tokens:
            return ""
        if self.task_type == "latent_world_modeling":
            return "[TASK:LATENT_WORLD_MODELING]"
        if self.task_type == "conversation":
            return "[TASK:CONVERSATION]"
        task_name = (((sample.get("targets") or {}).get("task") or {}).get("task_name") or "UNKNOWN")
        return f"[TASK:DOWNSTREAM_{str(task_name).upper()}]"

    def _build_messages(self, sample):
        inp = sample.get("input") or {}
        tgt = sample.get("targets") or {}
        system_prompt = inp.get("system_prompt") or self.default_system
        instruction = inp.get("instruction") or ""
        history = inp.get("conversation_history") or []
        text_target = tgt.get("text")
        if not isinstance(text_target, str):
            text_target = ""
        if self.task_type == "downstream" and text_target.strip() == "":
            task = tgt.get("task") or {}
            label_text = task.get("label_text")
            label = task.get("label")
            if isinstance(label_text, str) and len(label_text) > 0:
                text_target = label_text
            elif label in (0, 1):
                text_target = "Yes" if int(label) == 1 else "No"

        task_token = self._task_token(sample)
        user_content = f"{task_token}\n{self.mol_prompt}\n{instruction}".strip()
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and len(content) > 0:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": text_target})
        return messages

    def _extract_other_info(self, sample, smiles):
        targets = sample.get("targets") or {}
        latent_slots = ((targets.get("latent") or {}).get("slots") or [])
        property_reg = (targets.get("properties") or {}).get("regression") or {}
        property_cls = (targets.get("properties") or {}).get("classification") or {}
        task_obj = targets.get("task") or {}
        quality_obj = sample.get("quality") or {}
        return {
            "sample_id": sample.get("sample_id"),
            "source_dataset": sample.get("source_dataset", self.source_name),
            "task_type": sample.get("task_type", self.task_type),
            "smiles": smiles,
            "latent_slots": latent_slots[: self.max_latent_slots],
            "property_regression": property_reg,
            "property_classification": property_cls,
            "task": task_obj,
            "quality": quality_obj,
            "meta": sample.get("meta") or {},
        }

    def __getitem__(self, index):
        # Retry a few times when a sample has invalid SMILES / geometry.
        for shift in range(min(8, len(self.samples))):
            idx = (index + shift) % len(self.samples)
            sample = self.samples[idx]
            molecule = sample.get("molecule") or {}
            smiles = molecule.get("canonical_smiles") or molecule.get("smiles")
            if not isinstance(smiles, str) or len(smiles) == 0:
                continue
            try:
                data_graph = self._build_graphs(smiles)
            except Exception:
                continue
            messages = self._build_messages(sample)
            other_info = self._extract_other_info(sample, smiles)
            return data_graph, messages, other_info
        raise RuntimeError("Unable to fetch a valid sample after retries.")