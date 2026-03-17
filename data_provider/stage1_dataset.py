import json
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
        jsonl_path,
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
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

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