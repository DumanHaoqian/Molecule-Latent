import argparse
import os
import json
from collections import defaultdict
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data_provider.collaters import Mol3DCollater
from data_provider.instruction_dataset import InstructionDataset
from data_provider.mol_dataset import MolDataset_cid
from data_provider.mol_dataset import smiles2graph, get_unimol_data

from torch_geometric.data import Batch
from torch_geometric.data import Data

from data_provider.tokenization_utils import batch_tokenize_messages_list
from datasets import load_dataset

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel



class Stage2Collater:
    def __init__(self, tokenizer, llama_version, pad_idx, encoder_types):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)

    def __call__(self, batch):
        data_graphs, messages_list, other_infos = zip(*batch)

        graph_batch = {}
        if 'unimol' in self.encoder_types:
            data_unimol = []
            for data in data_graphs:
                data_unimol.extend(data['unimol'])
            graph_batch['unimol'] = self.d3_collater(data_unimol)
        if 'moleculestm' in self.encoder_types:
            data_moleculestm = []
            for data in data_graphs:
                data_moleculestm.extend(data['moleculestm'])
            graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)

        tokenized = batch_tokenize_messages_list(messages_list, self.tokenizer, 
                                                self.llama_version, padding_side='left')
        text_batch = tokenized

        other_infos_ = defaultdict(list)
        for key in other_infos[0].keys():
            for info in other_infos:
                other_infos_[key].append(info[key])

        return graph_batch, text_batch, other_infos_


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
    except Exception:
        return None, None

    if mol.GetNumConformers() == 0:
        return None, None
    if num_atoms != mol.GetNumAtoms():
        return None, None
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = np.array(mol.GetConformer().GetPositions())
    return atoms, coordinates


def _gen_3d_conformation_from_openbabel(smiles):
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D(forcefield="mmff94", steps=300)
        mol.OBMol.DeleteHydrogens()
        atomic_nums = [atom.atomicnum for atom in mol.atoms]
        pt = Chem.GetPeriodicTable()
        atoms = [pt.GetElementSymbol(num) for num in atomic_nums]
        coordinates = np.array([atom.coords for atom in mol.atoms])
        return atoms, coordinates
    except Exception:
        return None, None


def _gen_3d_conformation(smiles):
    atoms, coordinates = _gen_3d_conformation_from_rdkit(smiles)
    if atoms is None or coordinates is None:
        atoms, coordinates = _gen_3d_conformation_from_openbabel(smiles)
    return atoms, coordinates


class LatentReasoningDataset:
    def __init__(self, json_path, unimol_dictionary, encoder_types):
        self.samples = load_dataset("json", data_files=[json_path])["train"]
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"

    def __len__(self):
        return len(self.samples)

    def _build_graphs(self, smiles):
        data_graph = defaultdict(list)
        if "unimol" in self.encoder_types:
            atoms, coordinates = _gen_3d_conformation(smiles)
            if atoms is None or coordinates is None:
                raise ValueError(f"Cannot generate 3D conformation for SMILES: {smiles}")
            data_graph["unimol"].append(
                get_unimol_data(np.array(atoms), np.array(coordinates), self.unimol_dictionary, remove_Hs=True)
            )

        if "moleculestm" in self.encoder_types:
            graph = smiles2graph(smiles)
            data_graph["moleculestm"].append(
                Data(x=graph["node_feat"], edge_index=graph["edge_index"], edge_attr=graph["edge_feat"])
            )
        return data_graph

    def __getitem__(self, index):
        sample = self.samples[index]
        smiles = sample["smiles"]
        data_graphs = self._build_graphs(smiles)

        system = sample.get("system", "You are a molecular reasoning assistant.")
        user = sample.get("query", "Analyze this molecule.")
        if "<mol>" in user:
            user = user.replace("<mol>", self.mol_prompt)
        else:
            # Ensure molecule placeholders always exist for encoder feature injection.
            user = f"{self.mol_prompt}\n{user}"
        rationale = sample.get("rationale_text", "")
        answer_text = sample.get("answer_text", str(sample.get("label", "")))

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"{rationale}\n\nAnswer: {answer_text}".strip()},
        ]

        candidate_raw = sample.get("candidate_subgraphs", [])
        candidate_subgraphs = []
        for item in candidate_raw:
            if isinstance(item, dict):
                smi = item.get("smiles", "")
                if isinstance(smi, str) and len(smi) > 0:
                    candidate_subgraphs.append(smi)
            elif isinstance(item, str) and len(item) > 0:
                candidate_subgraphs.append(item)

        latent_raw = sample.get("latent_targets", [])
        latent_targets = []
        for item in latent_raw:
            if isinstance(item, dict):
                latent_targets.append(int(item.get("candidate_id", -1)))
            else:
                latent_targets.append(int(item))

        other_info = {
            "id": sample.get("id", index),
            "task_type": sample.get("task", "latent_reasoning"),
            "candidate_subgraphs": candidate_subgraphs,
            "latent_targets": latent_targets,
            "class_label": sample.get("label", None),
            "smiles": smiles,
        }
        return data_graphs, messages, other_info


class Stage2DM(LightningDataModule):
    def __init__(
            self,
            tokenizer,
            llama_version,
            num_workers,
            batch_size,
            root,
            unimol_dictionary,
            encoder_types,
            data_types,
            train_json_path=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types
        
        if train_json_path is not None and len(str(train_json_path)) > 0:
            print(f"Loading latent reasoning dataset from {train_json_path} ...")
            self.train_dataset = LatentReasoningDataset(
                json_path=train_json_path,
                unimol_dictionary=unimol_dictionary,
                encoder_types=encoder_types,
            )
        else:
            print('Loading molecule data...')
            data_list = json.load(open(root + 'pubchem-molecules.json'))
            mol_dataset = MolDataset_cid(data_list, unimol_dictionary, encoder_types)
            json_paths = [os.path.join(root, f'{data_type}.json') for data_type in data_types]

            self.train_dataset = InstructionDataset(
                json_paths=json_paths,
                mol_dataset = mol_dataset
            )
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=False,
                            drop_last=True,
                            persistent_workers=True,
                            collate_fn=Stage2Collater(self.tokenizer,
                                                    self.llama_version,
                                                    self.unimol_dictionary.pad(),
                                                    self.encoder_types)
                            )
        return loader