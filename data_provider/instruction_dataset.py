from torch.utils.data import Dataset
from datasets import load_dataset

class InstructionDataset(Dataset):
    def __init__(self, json_paths, mol_dataset):
        super(InstructionDataset, self).__init__()

        self.instruction_dataset = load_dataset("json", data_files=json_paths)['train']
        self.mol_dataset = mol_dataset
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
    
    def __len__(self):
        return len(self.instruction_dataset)


    def __getitem__(self, index):
        text_data = self.instruction_dataset[index]
    
        cid = text_data['cid']
        data_graphs, data_others = self.mol_dataset[cid]
        num_mols = len(data_graphs[list(data_graphs.keys())[0]])

        messages = []
        messages.append({"role": "system", "content": text_data['system']})
        for turn in text_data['conversations']:
            messages.append({"role": "user", "content": turn['user'].replace('<mol>', self.mol_prompt)})
            messages.append({"role": "assistant", "content": turn['assistant']})

        other_info = {
            "cid": cid,
            "names": [text_data['iupac_name']],
            "task_type": text_data['type'],
            "num_turns": len(text_data['conversations']),
            # Optional fields for decoder-side latent reasoning.
            # Keep defaults so baseline datasets remain fully compatible.
            "candidate_subgraphs": text_data.get("candidate_subgraphs", []),
            "latent_targets": text_data.get("latent_targets", []),
            "class_label": text_data.get("class_label", text_data.get("label", None)),
        }        
        return data_graphs, messages, other_info
