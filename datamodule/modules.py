import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from datamodule.dataset import (
    MoleculeDatasetCancer,
    MoleculeDatasetSSI,
    MoleculeDatasetDDI,
    MoleculeDatasetCellLine,
    MoleculeDatasetTargetCellLine,
    MoleculeDatasetCombined,
)
import re
from torch_geometric.data import Data, Batch

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"
SPLIT_MARKER = f"|"

def _insert_split_marker(m: re.Match):
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph, mode="val"):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list

def cell_handler(text, cell_ph, mode="val"):
    text = re.sub(r'\[START_CELL\]\[END_CELL\]', f'[START_CELL]{cell_ph}[END_CELL]', text)
    return text


def escape_custom_split_sequence(text):
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, texts ,smiles_prompt= zip(*batch)

        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)

        smiles_prompt = [smiles_handler(p, self.mol_ph, mode="train")[0] for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)   
        

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        
        return graphs1, graphs2, smiles_prompt_tokens, text_tokens   

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs1,graphs2, texts ,smiles_prompt = zip(*batch)

        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)

        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        return graphs1,graphs2, smiles_prompt_tokens, texts       

class CancerTrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, texts ,smiles_prompt= zip(*batch)

        valid1 = self.collater([g["Valid"] for g in graphs1])
        if valid1.any():
            data1 = self.collater([g["Graph"] for g in graphs1 if g["Valid"]])
        else:
            data1 = None
        graphs1 = {
            "Valid": valid1,
            "Graph": data1
        }
        valid2 = self.collater([g["Valid"] for g in graphs2])
        if valid2.any():
            data2 = self.collater([g["Graph"] for g in graphs2 if g["Valid"]])
        else:
            data2 = None
        graphs2 = {
            "Valid": valid2,
            "Graph": data2
        }

        smiles_prompt = [smiles_handler(p, self.mol_ph, mode="train")[0] for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)      

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        
        return graphs1, graphs2, smiles_prompt_tokens, text_tokens 

class CancerInferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, texts ,smiles_prompt = zip(*batch)

        valid1 = self.collater([g["Valid"] for g in graphs1])
        if valid1.any():
            data1 = self.collater([g["Graph"] for g in graphs1 if g["Valid"]])
        else:
            data1 = None
        graphs1 = {
            "Valid": valid1,
            "Graph": data1
        }
        valid2 = self.collater([g["Valid"] for g in graphs2])
        if valid2.any():
            data2 = self.collater([g["Graph"] for g in graphs2 if g["Valid"]])
        else:
            data2 = None
        graphs2 = {
            "Valid": valid2,
            "Graph": data2
        }

        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)
        

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        return graphs1,graphs2, smiles_prompt_tokens, texts  
    
def show(s):
    print(s)
    

class CellLineTrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, cell_ph, cell_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.cell_ph = cell_ph
        self.cell_token_id = cell_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, genes, texts, smiles_prompt= zip(*batch)

        valid1 = self.collater([g["Valid"] for g in graphs1])
        if valid1.any():
            data1 = self.collater([g["Graph"] for g in graphs1 if g["Valid"]])
        else:
            data1 = None
        graphs1 = {
            "Valid": valid1,
            "Graph": data1
        }
        valid2 = self.collater([g["Valid"] for g in graphs2])
        if valid2.any():
            data2 = self.collater([g["Graph"] for g in graphs2 if g["Valid"]])
        else:
            data2 = None
        graphs2 = {
            "Valid": valid2,
            "Graph": data2
        }

        smiles_prompt = [smiles_handler(p, self.mol_ph, mode="train")[0] for p in smiles_prompt]
        smiles_prompt = [cell_handler(p, self.cell_ph) for p in smiles_prompt]

        genes = self.collater(genes)
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)      

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        is_cell_token = smiles_prompt_tokens.input_ids == self.cell_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        smiles_prompt_tokens['is_cell_token'] = is_cell_token

        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        
        return graphs1, graphs2, genes, smiles_prompt_tokens, text_tokens 

class CellLineInferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, cell_ph, cell_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.cell_ph = cell_ph
        self.cell_token_id = cell_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, genes, texts ,smiles_prompt = zip(*batch)

        valid1 = self.collater([g["Valid"] for g in graphs1])
        if valid1.any():
            data1 = self.collater([g["Graph"] for g in graphs1 if g["Valid"]])
        else:
            data1 = None
        graphs1 = {
            "Valid": valid1,
            "Graph": data1
        }
        valid2 = self.collater([g["Valid"] for g in graphs2])
        if valid2.any():
            data2 = self.collater([g["Graph"] for g in graphs2 if g["Valid"]])
        else:
            data2 = None
        graphs2 = {
            "Valid": valid2,
            "Graph": data2
        }

        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]
        smiles_prompt = [cell_handler(p, self.cell_ph) for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)
        
        genes = self.collater(genes)
        
        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        is_cell_token = smiles_prompt_tokens.input_ids == self.cell_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        smiles_prompt_tokens['is_cell_token'] = is_cell_token

        return graphs1, graphs2, genes, smiles_prompt_tokens, texts  


class CombinedTrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, cell_ph, cell_token_id, cell_num_features):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.cell_ph = cell_ph
        self.cell_token_id = cell_token_id
        self.cell_num_features = cell_num_features

    def _collate_graphs(self, graphs):
        valid = self.collater([g["Valid"] for g in graphs])
        out = {"Valid": valid}
        if valid.any():
            data = self.collater([g["Graph"] for g in graphs if g["Valid"]])
        else:
            data = None
        out["Graph"] = data

        valid_graphs = [g for g in graphs if g["Valid"]]
        if valid.any() and any("Transform" in g for g in valid_graphs):
            transforms = []
            for g in valid_graphs:
                if "Transform" in g:
                    transforms.append(g["Transform"])
                else:
                    transformed = g["Graph"].clone()
                    zeros = torch.zeros(transformed.num_nodes, 2, dtype=torch.float)
                    transformed.x = torch.cat([zeros, transformed.x.to(torch.float)], dim=1)
                    transforms.append(transformed)
            out["Transform"] = self.collater(transforms)
        if valid.any() and any("Target" in g for g in valid_graphs):
            targets = []
            for i, g in enumerate(valid_graphs):
                target = g.get("Target")
                if not isinstance(target, torch.Tensor) or target.numel() == 0:
                    target = torch.zeros(1, 1280, dtype=torch.float)
                if target.dim() == 1:
                    target = target.unsqueeze(0)
                targets.append(Data(x=target.to(torch.float), idx=torch.tensor([i])))
            out["Target"] = self.collater(targets)
        return out

    def _stack_genes(self, genes):
        stacked = []
        for gene in genes:
            if isinstance(gene, torch.Tensor) and gene.numel() > 0:
                stacked.append(gene.to(torch.float))
            else:
                stacked.append(torch.zeros(self.cell_num_features, dtype=torch.float))
        return torch.stack(stacked, dim=0)

    def __call__(self, batch):
        graphs1, graphs2, genes, texts, smiles_prompt, task_ids = zip(*batch)

        graphs1 = self._collate_graphs(graphs1)
        graphs2 = self._collate_graphs(graphs2)
        genes = self._stack_genes(genes)
        task_ids = torch.tensor(task_ids, dtype=torch.long)

        processed_prompt = []
        for prompt, task_id in zip(smiles_prompt, task_ids.tolist()):
            text = smiles_handler(prompt, self.mol_ph, mode="train")[0]
            if task_id == 1:
                text = cell_handler(text, self.cell_ph)
            processed_prompt.append(text)

        smiles_prompt_tokens = self.tokenizer(
            text=processed_prompt,
            truncation=False,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        is_cell_token = smiles_prompt_tokens.input_ids == self.cell_token_id
        smiles_prompt_tokens["is_mol_token"] = is_mol_token
        smiles_prompt_tokens["is_cell_token"] = is_cell_token

        text_tokens = self.tokenizer(
            text=texts,
            truncation=True,
            padding="longest",
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return graphs1, graphs2, genes, task_ids, smiles_prompt_tokens, text_tokens


class CombinedInferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, cell_ph, cell_token_id, cell_num_features):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.cell_ph = cell_ph
        self.cell_token_id = cell_token_id
        self.cell_num_features = cell_num_features

    def _collate_graphs(self, graphs):
        valid = self.collater([g["Valid"] for g in graphs])
        out = {"Valid": valid}
        if valid.any():
            data = self.collater([g["Graph"] for g in graphs if g["Valid"]])
        else:
            data = None
        out["Graph"] = data

        valid_graphs = [g for g in graphs if g["Valid"]]
        if valid.any() and any("Transform" in g for g in valid_graphs):
            transforms = []
            for g in valid_graphs:
                if "Transform" in g:
                    transforms.append(g["Transform"])
                else:
                    transformed = g["Graph"].clone()
                    zeros = torch.zeros(transformed.num_nodes, 2, dtype=torch.float)
                    transformed.x = torch.cat([zeros, transformed.x.to(torch.float)], dim=1)
                    transforms.append(transformed)
            out["Transform"] = self.collater(transforms)
        if valid.any() and any("Target" in g for g in valid_graphs):
            targets = []
            for i, g in enumerate(valid_graphs):
                target = g.get("Target")
                if not isinstance(target, torch.Tensor) or target.numel() == 0:
                    target = torch.zeros(1, 1280, dtype=torch.float)
                if target.dim() == 1:
                    target = target.unsqueeze(0)
                targets.append(Data(x=target.to(torch.float), idx=torch.tensor([i])))
            out["Target"] = self.collater(targets)
        return out

    def _stack_genes(self, genes):
        stacked = []
        for gene in genes:
            if isinstance(gene, torch.Tensor) and gene.numel() > 0:
                stacked.append(gene.to(torch.float))
            else:
                stacked.append(torch.zeros(self.cell_num_features, dtype=torch.float))
        return torch.stack(stacked, dim=0)

    def __call__(self, batch):
        graphs1, graphs2, genes, texts, smiles_prompt, task_ids = zip(*batch)

        graphs1 = self._collate_graphs(graphs1)
        graphs2 = self._collate_graphs(graphs2)
        genes = self._stack_genes(genes)
        task_ids = torch.tensor(task_ids, dtype=torch.long)

        processed_prompt = []
        for prompt, task_id in zip(smiles_prompt, task_ids.tolist()):
            text = smiles_handler(prompt, self.mol_ph)[0]
            if task_id == 1:
                text = cell_handler(text, self.cell_ph)
            processed_prompt.append(text)

        smiles_prompt_tokens = self.tokenizer(
            processed_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            return_attention_mask=True,
        )

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        is_cell_token = smiles_prompt_tokens.input_ids == self.cell_token_id
        smiles_prompt_tokens["is_mol_token"] = is_mol_token
        smiles_prompt_tokens["is_cell_token"] = is_cell_token

        return graphs1, graphs2, genes, task_ids, smiles_prompt_tokens, texts
    

class TargetCellLineTrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, cell_ph, cell_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.cell_ph = cell_ph
        self.cell_token_id = cell_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, target1, target2, genes, texts, smiles_prompt= zip(*batch)

        valid1 = self.collater([g["Valid"] for g in graphs1])
        if valid1.any():
            data1 = self.collater([g["Graph"] for g in graphs1 if g["Valid"]])
            env1 = self.collater([g["Transform"] for g in graphs1 if g["Valid"]])
        else:
            data1 = None
        graphs1 = {
            "Valid": valid1,
            "Graph": data1,
            "Transform": env1
        }
        valid2 = self.collater([g["Valid"] for g in graphs2])
        if valid2.any():
            data2 = self.collater([g["Graph"] for g in graphs2 if g["Valid"]])
            env2 = self.collater([g["Transform"] for g in graphs2 if g["Valid"]])
        else:
            data2 = None
        graphs2 = {
            "Valid": valid2,
            "Graph": data2,
            "Transform": env2
        }
        targets = []
        for i, t in enumerate(target1):
            data = Data(x=t, idx=torch.tensor([i]))
            targets.append(data)
        target1 = self.collater(targets)

        targets = []
        for i, t in enumerate(target2):
            data = Data(x=t, idx=torch.tensor([i]))
            targets.append(data)
        target2 = self.collater(targets)

        smiles_prompt = [smiles_handler(p, self.mol_ph, mode="train")[0] for p in smiles_prompt]
        smiles_prompt = [cell_handler(p, self.cell_ph) for p in smiles_prompt]

        genes = self.collater(genes)

        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)      

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        is_cell_token = smiles_prompt_tokens.input_ids == self.cell_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        smiles_prompt_tokens['is_cell_token'] = is_cell_token

        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        
        return graphs1, graphs2, target1, target2, genes, smiles_prompt_tokens, text_tokens 


class TargetCellLineInferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, cell_ph, cell_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.cell_ph = cell_ph
        self.cell_token_id = cell_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, target1, target2, genes, texts ,smiles_prompt = zip(*batch)

        valid1 = self.collater([g["Valid"] for g in graphs1])
        if valid1.any():
            data1 = self.collater([g["Graph"] for g in graphs1 if g["Valid"]])
            env1 = self.collater([g["Transform"] for g in graphs1 if g["Valid"]])
        else:
            data1 = None
        graphs1 = {
            "Valid": valid1,
            "Graph": data1,
            "Transform": env1
        }
        valid2 = self.collater([g["Valid"] for g in graphs2])
        if valid2.any():
            data2 = self.collater([g["Graph"] for g in graphs2 if g["Valid"]])
            env2 = self.collater([g["Transform"] for g in graphs2 if g["Valid"]])
        else:
            data2 = None
        graphs2 = {
            "Valid": valid2,
            "Graph": data2,
            "Transform": env2
        }
        targets = []
        for i, t in enumerate(target1):
            data = Data(x=t, idx=torch.tensor([i]))
            targets.append(data)
        target1 = self.collater(targets)

        targets = []
        for i, t in enumerate(target2):
            data = Data(x=t, idx=torch.tensor([i]))
            targets.append(data)
        target2 = self.collater(targets)

        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]
        smiles_prompt = [cell_handler(p, self.cell_ph) for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)
        
        genes = self.collater(genes)
        
        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        is_cell_token = smiles_prompt_tokens.input_ids == self.cell_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        smiles_prompt_tokens['is_cell_token'] = is_cell_token

        return graphs1, graphs2, target1, target2, genes, smiles_prompt_tokens, texts  


class DataModuleSSI(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeDatasetSSI(root, text_max_len, self.prompt, self.args.transform)
        self.train_dataset =  MoleculeDatasetSSI(root, text_max_len, self.prompt, self.args.transform)
        self.val_dataset =  MoleculeDatasetSSI(args.valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeDatasetSSI(args.valid_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id


    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
    

class DataModuleDDI(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeDatasetDDI(root, text_max_len, self.prompt, self.args.transform)
        self.train_dataset =  MoleculeDatasetDDI(root, text_max_len, self.prompt, self.args.transform)
        self.val_dataset =  MoleculeDatasetDDI(self.args.valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeDatasetDDI(self.args.test_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        parser.add_argument('--transform', type=str, default=None)
        return parent_parser
    

class DataModuleCancer(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeDatasetCancer(root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, args=self.args)
        self.train_dataset =  MoleculeDatasetCancer(root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, args=self.args)  
        self.val_dataset =  MoleculeDatasetCancer(self.args.valid_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, args=self.args)    
        self.test_dataset = MoleculeDatasetCancer(self.args.test_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, args=self.args)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=CancerTrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=CancerTrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=CancerInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=CancerInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=CancerInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        parser.add_argument('--transform', type=str, default=None)
        return parent_parser
    

class DataModuleCellLine(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeDatasetCellLine(root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=True)
        self.train_dataset =  MoleculeDatasetCellLine(root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=False)
        self.val_dataset =  MoleculeDatasetCellLine(self.args.valid_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=False)  
        self.test_dataset = MoleculeDatasetCellLine(self.args.test_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=False)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        self.cell_ph_token = '<cell>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        self.cell_token_id = self.tokenizer.cell_token_id

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=CellLineTrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id)
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=CellLineTrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
            )
        else:
            raise NotImplementedError
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=CellLineInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=CellLineInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=CellLineInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        parser.add_argument('--transform', type=str, default=None)
        return parent_parser
   

class DataModuleTargetCellLine(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeDatasetTargetCellLine(root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=True)
        self.train_dataset =  MoleculeDatasetTargetCellLine(self.args.train_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=False)
        self.val_dataset =  MoleculeDatasetTargetCellLine(self.args.valid_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=False)
        self.test_dataset = MoleculeDatasetTargetCellLine(self.args.test_root, text_max_len, self.prompt, desc=self.args.desc, question=self.args.question, pretrain=False)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        self.cell_ph_token = '<cell>' * self.args.num_query_token
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        self.cell_token_id = self.tokenizer.cell_token_id

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TargetCellLineTrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id)
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TargetCellLineTrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
            )
        else:
            raise NotImplementedError
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TargetCellLineInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TargetCellLineInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TargetCellLineInferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.cell_ph_token, self.cell_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        parser.add_argument('--transform', type=str, default=None)
        return parent_parser


class DataModuleCombined(LightningDataModule):
    def __init__(
        self,
        mode: str = "pretrain",
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = "data/",
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        valid_root = self.args.valid_root if self.args.valid_root else root
        test_root = self.args.test_root if self.args.test_root else root

        self.pretrain_dataset = MoleculeDatasetCombined(
            root,
            text_max_len,
            self.prompt,
            ddi_question=self.args.combined_ddi_question,
            dsp_question=self.args.combined_dsp_question,
        )
        self.train_dataset = MoleculeDatasetCombined(
            root,
            text_max_len,
            self.prompt,
            ddi_question=self.args.combined_ddi_question,
            dsp_question=self.args.combined_dsp_question,
        )
        self.val_dataset = MoleculeDatasetCombined(
            valid_root,
            text_max_len,
            self.prompt,
            ddi_question=self.args.combined_ddi_question,
            dsp_question=self.args.combined_dsp_question,
        )
        self.test_dataset = MoleculeDatasetCombined(
            test_root,
            text_max_len,
            self.prompt,
            ddi_question=self.args.combined_ddi_question,
            dsp_question=self.args.combined_dsp_question,
        )

        self.init_tokenizer(tokenizer)
        self.mol_ph_token = "<mol>" * self.args.num_query_token
        self.cell_ph_token = "<cell>" * self.args.num_query_token

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        self.cell_token_id = self.tokenizer.cell_token_id

    def train_dataloader(self):
        persistent_workers = self.num_workers > 0
        collate_fn = CombinedTrainCollater(
            self.tokenizer,
            self.text_max_len,
            self.mol_ph_token,
            self.mol_token_id,
            self.cell_ph_token,
            self.cell_token_id,
            self.args.cell_num_features,
        )
        if self.mode == "pretrain":
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        elif self.mode == "ft":
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        persistent_workers = self.num_workers > 0
        collate_fn = CombinedInferenceCollater(
            self.tokenizer,
            self.text_max_len,
            self.mol_ph_token,
            self.mol_token_id,
            self.cell_ph_token,
            self.cell_token_id,
            self.args.cell_num_features,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )
        return [val_loader, test_loader]

    def test_dataloader(self):
        persistent_workers = self.num_workers > 0
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=CombinedInferenceCollater(
                self.tokenizer,
                self.text_max_len,
                self.mol_ph_token,
                self.mol_token_id,
                self.cell_ph_token,
                self.cell_token_id,
                self.args.cell_num_features,
            ),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--inference_batch_size", type=int, default=4)
        parser.add_argument("--use_smiles", action="store_true", default=False)
        parser.add_argument("--root", type=str, default="")
        parser.add_argument("--text_max_len", type=int, default=128)
        parser.add_argument(
            "--prompt",
            type=str,
            default="The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ",
        )
        parser.add_argument("--transform", type=str, default=None)
        parser.add_argument(
            "--combined_ddi_question",
            type=str,
            default="What are the side effects of these two drugs?",
        )
        parser.add_argument(
            "--combined_dsp_question",
            type=str,
            default="What is the drug synergy score of this drug pair in this cell line?",
        )
        parser.add_argument("--combined_ddi_ood", type=str, default="none", choices=["none", "dynas"])
        parser.add_argument("--combined_dsp_ood", type=str, default="none", choices=["none", "disen"])
        return parent_parser
