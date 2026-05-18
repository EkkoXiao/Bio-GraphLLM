import torch
import torch_geometric
from torch_geometric.data import Dataset
import os
import pandas as pd
import numpy as np

def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
        return len(subdirectories)
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not exist.")
        return -1
    except Exception as e:
        print(f"Error: {e}")
        return -2 

class DegreeDistribution(object):
    def __call__(self, g):
        '''
        if g.is_undirected():
            edges = g.edge_index[0]
        else:
            edges = torch.cat((g.edge_index[0], g.edge_index[1]))
        '''
        edges = g.edge_index[1]
        if edges.numel() == 0:
            deratio = torch.tensor([0.0, 0.0, 0.0])
        else:
            degrees = torch_geometric.utils.degree(edges).to(torch.long).numpy().tolist()
            deratio = [degrees.count(i) for i in range(1, 4)]
            deratio = torch.tensor(deratio) / g.num_nodes
        g.deratio = deratio
        return g

class MoleculeDatasetSSI(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None, pretrain=False):
        super(MoleculeDatasetSSI, self).__init__(root)
        self.root = root if not pretrain else os.path.join(root, min(next(os.walk(root))[1])) + "/"
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+" What is the solvation Gibbs free energy of this pair of molecules?"
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)

        if self.transform == "Degree":
            data_graph1 = DegreeDistribution()(data_graph1)
            data_graph2 = DegreeDistribution()(data_graph2)
        
        return data_graph1, data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token    
    
class MoleculeDatasetDDI(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None, pretrain=False):
        super(MoleculeDatasetDDI, self).__init__(root)
        self.root = root if not pretrain else os.path.join(root, min(next(os.walk(root))[1])) + "/"
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")


    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)

        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+" What are the side effects of these two drugs?"
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)

        if self.transform == "Degree":
            data_graph1 = DegreeDistribution()(data_graph1)
            data_graph2 = DegreeDistribution()(data_graph2)

        return data_graph1, data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
    
class MoleculeDatasetCancer(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None, desc=False, question="What are the side effects of these two drugs?", pretrain=False):
        super(MoleculeDatasetCancer, self).__init__(root)
        self.root = root if not pretrain else os.path.join(root, min(next(os.walk(root))[1])) + "/"
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        self.desc = desc
        self.question = question
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")


    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        property1_name_list = os.listdir(self.root+'property1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        property2_name_list = os.listdir(self.root+'property2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        if self.desc:
            desc_name_list = os.listdir(self.root+'description/'+str(index)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        property_path = os.path.join(self.root, 'property1/'+str(index)+'/', property1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt

        with open(property_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            drug_property = lines[0].strip() if len(lines) else ""
        
        smiles_prompt1 = drug_property + ("" if smiles == "None" else smiles_prompt1)
        
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)

        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        property_path = os.path.join(self.root, 'property2/'+str(index)+'/', property2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt

        with open(property_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            drug_property = lines[0].strip() if len(lines) else ""
        
        smiles_prompt2 = drug_property + ("" if smiles == "None" else smiles_prompt2)

        desc = ""

        if self.desc:
            desc_path = os.path.join(self.root, 'description/'+str(index)+'/', desc_name_list[0])
            with open(desc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) <= 1
                desc = lines[0].strip()
            
        smiles_prompt = '</s> '+smiles_prompt1+' </s>'+' </s>'+smiles_prompt2+' </s> .'+ desc + " " + self.question

        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)

        return data_graph1, data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
    

class MoleculeDatasetCellLine(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None, desc=False, question="What are the side effects of these two drugs?", pretrain=False):
        super(MoleculeDatasetCellLine, self).__init__(root)
        self.root = root if not pretrain else os.path.join(root, min(next(os.walk(root))[1])) + "/"
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        self.desc = desc
        self.question = question
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")


    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        property1_name_list = os.listdir(self.root+'property1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        property2_name_list = os.listdir(self.root+'property2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        gene_name_list = os.listdir(self.root+'genes/'+str(index)+'/')
        gene_path = os.path.join(self.root, 'genes/'+str(index)+'/', gene_name_list[0])
        genes = torch.load(gene_path)

        if self.desc:
            desc_name_list = os.listdir(self.root+'description/'+str(index)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        property_path = os.path.join(self.root, 'property1/'+str(index)+'/', property1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt

        with open(property_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            drug_property = lines[0].strip()
        
        smiles_prompt1 = drug_property + ("" if smiles == "None" else smiles_prompt1)
        
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)

        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        property_path = os.path.join(self.root, 'property2/'+str(index)+'/', property2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt

        with open(property_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            drug_property = lines[0].strip()
        
        smiles_prompt2 = drug_property + ("" if smiles == "None" else smiles_prompt2)

        desc = ""

        if self.desc:
            desc_path = os.path.join(self.root, 'description/'+str(index)+'/', desc_name_list[0])
            with open(desc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) <= 1
                desc = lines[0].strip()

        cell_line_prompt = "The cell line of this drug pair is [START_CELL][END_CELL]. "
        
        smiles_prompt = '</s> '+ cell_line_prompt + '< /s>' + smiles_prompt1 + ' </s>'+' </s>'+smiles_prompt2+' </s> .'+ desc + " " + self.question

        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)

        return data_graph1, data_graph2, genes, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
    

class MoleculeDatasetTargetCellLine(Dataset):
    def __init__(self, root, text_max_len, prompt=None, transform=None, desc=False, question="What are the side effects of these two drugs?", pretrain=False):
        super(MoleculeDatasetTargetCellLine, self).__init__(root)
        self.root = root if not pretrain else os.path.join(root, min(next(os.walk(root))[1])) + "/"
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        self.desc = desc
        self.question = question
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")


    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        property1_name_list = os.listdir(self.root+'property1/'+str(index)+'/')
        target1_name_list = os.listdir(self.root+'target1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        property2_name_list = os.listdir(self.root+'property2/'+str(index)+'/')
        target2_name_list = os.listdir(self.root+'target2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        gene_name_list = os.listdir(self.root+'genes/'+str(index)+'/')
        gene_path = os.path.join(self.root, 'genes/'+str(index)+'/', gene_name_list[0])
        genes = torch.load(gene_path)

        target1_path = os.path.join(self.root, 'target1/'+str(index)+'/', target1_name_list[0])
        data_target1 = torch.load(target1_path)

        target2_path = os.path.join(self.root, 'target2/'+str(index)+'/', target2_name_list[0])
        data_target2 = torch.load(target2_path)

        if self.desc:
            desc_name_list = os.listdir(self.root+'description/'+str(index)+'/')
        
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        property_path = os.path.join(self.root, 'property1/'+str(index)+'/', property1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt

        with open(property_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            drug_property = lines[0].strip()
        
        smiles_prompt1 = drug_property + ("" if smiles == "None" else smiles_prompt1)
        
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)

        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        property_path = os.path.join(self.root, 'property2/'+str(index)+'/', property2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt

        with open(property_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) <= 1
            drug_property = lines[0].strip()
        
        smiles_prompt2 = drug_property + ("" if smiles == "None" else smiles_prompt2)

        desc = ""

        if self.desc:
            desc_path = os.path.join(self.root, 'description/'+str(index)+'/', desc_name_list[0])
            with open(desc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) <= 1
                desc = lines[0].strip()

        cell_line_prompt = "The cell line of this drug pair is [START_CELL][END_CELL]. "
        
        smiles_prompt = '</s> '+ cell_line_prompt + '< /s>' + smiles_prompt1 + ' </s>'+' </s>'+smiles_prompt2+' </s> .'+ desc + " " + self.question

        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)

        return data_graph1, data_graph2, data_target1, data_target2, genes, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token


class MoleculeDatasetCombined(Dataset):
    def __init__(
        self,
        root,
        text_max_len,
        prompt=None,
        transform=None,
        ddi_question="What are the side effects of these two drugs?",
        dsp_question="What is the drug synergy score of this drug pair in this cell line?",
    ):
        super(MoleculeDatasetCombined, self).__init__(root)
        self.root = root if root.endswith("/") else root + "/"
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.transform = transform
        self.ddi_question = ddi_question
        self.dsp_question = dsp_question

        if not prompt:
            self.prompt = "The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. "
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return count_subdirectories(self.root + "text/")

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root + "graph1/" + str(index) + "/")
        smiles1_name_list = os.listdir(self.root + "smiles1/" + str(index) + "/")
        graph2_name_list = os.listdir(self.root + "graph2/" + str(index) + "/")
        smiles2_name_list = os.listdir(self.root + "smiles2/" + str(index) + "/")
        text_name_list = os.listdir(self.root + "text/" + str(index) + "/")
        task_name_list = os.listdir(self.root + "task/" + str(index) + "/")

        graph_path = os.path.join(self.root, "graph1/" + str(index) + "/", graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, "smiles1/" + str(index) + "/", smiles1_name_list[0])
        with open(smiles_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles1 = lines[0].strip()
        if self.prompt.find("{}") >= 0:
            smiles_prompt1 = self.prompt.format(smiles1[:128])
        else:
            smiles_prompt1 = self.prompt

        graph_path = os.path.join(self.root, "graph2/" + str(index) + "/", graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        smiles_path = os.path.join(self.root, "smiles2/" + str(index) + "/", smiles2_name_list[0])
        with open(smiles_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) <= 1
            smiles2 = lines[0].strip()
        if self.prompt.find("{}") >= 0:
            smiles_prompt2 = self.prompt.format(smiles2[:128])
        else:
            smiles_prompt2 = self.prompt

        task_path = os.path.join(self.root, "task/" + str(index) + "/", task_name_list[0])
        with open(task_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) <= 1
            task = lines[0].strip().upper()

        gene_path = os.path.join(self.root, "genes", str(index), "gene_data.pt")
        if os.path.exists(gene_path) and os.path.getsize(gene_path) > 0:
            try:
                genes = torch.load(gene_path)
            except (EOFError, RuntimeError, OSError, ValueError):
                genes = torch.empty(0)
        else:
            genes = torch.empty(0)

        if not isinstance(genes, torch.Tensor):
            genes = torch.tensor(genes)

        if task == "DSP":
            cell_line_prompt = "The cell line of this drug pair is [START_CELL][END_CELL]. "
            question = self.dsp_question
            task_id = 1
            smiles_prompt = (
                "</s> "
                + cell_line_prompt
                + " </s>"
                + " </s>"
                + smiles_prompt1
                + " </s>"
                + " </s>"
                + smiles_prompt2
                + " </s> . "
                + question
            )
        else:
            question = self.ddi_question
            task_id = 0
            smiles_prompt = (
                "</s> "
                + smiles_prompt1
                + " </s>"
                + " </s>"
                + smiles_prompt2
                + " </s> . "
                + question
            )

        text_path = os.path.join(self.root, "text/" + str(index) + "/", text_name_list[0])
        text_list = []
        count = 0
        for line in open(text_path, "r", encoding="utf-8"):
            count += 1
            text_list.append(line.strip("\n"))
            if count > 100:
                break
        text = " ".join(text_list)

        return data_graph1, data_graph2, genes, text, smiles_prompt, task_id

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(
            text=text,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return sentence_token
