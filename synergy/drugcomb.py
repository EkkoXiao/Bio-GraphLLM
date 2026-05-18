import copy
import math
import os
import re
import torch
import random
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(smiles):
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def calculate_bounds(number):
    lower_bound = math.floor(number * 2) / 2  # 下界
    upper_bound = math.ceil(number * 2) / 2   # 上界
    return lower_bound, upper_bound

import pandas as pd
import numpy as np
import os 
import csv
import shutil

fetch_data_path = "../../data/synergy_data/"
dataset_name = "ScaffoldLoewe/"
data_path = "/DATA/DATANAS1/xlx21/data/"

if os.path.exists(data_path + dataset_name):
    shutil.rmtree(data_path + dataset_name)

empty_graph = Data(
    x=torch.empty(0, 0),  # 没有节点特征
    edge_index=torch.empty(2, 0, dtype=torch.long),  # 没有边
    edge_attr=torch.empty(0, 0)  # 没有边特征
)

genes = pd.read_csv("../dicts/df_rma_landm.tsv", sep='\t')
genes_pca = pd.read_csv("../dicts/df_rma_landm_pca2.tsv", sep='\t')

model_path = "../modelscope/ESM"

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)

uniprot_df = pd.read_csv(fetch_data_path + "uniprot_extracted.csv")
name_to_seq = {name.lower(): seq for name, seq in zip(uniprot_df["name"], uniprot_df["sequence"])}

def save_targets(targets, path):
    if pd.isna(targets):  
        torch.save(torch.empty(0), path)
    else:
        targets = [t.strip().lower() for t in targets.split(";")]
        embeddings = []

        for t in targets:
            if t in name_to_seq:
                seq = name_to_seq[t]
                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # 取 CLS token 表示
                last_hidden_state = outputs.last_hidden_state[:, 0, :]  # shape: (1, 1280)
                embeddings.append(last_hidden_state.cpu())

        if embeddings:
            tensor_out = torch.cat(embeddings, dim=0)  # shape: (n_targets, 1280)
        else:
            tensor_out = torch.empty(0)  # 如果没找到任何匹配的sequence

        torch.save(tensor_out, path)

def process(data, dir):
    i = 0
    for j in range(len(data)):
        smiles1 = data[j][4].split(";")[1] if (";" in data[j][4]) else data[j][4]
        smiles2 = data[j][5].split(";")[1] if (";" in data[j][5]) else data[j][5]
        flag = True
        try:
            graph1 = mol_to_graph_data_obj_simple(smiles1)
            data1 = {"Valid": True, "Graph": graph1, "Drug": data[j][0]}
        except Exception as e:
            print(f"Error occurred while processing {data[j][0]}:")
            print(str(e))
            flag = False
            data1 = {"Valid": False, "Graph": empty_graph, "Drug": data[j][0]}
        try:
            graph2 = mol_to_graph_data_obj_simple(smiles2)
            data2 = {"Valid": True, "Graph": graph2, "Drug": data[j][1]}
        except Exception as e:
            print(f"Error occurred while processing {data[j][1]}:")
            print(str(e))
            flag = False
            data2 = {"Valid": False, "Graph": empty_graph, "Drug": data[j][1]}
        if not flag:
            continue
        os.makedirs(data_path + dataset_name + dir + "/smiles1/" + str(i))
        os.makedirs(data_path + dataset_name + dir + "/smiles2/" + str(i))
        os.makedirs(data_path + dataset_name + dir + "/property1/" + str(i))
        os.makedirs(data_path + dataset_name + dir + "/property2/" + str(i))
        os.makedirs(data_path + dataset_name + dir + "/text/" + str(i))
        cell_line = data[j][3]
        tensor = torch.tensor(genes['DATA.'+ str(cell_line)].astype(float).values)
        os.makedirs(data_path + dataset_name + dir + "/genes/" + str(i))
        torch.save(tensor, data_path + dataset_name + dir + "/genes/" + str(i) + '/gene_data.pt')
        gene_2d = torch.tensor(genes_pca['DATA.'+ str(cell_line)].astype(float).values)
        os.makedirs(data_path + dataset_name + dir + "/graph1/" + str(i))
        transformed = copy.deepcopy(data1["Graph"])
        transformed.x = torch.cat([gene_2d.repeat(data1["Graph"].num_nodes, 1), data1["Graph"].x], dim=1).to(torch.bfloat16)
        data1["Transform"] = transformed
        torch.save(data1, data_path + dataset_name + dir + "/graph1/" + str(i) + '/graph_data.pt')
        os.makedirs(data_path + dataset_name + dir + "/graph2/" + str(i))
        transformed = copy.deepcopy(data2["Graph"])
        transformed.x = torch.cat([gene_2d.repeat(data2["Graph"].num_nodes, 1), data2["Graph"].x], dim=1).to(torch.bfloat16)
        data2["Transform"] = transformed
        torch.save(data2, data_path + dataset_name + dir + "/graph2/" + str(i) + '/graph_data.pt')
        print(i)
        file = open(data_path + dataset_name + dir + "/text/" + str(i) + "/text.txt","w")
        value = data[j][9]
        lower, upper = calculate_bounds(value)
        if value > 0:
            text = "Yes. The absolute value is above "+ str(abs(lower)) + " and below " + str(abs(upper)) +", thus the accurate value is "+ str('%.2f'%(abs(value))) + "."
        else:
            text = "No. The absolute value is above "+ str(abs(upper)) +" and below "+ str(abs(lower)) +", thus the accurate value is "+ str('%.2f'%(abs(value))) + "."
        file.write(text)
        file.close()
        file = open(data_path + dataset_name + dir + "/smiles1/" + str(i) + "/text.txt","w")
        if pd.notna(smiles1) and smiles1 != "Not Available":
            if not data1["Valid"]:
                print("ERROR!")
                print(smiles1)
                print(data[j][0]) 
                break
            file.write(smiles1)
        else:
            file.write("None")
        file.close()
        file = open(data_path + dataset_name + dir + "/smiles2/" + str(i) + "/text.txt","w")
        if pd.notna(smiles2) and smiles2 != "Not Available":
            if not data2["Valid"]:
                print("ERROR!")
                print(smiles2)
                print(data[j][1]) 
                break
            file.write(smiles2)
        else:
            file.write("None")
        file.close()
        file = open(data_path + dataset_name + dir + "/property1/" + str(i) + "/text.txt","w")
        property1 = "#Drug1 is " + data[j][0]
        file.write(property1)
        file.close()
        file = open(data_path + dataset_name + dir + "/property2/" + str(i) + "/text.txt","w")
        property2 = "#Drug2 is " + data[j][2]
        file.write(property2)
        file.close()
        os.makedirs(data_path + dataset_name + dir + "/target1/" + str(i))
        save_targets(data[j][6], os.path.join(data_path, dataset_name, dir, "target1", str(i), "target_data.pt"))
        os.makedirs(data_path + dataset_name + dir + "/target2/" + str(i))
        save_targets(data[j][7], os.path.join(data_path, dataset_name, dir, "target2", str(i), "target_data.pt"))
        i += 1

train_data = pd.read_csv(fetch_data_path + 'OOD_loewe/scaffold_train.csv')
train_data = np.array(train_data)

process(train_data, "train")

valid_data = pd.read_csv(fetch_data_path + 'OOD_loewe/scaffold_valid.csv')
valid_data = np.array(valid_data)

process(valid_data, "valid")

test_data = pd.read_csv(fetch_data_path + 'OOD_loewe/scaffold_test.csv')
test_data = np.array(test_data)

process(test_data, "test")
