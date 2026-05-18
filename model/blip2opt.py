import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
from rdkit import Chem

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

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

def mask_by_len(input, lens, fill_value=0):
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input

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

    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
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
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

import re
SPLIT_MARKER = f"|"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


def _insert_split_marker(m: re.Match):
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def escape_custom_split_sequence(text):
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

def smiles_handler(text, mol_ph):
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

class Blip2OPT(Blip2Base):
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune='freeze',
        peft_dir='',
        opt_model="modelscope/galactica-1.3b",
        prompt="",
        use_nas=False,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.use_nas = use_nas
        self.is_cell_line = args.cell
        self.is_cancer = args.cancer
        self.is_string = args.string
        self.sslloss_fn = torch.nn.L1Loss()
        self.num_query_token = num_query_token
        if self.use_nas and self.is_cell_line:
            self.graph_encoder, self.ln_graph = self.init_disen_encoder(
                args.input_dim,
                args.env_dim,
                mol=True,
                virtual=True,
                args=args,
                use_forward=tune_gnn
            )
            self.tune_gnn = tune_gnn
            if not tune_gnn:
                for name, param in self.graph_encoder.named_parameters():
                    param.requires_grad = False
                self.graph_encoder = self.graph_encoder.eval()
                logging.info("freeze nas encoder")

            self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, 
                                                                self.graph_encoder.supernet.hidden_size * (self.graph_encoder.supernet.num_layers + 1), cross_attention_freq)
        elif self.use_nas:
            self.graph_encoder, self.ln_graph = self.init_nas_encoder(
                args.input_dim,
                mol=True,
                virtual=True,
                args=args,
                use_forward=tune_gnn
            )
            self.tune_gnn = tune_gnn
            if not tune_gnn:
                for name, param in self.graph_encoder.named_parameters():
                    param.requires_grad = False
                self.graph_encoder = self.graph_encoder.eval()
                logging.info("freeze nas encoder")

            self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, 
                                                                self.graph_encoder.supernet.hidden_size * (self.graph_encoder.supernet.num_layers + 1), cross_attention_freq)
        elif not self.is_string:
            self.graph_encoder, self.ln_graph = self.init_graph_encoder(
                gin_num_layers, 
                gin_hidden_dim, 
                gin_drop_ratio,
                not self.args.no_batch_norm
            )
            self.tune_gnn = tune_gnn
            if not tune_gnn:
                for name, param in self.graph_encoder.named_parameters():
                    param.requires_grad = False
                self.graph_encoder = self.graph_encoder.eval()
                self.graph_encoder.train = disabled_train
                logging.info("freeze graph encoder")

            self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)

        if self.args.cell:
            self.cell_Qformer, self.cell_query_tokens = self.init_Qformer(bert_name, num_query_token, self.args.cell_num_features, cross_attention_freq)

            self.cell_Qformer.cls = None
            self.cell_Qformer.bert.embeddings.word_embeddings = None
            self.cell_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.cell_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        print("##### INIT #####")
        if not self.is_string:
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})
        self.opt_tokenizer.add_tokens('<mol>')
        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]
        if self.args.cell:
            self.opt_tokenizer.add_tokens('<cell>')
            self.cell_token = '<cell>'
            self.opt_tokenizer.cell_token_id = self.opt_tokenizer("<cell>", add_special_tokens=False).input_ids[0]

        if self.args.category != -1:
            self.cat_tokens = [f"<CAT{str(i).zfill(3)}>" for i in range(1, self.args.category + 1)]
            cat_token_ids = self.opt_tokenizer.add_tokens(self.cat_tokens)
            self.cat_token_dict = {self.opt_tokenizer.convert_tokens_to_ids(token): token for token in self.cat_tokens}

        self.collater = Collater([], [])
        new_tokens = []

        if self.args.cell and self.args.category == -1:
            new_tokens.extend([token for token in ["Yes.", "No."] if token not in self.opt_tokenizer.get_vocab()])
        if new_tokens:
            self.opt_tokenizer.add_tokens(new_tokens)
        
        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
        else:
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))

        self.llm_tune = llm_tune
        if llm_tune == 'lora':
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
            self.opt_model.print_trainable_parameters()
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        if not self.is_string:
            self.opt_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
            )
        
        if self.args.cell:
            self.cell_proj = nn.Linear(
                self.cell_Qformer.config.hidden_size, self.opt_model.config.hidden_size
            )

        self.prompt = prompt

    def forward(self, batch):
        if self.is_cell_line and self.use_nas:
            return self.forwardCellLineTarget(batch)
        elif self.is_cell_line:
            return self.forwardCellLine(batch)
        elif self.is_cancer:
            return self.forwardCancer(batch)
        else:
            return self.forwardRaw(batch)

    def forwardRaw(self, batch):
        graphs1, graphs2, prompt_tokens, text_tokens= batch
        # Process graphs1
        if self.use_nas:
            cosloss1, ssloutput1, graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
        else:
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) 
        if not self.tune_gnn:
            graph_embeds1 = graph_embeds1.detach()
        graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1) 
        device = graph_embeds1.device
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1) 

        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        
        # Process graphs2
        if self.use_nas:
            cosloss2, ssloutput2, graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        else:
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2) 
        if not self.tune_gnn:
            graph_embeds2 = graph_embeds2.detach()
        graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2)

        device = graph_embeds2.device 
        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)

        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        
        mol_tokens = torch.cat([mol_tokens1,mol_tokens2], dim=1)

        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        if mol_tokens is not None:
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        if self.use_nas:
            ssltarget1 = graphs1.deratio.view(-1, 3)
            sslloss1 = self.sslloss_fn(ssloutput1, ssltarget1)

            ssltarget2 = graphs2.deratio.view(-1, 3)
            sslloss2 = self.sslloss_fn(ssloutput2, ssltarget2)

            sslloss = sslloss1 + sslloss2
            cosloss = cosloss1 + cosloss2
            total_loss = loss + self.args.gamma * sslloss + self.args.beta * cosloss
        else:
            total_loss = loss

        return {"loss": total_loss}
    
    def forwardCancer(self, batch):
        graphs1, graphs2, prompt_tokens, text_tokens= batch

        if not self.is_string:
            valid1 = graphs1["Valid"]
            graphs1 = graphs1["Graph"]
            valid2 = graphs2["Valid"]
            graphs2 = graphs2["Graph"]
            if valid1.any():
                graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)

                graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1) 

                device = graph_embeds1.device
                query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1) 

                query_output1 = self.Qformer.bert(
                    query_embeds=query_tokens1,
                    encoder_hidden_states=graph_embeds1,
                    encoder_attention_mask=graph_masks1,
                    return_dict=True,
                )
                mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

            if valid2.any():
                graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)

                graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2) 

                device = graph_embeds2.device
                query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1) 

                query_output2 = self.Qformer.bert(
                    query_embeds=query_tokens2,
                    encoder_hidden_states=graph_embeds2,
                    encoder_attention_mask=graph_masks2,
                    return_dict=True,
                )
                mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

            mol_tokens = []
            graph1_pointer = 0
            graph2_pointer = 0
            assert(valid1.shape[0] == valid2.shape[0])
            for i in range(valid1.shape[0]):
                if valid1[i] and valid2[i]:
                    mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                    graph1_pointer += 1
                    graph2_pointer += 1
                elif valid1[i] and not valid2[i]:
                    mol_tokens.append(mol_tokens1[graph1_pointer])
                    graph1_pointer += 1
                elif valid2[i]:
                    mol_tokens.append(mol_tokens2[graph2_pointer])
                    graph2_pointer += 1

            mol_tokens = torch.cat(mol_tokens, dim=0) if len(mol_tokens) > 0 else None
        else:
            device = prompt_tokens.input_ids.device
            mol_tokens = None

        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        if mol_tokens is not None:
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}
    
    def forwardCellLine(self, batch):
        graphs1, graphs2, genes, prompt_tokens, text_tokens= batch

        valid1 = graphs1["Valid"]
        graphs1 = graphs1["Graph"]
        valid2 = graphs2["Valid"]
        graphs2 = graphs2["Graph"]

        if valid1.any():
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)

            graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1) 

            device = graph_embeds1.device
            query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1) 

            query_output1 = self.Qformer.bert(
                query_embeds=query_tokens1,
                encoder_hidden_states=graph_embeds1,
                encoder_attention_mask=graph_masks1,
                return_dict=True,
            )
            mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        if valid2.any():
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)

            graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2) 

            device = graph_embeds2.device
            query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1) 

            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens2,
                encoder_hidden_states=graph_embeds2,
                encoder_attention_mask=graph_masks2,
                return_dict=True,
            )
            mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        mol_tokens = []
        graph1_pointer = 0
        graph2_pointer = 0
        assert(valid1.shape[0] == valid2.shape[0])
        for i in range(valid1.shape[0]):
            if valid1[i] and valid2[i]:
                mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                graph1_pointer += 1
                graph2_pointer += 1
            elif valid1[i] and not valid2[i]:
                mol_tokens.append(mol_tokens1[graph1_pointer])
                graph1_pointer += 1
            elif valid2[i]:
                mol_tokens.append(mol_tokens2[graph2_pointer])
                graph2_pointer += 1

        mol_tokens = torch.cat(mol_tokens, dim=0)

        genes = genes.unsqueeze(1).to(torch.float)

        query_token_gene = self.cell_query_tokens.expand(genes.shape[0], -1, -1)
        gene_attention_mask = torch.ones(genes.shape[0], genes.shape[1], dtype=torch.long).to(device)
        gene_output = self.cell_Qformer.bert(
            query_embeds=query_token_gene,
            encoder_hidden_states=genes,
            encoder_attention_mask=gene_attention_mask,
            return_dict=True,
        )
        gene_tokens = self.cell_proj(gene_output.last_hidden_state).flatten(0, 1)

        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens
        prompt_embeds[prompt_tokens.is_cell_token] = gene_tokens

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    def forwardCellLineTarget(self, batch):
        graphs1, graphs2, target1, target2, genes, prompt_tokens, text_tokens= batch
        valid1 = graphs1["Valid"]
        env1 = graphs1["Transform"]
        graphs1 = graphs1["Graph"]
        valid2 = graphs2["Valid"]
        env2 = graphs2["Transform"]
        graphs2 = graphs2["Graph"]

        if valid1.any():
            disenloss1, cosloss1, graph_embeds1, graph_masks1 = self.graph_encoder(graphs1, env1, target1)

            graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1) 

            device = graph_embeds1.device
            query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1) 

            query_output1 = self.Qformer.bert(
                query_embeds=query_tokens1,
                encoder_hidden_states=graph_embeds1,
                encoder_attention_mask=graph_masks1,
                return_dict=True,
            )
            mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        if valid2.any():
            disenloss2, cosloss2, graph_embeds2, graph_masks2 = self.graph_encoder(graphs2, env2, target2)

            graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2) 

            device = graph_embeds2.device
            query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1) 

            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens2,
                encoder_hidden_states=graph_embeds2,
                encoder_attention_mask=graph_masks2,
                return_dict=True,
            )
            mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        mol_tokens = []
        graph1_pointer = 0
        graph2_pointer = 0
        assert(valid1.shape[0] == valid2.shape[0])
        for i in range(valid1.shape[0]):
            if valid1[i] and valid2[i]:
                mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                graph1_pointer += 1
                graph2_pointer += 1
            elif valid1[i] and not valid2[i]:
                mol_tokens.append(mol_tokens1[graph1_pointer])
                graph1_pointer += 1
            elif valid2[i]:
                mol_tokens.append(mol_tokens2[graph2_pointer])
                graph2_pointer += 1

        mol_tokens = torch.cat(mol_tokens, dim=0)

        genes = genes.unsqueeze(1).to(torch.float)

        query_token_gene = self.cell_query_tokens.expand(genes.shape[0], -1, -1)
        gene_attention_mask = torch.ones(genes.shape[0], genes.shape[1], dtype=torch.long).to(device)
        gene_output = self.cell_Qformer.bert(
            query_embeds=query_token_gene,
            encoder_hidden_states=genes,
            encoder_attention_mask=gene_attention_mask,
            return_dict=True,
        )
        gene_tokens = self.cell_proj(gene_output.last_hidden_state).flatten(0, 1)

        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens
        prompt_embeds[prompt_tokens.is_cell_token] = gene_tokens

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = self.args.gamma * (disenloss1 + disenloss2) + self.args.beta * (cosloss1 + cosloss2) + outputs.loss

        return {"loss": loss}
    
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
    ):
        if self.is_cell_line and self.use_nas:
            return self.generateCellLineTarget(
                samples,
                do_sample,
                num_beams,
                max_length,
                min_length,
                top_p,
                repetition_penalty,
                length_penalty,
                num_captions,
                temperature,
                output_scores,
        )
        elif self.is_cell_line:
            return self.generateCellLine(
                samples,
                do_sample,
                num_beams,
                max_length,
                min_length,
                top_p,
                repetition_penalty,
                length_penalty,
                num_captions,
                temperature,
                output_scores=output_scores,
            )
        elif self.is_cancer:
            return self.generateCancer(
                samples,
                do_sample,
                num_beams,
                max_length,
                min_length,
                top_p,
                repetition_penalty,
                length_penalty,
                num_captions,
                temperature,
            )
        else:
            return self.generateRaw(
                samples,
                do_sample,
                num_beams,
                max_length,
                min_length,
                top_p,
                repetition_penalty,
                length_penalty,
                num_captions,
                temperature,
            )
    
    @torch.no_grad()
    def generateRaw(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
    ):
        graphs1 = samples['graphs1']
        prompt_tokens = samples['prompt_tokens']
        if self.use_nas:
            cosloss1, ssloutput1, graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
        else:
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) 
        graph_embeds1 = self.ln_graph(graph_embeds1)

        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        
        graphs2 = samples['graphs2']
        if self.use_nas:
            cosloss2, ssloutput2, graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        else:
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2) 
        graph_embeds2 = self.ln_graph(graph_embeds2)

        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        mol_tokens=torch.cat([mol_tokens1,mol_tokens2],dim=1)
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs
    
    @torch.no_grad()
    def generateCancer(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
    ):
        if not self.is_string:
            graphs1 = samples['graphs1']
            valid1 = graphs1["Valid"]
            graphs1 = graphs1["Graph"]

            if valid1.any():
                graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) 
                graph_embeds1 = self.ln_graph(graph_embeds1)

                query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
                query_output1 = self.Qformer.bert(
                    query_embeds=query_tokens1,
                    encoder_hidden_states=graph_embeds1,
                    encoder_attention_mask=graph_masks1,
                    return_dict=True,
                )
                mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
            
            graphs2 = samples['graphs2']
            valid2 = graphs2["Valid"]
            graphs2 = graphs2["Graph"]

            if valid2.any():
                graph_embeds2, graph_masks2 = self.graph_encoder(graphs2) 
                graph_embeds2 = self.ln_graph(graph_embeds2)

                query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
                query_output2 = self.Qformer.bert(
                    query_embeds=query_tokens2,
                    encoder_hidden_states=graph_embeds2,
                    encoder_attention_mask=graph_masks2,
                    return_dict=True,
                )
                mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

            prompt_tokens = samples['prompt_tokens']
            mol_tokens = []
            graph1_pointer = 0
            graph2_pointer = 0
            assert(valid1.shape[0] == valid2.shape[0])
            for i in range(valid1.shape[0]):
                if valid1[i] and valid2[i]:
                    mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                    graph1_pointer += 1
                    graph2_pointer += 1
                elif valid1[i] and not valid2[i]:
                    mol_tokens.append(mol_tokens1[graph1_pointer])
                    graph1_pointer += 1
                elif valid2[i]:
                    mol_tokens.append(mol_tokens2[graph2_pointer])
                    graph2_pointer += 1

            mol_tokens = torch.cat(mol_tokens, dim=0)
        else:
            prompt_tokens = samples['prompt_tokens']
            device = prompt_tokens.input_ids.device
            mol_tokens = None
    
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        if not self.is_string:
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens

        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs

    @torch.no_grad()
    def generateCellLine(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
    ):
        graphs1 = samples['graphs1']
        valid1 = graphs1["Valid"]
        graphs1 = graphs1["Graph"]

        if valid1.any():
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) 
            graph_embeds1 = self.ln_graph(graph_embeds1)

            query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
            query_output1 = self.Qformer.bert(
                query_embeds=query_tokens1,
                encoder_hidden_states=graph_embeds1,
                encoder_attention_mask=graph_masks1,
                return_dict=True,
            )
            mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        
        graphs2 = samples['graphs2']
        valid2 = graphs2["Valid"]
        graphs2 = graphs2["Graph"]

        if valid2.any():
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2) 
            graph_embeds2 = self.ln_graph(graph_embeds2)

            query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens2,
                encoder_hidden_states=graph_embeds2,
                encoder_attention_mask=graph_masks2,
                return_dict=True,
            )
            mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        prompt_tokens = samples['prompt_tokens']
        mol_tokens = []
        graph1_pointer = 0
        graph2_pointer = 0
        assert(valid1.shape[0] == valid2.shape[0])
        for i in range(valid1.shape[0]):
            if valid1[i] and valid2[i]:
                mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                graph1_pointer += 1
                graph2_pointer += 1
            elif valid1[i] and not valid2[i]:
                mol_tokens.append(mol_tokens1[graph1_pointer])
                graph1_pointer += 1
            elif valid2[i]:
                mol_tokens.append(mol_tokens2[graph2_pointer])
                graph2_pointer += 1

        mol_tokens = torch.cat(mol_tokens, dim=0)

        genes = samples['genes']

        device = genes.device
        genes = genes.unsqueeze(1).to(torch.float)

        query_token_gene = self.cell_query_tokens.expand(genes.shape[0], -1, -1)
        gene_attention_mask = torch.ones(genes.shape[0], genes.shape[1], dtype=torch.long).to(device)
        gene_output = self.cell_Qformer.bert(
            query_embeds=query_token_gene,
            encoder_hidden_states=genes,
            encoder_attention_mask=gene_attention_mask,
            return_dict=True,
        )
        gene_tokens = self.cell_proj(gene_output.last_hidden_state).flatten(0, 1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens
        prompt_embeds[prompt_tokens.is_cell_token] = gene_tokens


        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs

    @torch.no_grad()
    def generateCellLineTarget(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
    ):
        graphs1 = samples['graphs1']
        valid1 = graphs1["Valid"]
        env1 = graphs1["Transform"]
        graphs1 = graphs1["Graph"]
        target1 = samples['target1']

        if valid1.any():
            _, _, graph_embeds1, graph_masks1 = self.graph_encoder(graphs1, env1, target1) 
            graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1)

            query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
            query_output1 = self.Qformer.bert(
                query_embeds=query_tokens1,
                encoder_hidden_states=graph_embeds1,
                encoder_attention_mask=graph_masks1,
                return_dict=True,
            )
            mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        
        graphs2 = samples['graphs2']
        valid2 = graphs2["Valid"]
        env2 = graphs2["Transform"]
        graphs2 = graphs2["Graph"]
        target2 = samples['target2']

        if valid2.any():
            _, _, graph_embeds2, graph_masks2 = self.graph_encoder(graphs2, env2, target2) 
            graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2)

            query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens2,
                encoder_hidden_states=graph_embeds2,
                encoder_attention_mask=graph_masks2,
                return_dict=True,
            )
            mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        prompt_tokens = samples['prompt_tokens']
        mol_tokens = []
        graph1_pointer = 0
        graph2_pointer = 0
        assert(valid1.shape[0] == valid2.shape[0])
        for i in range(valid1.shape[0]):
            if valid1[i] and valid2[i]:
                mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                graph1_pointer += 1
                graph2_pointer += 1
            elif valid1[i] and not valid2[i]:
                mol_tokens.append(mol_tokens1[graph1_pointer])
                graph1_pointer += 1
            elif valid2[i]:
                mol_tokens.append(mol_tokens2[graph2_pointer])
                graph2_pointer += 1

        mol_tokens = torch.cat(mol_tokens, dim=0)

        genes = samples['genes']

        device = genes.device
        genes = genes.unsqueeze(1).to(torch.float)

        query_token_gene = self.cell_query_tokens.expand(genes.shape[0], -1, -1)
        gene_attention_mask = torch.ones(genes.shape[0], genes.shape[1], dtype=torch.long).to(device)
        gene_output = self.cell_Qformer.bert(
            query_embeds=query_token_gene,
            encoder_hidden_states=genes,
            encoder_attention_mask=gene_attention_mask,
            return_dict=True,
        )
        gene_tokens = self.cell_proj(gene_output.last_hidden_state).flatten(0, 1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens
        prompt_embeds[prompt_tokens.is_cell_token] = gene_tokens


        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs

    @torch.no_grad()
    def opt_qa(
        self, 
        samples,
        device='cpu',
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=16,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1.,
        output_scores=False,
        ):
        prompts = samples['prompts']
        prepared_prompts = []
        mol_list = []
        for p in prompts:
            text, smiles = smiles_handler(p, self.mol_token * self.num_query_token)
            prepared_prompts.append(text)
            mol_list.extend([mol_to_graph_data_obj_simple(s) for s in smiles])

        graphs1 = self.collater((mol_list[0],)).to(device)
        graphs2 = self.collater((mol_list[1],)).to(device)
        
        prompt_tokens = self.opt_tokenizer(prepared_prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        is_mol_token = prompt_tokens.input_ids == self.opt_tokenizer.mol_token_id
        prompt_tokens['is_mol_token'] = is_mol_token
        
        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
        graph_embeds1 = self.ln_graph(graph_embeds1)

        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        graph_embeds2 = self.ln_graph(graph_embeds2)

        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        mol_tokens=torch.cat([mol_tokens1,mol_tokens2],dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1).to(prompt_embeds.dtype)
        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs

    @torch.no_grad()  
    def cellline_qa(
        self, 
        samples,
        device='cpu',
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.1,
        output_scores=False,
        ):

        prompts = samples['prompts']
        prepared_prompts = []
        mol_ph = self.mol_token * self.num_query_token
        cell_ph = self.cell_token * self.num_query_token
        for p in prompts:
            smiles_prompt = smiles_handler(p, mol_ph)[0]
            smiles_prompt = cell_handler(smiles_prompt, cell_ph)
            prepared_prompts.append(smiles_prompt)

        prompt_tokens = self.opt_tokenizer(prepared_prompts, 
                                           return_tensors="pt", 
                                           padding='longest', 
                                           truncation=False, 
                                           return_attention_mask=True).to(device)
        
        is_mol_token = prompt_tokens.input_ids == self.opt_tokenizer.mol_token_id
        is_cell_token = prompt_tokens.input_ids == self.opt_tokenizer.cell_token_id
        prompt_tokens['is_mol_token'] = is_mol_token
        prompt_tokens['is_cell_token'] = is_cell_token

        graphs1 = samples['graphs1']
        valid1 = graphs1["Valid"]
        graphs1 = graphs1["Graph"]

        if valid1.any():
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) 
            graph_embeds1 = self.ln_graph(graph_embeds1)

            query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
            query_output1 = self.Qformer.bert(
                query_embeds=query_tokens1,
                encoder_hidden_states=graph_embeds1,
                encoder_attention_mask=graph_masks1,
                return_dict=True,
            )
            mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        
        graphs2 = samples['graphs2']
        valid2 = graphs2["Valid"]
        graphs2 = graphs2["Graph"]

        if valid2.any():
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2) 
            graph_embeds2 = self.ln_graph(graph_embeds2)

            query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens2,
                encoder_hidden_states=graph_embeds2,
                encoder_attention_mask=graph_masks2,
                return_dict=True,
            )
            mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        mol_tokens = []
        graph1_pointer = 0
        graph2_pointer = 0
        assert(valid1.shape[0] == valid2.shape[0])
        for i in range(valid1.shape[0]):
            if valid1[i] and valid2[i]:
                mol_tokens.append(torch.cat([mol_tokens1[graph1_pointer], mol_tokens2[graph2_pointer]], dim=0))
                graph1_pointer += 1
                graph2_pointer += 1
            elif valid1[i] and not valid2[i]:
                mol_tokens.append(mol_tokens1[graph1_pointer])
                graph1_pointer += 1
            elif valid2[i]:
                mol_tokens.append(mol_tokens2[graph2_pointer])
                graph2_pointer += 1

        mol_tokens = torch.cat(mol_tokens, dim=0)

        genes = samples['genes']

        device = genes.device
        genes = genes.unsqueeze(1).to(torch.float)

        query_token_gene = self.cell_query_tokens.expand(genes.shape[0], -1, -1)
        gene_attention_mask = torch.ones(genes.shape[0], genes.shape[1], dtype=torch.long).to(device)
        gene_output = self.cell_Qformer.bert(
            query_embeds=query_token_gene,
            encoder_hidden_states=genes,
            encoder_attention_mask=gene_attention_mask,
            return_dict=True,
        )
        gene_tokens = self.cell_proj(gene_output.last_hidden_state).flatten(0, 1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.to(prompt_embeds.dtype)
        prompt_embeds[prompt_tokens.is_cell_token] = gene_tokens.to(prompt_embeds.dtype)

        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs
        

    @torch.no_grad()  
    def cancer_qa(
        self, 
        samples,
        valid,
        device='cpu',
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1.,
        output_scores=False,
        ):
        prompts = samples['prompts']
        prepared_prompts = []
        mol_list = []
        for p in prompts:
            text, smiles = smiles_handler(p, self.mol_token * self.num_query_token)
            prepared_prompts.append(text)
            mol_list.extend([mol_to_graph_data_obj_simple(s) for s in smiles])

        if valid[0] and not valid[1]:
            mol_list.append(None)
        elif not valid[0] and valid[1]:
            mol_list.append(mol_list[0])
            mol_list[0] = None
        elif not valid[0] and not valid[1]:
            mol_list.append(None)
            mol_list.append(None)
        
        prompt_tokens = self.opt_tokenizer(prepared_prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        is_mol_token = prompt_tokens.input_ids == self.opt_tokenizer.mol_token_id
        prompt_tokens['is_mol_token'] = is_mol_token
        
        if valid[0]:
            graphs1 = self.collater((mol_list[0],)).to(device)
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
            graph_embeds1 = self.ln_graph(graph_embeds1)

            query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
            query_output1 = self.Qformer.bert(
                query_embeds=query_tokens1,
                encoder_hidden_states=graph_embeds1,
                encoder_attention_mask=graph_masks1,
                return_dict=True,
            )
            mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        if valid[1]:
            graphs2 = self.collater((mol_list[1],)).to(device)
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
            graph_embeds2 = self.ln_graph(graph_embeds2)

            query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
            query_output2 = self.Qformer.bert(
                query_embeds=query_tokens2,
                encoder_hidden_states=graph_embeds2,
                encoder_attention_mask=graph_masks2,
                return_dict=True,
            )
            mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        
        if valid[0] and valid[1]:
            mol_tokens = torch.cat([mol_tokens1, mol_tokens2], dim=1)
        elif valid[0] and not valid[1]:
            mol_tokens = mol_tokens1
        elif not valid[0] and valid[1]:
            mol_tokens = mol_tokens2          

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        if valid[0] or valid[1]:
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1).to(prompt_embeds.dtype)
        if not output_scores:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text
        else:
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs
