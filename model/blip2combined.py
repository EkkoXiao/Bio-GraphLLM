import logging
import re

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch_geometric.loader.dataloader import Collater
from transformers import AutoTokenizer, OPTForCausalLM

from model.blip2 import Blip2Base, disabled_train


CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


def _insert_split_marker(m: re.Match):
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", r"|\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}|{end_token}"


def escape_custom_split_sequence(text):
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


def smiles_handler(text, mol_ph):
    text = CUSTOM_SEQ_RE.sub(r"\1\3\4%s" % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text


def cell_handler(text, cell_ph):
    return re.sub(r"\[START_CELL\]\[END_CELL\]", f"[START_CELL]{cell_ph}[END_CELL]", text)


class Blip2OPTCombined(Blip2Base):
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune="freeze",
        peft_dir="",
        opt_model="modelscope/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args
        self.num_query_token = num_query_token
        self.collater = Collater([], [])

        # DDI-specific graph branch
        self.graph_encoder_ddi, self.ln_graph_ddi = self.init_graph_encoder(
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio,
            not self.args.no_batch_norm,
        )
        if not tune_gnn:
            for _, param in self.graph_encoder_ddi.named_parameters():
                param.requires_grad = False
            self.graph_encoder_ddi = self.graph_encoder_ddi.eval()
            self.graph_encoder_ddi.train = disabled_train

        self.Qformer_ddi, self.query_tokens_ddi = self.init_Qformer(
            bert_name,
            num_query_token,
            self.graph_encoder_ddi.num_features,
            cross_attention_freq,
        )

        # DSP-specific graph branch
        self.graph_encoder_dsp, self.ln_graph_dsp = self.init_graph_encoder(
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio,
            not self.args.no_batch_norm,
        )
        if not tune_gnn:
            for _, param in self.graph_encoder_dsp.named_parameters():
                param.requires_grad = False
            self.graph_encoder_dsp = self.graph_encoder_dsp.eval()
            self.graph_encoder_dsp.train = disabled_train

        self.Qformer_dsp, self.query_tokens_dsp = self.init_Qformer(
            bert_name,
            num_query_token,
            self.graph_encoder_dsp.num_features,
            cross_attention_freq,
        )

        # DSP cell-line branch
        self.cell_Qformer, self.cell_query_tokens = self.init_Qformer(
            bert_name,
            num_query_token,
            self.args.cell_num_features,
            cross_attention_freq,
        )

        # Keep only query cross-attention blocks for all Qformers.
        for qformer in [self.Qformer_ddi, self.Qformer_dsp, self.cell_Qformer]:
            qformer.cls = None
            qformer.bert.embeddings.word_embeddings = None
            qformer.bert.embeddings.position_embeddings = None
            for layer in qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side="right")
        self.opt_tokenizer.add_special_tokens({"pad_token": "<pad>", "sep_token": "</s>"})
        self.opt_tokenizer.add_tokens("<mol>")
        self.opt_tokenizer.add_tokens("<cell>")
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]
        self.opt_tokenizer.cell_token_id = self.opt_tokenizer("<cell>", add_special_tokens=False).input_ids[0]

        if opt_model == "facebook/galactica-125m":
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
        else:
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))

        self.llm_tune = llm_tune
        if llm_tune == "lora":
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                    )
                self.opt_model = get_peft_model(self.opt_model, peft_config)
            self.opt_model.print_trainable_parameters()
        elif llm_tune == "freeze":
            for _, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == "full":
            pass
        else:
            raise NotImplementedError()

        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]

        self.opt_proj_ddi = nn.Linear(self.Qformer_ddi.config.hidden_size, self.opt_model.config.hidden_size)
        self.opt_proj_dsp = nn.Linear(self.Qformer_dsp.config.hidden_size, self.opt_model.config.hidden_size)
        self.cell_proj = nn.Linear(self.cell_Qformer.config.hidden_size, self.opt_model.config.hidden_size)

        self.prompt = prompt

    def _extract_graph_by_index(self, valid_mask, batched_graphs):
        out = {}
        if batched_graphs is None:
            return out
        graph_list = batched_graphs.to_data_list()
        ptr = 0
        for idx in range(valid_mask.shape[0]):
            if valid_mask[idx]:
                out[idx] = graph_list[ptr]
                ptr += 1
        return out

    def _encode_task_graphs(self, graph_dict, encoder, ln_graph, qformer, query_tokens, projector):
        if len(graph_dict) == 0:
            return {}

        sample_indices = list(graph_dict.keys())
        graph_batch = self.collater([graph_dict[i] for i in sample_indices]).to(query_tokens.device)
        graph_embeds, graph_masks = encoder(graph_batch)
        graph_embeds = ln_graph(graph_embeds, graph_masks)

        qtokens = query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = qformer.bert(
            query_embeds=qtokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks,
            return_dict=True,
        )
        mol_tokens = projector(query_output.last_hidden_state)

        token_map = {}
        for i, sample_idx in enumerate(sample_indices):
            token_map[sample_idx] = mol_tokens[i]
        return token_map

    def _build_mol_tokens(self, graphs1, graphs2, task_ids):
        valid1 = graphs1["Valid"]
        valid2 = graphs2["Valid"]
        graph_map1 = self._extract_graph_by_index(valid1, graphs1["Graph"])
        graph_map2 = self._extract_graph_by_index(valid2, graphs2["Graph"])

        ddi_graphs1 = {i: g for i, g in graph_map1.items() if task_ids[i].item() == 0}
        ddi_graphs2 = {i: g for i, g in graph_map2.items() if task_ids[i].item() == 0}
        dsp_graphs1 = {i: g for i, g in graph_map1.items() if task_ids[i].item() == 1}
        dsp_graphs2 = {i: g for i, g in graph_map2.items() if task_ids[i].item() == 1}

        ddi_tokens1 = self._encode_task_graphs(
            ddi_graphs1,
            self.graph_encoder_ddi,
            self.ln_graph_ddi,
            self.Qformer_ddi,
            self.query_tokens_ddi,
            self.opt_proj_ddi,
        )
        ddi_tokens2 = self._encode_task_graphs(
            ddi_graphs2,
            self.graph_encoder_ddi,
            self.ln_graph_ddi,
            self.Qformer_ddi,
            self.query_tokens_ddi,
            self.opt_proj_ddi,
        )
        dsp_tokens1 = self._encode_task_graphs(
            dsp_graphs1,
            self.graph_encoder_dsp,
            self.ln_graph_dsp,
            self.Qformer_dsp,
            self.query_tokens_dsp,
            self.opt_proj_dsp,
        )
        dsp_tokens2 = self._encode_task_graphs(
            dsp_graphs2,
            self.graph_encoder_dsp,
            self.ln_graph_dsp,
            self.Qformer_dsp,
            self.query_tokens_dsp,
            self.opt_proj_dsp,
        )

        mol_token_per_sample = {}
        batch_size = valid1.shape[0]
        for i in range(batch_size):
            task = task_ids[i].item()
            t1 = ddi_tokens1.get(i) if task == 0 else dsp_tokens1.get(i)
            t2 = ddi_tokens2.get(i) if task == 0 else dsp_tokens2.get(i)
            if t1 is not None and t2 is not None:
                mol_token_per_sample[i] = torch.cat([t1, t2], dim=0)
            elif t1 is not None:
                mol_token_per_sample[i] = t1
            elif t2 is not None:
                mol_token_per_sample[i] = t2

        mol_tokens = [mol_token_per_sample[i] for i in range(batch_size) if i in mol_token_per_sample]
        if len(mol_tokens) == 0:
            return None
        return torch.cat(mol_tokens, dim=0)

    def _inject_prompt_embeddings(self, prompt_tokens, graphs1, graphs2, genes, task_ids):
        mol_tokens = self._build_mol_tokens(graphs1, graphs2, task_ids)
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        if mol_tokens is not None:
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.to(prompt_embeds.dtype)

        dsp_mask = task_ids == 1
        if dsp_mask.any():
            device = prompt_embeds.device
            genes_dsp = genes[dsp_mask].to(device).unsqueeze(1).to(torch.float)
            query_token_gene = self.cell_query_tokens.expand(genes_dsp.shape[0], -1, -1)
            gene_attention_mask = torch.ones(genes_dsp.shape[0], genes_dsp.shape[1], dtype=torch.long).to(device)
            gene_output = self.cell_Qformer.bert(
                query_embeds=query_token_gene,
                encoder_hidden_states=genes_dsp,
                encoder_attention_mask=gene_attention_mask,
                return_dict=True,
            )
            gene_tokens = self.cell_proj(gene_output.last_hidden_state).flatten(0, 1)

            cell_mask = prompt_tokens.is_cell_token.clone()
            cell_mask[~dsp_mask.to(cell_mask.device)] = False
            prompt_embeds[cell_mask] = gene_tokens.to(prompt_embeds.dtype)

        return prompt_embeds

    def forward(self, batch):
        graphs1, graphs2, genes, task_ids, prompt_tokens, text_tokens = batch
        task_ids = task_ids.to(prompt_tokens.input_ids.device)

        prompt_embeds = self._inject_prompt_embeddings(prompt_tokens, graphs1, graphs2, genes, task_ids)

        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(prompt_embeds.device).fill_(-100)
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        return {"loss": outputs.loss}

    @torch.no_grad()
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
        graphs1 = samples["graphs1"]
        graphs2 = samples["graphs2"]
        genes = samples["genes"]
        task_ids = samples["task_ids"].to(samples["prompt_tokens"].input_ids.device)
        prompt_tokens = samples["prompt_tokens"]

        prompt_embeds = self._inject_prompt_embeddings(prompt_tokens, graphs1, graphs2, genes, task_ids)

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
            return [text.strip() for text in output_text]

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
            return_dict_in_generate=True,
        )
        return outputs
