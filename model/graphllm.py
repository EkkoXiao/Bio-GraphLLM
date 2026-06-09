import os
from typing import Any, Dict
import torch
from model.blip2opt import Blip2OPT
from model.blip2combined import Blip2OPTCombined
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class GraphLLM(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return super().on_save_checkpoint(checkpoint)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        self.previous_state_dict = {}
        self.previous_unchanged_keys = None
        if args.opt_model.find('galactica') >= 0:
            if hasattr(args, "combined") and args.combined:
                self.blip2opt = Blip2OPTCombined(
                    args.bert_name,
                    args.gin_num_layers,
                    args.gin_hidden_dim,
                    args.drop_ratio,
                    args.tune_gnn,
                    args.num_query_token,
                    args.cross_attention_freq,
                    args.llm_tune,
                    args.peft_dir,
                    args.opt_model,
                    args.prompt,
                    args=args,
                )
            else:
                self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, use_nas=args.NAS, args=args)
            self.tokenizer = self.blip2opt.opt_tokenizer
        else:
            from model.blip2llama import Blip2Llama

            if hasattr(args, "combined") and args.combined:
                raise NotImplementedError("Combined multitask mode currently supports Galactica/Blip2OPT backbones only.")
            self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune == 'lora', args.peft_dir, args.opt_model, args.prompt, use_nas=args.NAS, args=args)
            self.tokenizer = self.blip2opt.llm_tokenizer
        self.save_hyperparameters(args)
    
    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        if self.args.NAS and self.args.cell:
            encoder_params = []
            ag_params = []
            params = []

            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "graph_encoder" in name and "encoder" in name:
                        encoder_params.append(param)
                    elif "graph_encoder" in name and "ag" in name:
                        ag_params.append(param)
                    else:
                        params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": params, "lr": self.args.init_lr, "weight_decay": self.args.weight_decay},
                {"params": encoder_params, "lr": self.args.encoder_learning_rate, "weight_decay": self.args.encoder_weight_decay},
                {"params": ag_params, "lr": self.args.arch_learning_rate, "weight_decay": self.args.arch_weight_decay},
            ])

            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                self.args.max_epochs, 
                self.args.min_lr, 
                self.args.init_lr, 
                warmup_steps, 
                self.args.warmup_lr
            )

            return optimizer
        elif self.args.NAS:
            encoder_params = []
            ag_params = []
            params = []

            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "graph_encoder" in name and "supernet0" in name:
                        encoder_params.append(param)
                    elif "graph_encoder" in name and "ag" in name:
                        ag_params.append(param)
                    else:
                        params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": params, "lr": self.args.init_lr, "weight_decay": self.args.weight_decay},
                {"params": encoder_params, "lr": self.args.encoder_learning_rate, "weight_decay": self.args.encoder_weight_decay},
                {"params": ag_params, "lr": self.args.arch_learning_rate, "weight_decay": self.args.arch_weight_decay},
            ])

            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                self.args.max_epochs, 
                self.args.min_lr, 
                self.args.init_lr, 
                warmup_steps, 
                self.args.warmup_lr
            )

            return optimizer
        else:
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                self.args.init_lr, 
                weight_decay=self.args.weight_decay
            )
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer, 
                self.args.max_epochs, 
                self.args.min_lr, 
                self.args.init_lr, 
                warmup_steps, 
                self.args.warmup_lr
            )
            return optimizer

    def test_epoch_end(self, outputs):
        list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        if self.trainer.world_size > 1:  # Only gather when in distributed mode
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_targets = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)

            if self.global_rank == 0:
                all_predictions = [i for ii in all_predictions for i in ii]
                all_targets = [i for ii in all_targets for i in ii]
                self.save_predictions_test(all_predictions, all_targets)
        else:
            # For single GPU, no need for dist.all_gather_object
            self.save_predictions_test(predictions, targets)

    def save_predictions_test(self, predictions, targets):
        ## print("##### Save Prediction Test #####")
        assert len(predictions) == len(targets)
        predict_dir = 'result/' + self.args.filename
        os.makedirs(predict_dir, exist_ok=True)
        predict_file = os.path.join(predict_dir, f'predictions_epoch={self.current_epoch + 1}.txt')
        with open(predict_file, 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if hasattr(self.args, "combined") and self.args.combined:
            graphs1, graphs2, genes, task_ids, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2, 'genes': genes, 'task_ids': task_ids, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]
            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))
            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        elif self.args.cell and self.args.NAS:
            graphs1, graphs2, target1, target2, genes, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2, 'target1': target1, 'target2': target2, 'genes': genes, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]

            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))

            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        elif self.args.cell == True:
            graphs1, graphs2, genes, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2, 'genes': genes, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]

            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))

            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        else:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            return predictions, texts
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if hasattr(self.args, "combined") and self.args.combined:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return
            graphs1, graphs2, genes, task_ids, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2, 'genes': genes, 'task_ids': task_ids, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]
            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))
            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        elif self.args.cell and self.args.NAS:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            
            graphs1, graphs2, target1, target2, genes, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2, 'target1': target1, 'target2': target2, 'genes': genes, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]

            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))

            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        elif self.args.cell == True:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1, graphs2, genes, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2, 'genes': genes, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )

            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]

            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))

            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        elif self.args.DDI == True or self.args.double == True or self.args.SSI == True or self.args.cancer == True:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            return predictions, texts
        else:
            if dataloader_idx == 0:
                _, _, text_tokens = batch
                text_tokens=text_tokens
                batch_size = text_tokens.input_ids.shape[0]
                loss = self.blip2opt(batch)
                self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            elif dataloader_idx == 1:
                if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                    return 
                graphs,prompt_tokens, texts = batch
                samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
                predictions = self.blip2opt.generate(
                    samples, 
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    max_length=self.max_len,
                    min_length=self.min_len
                )
                return predictions, texts
            elif dataloader_idx == 2:
                reaction_tokens, _, _ = batch
                batch_size = reaction_tokens.input_ids.shape[0]
                loss = self.blip2opt.forward_reaction(batch)
                self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            else:
                raise NotImplementedError

    def validation_epoch_end(self, outputs):
        if self.current_epoch != 0:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            caption_outputs = outputs[1]
            list_predictions, list_targets = zip(*caption_outputs)
            predictions = [i for ii in list_predictions for i in ii]
            targets = [i for ii in list_targets for i in ii]
            if self.trainer.world_size > 1:  # Only gather when in distributed mode
                all_predictions = [None for _ in range(self.trainer.world_size)]
                all_targets = [None for _ in range(self.trainer.world_size)]
                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_targets, targets)

                if self.global_rank == 0:
                    all_predictions = [i for ii in all_predictions for i in ii]
                    all_targets = [i for ii in all_targets for i in ii]
                    self.save_predictions_test(all_predictions, all_targets)
            else:
                # For single GPU, no need for dist.all_gather_object
                self.save_predictions_test(predictions, targets)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        if isinstance(batch, list) and len(batch) == 2:
            molecule_batch, reaction_batch = batch
            batch_size = molecule_batch[-1].size(0)
            ###============== molecule Loss ===================###
            molecule_loss = self.blip2opt(molecule_batch)['loss']
            self.log("molecule loss", float(molecule_loss), batch_size=batch_size, sync_dist=True)
            
            ###============== reaction Loss ===================###
            reaction_loss = self.blip2opt.forward_reaction(reaction_batch)['loss']
            self.log("reaction loss", float(reaction_loss), batch_size=batch_size, sync_dist=True)

            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return molecule_loss + self.reaction_weight * reaction_loss
        elif isinstance(batch, list):
            batch_size = batch[-1].input_ids.size(0)
            ###============== Overall Loss ===================###
            loss = self.blip2opt(batch)

            self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return loss['loss']
        else:
            batch_size = batch[-1].input_ids.size(0)
            ###============== Overall Loss ===================###
            loss = self.blip2opt(batch)

            self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Blip2Stage2")
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=True)
        parser.add_argument('--no_batch_norm', type=bool, default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT Open Pre-trained Transformer
        parser.add_argument('--opt_model', type=str, default="modelscope/galactica-1.3b")
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')
        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)
        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)
        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        # cell line
        parser.add_argument('--cell_num_features', type=int, default=908)
        # Combined OOD branches are opt-in so existing checkpoints keep their original module keys.
        parser.add_argument('--combined_ddi_ood', type=str, default='none', choices=['none', 'dynas'],
                            help='DDI OOD graph encoder: none keeps V1.0 GIN, dynas enables DyNAS-DDI.')
        parser.add_argument('--combined_dsp_ood', type=str, default='none', choices=['none', 'disen'],
                            help='DSP OOD graph encoder: none keeps V1.0 GIN, disen enables OOD-GraphLLM disentangle/env branch.')
        parser.add_argument('--ddi_ood', type=str, default='none', choices=['none', 'dynas'],
                            help='Single-task DDI OOD graph encoder.')
        parser.add_argument('--dsp_ood', type=str, default='none', choices=['none', 'disen'],
                            help='Single-task DSP OOD graph encoder.')
        return parent_parser
