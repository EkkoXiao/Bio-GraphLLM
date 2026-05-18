import argparse
import datetime
import itertools
import gradio as gr
import pandas as pd
import warnings
import os

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import torch
import pytorch_lightning as pl
from model.moltc import MolTC
import re
from difflib import SequenceMatcher
import sys
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

def process_text(text):
    sentences = [s.strip() for s in re.split(r'\.', text) if s.strip()]
    filtered_sentences = []

    for sentence in sentences:
        if not filtered_sentences:
            filtered_sentences.append(sentence)
        else:
            duplicate = False
            for filtered in filtered_sentences:
                if SequenceMatcher(None, sentence, filtered).ratio() > 0.5:
                    duplicate = True
                    break
            if not duplicate:
                filtered_sentences.append(sentence)

    return '. '.join(filtered_sentences)

sys_prompt = ""

def main(args):
    device = f"cuda:{args.device}"
    llm_device = f"cuda:{args.llm_device}"

    desc_model = MolTC(args)
    ckpt = torch.load("all_checkpoints/ft_drugbankCancer_galactica_category_all/last.ckpt", map_location=device)
    desc_model.load_state_dict(ckpt['state_dict'], strict=False)

    args.category = -1

    extend_model = MolTC(args)
    ckpt = torch.load("all_checkpoints/ft_drugbankCancer_galactica_desc_all/last.ckpt", map_location=device)
    extend_model.load_state_dict(ckpt['state_dict'], strict=False)

    severity_model = MolTC(args)
    ckpt = torch.load("all_checkpoints/ft_drugbankCancer_galactica_severity_all/last.ckpt", map_location=device)
    severity_model.load_state_dict(ckpt['state_dict'], strict=False)
    
    desc_model.to(device)
    extend_model.to(device)
    severity_model.to(device)

    chat_model_path = "modelscope/vicuna-7b-v1.5"
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_path)
    chat_model = AutoModelForCausalLM.from_pretrained(chat_model_path)
    chat_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})

    chat_model.resize_token_embeddings(len(chat_tokenizer))
    
    chat_model.to(llm_device)

    dict_path = "./cancer/"
    category_df = pd.read_csv(dict_path + 'category_dict.csv')
    category_dict = pd.Series(category_df['Item'].values, index=category_df['Category']).to_dict()

    cancer_drug = pd.read_excel(dict_path + "propertyDrugbank.xlsx")
    drug_id_dict = pd.Series(cancer_drug['DrugBank ID'].values, index=cancer_drug['Common name'].str.lower()).to_dict()
    drug_dict = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['SMILES']))
    drug_dict_property = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['description']))

    target_df = pd.read_excel(dict_path + "targetDrugbank.xlsx")
    target_dict = target_df.groupby(target_df.columns[1])[target_df.columns[2]].agg(list).to_dict()

    def drug_interaction(drug1, drug2, 
                    drug1_property, drug1_target, drug1_smiles,
                    drug2_property, drug2_target, drug2_smiles):
        with open("log.txt", 'a') as log:
            log.write(f"[{datetime.datetime.now()}]\n")
            log.write(f"    #Drug1: {drug1}\n")
            log.write(f"    #Drug2: {drug2}\n")
            log.write(f"    #Drug1 Property: {drug1_property}\n")
            log.write(f"    #Drug1 Target: {drug1_target}\n")
            log.write(f"    #Drug1 SMILES: {drug1_smiles}\n")
            log.write(f"    #Drug2 Property: {drug2_property}\n")
            log.write(f"    #Drug2 Target: {drug2_target}\n")
            log.write(f"    #Drug2 SMILES: {drug2_smiles}\n")
        try:
            if not bool(re.search(r'[a-zA-Z]', drug1)) or not bool(re.search(r'[a-zA-Z]', drug2)):
                raise NotImplementedError
            if drug1_property == "":
                drug1_property = "" if drug_dict_property.get(drug_id_dict.get(drug1.lower()), "").strip().endswith("Overview") else drug_dict_property.get(drug_id_dict.get(drug1.lower()), "")
            if drug1_target == "":
                drug1_target = ",".join(target_dict.get(drug_id_dict.get(drug1.lower()), []))
            if drug2_property == "":
                drug2_property = "" if drug_dict_property.get(drug_id_dict.get(drug2.lower()), "").strip().endswith("Overview") else drug_dict_property.get(drug_id_dict.get(drug2.lower()), "")
            if drug2_target == "":
                drug2_target = ",".join(target_dict.get(drug_id_dict.get(drug2.lower()), []))
            valid = [True, True]
            if drug1_smiles == "":
                smiles1 = drug_dict.get(drug_id_dict.get(drug1.lower(),"NAN"), "Not Available")
            else:
                smiles1 = drug1_smiles
            if pd.isna(smiles1) or smiles1 == "Not Available":
                smiles1 = ""
                valid[0] = False
            if drug2_smiles == "":
                smiles2 = drug_dict.get(drug_id_dict.get(drug1.lower(),"NAN"), "Not Available")
            else:
                smiles2 = drug2_smiles
            if pd.isna(smiles2) or smiles2 == "Not Available":
                smiles2 = ""
                valid[1] = False
            ## 顺序 Drug 1/2
            prompt = f"</s> #Drug1 is {drug1}."
            if drug1_property != "": 
                prompt += f"[START_PROPERTY]{drug1_property}[END_PROPERTY]"
            if drug1_target != "": 
                prompt += f"[START_TARGET]{drug1_target}[END_TARGET]"
            if smiles1 != "": 
                prompt += f"[START_SMILES]{smiles1}[END_SMILES]"
            prompt += " </s>"
            prompt += f"</s> #Drug2 is {drug2}."
            if drug2_property != "": 
                prompt += f"[START_PROPERTY]{drug2_property}[END_PROPERTY]"
            if drug2_target != "": 
                prompt += f"[START_TARGET]{drug2_target}[END_TARGET]"
            if smiles2 != "": 
                prompt += f"[START_SMILES]{smiles2}[END_SMILES]"
            prompt += " </s>"
            desc_prompt = prompt + "What are the side effects of these two drugs?"

            output = desc_model.blip2opt.cancer_qa({'prompts': [desc_prompt]}, valid=valid, device=device, num_beams=1, output_scores=True)
            cat_token_dict = desc_model.blip2opt.cat_token_dict

            probabilities = torch.nn.functional.softmax(output["scores"][0][0], dim=-1)
            probabilities, indices = torch.topk(probabilities[50000:], 5)
            indices += 50000

            interaction = category_dict[cat_token_dict[indices[0].item()]]
            extend_prompt = prompt + interaction + " What additional information is available about this interaction?"

            output = extend_model.blip2opt.cancer_qa({'prompts': [extend_prompt]}, valid=valid, device=device, num_beams=1)
            extend_desc1 = process_text(output[0][:output[0].rfind('.')] + '.') + '.'

            severity_prompt = prompt + interaction + "What is the severity of this interaction?"
            output = severity_model.blip2opt.cancer_qa({'prompts': [severity_prompt]}, valid=valid, device=device, num_beams=1)
            severity1 = output[0].split('.')[0] + '.'

            interaction_dict1 = {category_dict[cat_token_dict[idx.item()]]: prob.item() for idx, prob in zip(indices, probabilities)}

            ## 逆序 Drug 1/2
            prompt = f"</s> #Drug1 is {drug2}."
            if drug2_property != "": 
                prompt += f"[START_PROPERTY]{drug2_property}[END_PROPERTY]"
            if drug2_target != "": 
                prompt += f"[START_TARGET]{drug2_target}[END_TARGET]"
            if smiles2 != "": 
                prompt += f"[START_SMILES]{smiles2}[END_SMILES]"
            prompt += " </s>"
            prompt += f"</s> #Drug2 is {drug1}."
            if drug1_property != "": 
                prompt += f"[START_PROPERTY]{drug1_property}[END_PROPERTY]"
            if drug1_target != "": 
                prompt += f"[START_TARGET]{drug1_target}[END_TARGET]"
            if smiles1 != "": 
                prompt += f"[START_SMILES]{smiles1}[END_SMILES]"
            prompt += " </s>"
            desc_prompt = prompt + "What are the side effects of these two drugs?"
            valid = valid[::-1]

            output = desc_model.blip2opt.cancer_qa({'prompts': [desc_prompt]}, valid=valid, device=device, num_beams=1, output_scores=True)
            cat_token_dict = desc_model.blip2opt.cat_token_dict

            probabilities = torch.nn.functional.softmax(output["scores"][0][0], dim=-1)
            probabilities, indices = torch.topk(probabilities[50000:], 5)
            indices += 50000

            interaction = category_dict[cat_token_dict[indices[0].item()]]
            extend_prompt = prompt + interaction + " What additional information is available about this interaction?"

            output = extend_model.blip2opt.cancer_qa({'prompts': [extend_prompt]}, valid=valid, device=device, num_beams=1)
            extend_desc2 = process_text(output[0][:output[0].rfind('.')] + '.') + '.'

            severity_prompt = prompt + interaction + "What is the severity of this interaction?"
            output = severity_model.blip2opt.cancer_qa({'prompts': [severity_prompt]}, valid=valid, device=device, num_beams=1)
            severity2 = output[0].split('.')[0] + '.'

            interaction_dict2 = {category_dict[cat_token_dict[idx.item()]]: prob.item() for idx, prob in zip(indices, probabilities)}

            interaction_dict = {}

            for key1, key2 in itertools.product(interaction_dict1, interaction_dict2):
                swapped_key2 = key2.replace("#Drug1", "__TEMP_DRUG1__").replace("#Drug2", "#Drug1").replace("__TEMP_DRUG1__", "#Drug2")
                
                if key1 != swapped_key2:
                    sorted_keys = sorted([key1, swapped_key2])
                    combined_key = f"{sorted_keys[0]}{sorted_keys[1]}"
                    
                    combined_key = combined_key.replace("#Drug1", drug1).replace("#Drug2", drug2)
                    interaction_dict[combined_key] = (interaction_dict1[key1] + interaction_dict2[key2]) / 2
                else:
                    combined_key = key1.replace("#Drug1", drug1).replace("#Drug2", drug2)
                    interaction_dict[combined_key] = (interaction_dict1[key1] + interaction_dict2[key2]) / 2

            extend_desc1 = extend_desc1.replace("#Drug1", "__TEMP_DRUG1__").replace("#Drug2", "#Drug1").replace("__TEMP_DRUG1__", "#Drug2")
            extend_desc2 = extend_desc2.replace("#Drug1", "__TEMP_DRUG1__").replace("#Drug2", "#Drug1").replace("__TEMP_DRUG1__", "#Drug2")
            extend_desc = "".join(extend_desc1.split(".")[:3]) + "".join(extend_desc2.split(".")[:3])
            extend_desc = extend_desc.replace("#Drug1", drug1).replace("#Drug2", drug2)
            severity = severity1 + "/" + severity2
            with open("log.txt", 'a') as log:
                log.write(f"    Interaction: {interaction_dict}\n")
                log.write(f"    Description: {extend_desc}\n")
                log.write(f"    Severity: {severity}\n")

            return interaction_dict, extend_desc, severity
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(f"Error type: {exc_type}")
            print(f"Error value: {exc_value}")
            print("Detail Traceback information:")
            traceback.print_tb(exc_tb)
            return "Processing Error! Please check your input and try again later.", "", ""

    with gr.Blocks(title="智能药物分子相互作用预测平台", theme=gr.themes.Soft(primary_hue="sky", font=["Palatino", "Arial", "sans-serif"], 
                    font_mono=["Palatino", "Arial", "sans-serif"]), css="footer {visibility: hidden}") as demo:
        with gr.Row():
            with gr.Column():
                drug1_input = gr.Textbox(label="药物1（请输入标准药物英文名称）", placeholder="Enter the name of Drug1", lines=1)
                drug1_smiles_input = gr.Textbox(label="药物1 SMILES 序列（可选，如不指定则查询数据库获取）", placeholder="Enter the SMILES of Drug1 (optional)", lines=1)
                drug1_property_input = gr.Textbox(label="药物1 药物性质 (可选，如不指定则查询数据库获取)", placeholder="Enter the properties of Drug1 (optional)", lines=1)
                drug1_target_input = gr.Textbox(label="药物1 靶点信息 (可选，如不指定则查询数据库获取)", placeholder="Enter the target information of Drug1 (optional)", lines=1)

            with gr.Column():
                drug2_input = gr.Textbox(label="药物2（请输入标准药物英文名称）", placeholder="Enter the name of Drug2", lines=1)
                drug2_smiles_input = gr.Textbox(label="药物2 SMILES 序列（可选，如不指定则查询数据库获取）", placeholder="Enter the SMILES of Drug2 (optional)", lines=1)
                drug2_property_input = gr.Textbox(label="药物2 药物性质 (可选，如不指定则查询数据库获取)", placeholder="Enter the properties of Drug2 (optional)", lines=1)
                drug2_target_input = gr.Textbox(label="药物2 靶点信息 (可选，如不指定则查询数据库获取)", placeholder="Enter the target information of Drug2 (optional)", lines=1)

        interact_btn = gr.Button("药物反应预测", variant="primary")
        
        output_label = gr.Label(num_top_classes=5, label="反应预测与可能性")
        with gr.Row():
            output_desc = gr.Textbox(label="反应补充信息", placeholder="Extend description of the most possible interaction", lines=3)
            output_severity = gr.Label(label="反应强度")

        def del_history(chat_history):
            chat_history.clear()
            return chat_history

        def on_click(drug1, drug2, 
                    drug1_property, drug1_target, drug1_smiles,
                    drug2_property, drug2_target, drug2_smiles,
                    chat_history):

            desc, extend_desc, severity = drug_interaction(drug1, drug2, 
                                                        drug1_property, drug1_target, drug1_smiles,
                                                        drug2_property, drug2_target, drug2_smiles)
            if not isinstance(desc, str):
                description = "\n".join([f"- {key}: {value*100:.2f}%" for key, value in desc.items()])

            global sys_prompt
            sys_prompt = "" if isinstance(desc, str) else f"""
Drug 1:
- Name: {drug1}
- Properties: {drug1_property if drug1_property else "None"}
- Target Information: {drug1_target if drug1_target else "None"}

Drug 2:
- Name: {drug2}
- Properties: {drug2_property if drug2_property else "None"}
- Target Information: {drug2_target if drug2_target else "None"}

Interaction Prediction:
- The five most likely interaction types and their confidence levels:
{description}

Additional Description:
- {extend_desc}

Reaction Severity:
- {severity}

You are a drug interaction prediction assistant. Above are some background knowledge, please infer the answer of the following questions and give explanations:
            """
            return desc, extend_desc, severity, chat_history
        
        def add_text(user_input, chat_history):
            chat_history.append((user_input, None))
            return chat_history
            
        def chat_with_bot(user_input, chat_history):
            if sys_prompt == "":
                chat_history.pop()
                chat_history.append((user_input, "Please specify the interaction drugs first!"))
            else:
                history_text = "\n".join([f"USER:{user}\nASSISTANT:{assistant}" for user, assistant in chat_history[:-1]])
                prompt = f"{sys_prompt}\n{history_text}\nUSER:{user_input}\nASSISTANT:"
                inputs = chat_tokenizer(prompt, return_tensors="pt").to(llm_device)

                with torch.no_grad():
                    outputs = chat_model.generate(inputs["input_ids"], max_new_tokens=1024, num_return_sequences=1, pad_token_id=chat_tokenizer.eos_token_id)
                
                generated_text = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
                assistant_reply = generated_text.split("ASSISTANT:")[-1].strip()
                chat_history.pop()
                with open('log.txt', 'a') as log:
                    log.write(f"    [USER]Input: {user_input}\n")
                    log.write(f"    [ASSISTANT]Reply: {assistant_reply}\n")
                chat_history.append((user_input, assistant_reply))


            return "", chat_history
        
        chatbot = gr.Chatbot(label="药物反应预测对话 (根据上述交互预测背景信息)", height=450, show_copy_button=True)
        with gr.Row():
            msg = gr.Textbox(show_label=False, placeholder="Please input your message", lines=1, interactive=True)
            send_btn = gr.Button(value="发送", variant="primary", scale=0)

        interact_btn.click(
            fn=del_history,
            inputs=chatbot,
            outputs=chatbot
        ).then(
            fn=on_click,
            inputs=[
                drug1_input, drug2_input,
                drug1_property_input, drug1_target_input, drug1_smiles_input,
                drug2_property_input, drug2_target_input, drug2_smiles_input,
                chatbot
            ],
            outputs=[
                output_label,
                output_desc,
                output_severity,
                chatbot
            ]
        )

        send_btn.click(fn=add_text, inputs=[msg, chatbot], outputs=chatbot
                       ).then(fn=chat_with_bot, inputs=[msg, chatbot], outputs=[msg, chatbot])

        

    demo.launch(share=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser = MolTC.add_model_specific_args(parser)
    parser.add_argument('--prompt', type=str, default='[START_SMILES]{}[END_SMILES]')
    parser.add_argument('--cancer', type=bool, default=True)
    parser.add_argument('--NAS', type=bool, default=False)
    parser.add_argument('--category', type=int, default=203)
    parser.add_argument('--device', type=int, default=6)
    parser.add_argument('--llm_device', type=int, default=0)
    args = parser.parse_args()
    args.tune_gnn = True
    args.llm_tune = "lora"
    return args

if __name__ == '__main__':
    main(get_args())
