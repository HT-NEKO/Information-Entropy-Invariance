import json
import argparse
import math
import ipdb
import numpy as np
import torch
from my_gau_alpha_eval.modeling_gau_alpha import GAUAlphaForMaskedLM, GAUAlphaTokenizerFast, GAUAlphaConfig
from my_gau_alpha_eval.layer import GatedAttentionUnit
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import json
import os
from datetime import datetime
import re
from my_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type = str,
    )
    parser.add_argument(
        "--max_length",
        type = int,
    )
    parser.add_argument(
        "--eval_norm_method",
        type = str,
        default = None,
    )
    parser.add_argument(
        "--eval_normalization",
        type = str,
        default = None,
    )
    parser.add_argument(
        "--dataset_name",
        type = str,
        default = "WanJuan",
    )
    parser.add_argument(
        "--algorithm",
        type = str,
        default = None,
    )
    parser.add_argument(
        "--CosScale_value",
        type = int,
        default = None,
    )
    
    parser.add_argument(    # rerope
        "--rerope_window_size",
        type = int,
        default = 64,
    )
    args = parser.parse_args()
    return args

print("-------------------------------------------------------------------")
print_running_command()
args = parse_args()

model_name = args.model_name
max_length = args.max_length
eval_norm_method = args.eval_norm_method
eval_normalization = args.eval_normalization
dataset_name = args.dataset_name
algorithm = args.algorithm
CosScale_value = args.CosScale_value
rerope_window_size = args.rerope_window_size

assert eval_norm_method in ["CosScale", "noCosScale",None]
assert eval_norm_method != "noCosScale" or CosScale_value == None
assert eval_normalization in ["InfoScale", "softmax", "softmax_plus",None]
assert dataset_name in ["WanJuan"]
assert algorithm in ["PI", "PoSE", "YaRN", "ALiBi", "StreamingLLM", "LM-Infinite", "ReRoPE", "Windowed-Attention",None]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, train_len, seq_len):
        self.data = []
        self.label = []
        with open(json_file, 'r', encoding='utf-8') as infile:
            for line_number, line in enumerate(tqdm(infile, desc="Loading Dataset")):
                if dataset_name == "WanJuan" and line_number not in [492, 2627, 1331, 1292, 1421, 419, 1122, 2184, 3024, 470]:
                        continue
                data = json.loads(line)
                sentence = data['text']
                input_ids = tokenizer(sentence, padding=True, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze(0)

                if input_ids.size(0) < seq_len:
                    repetitions = (seq_len // input_ids.size(0)) + 1
                    input_ids = (input_ids.repeat(repetitions))[:seq_len]
                else:
                    input_ids = input_ids[:seq_len]
                for i in range(1, seq_len+1):
                    input_ids_temp = input_ids.clone()
                    input_ids_temp[-i] = tokenizer.mask_token_id
                    self.data.append(input_ids_temp)
                    self.label.append(input_ids[-i])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def get_info_dict(config):
    info_dict = {"task":"eval", "task_version":"old-v1", "dataset":dataset_name, "model_name":model_name,
                 "train_len":config.max_position_embeddings, "seq_len":max_length,
                 "train_norm_method":config.norm_method, "train_normalization":config.normalization,
                 "eval_norm_method":{"name":eval_norm_method},
                 "eval_normalization":{"name":eval_normalization},}
    
    if eval_norm_method == "CosScale":
        info_dict["eval_norm_method"]["CosScale_value"] = CosScale_value

    info_dict["algorithm"] = algorithm
    if algorithm == "ReRoPE":
        info_dict["rerope_window_size"] = rerope_window_size

    return info_dict



config = GAUAlphaConfig.from_pretrained(model_name)

eval_norm_method = config.norm_method if eval_norm_method == None else eval_norm_method
eval_normalization = config.eval_normalization if eval_normalization == None else eval_normalization

if algorithm != None and algorithm[:4] == "yarn": algorithm = "yarn"

if CosScale_value == None and eval_norm_method == "CosScale":
    CosScale_value=config.my_info_dict["train_norm_method"]["CosScale_value"]


my_info_dict = get_info_dict(config)
config.my_info_dict = my_info_dict
config.max_position_embeddings = max_length
config.normalization = eval_normalization
config.norm_method = eval_norm_method

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAUAlphaForMaskedLM.from_pretrained(model_name, config=config).to(device)

model = torch.nn.DataParallel(model)
num_gpus = torch.cuda.device_count()
print(f"Total GPUs available: {num_gpus}")
print(f"GPUs used by DataParallel: {model.device_ids}")

tokenizer = GAUAlphaTokenizerFast.from_pretrained(model_name)

model.eval()
criterion = torch.nn.CrossEntropyLoss()

json_file = f"../datasets/{dataset_name}_deal_validation.json"
dataset = CustomDataset(json_file, train_len = my_info_dict["train_len"], seq_len=max_length)
batch_size = 32
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

total_correct = 0
total_predictions = 0
total_loss = 0
for batch in tqdm(dataloader):
    pt_inputs = {
        "input_ids": batch[0],
        "attention_mask": torch.ones(batch[0].shape, dtype=torch.long)
    }
    with torch.no_grad():
        outputs=model(**pt_inputs)
        pt_outputs = outputs.logits

    mask_positions = (pt_inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    for i, input_ids in enumerate(pt_inputs['input_ids']):  # bsz
        true_label_my=batch[1][i]
        predicted_ids = torch.argmax(pt_outputs[i], dim=-1)
        mask_index = mask_positions[1][i].item()
        total_predictions += 1
        if predicted_ids[mask_index] == true_label_my:
            total_correct += 1
    logits_at_mask = pt_outputs[mask_positions[0], mask_positions[1], :]
    mask_labels = batch[1].to(logits_at_mask.device)
    loss = criterion(logits_at_mask.view(-1, logits_at_mask.size(-1)), mask_labels.view(-1))
    total_loss+=loss


average_loss = total_loss / len(dataloader)
average_loss = average_loss.cpu().item()
perplexity = np.exp(average_loss)
accuracy = total_correct / total_predictions if total_predictions > 0 else 0
print(f"Average Loss: {average_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
print(f"Accuracy: {accuracy:.4f}")