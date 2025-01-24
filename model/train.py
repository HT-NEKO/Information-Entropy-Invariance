from datasets import load_dataset, DatasetDict, Dataset
import ipdb
import math
from datasets import load_dataset,DatasetDict
import ipdb
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments
from overlap_trainer import *
from transformers import DataCollatorWithPadding
import torch
from tqdm import tqdm
from my_gau_alpha_eval.modeling_gau_alpha import GAUAlphaForMaskedLM, GAUAlphaTokenizerFast, GAUAlphaConfig
from my_gau_alpha_eval.layer import GatedAttentionUnit
import argparse
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback
from my_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type = str,
    )
    parser.add_argument(
        "--max_position_embeddings",
        type = int,
        default=512
    )
    parser.add_argument(
        "--num_hidden_layers",
        type = int,
        default = 24
    )
    parser.add_argument(
        "--normalization",
        type = str,
        default = "softmax"
    )
    parser.add_argument(
        "--norm_method",
        type = str,
        default = "noCosScale"
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 128
    )
    parser.add_argument(
        "--CosScale_value",
        type = int,
        default = None,
    )
    args = parser.parse_args()
    return args


print("-------------------------------------------------------------------")
print_running_command()
args = parse_args()

dataset_name = args.dataset_name
max_position_embeddings = args.max_position_embeddings
num_hidden_layers = args.num_hidden_layers
normalization = args.normalization
norm_method = args.norm_method
batch_size = args.batch_size
CosScale_value = args.CosScale_value
assert norm_method != "noCosScale" or CosScale_value == None
assert dataset_name == "WanJuan"



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).cpu()
    labels = torch.tensor(labels).cpu()
    accuracy = (predictions.argmax(axis=-1) == labels).float().mean().item()
    return {'accuracy': accuracy}

def get_info_dict(config):
    info_dict = {"task":"train","dataset":dataset_name,
            "train_len":max_position_embeddings,
            "train_norm_method":{"name":norm_method},
            "train_normalization":{"name":normalization},}
    if norm_method == "CosScale":
        info_dict["train_norm_method"]["CosScale_value"] = CosScale_value
    return info_dict



full_dataset = load_dataset("json", data_files=f"../datasets/{dataset_name}_deal.json", split="train")
total_rows = len(full_dataset)
train_size = int(0.8 * total_rows)
dataset = DatasetDict({
    "train": full_dataset.select(range(train_size)),
    "validation": full_dataset.select(range(train_size, total_rows))
        .filter(lambda _, idx: idx not in [492, 2627, 1331, 1292, 1421, 419, 1122, 2184, 3024, 470], with_indices=True),

})


model_name = "../junnyu/chinese_GAU-alpha-char_L-24_H-768"

config = GAUAlphaConfig.from_pretrained(model_name)
config.my_info_dict=get_info_dict(config)
config.max_position_embeddings = max_position_embeddings
config.normalization=normalization
config.num_hidden_layers=num_hidden_layers
config.norm_method=norm_method
config.use_cache = (False)

tokenizer = GAUAlphaTokenizerFast.from_pretrained(model_name,config=config)


def tokenize_function(examples):
    outputs = tokenizer(examples["text"], padding="max_length", truncation=True, 
                        max_length=max_position_embeddings, return_special_tokens_mask=True)
    return outputs


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

model = GAUAlphaForMaskedLM.from_pretrained(model_name,config=config,ignore_mismatched_sizes=True)


num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

bsz=batch_size // num_gpus if num_gpus > 0 else batch_size
epochs=400


train_dataset=tokenized_datasets["train"]
eval_dataset=tokenized_datasets["validation"]

total_steps = (1.0*len(train_dataset) // batch_size) * epochs
warmup_steps = int(total_steps * 0.1)
print(f"warmup_steps: {warmup_steps}")

output_dir=f"./models_{dataset_name}/"+datetime.now().strftime("%y-%m-%d-%H-%M-%S")+"_len" + str(max_position_embeddings) + "_" + norm_method + "_hl" + str(num_hidden_layers) + "_normalization" + normalization
print(output_dir)
training_args = TrainingArguments(
    output_dir = output_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    save_strategy="epoch",
    # save_strategy="steps",

    evaluation_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_steps=1,
    # evaluation_strategy="no",

    logging_strategy="epoch",

    save_steps=0,
    save_total_limit=1,

    learning_rate=1e-5,
    weight_decay=0.01,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    num_train_epochs=epochs,
    save_safetensors=False,
    ddp_backend='nccl',
    # fp16=True,
    warmup_steps=warmup_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    
)

trainer.train()
torch.cuda.empty_cache()
with torch.no_grad():
    eval_results = trainer.evaluate()


eval_perplexity = math.exp(eval_results['eval_loss'])
print(f"Perplexity: {eval_perplexity}")
print(f"Accuracy: {eval_results['eval_accuracy']}")

