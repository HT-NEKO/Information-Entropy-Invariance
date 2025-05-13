from train_kyw import *
import torch
import transformers
import os
from datetime import datetime
import torch.distributed as dist
from sklearn.metrics import roc_auc_score,average_precision_score
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(" ".join(sys.argv))

    model_max_length=training_args.model_max_length

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if training_args.output_dir in ["output/variant_effect_causal_eqtl","output/vep_pathogenic_coding"]:
        if model_max_length==1000:
            model_max_length=512    # 500
        elif model_max_length==3000:
            model_max_length=1024   # 1500
        elif model_max_length==5000:
            model_max_length=2048   # 2500
        elif model_max_length==7000:
            model_max_length=3072   # 3500
        elif model_max_length==10000:
            model_max_length=4096   # 5000
        else: assert 0
    else:
        assert 0
    training_args.model_max_length=model_max_length


    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token


    # 加载验证集
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    # sub_test_dataset = torch.utils.data.Subset(test_dataset, indices=range(5))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # 加载模型配置
    config = BertConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=test_dataset.num_labels
    )
    config.kyw_normalization = training_args.kyw_normalization
    config.kyw_norm_method = training_args.kyw_norm_method
    config.kyw_norm_scale = training_args.kyw_norm_scale


    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,  # 加载保存的checkpoint目录
        config=config,
        trust_remote_code=True
    )
    model.to(device)



    # LoRA 模型可选加载（如使用）
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=True,
        )
        model = get_peft_model(model, lora_config)



    # 构建 trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        eval_dataset=test_dataset,# sub_test_dataset,
        data_collator=data_collator,
    )



    # 执行验证
    metrics = trainer.evaluate()
    print("Evaluation results:")
    print(metrics)

    # Only save in the main process (rank 0)
    if not dist.is_initialized() or dist.get_rank() == 0:
        my_output_dir = f"results/{data_args.data_path.split('/')[-2]}/eval-results_{training_args.kyw_normalization}_{training_args.kyw_norm_method}{'' if training_args.kyw_norm_scale==None else '-'+str(training_args.kyw_norm_scale)}.json"
        os.makedirs(os.path.dirname(my_output_dir), exist_ok=True)
        with open(my_output_dir, "a") as f:
            json.dump(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), f)
            f.write("\n")
            json.dump(" ".join(sys.argv), f)
            f.write("\n")
            json.dump(metrics, f, indent=4)
            f.write("\n\n")
        print(f"Saved evaluation results to {my_output_dir}")
    # my_output_dir=f"{training_args.output_dir}/{training_args.kyw_normalization}_{training_args.kyw_norm_method}{'' if training_args.kyw_norm_scale==None else '-'+str(training_args.kyw_norm_scale)}"
    # # 保存结果
    # results_path = os.path.join(my_output_dir, "eval_results.json")
    # with open(results_path, "a") as f:
    #     json.dump(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),f)
    #     f.write("\n")
    #     json.dump(" ".join(sys.argv), f)
    #     f.write("\n")
    #     json.dump(metrics, f, indent=4)
    #     f.write("\n")
    #     f.write("\n")
    # print(f"Saved evaluation results to {results_path}")

if __name__ == "__main__":
    main()