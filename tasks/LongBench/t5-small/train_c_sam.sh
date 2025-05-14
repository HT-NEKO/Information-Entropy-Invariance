#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.8 
module load cudnn/8.6.0_cuda11.x
source activate bart-small
export PYTHONUNBUFFERED=1
export num_gpu=4
DATASET=SAMsum
torchrun --nproc_per_node=${num_gpu} --master_port=60600 run_summarization.py \
    --model_name_or_path /data/home/scv6872/run/kwli/infoscale/text-to-text-transfer-transformer-main/BART-SMALL/checkpoint/t5-small \
    --do_train \
    --do_eval \
    --dataset_name $DATASET \
    --dataset_config_name "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /data/home/scv6872/run/kwli/infoscale/text-to-text-transfer-transformer-main/BART-SMALL/t5_checkpoint/$DATASET \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --kyw_norm_method "CosScale" \
    --kyw_norm_scale 256