#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.8 
module load cudnn/8.6.0_cuda11.x
source activate bart-small
export PYTHONUNBUFFERED=1
export num_gpu=4
DATASET=multi_news
CHECKPOINT=
for SEQ_LEN in 1000 2000 3000
do
    torchrun --nproc_per_node=${num_gpu} --master_port=54000 run_summarization.py \
    --model_name_or_path $CHECKPOINT \
    --do_predict \
    --dataset_name $DATASET \
    --dataset_config_name "3.0.0" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --kyw_normalization "InfoScale" \
    --kyw_norm_method "CosScale" \
    --kyw_norm_scale 256 \
    --max_source_length $SEQ_LEN
done