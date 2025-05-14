#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.8 
module load cudnn/8.6.0_cuda11.x
source activate dna
export PYTHONUNBUFFERED=1
export num_gpu=4
data_path=/data/run01/scv6872/kwli/infoscale/LRB/variant_effect_causal_eqtl
# eval_len=1000
# eval_len=3000
# eval_len=5000
model_path=/data/home/scv6872/run/kwli/infoscale/DNABERT_2/output/variant_effect_causal_eqtl/softmax_CosScale-32/checkpoint-4400
# model_path=/data/home/scv6872/run/kwli/infoscale/DNABERT_2/output/variant_effect_causal_eqtl/softmax_CosScale-128/checkpoint-4400
lr=3e-5

echo "The provided data_path is $data_path/$eval_len"
# 
for seed in 42
do
    for data in covid
    do
        for eval_len in 1000 3000 5000
        do
            torchrun --nproc_per_node=${num_gpu} eval_kyw.py \
                --model_name_or_path $model_path \
                --data_path  $data_path/$eval_len \
                --kmer -1 \
                --model_max_length $eval_len \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 1 \
                --learning_rate ${lr} \
                --num_train_epochs 8 \
                --save_steps 200 \
                --fp16 \
                --output_dir output/variant_effect_causal_eqtl \
                --evaluation_strategy steps \
                --eval_steps 200 \
                --warmup_steps 50 \
                --logging_steps 100000 \
                --overwrite_output_dir False \
                --log_level info \
                --find_unused_parameters False \
                --kyw_norm_method "CosScale" \
                --kyw_norm_scale 32
        done
    done

done
