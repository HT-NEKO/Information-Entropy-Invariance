#!/bin/bash
module load miniforge gcc/11.1.0 cuda/11.8 
module load cudnn/8.6.0_cuda11.x
source activate dna
export PYTHONUNBUFFERED=1
export num_gpu=4
model_path=/data/run01/scv6872/kwli/infoscale/DNABERT_2/
data_path=/data/run01/scv6872/kwli/infoscale/LRB/variant_effect_causal_eqtl/500/
lr=3e-5

echo "The provided data_path is $data_path"
# 
for seed in 42
do
    for data in covid
    do
        torchrun --master_port 32000 --nproc_per_node=${num_gpu} train_kyw.py \
            --model_name_or_path $model_path/DNABERT_2_117M \
            --data_path  $data_path \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_virus_${data}_seed${seed} \
            --model_max_length 256 \
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
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --kyw_norm_method "CosScale" \
            --kyw_norm_scale 128
    done

done
