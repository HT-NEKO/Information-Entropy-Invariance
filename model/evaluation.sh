#!/bin/bash
module load miniforge/24.1.2 gcc/11.1.0 cuda/11.1
module load cudnn/8.6.0_cuda11.x
source activate gau-alpha-pt-v2
export PYTHONUNBUFFERED=1

MAXLENGTHs=(128 256 512 1024 2048 4096)
EVALNORMALIZATIONs=("InfoScale" "softmax")
# MODELNAME="./models_WanJuan/24-12-13-13-42-11_len64_norm_hl6_normalizationsoftmax/checkpoint-36225"
MODELNAME="./models_WanJuan/24-12-03-19-57-58_len64_nonorm_hl6_normalizationsoftmax/checkpoint-6615"
for EVALNORMALIZATION in "${EVALNORMALIZATIONs[@]}"; do 
    for MAXLENGTH in "${MAXLENGTHs[@]}"; do 
        python evaluation.py \
            --model_name $MODELNAME \
            --max_length $MAXLENGTH \
            --eval_normalization $EVALNORMALIZATION
    done
done