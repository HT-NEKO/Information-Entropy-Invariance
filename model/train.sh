#!/bin/bash
module load miniforge/24.1.2 gcc/11.1.0 cuda/11.1
module load cudnn/8.6.0_cuda11.x
source activate gau-alpha-pt-v2
export PYTHONUNBUFFERED=1


DATASETNAME="WanJuan"

python train_WanJuan.py \
    --dataset_name $DATASETNAME \
    --max_position_embeddings 64 \
    --num_hidden_layers 6 \
    --normalization "softmax" \
    --norm_method "CosScale" \
    --batch_size 128 \
    --CosScale_value 128

python train_WanJuan.py \
    --dataset_name $DATASETNAME \
    --max_position_embeddings 64 \
    --num_hidden_layers 6 \
    --normalization "softmax" \
    --batch_size 128
