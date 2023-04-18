#!/bin/bash

# stage -1: download the data
# stage 0: preprocess the data
# stage 1: train the model

stage=-1
stop_stage=2
token_level="char"
batch_size=6
lr=0.00005
epochs=350
d_model=512
n_layers=6
n_heads=8
dropout=0.1
d_ff=2048
accumulation_steps=8
max_len=5000
save_interval=4

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Download the data"
    python utils/download.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Preprocess the data"
    python utils/preprocessing.py --token_level "$token_level"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Train the model"
    # train step
    python train.py \
        --src_dict "data/wmt" \
        --tgt_dict "data/wmt" \
        --train_data "data/wmt" \
        --dev_data "data/wmt" \
        --token_level "$token_level" \
        --batch_size "$batch_size" \
        --lr "$lr" \
        --epochs "$epochs" \
        --d_model "$d_model" \
        --n_layers "$n_layers" \
        --n_heads "$n_heads" \
        --dropout "$dropout" \
        --d_ff "$d_ff" \
        --accumulation_steps "$accumulation_steps" \
        --max_len "$max_len" \
        --verbose_interval 100 \
        --save_interval "$save_interval"
fi
