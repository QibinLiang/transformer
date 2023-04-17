#!/bin/bash

num_gpus=4
ddp_file=ckpt/ddp_init_char
use_ddp=true
token_level="char"
batch_size=16
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

for((i=0; i<$num_gpus; ++i)); do
{
  gpu_id=$i
  init_file=file://$(readlink -f $ddp_file)
  echo running on rank "$gpu_id"
  python train.py \
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
      --save_interval "$save_interval" \
      --use_ddp "$use_ddp" \
      --rank "$gpu_id" \
      --init_method "$init_file" \
      --world_size "$num_gpus"
} &
done
wait
