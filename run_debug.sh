#!/bin/bash

source /home/ubuntu/py311cuda/bin/activate


## Sweep for lr
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

loglrs=(-10 -9 -8 -7 -6 -5 -4)

for loglr in ${loglrs[@]}; do
    lr=$(python -c "print(2 ** $loglr)")
    torchrun --nproc_per_node=6 train.py \
        --batch_size 8 \
        --run_name lr${lr}_width512_power \
        --num_epochs 100 \
        --learning_rate ${lr} \
        --max_steps 6004 \
        --evaluate_every 400 \
        --model_width 512 \
        --model_depth 24 \
        --model_head_dim 128 \
        --optimizer_type mup_adam \
        --lr_scheduler_type power \
        --project_name openvid-diffusion-sweeplr \
        --compile_models True 
        
done
