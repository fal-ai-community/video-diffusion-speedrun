#!/bin/bash

source /home/ubuntu/py311cuda/bin/activate


## Sweep for lr

loglrs=(-8 -7 -6 -5 -4 -3 -2)

for loglr in ${loglrs[@]}; do
    lr=$(python -c "print(2 ** $loglr)")
    torchrun --nproc_per_node=8 train.py \
        --batch_size 8 \
        --run_name lr${lr}_width512 \
        --num_epochs 100 \
        --learning_rate ${lr} \
        --max_steps 5004 \
        --evaluate_every 500 \
        --model_width 512 \
        --model_depth 24 \
        --model_head_dim 128 \
        --optimizer_type mup_adam \
        --lr_scheduler_type linear \
        --project_name openvid-diffusion-sweeplr \
        --compile_models True 
        
done
