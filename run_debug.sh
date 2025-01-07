#!/bin/bash

source /home/ubuntu/py311cuda/bin/activate

torchrun --nproc_per_node=8 train.py \
    --batch_size 2 \
    --run_name debug \
    --num_epochs 2 \
    --learning_rate 0.01 \
    --max_steps 100000 \
    --evaluate_every 2000 \
    --model_width 2048 \
    --model_depth 24 \
    --model_head_dim 128 \
    --optimizer_type mup_adam \
    --lr_scheduler_type linear \
    --project_name openvid-diffusion \
    --compile_models True \