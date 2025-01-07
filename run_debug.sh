#!/bin/bash

source /home/ubuntu/py311cuda/bin/activate

torchrun --nproc_per_node=2 train.py \
    --batch_size 4 \
    --run_name debug \
    --dataset_url /home/ubuntu/cluster-simo-cluster-freepik/dataside/input_0.csv \
    --test_dataset_url /home/ubuntu/cluster-simo-cluster-freepik/dataside/input_head.csv \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --max_steps 10000 \
    --evaluate_every 20 \
    --model_width 512 \
    --model_depth 9 \
    --model_head_dim 128 \
    --batch_multiplicity 1 \
    --optimizer_type mup_adam \
    --lr_scheduler_type cosine \
    --project_name test_diffusion_test