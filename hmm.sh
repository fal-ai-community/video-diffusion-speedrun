#!/bin/bash
# L=24
# D=1024
# bs=4
# lr=0.001
# compile=full
# dataset=real

L=16
D=1024
bs=2
lr=0.001
compile=block
dataset=real
cp="${1:-1}"
fs="${2:-1}"

[ -z "$HF_HUB_TOKEN" ] && echo "warning: HF_HUB_TOKEN is not set"
[ -z "$HF_HOME" ] && echo "warning: HF_HOME is not set"
export HF_HUB_USE_HF_TRANSFER=1
if [ -z "$WANDB_API_KEY" ]
then export WANDB_MODE=offline
else export WANDB_ENTITY=main-horse-org
fi

. .venv/bin/activate
export OMP_NUM_THREADS=32

if [ -z "$SLURM_JOB_ID" ]
then cmd='torchrun --standalone --nproc_per_node=4 train.py'
else cmd='./slurm.sh'
fi

$cmd \
    --batch_size $bs \
    --run_name lr${lr}_width$D \
    --num_epochs 100 \
    --learning_rate ${lr} \
    --max_steps 5004 \
    --evaluate_every 500 \
    --model_width $D \
    --model_depth $L \
    --model_head_dim 128 \
    --optimizer_type mup_adam \
    --lr_scheduler_type linear \
    --project_name openvid-diffusion-dbg \
    --dataset $dataset \
    --compile-strat $compile \
    --cp $cp --fs $fs
    # --profile-dir ./profile-train \
