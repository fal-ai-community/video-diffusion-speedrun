lr=0.002
export TOKENIZERS_PARALLELISM=false
torchrun --nproc_per_node=8 train.py \
        --batch_size 2 \
        --run_name new-baseline-large-hs2 \
        --num_epochs 100 \
        --learning_rate ${lr} \
        --max_steps 80004 \
        --evaluate_every 2000 \
        --model_width 2048 \
        --model_depth 24 \
        --model_head_dim 128 \
        --optimizer_type mup_adam \
        --lr_scheduler_type linear \
        --project_name openvid-diffusion-tuned-exp \
        --compile_models True 
        