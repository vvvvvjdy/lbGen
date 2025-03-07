#!/usr/bin bash

accelerate launch --config_file configs/default.yaml --main_process_port 29999 training/train.py \
    --pretrain_model sd-legacy/stable-diffusion-v1-5 --resolution 512 \
    --train_batch_size 4 --gradient_accumulation_steps 1 \
    --max_train_steps 500  --loops 3 \
    --learning_rate 2e-5 --max_grad_norm 0.1 --lr_scheduler constant --lr_warmup_steps 0 \
    --output_dir output/lbGen \
    --gradient_checkpointing \
    --mixed_precision  fp16 --validation_prompts "goldfish" \
    --seed 42 --K 5 --lora_rank 128 \
    --total_step 50 --scheduler DDPM \
    --gan_loss_weights 1 --learning_rate_D 1e-5 --adam_beta1_D 0 --max_grad_norm_D 1 \
    --validation_steps 10 --pretrain_model_name sd_1_5 \
    --clip_reward_weight 1 --q_align_reward_weight 0.1 \
    --version_clip_model openai/clip-vit-large-patch14 \
    --d_epoch_start 1



