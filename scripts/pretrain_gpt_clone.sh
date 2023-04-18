#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain-gpt
export CUDA_VISIBLE_DEVICES=0
python gpt_pretrain.py    \
    --save_prefix final_gpt2_base_bs512_lr0.0021_warmup3k_seqlen1024_gpu1 \
    --gpu_count 1 \
    --model gpt2 \
    --tokenizer gpt2 \
    --batch_size 16 \
    --grad_accum 32 \
    --lr 0.002132892651963921 \
    --warmup_steps 3000 \
    --train_steps 250000 \
    --val_every 50 \
    --val_batches 50 \
    --fp16 \
    --seqlen 1024 \
    --log_rate 10 \
    --num_workers 12 \
    --size GPT2_base \
    --random \
    --data_type raw_text \
    --input_dir /home/wk247/data/c4 \
    --output_dir /home/wk247/data/c4