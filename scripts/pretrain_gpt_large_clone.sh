#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain-gpt
export CUDA_VISIBLE_DEVICES=0,1
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python gpt_pretrain.py    \
    --save_prefix gpt2_large_bs512_lr0.0020_warmup3k_seqlen1024_2gpu \
    --gpu_count 2 \
    --model gpt2 \
    --tokenizer gpt2 \
    --batch_size 4 \
    --grad_accum 64 \
    --lr 0.002006911598778545 \
    --warmup_steps 3000 \
    --train_steps 250000 \
    --val_every 50 \
    --val_batches 50 \
    --fp16 \
    --seqlen 1024 \
    --log_rate 10 \
    --num_workers 8 \
    --size GPT2_large \
    --random \
    --data_type raw_text \
    --input_dir /home/wk247/data/c4 \
    --output_dir /home/wk247/data/c4