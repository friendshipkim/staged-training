#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain-gpt
python gpt_pretrain.py \
    --tokenizer gpt2 \
    --input_dir /home/wk247/data/c4 \
    --output_dir /home/wk247/data/c4 \
    --train_dev_split 0.05 \
    --num_preprocessing_workers 12 \
    --data_type raw_text