#!/usr/bin/env bash
export HOME=/home/dlf/pyprojects/
export PROJECT_DIR=/home/dlf/pyprojects/BertDistillNLI
export LARGE_DIR=$HOME/pretrain_models/bert-large-uncased
export BASE_DIR=$HOME/pretrain_models/bert-base-uncased


python train_stu.py \
    --depth=2 \
    --data_dir=/home/dlf/pyprojects/BertDistillNLI/dataset/dnli \
    --dataset="dnli" \
    --dev_name=dev \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --num_train_epochs=10 \
    --log_step=200 \
    --warmup_proportion=0.05 \
    --gradient_accumulation_steps=1 \
    --max_seq_length=80 \
    --learning_rate=2e-5 \
    --bert_config_file=$BASE_DIR/bert_config.json \
    --vocab_file=$BASE_DIR/vocab.txt \
    --seed=0 \
    --output_dir=$PROJECT_DIR/saved_models/dnli/L-2_S-0_C \
    --load_model=$PROJECT_DIR/saved_models/dnli/L-4_S-0/best_model.pt
#--load_model=$BASE_DIR/pytorch_model.bin \





