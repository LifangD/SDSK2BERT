#!/usr/bin/env bash
export HOME=/home/dlf/pyprojects/
export PROJECT_DIR=/home/dlf/pyprojects/BertDistillNLI
export LARGE_DIR=$HOME/pretrain_models/bert-large-uncased
export BASE_DIR=$HOME/pretrain_models/bert-base-uncased


python train_tea.py \
    --data_dir=/home/dlf/pyprojects/BertDistillNLI/dataset/mnli \
    --dataset="mnli" \
    --dev_name=dev_matched \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --num_train_epochs=10 \
    --warmup_proportion=0.05 \
    --gradient_accumulation_steps=2 \
    --max_seq_length=128 \
    --learning_rate=2e-5 \
    --bert_config_file=$LARGE_DIR/bert_config.json \
    --load_model=$LARGE_DIR/pytorch_model.bin \
    --vocab_file=$BASE_DIR/vocab.txt \
    --output_dir=$PROJECT_DIR/saved_models/mnli/test

#python train_tea.py --data_dir=/home/dlf/pyprojects/BertDistill/dataset/mnli
#--dataset="mnli"
#--dev_name=dev_matched
#--train_batch_size=32
#--eval_batch_size=32
#--num_train_epochs=10
#--warmup_proportion=0.05
#--gradient_accumulation_steps=2
#--max_seq_length=128
#--learning_rate=2e-5
#--bert_config_file=/home/dlf/pyprojects/pretrain_models/bert-large-uncased/bert_config.json
#--load_model=/home/dlf/pyprojects/pretrain_models/bert-large-uncased/pytorch_model.bin
#--vocab_file=/home/dlf/pyprojects/pretrain_models/bert-large-uncased/vocab.txt
#--output_dir=/home/dlf/pyprojects/saved_models/mnli/test

