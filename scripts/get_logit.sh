#!/usr/bin/env bash



python get_logit.py \
    --depth=24 \
    --data_dir=/home/dlf/pyprojects/BertDistillNLI/dataset/dnli \
    --dataset="dnli" \
    --dev_name=dev \
    --test_name=test \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --max_seq_length=128 \
    --learning_rate=2e-5 \
    --bert_config_file=/home/dlf/pyprojects/pretrain_models/bert-large-uncased/bert_config.json \
    --vocab_file=/home/dlf/pyprojects/pretrain_models/bert-base-uncased/vocab.txt \
    --load_model=/home/dlf/pyprojects/BertDistillNLI/saved_models/dnli_best_model.pt