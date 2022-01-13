#!/bin/bash
python train.py \
--dataset Tweets \
--emb_type bert \
--gcn_dropout 0.4 \
--head_num 4 \
--input_dropout 0.3 \
--lr 1e-5 \
--mhsa_dropout 0.8 \
--num_layers 2 \
--seed 908 \
--sem_srd 5 \
--syn_srd 5 \
--hidden_dim 204 \
--rnn_hidden 204 \
--batch_size 32 \
--bert_lr 5e-5 \