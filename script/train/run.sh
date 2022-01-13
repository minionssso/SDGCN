#!/bin/bash

# rest
#python train.py --model_name semmhsa --dataset Restaurants --emb_type bert --gcn_dropout 0.3 --head_num 4 --input_dropout 0.3 --lr 1e-4 --mhsa_dropout 0.1 --num_layers 2 --sem_srd 3 --syn_srd 1

# laptop
#python train.py --dataset Laptops --emb_type bert --gcn_dropout 0.4 --head_num 8 --input_dropout 0.1 --lr 1e-4 --mhsa_dropout 0.2 --num_layers 3 --sem_srd 2 --syn_srd 3

# * tweets
#python train.py --dataset Tweets --emb_type bert --gcn_dropout 0.4 --head_num 4 --input_dropout 0.3 --lr 1e-5 --mhsa_dropout 0.8 --num_layers 2 --seed 908 --sem_srd 5 --syn_srd 5
# best
python train.py --dataset Tweets --emb_type bert --gcn_dropout 0.5 --head_num 8 --input_dropout 0.3 --lr 1e-4 --mhsa_dropout 0.5 --num_layers 2 --sem_srd 1 --syn_srd 9



# MAMS
#python train.py --model_name dssgcn --dataset MAMS --emb_type bert --gcn_dropout 0.5 --head_num 4 --input_dropout 0.1 --lr 1e-4 --mhsa_dropout 0.6 --num_layers 1 --sem_srd 4 --syn_srd 5
