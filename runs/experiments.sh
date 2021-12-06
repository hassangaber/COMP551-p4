 #!/bin/bash

# Project repository path
cd ~/fake-news-reasoning/code-acl/bias

# Random forrest model runs
sudo python3 main.py --model bow --dataset pomt --inputtype EVIDENCE_ONLY
sudo python3 main.py --model bow --dataset pomt --inputtype CLAIM_ONLY
sudo python3 main.py --model bow --dataset pomt --inputtype CLAIM_AND_EVIDENCE
sudo python3 main.py --model bow --dataset snes --inputtype CLAIM_AND_EVIDENCE
sudo python3 main.py --model bow --dataset snes --inputtype CLAIM_ONLY
sudo python3 main.py --model bow --dataset snes --inputtype EVIDENCE_ONLY

# BERT model runs
sudo python3 main.py --model bert --dataset snes --inputtype EVIDENCE_ONLY --batchsize 8 --lr 3e-6
sudo python3 main.py --model bert --dataset snes --inputtype CLAIM_AND_EVIDENCE --batchsize 8 --lr 3e-6

# LSTM model runs
sudo python3 main.py --model lstm --dataset snes --inputtype CLAIM_AND_EVIDENCE --batchsize 16 --lr 0.0001 --lstm_hidden_dim 128 --lstm_layers 2 --lstm_dropout 0.1
sudo python3 main.py --model lstm --dataset snes --inputtype EVIDENCE_ONLY --batchsize 16 --lr 0.0001 --lstm_hidden_dim 128 --lstm_layers 2 --lstm_dropout 0.1
sudo python3 main.py --model lstm --dataset pomt --inputtype CLAIM_AND_EVIDENCE --batchsize 16 --lr 0.0001 --lstm_hidden_dim 128 --lstm_layers 2 --lstm_dropout 0.1
sudo python3 main.py --model lstm --dataset pomt --inputtype EVIDENCE_ONLY --batchsize 16 --lr 0.0001 --lstm_hidden_dim 128 --lstm_layers 2 --lstm_dropout 0.1