#!/usr/bin/env bash
python3 train.py -c text8 -m CNNWord2Vec -s model_files/text8_cnn --batch_size 500
python3 train.py -c text8 -m PoolGRUWord2Vec -s model_files/text8_pool_gru --batch_size 500
python3 train.py -c text8 -m GRUWord2Vec -s model_files/text8_gru --batch_size 500