#!/usr/bin/env bash

## This will run the models using a gensim decoder and more layers with lstms
python3 train.py -c text8 -m RNNWord2Vec -s model_files/text8_lstm5_512_char_emb_gensim_decoders --batch_size 500  --use_adam --gensim_decoders gensim_word2vec.w2v --trainable_char_embeddings --num_encoder_layers=5 --linear_size=512
python3 train.py -c text8 -m PoolRNNWord2Vec -s model_files/text8_pool_lstm5_512_char_emb_gensim_decoders --batch_size 500  --use_adam --gensim_decoders gensim_word2vec.w2v --trainable_char_embeddings --num_encoder_layers=5 --linear_size=512
python3 train.py -c text8 -m CNNWord2Vec -s model_files/text8_cnn5_512_char_emb_gensim_decoders --batch_size 500  --use_adam --gensim_decoders gensim_word2vec.w2v --trainable_char_embeddings --num_encoder_layers=5 --linear_size=512

## This will run the models using a gensim decoder
python3 train.py -c text8 -m RNNWord2Vec -s model_files/text8_gru_char_emb_gensim_decoders --batch_size 500  --use_adam --trainable_char_embeddings --gensim_decoders gensim_word2vec.w2v --use_gru
python3 train.py -c text8 -m PoolRNNWord2Vec -s model_files/text8_pool_gru_char_emb_gensim_decoders --batch_size 500 --use_adam --trainable_char_embeddings --gensim_decoders gensim_word2vec.w2v --use_gru
python3 train.py -c text8 -m CNNWord2Vec -s model_files/text8_cnn_char_emb_gensim_decoders --batch_size 500 --use_adam --trainable_char_embeddings --gensim_decoders gensim_word2vec.w2v

## This will run the models with trainable character embeddings
#python3 train.py -c text8 -m GRUWord2Vec -s model_files/text8_gru_char_emb --batch_size 500  --use_adam --trainable_char_embeddings
#python3 train.py -c text8 -m PoolGRUWord2Vec -s model_files/text8_pool_gru_char_emb --batch_size 500 --use_adam --trainable_char_embeddings
#python3 train.py -c text8 -m CNNWord2Vec -s model_files/text8_cnn_char_emb --batch_size 500 --use_adam --trainable_char_embeddings
#
#python3 train.py -c text8 -m GRUWord2Vec -s model_files/text8_gru_no_adam --batch_size 500 --trainable_char_embeddings
#python3 train.py -c text8 -m PoolGRUWord2Vec -s model_files/text8_pool_gru_no_adam --batch_size 500 --trainable_char_embeddings





