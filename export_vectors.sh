#!/usr/bin/env bash

# This is for the models with gensim decoders
python3 export.py -m model_files/text8_gru_char_emb_gensim_decoders -e text8_gru_char_emb_gensim_decoders.txt
python3 export.py -m model_files/text8_pool_gru_char_emb_gensim_decoders -e text8_pool_gru_char_emb_gensim_decoders.txt



# python3 export.py -m model_files/text8_gru_no_adam -e text8_gru_no_adam_vectors.txt
# python3 export.py -m model_files/text8_gru -e text8_gru_vectors_char_emb.txt
# python3 export.py -m model_files/text8_pool_gru_no_adam -e text8_pool_gru_no_adam_vectors.txt
# python3 export.py -m model_files/text8_pool_gru -e text8_pool_gru_vectors.txt

# python3 export.py -m model_files/text8_cnn -e text8_cnn_vectors_char_emb.txt
