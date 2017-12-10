#!/usr/bin/env bash

python3 export.py -m model_files/text8_gru_no_adam -e text8_gru_no_adam_vectors.txt
python3 export.py -m model_files/text8_gru -e text8_gru_vectors.txt
python3 export.py -m model_files/text8_pool_gru_no_adam -e text8_pool_gru_no_adam_vectors.txt
python3 export.py -m model_files/text8_pool_gru -e text8_pool_gru_vectors.txt

python3 export.py -m model_files/text8_cnn -e text8_cnn_vectors.txt
