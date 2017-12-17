#!/usr/bin/env bash

STRING="Running vector exporter..."
PYTHON="/usr/bin/python3.5"
PROJECT_ROOT="/~/PycharmProjects/576FinalProject/"
EXPORT_SCRIPT="export.py"

# Fixes import errors, see https://stackoverflow.com/questions/24727582/running-python-script-from-bash-file-causes-import-errors
pushd . > /dev/null 2>&1
cd ${PROJECT_ROOT}

echo ${STRING}

# This is for the multilayer lstms
${PYTHON} ${EXPORT_SCRIPT} -m model_files/text8_lstm2_512_char_emb_gensim_decoders -e vectors/text8_lstm2_512_char_emb_gensim_decoders.txt
${PYTHON} ${EXPORT_SCRIPT} -m model_files/text8_pool_lstm2_512_char_emb_gensim_decoders -e vectors/text8_pool_lstm2_512_char_emb_gensim_decoders.txt
${PYTHON} ${EXPORT_SCRIPT} -m model_files/text8_cnn2_512_char_emb_gensim_decoders -e vectors/text8_cnn2_512_char_emb_gensim_decoders_vectors.txt


# This is for the models with gensim decoders
#${PYTHON} ${EXPORT_SCRIPT} -m model_files/text8_lstm_char_emb_gensim_decoders -e vectors/text8_lstm_char_emb_gensim_decoders.txt
#${PYTHON} ${EXPORT_SCRIPT} -m model_files/text8_pool_lstm_char_emb_gensim_decoders -e vectors/text8_pool_lstm_char_emb_gensim_decoders.txt
#${PYTHON} ${EXPORT_SCRIPT} -m model_files/text8_cnn_char_emb_gensim_decoders -e vectors/text8_cnn_char_emb_gensim_decoders.txt



# python3 export.py -m model_files/text8_gru_no_adam -e text8_gru_no_adam_vectors.txt
# python3 export.py -m model_files/text8_gru -e text8_gru_vectors_char_emb.txt
# python3 export.py -m model_files/text8_pool_gru_no_adam -e text8_pool_gru_no_adam_vectors.txt
# python3 export.py -m model_files/text8_pool_gru -e text8_pool_gru_vectors.txt

# python3 export.py -m model_files/text8_cnn -e text8_cnn_vectors_char_emb.txt


popd > /dev/null 2>&1
