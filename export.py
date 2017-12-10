import argparse

import models
from gensim.models.word2vec import Text8Corpus
from word2vec import Word2Vec

parser = argparse.ArgumentParser()
# Main arguments
parser.add_argument("-m", "--model_path", required=True, help="The path to the model to export")
parser.add_argument("-e", "--export_path", required=True, help="The path to the output file of vectors")

args = parser.parse_args()

# Load the model
w2v = Word2Vec.load(args.model_path, Text8Corpus("text8"))
# Export it
w2v.export_vectors(args.export_path)