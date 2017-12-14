import argparse

from gensim.models.word2vec import Text8Corpus

import models
from word2vec import Word2Vec

SEED = 2415

parser = argparse.ArgumentParser()
# Main arguments
parser.add_argument("-c", "--corpus", required=True, help="The path to the corpus to train the model on.")
parser.add_argument("-m", "--model", required=True, help="The name of the model to train. The available models are: " +
                                                         ", ".join(models.get_available_models()))

# Optional arguments
# parser.add_argument("--token_min_len", type=int, default=2, help="The minimum length token to keep.")
# parser.add_argument("--token_max_len", type=int, default=15, help="The maximum length token to keep.")

parser.add_argument("--embedding_size", type=int, default=100, help="The size of the embeddings to train.")
parser.add_argument("--learning_rate", type=float, default=0.025, help="The initial learning rate of the gradient " +
                                                                       "descent.")
parser.add_argument("--min_learning_rate", type=float, default=0.0001, help="The minimum possible learning rate of " +
                                                                            "the gradient descent.")
parser.add_argument("--num_neg_samples", type=int, default=5, help="Number of noise (negative) words to sample per " +
                                                                    "token pair")
parser.add_argument("--batch_size", type=int, default=100, help="The minimum number of token pairs to pass through " +
                                                                  "the model before updating the weights")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model for.")
parser.add_argument("--window_size", type=int, default=5, help="Window size around a token that defines the context " +
                                                               "of a token")
parser.add_argument("--no_dynamic_window", action="store_false", help="Turns of the dynamic window on the model.")
parser.add_argument("--min_count", type=int, default=5, help="Only keeps tokens with at least `min_count` frequency.")
parser.add_argument("--subsample", type=float, default=0.001, help="Subsampling constant for frequent words.")
parser.add_argument("--use_adam", action="store_true", help="Whether or not to use adam to optimize the model.")
parser.add_argument("--gensim_decoders", default="", help="Path to gensim w2v file for decoders. If loaded will " +
                                                          "freeze the decoders")
parser.add_argument("--num_encoder_layers", default=1, type=int, help="Number of encoding layers to use")

parser.add_argument("-s", "--save", default="", help="File path to save the trained model to.")

parser.add_argument("--trainable_char_embeddings", action="store_true", help="Makes the characters trainable.")

# Test arguments
parser.add_argument("-t", "--test", type=float, default=float('inf'),
                    help="Runs the train script only using this many documents from the dataset.")
args = parser.parse_args()

# Get the model factory
model_func = models.load_model_func(args.model)

# Get corpus
corpus = Text8Corpus(args.corpus)
                     #, lemmatize=False, dictionary={}, token_min_len=args.token_min_len,
                    #token_max_len=args.token_max_len)


# Used for regeneration purposes
class CorpusGen(object):

    def __init__(self, corpus_obj, num_docs):
        self.corpus_obj = corpus_obj
        self.num_docs = num_docs

    def __iter__(self):
        for i, doc in enumerate(self.corpus_obj):
            if i >= self.num_docs:
                break
            yield doc


corpus = CorpusGen(corpus, args.test)

# Pass it to the trainer
w2v = Word2Vec(corpus, model_func, embedding_size=args.embedding_size, learning_rate=args.learning_rate,
               min_learning_rate=args.min_learning_rate, num_neg_samples=args.num_neg_samples,
               batch_size=args.batch_size, epochs=args.epochs, window_size=args.window_size,
               dynamic_window=args.no_dynamic_window, min_count=args.min_count, subsample=args.subsample, seed=SEED,
               use_adam=args.use_adam, save_fname=args.save, trainable_char_embeddings=args.trainable_char_embeddings,
               gensim_decoders=args.gensim_decoders, num_encoder_layers=args)

# Save the model
if args.save:
    w2v.save(args.save)
