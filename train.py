import argparse

from gensim.corpora.wikicorpus import WikiCorpus

from models import MODEL_DICT
from word2vec import Word2Vec

parser = argparse.ArgumentParser()
# Main arguments
parser.add_argument("-c", "--corpus", help="The path to the corpus to train the model on.")
parser.add_argument("-m", "--model", help="The name of the model to train.")

# Optional arguments
parser.add_argument("--token_min_len", type=int, default=2, help="The minimum length token to keep.")
parser.add_argument("--token_max_len", type=int, default=15, help="The maximum length token to keep.")
parser.add_argument("--embedding_size", type=int, default=100, help="The size of the embeddings to train.")

# Test arguments
parser.add_argument("-t", "--test", action='store_true', default=False,
                    help="Runs the train script in debug.")
args = parser.parse_args()

# Get the model factory
model_func = MODEL_DICT[args.model]

# Get corpus
corpus = WikiCorpus(args.corpus, lemmatize=False, dictionary={}, token_min_len=args.token_min_len,
                    token_max_len=args.token_max_len).get_texts()


# Used for debugging purposes
class TestCorpus(object):

    def __init__(self, corpus_obj, num_docs):
        self.corpus_obj = corpus_obj
        self.num_docs = num_docs

    def __iter__(self):
        for i, doc in enumerate(self.corpus_obj):
            if i >= self.num_docs:
                break
            yield doc


if args.test:
    corpus = TestCorpus(corpus, 1000)

# Pass it to the trainer
Word2Vec(corpus, model_func, embedding_size=args.embedding_size, min_count=1, window_size=5, learning_rate=0.025)
