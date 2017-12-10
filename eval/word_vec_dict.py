import pickle
from scipy.spatial.distance import cosine, euclidean
from operator import itemgetter
import heapq
from numpy.linalg import norm
import numpy as np
from functools import reduce
from gensim.models import KeyedVectors

class WordVecDict:

    def __init__(self):
        self.wv = None

    def get_dict(self):
        return self.word_vecs_dict

    def make_dict(self, dict_file_name, binary=True):
        """
        Because we don't train on a predetermined vocabulary, we need a method for generating a dictionary of words to vectors,
        given some arbitrary vocabulary. 

        Pass each character sequence (word in vocabulary) through the RNN layer of our model (encoder) and return the dictionary.
        We can utilize this dictionary for evaluation and analysis.

        Perhaps let this take vocabulary filename instead?
        """
        self.wv = KeyedVectors.load_word2vec_format(dict_file_name, binary=binary)

    def load_dict(self, dict_file_name):
        """Load the word-vector dictionary from the given pickle file."""
        with open(dict_file_name, 'rb') as dict_file:
            self.word_vecs_dict = pickle.load(dict_file)

        self.all_words = np.asarray(list(self.word_vecs_dict.keys()))

        vectors = list(self.word_vecs_dict.values())
        vectors = [w for w in vectors if w.shape == (300,)]

        self.all_vecs = np.asarray(vectors, dtype='float').T 
        print(self.all_vecs.shape)

    def has_dict(self):
        return self.wv != None

    def has_words(self, *words):
        """Determine whether all given words are in the dictionary."""
        # print(self.word_vecs_dict.keys()[:5])
        for word in words:
            if word not in self.wv:
                print(word, 'is not in the dictionary')
                return False
        return True

    def get_word_vec(self, word):
        return self.wv[word]

    def get_relation_vec(self, pair):
        """Given a pair of words, return the vector representing the relation between them."""
        vec1 = self.wv[pair[0]]
        vec2 = self.wv[pair[1]]
        return vec1 - vec2

    def get_d_vec(self, word_a, word_b, word_c):
        """Given A:B::C:?, return the vector representing word D."""
        vec_a = self.wv[word_a]
        vec_b = self.wv[word_b]
        vec_c = self.wv[word_c]
        vec_d = vec_b - vec_a + vec_c #are there other arithmetic operations to do?
        return vec_d

    def get_most_similar(self, word_a, word_b, word_c, topn):
        return self.wv.most_similar(positive=[word_c, word_b], negative=[word_a], topn=topn)

    def get_similarity_score(self, word1, word2):
        return self.wv.similarity(word1, word2)


    def check_top(self, top_words, key, num_to_check=3):
        for i in range(num_to_check):
            if key == top_words[i]:
                return True
        return False

