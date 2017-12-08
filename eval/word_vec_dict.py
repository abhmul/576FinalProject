import cPickle
from scipy.spatial.distance import cosine, euclidean
from operator import itemgetter
import heapq
from numpy.linalg import norm
import numpy as np

class WordVecDict:

    def __init__(self):
        self.word_vecs_dict = {}
        self.all_words = []
        self.all_vecs = []

    def generate_dict(vocabulary):
        """
        Because we don't train on a predetermined vocabulary, we need a method for generating a dictionary of words to vectors,
        given some arbitrary vocabulary. 

        Pass each character sequence (word in vocabulary) through the RNN layer of our model (encoder) and return the dictionary.
        We can utilize this dictionary for evaluation and analysis.

        Perhaps let this take vocabulary filename instead?
        """
        #TODO implement
        #self.word_vecs_dict = generate


        self.all_words = np.array(self.word_vecs_dict.keys())
        self.all_vecs = np.array(self.word_vecs_dict.values()).T #why transpose?

    def has_dict(self):
        return self.word_vecs_dict != None


    def has_words(self, *words):
        """Determine whether all given words are in the dictionary."""
        for word in words:
            if word not in self.word_vecs_dict:
                print word, 'is not in the dictionary'
                return False
        return True

    def get_word_vec(self, word):
        return self.word_vecs_dict[word]

    def get_relation_vec(self, pair):
        """Given a pair of words, return the vector representing the relation between them."""
        vec1 = self.word_vecs_dict[pair[0]]
        vec2 = self.word_vecs_dict[pair[1]]
        return vec1 - vec2

    def relational_sim(self, pair1, pair2, method='cosine'):
        """Calculate the relational similarity between two pairs of words."""
        rel_vec1 = self.get_relation_vec(pair1)
        rel_vec2 = self.get_relation_vec(pair2)
        if method == 'cosine':
            similarity = 1 - cosine(rel_vec1, rel_vec2)
        elif method == 'euclidean':
            similarity = 1 - euclidean(rel_vec1, rel_vec2)
        return similarity

    def get_d_vec(self, word_a, word_b, word_c):
        """Given A:B::C:?, return the vector representing word D."""
        vec_a = self.word_vecs_dict[word_a]
        vec_b = self.word_vecs_dict[word_b]
        vec_c = self.word_vecs_dict[word_c]
        vec_d = vecB - vecA + vecC #are there other arithmetic operations to do?
        return vec_d

    def get_closest_words(self, vec, num_results=50, similarity='cosine', remove_words=[]):
        if similarity == 'cosine':
            vec_norm = vec / norm(vec)
            all_sims = np.dot(vec_norm, self.all_vecs)
        elif similarity == 'euclidean':
            all_sims = 1 - norm(vec - self.all_vecs.T, axis = 1)

        # Remove remove_words from the list of results
        masks = [self.allWords != word for word in remove_words]
        init_mask = np.ones(len(allSims), dtype='bool')
        mask = reduce(np.logical_and, masks, init_mask)

        # Attach each similarity to the correct word
        word_sims = zip(self.all_words[mask], all_sims[mask])

        # Get the words with the highest similarities
        best_word_sims = heapq.nlargest(num_results, word_sims, key=itemgetter(1))

        return best_word_sims

    def get_analogy_completions(self, word_a, word_b, word_c, num_results=50, method='cosine', w=None):
        """Given A:B::C:?, return the top num_results completions for the D term."""

        # Get an imagined D vector and compute its distances to all words in the dictionary
        vec_d = self.get_d_vec(word_a, word_b, word_c)

        if method == 'cosine':
            # Calculate cosine similarities between vector D and all word vectors
            vec_d_norm = vec_d / norm(vec_d)
            all_sims = np.dot(vec_d_norm, self.all_vecs)
        else:
            all_sims = 1 - norm(vec_d - self.allVecs.T, axis = 1)

        # Remove words A, B, and C from the list of results
        mask = np.logical_and(self.all_words != word_a, np.logical_and(self.all_words != word_b,
            self.all_words != word_c))

        # Attach each similarity to the correct word
        word_sims = zip(self.allWords[mask], allSims[mask])

        best_word_sims = heapq.nlargest(num_results, word_sims, key=itemgetter(1))

        return best_word_sims

