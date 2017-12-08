import numpy as np
from numpy.linalg import norm
import cPickle

#requires a formatted text file of vectors -- TODO formatted how?
def store_word_vecs_dict(word_vecs_file_name, dict_file_name):
    """Store the word vectors given in a text file into a dict
    (after normalizing the vectors), and save the dict as a pickle
    file. The vectors are numpy float arrays.
    """
    with open(word_vecs_file_name, 'r') as word_vecs_file:
        line_num = 1
        word_vecs_dict = {}
        for line in word_vecs_file:
            # Skip over line 1 in the word2vec file -- why??
            if line_num != 1 or word_vecs_file_name.startswith('glove'):
                parts = line.split()
                word = parts[0]
                values = parts[1:]
                vector = np.array([float(v) for v in values])
                word_vecs_dict[word] = vector / norm(vector)

                if line_num % 100 == 0:
                    print 'Line', line_num, 'processed'

            line_num += 1
                    
        with open(dict_file_name, 'wb') as dict_file:
            cPickle.dump(word_vecs_dict, dict_file, cPickle.HIGHEST_PROTOCOL)

            
import os
if not os.path.isdir('dicts'):
    os.makedirs('dicts')

# Can choose either word2vec or GloVe vectors
word_vecs_file_name = 'GoogleNews-vectors-negative300.txt'#'glove/crawl/glove.840B.300d.txt'#
dict_file_name = 'dicts/word2vec-GoogleNews-vecs300-norm.pickle'#'dicts/glove-crawl840B-vecs300-norm.pickle'#
store_word_vecs_dict(word_vecs_file_name, dict_file_name)
