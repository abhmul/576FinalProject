import numpy as np
from numpy.linalg import norm
import pickle

#requires a formatted text file of vectors -- TODO formatted how?
def store_word_vecs_dict(word_vecs_file_name, dict_file_name):
    """Store the word vectors given in a text file into a dict
    (after normalizing the vectors), and save the dict as a pickle
    file. The vectors are numpy float arrays.
    """
    with open(word_vecs_file_name, 'r', encoding="utf-8") as word_vecs_file:
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

                if line_num % 100000 == 0:
                    print("Processed line: ", line_num)
                if line_num > 815380:
                    break

            line_num += 1
                    
        with open(dict_file_name, 'wb') as dict_file:
            pickle.dump(word_vecs_dict, dict_file, pickle.HIGHEST_PROTOCOL)
