import gensim
from gensim.models import KeyedVectors

import numpy as np

def load_w2v_word_map(filepath, binary = True):
    print('Loading word2vec formatted embeddings from [{0}] with binary={1}'.format(filepath, binary))
    
    w2v_model = KeyedVectors.load_word2vec_format(filepath, binary=binary)
    
    dimensions = w2v_model.vector_size
    
    # set up an empty matrix
    embeddings_matrix = np.zeros(shape=(len(w2v_model.vocab),dimensions))
    
    word_map = {}
    word_vector_list = []
    for word in w2v_model.vocab:
        word_index = w2v_model.vocab[word].index
        word_map[word] = word_index
        # replace this row in our empty matrix
        embeddings_matrix[word_index, :] = w2v_model[word]
    
    # now we need to return a numpy 2D array of embeddings and a mapping from word to integer index...
    return word_map, embeddings_matrix