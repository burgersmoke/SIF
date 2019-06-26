import sys
import os

import sklearn
from sklearn.decomposition import TruncatedSVD
# give this a different alias so that it does not conflict with SPACY
from sklearn.externals import joblib as sklearn_joblib

import data_io, params, SIF_embedding
from SIF_embedding import get_weighted_average

# helper for word2vec format 
from data_io_w2v import load_w2v_word_map

import numpy as np

from past.builtins import xrange

# this is a modified version of getWeight() from data_io.py
# the main difference is that there is an option here to look up word keys as
# different casings since the default 
def getWeightAlternate(words, word2weight, 
        attempt_other_cases = True,
        report_oov = True):
    weight4ind = {}
    
    word_oov_set = set()
    word_remapping_set = set()
    
    for word, ind in words.items():
    
        word_key = word
        if attempt_other_cases:
            if word_key not in word2weight:
                word_lower = word.lower()
                word_capital = word.capitalize()
                word_upper = word.upper()
                
                # let's try these in order
                if word_lower in word2weight:
                    word_key = word_lower
                elif word_capital in word2weight:
                    word_key = word_capital
                elif word_upper in word2weight:
                    word_key = word_upper
    
        if word_key in word2weight:
            weight4ind[ind] = word2weight[word_key]
            
            if word != word_key:
                word_remapping_set.add(word)
            
        else:
            weight4ind[ind] = 1.0
            word_oov_set.add(word)
            
    if report_oov:
    
        print('Total words remapped : {}'.format(len(word_remapping_set)))
        print('Total word-weight OOV : {}'.format(len(word_oov_set)))
    
        #word_oov_sorted = sorted(list(word_oov_set))
        #print('Out of vocabulary with respect to word weighting:')
        #print(word_oov_sorted)
        
    return weight4ind

def get_weighted_average_alternate(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in xrange(n_samples):
        denom = np.count_nonzero(w[i,:])
        
        if denom <= 0.0:
            print('WHOA!  Sample [{0}] attempted to compute a denominator of : [{1}]'.format(i, denom))
        else:
            emb[i,:] = w[i,:].dot(We[x[i,:],:]) / denom
    return emb

# This class serves as a means of fitting an SIF model and then being able to transform other sentence vectors later
# This also allows save/loading model components via scikit-learn's joblib implementation
class SIFModel(object):
    def __init__(self):
        self.trained = False
        self.svd = None
        self.word_map = None
        self.params = params
        self.sentence_count = -1
        self.lowercase_tokens = False
        self.embeddings_filepath = None
        self.embeddings_format = None

    def transform(self, We, sentences):
        x, m = data_io.sentences2idx(sentences, self.word_map) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = data_io.seq2weight(x, m, self.weight4ind) # get word weights
        weighted_emb = get_weighted_average(We, x, w)
        # now use the model we've already loaded
        return self.remove_pc(weighted_emb)
        
    def compute_pc(self, X):
        # this is what happens in compute_pc() in src/SIF_embedding.py
        self.svd = TruncatedSVD(n_components=self.params.rmpc, n_iter=7, random_state=0)
        self.svd.fit(X)
        
    def remove_pc(self, X):
        pc = self.svd.components_
        
        if self.params.rmpc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
            
        return XX
        
    def fit(self, sentences, We, lowercase_tokens, embeddings_format, embeddings_filepath, params, word_map, weight4ind):
        
        # store these off for pickling or extra transforms
        self.word_map = word_map
        self.weight4ind = weight4ind
        self.params = params
        self.lowercase_tokens = lowercase_tokens
        self.embeddings_format = embeddings_format
        self.embeddings_filepath = embeddings_filepath
        
        self.sentence_count = len(sentences)
        
        x, m = data_io.sentences2idx(sentences, self.word_map) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = data_io.seq2weight(x, m, self.weight4ind) # get word weights
        
        # now let's do some of what happens in src/SIF_embedding.py
        # but also keep some pieces along the way
        #weighted_emb = get_weighted_average(We, x, w)
        weighted_emb = get_weighted_average_alternate(We, x, w)
        
        self.compute_pc(weighted_emb)
        
        self.trained = True
        
        return self.remove_pc(weighted_emb)
        
    @staticmethod
    def embedding_loading_helper(embeddings_filepath, embeddings_format):
        words = None
        We = None
        if embeddings_format == 'GLOVE':
            print('Loading embeddings as GLOVE')
            (words, We) = data_io.load_glove_word_map(embeddings_filepath)
        elif embeddings_format == 'WORD2VEC_BIN':
            (words, We) = load_w2v_word_map(embeddings_filepath, binary = True)
        elif embeddings_format == 'WORD2VEC_TXT':
            (words, We) = load_w2v_word_map(embeddings_filepath, binary = False)
        else:
            print('Unknown embeddings format : {}'.format(embeddings_format))
            
        return words, We