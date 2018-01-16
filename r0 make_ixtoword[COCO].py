import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd


def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # function from Andre Karpathy's NeuralTalk
    print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        if nsents > 315098:
            print(nsents)
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    # ix = index
    ixtoword = {}
    ixtoword[0] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)

annotation_path = './data/results_COCO2014.token'


annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
captions = annotations['caption'].values
wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)
np.save('data/ixtoword', ixtoword)


