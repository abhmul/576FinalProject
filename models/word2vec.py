import os
import sys
import threading
import time
import random
from collections import defaultdict
import math

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import gensim

import pyjet.backend as J
from pyjet.losses import bce_with_logits
from pyjet.models import SLModel

from tqdm import tqdm

import models.utils as utils

class Word2VecModel(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(Word2VecModel, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self._emb = nn.Embedding(self.vocab_size, self.embedding_size)
        self._decoder = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, x):




class Word2Vec(object):

    def __init__(self, sentences, embedding_size=200, learning_rate=0.2, num_neg_samples=100, batch_size=16,
                 concurrent_steps=12, window_size=5, min_count=0, subsample=1e-3, seed=None):
        # super(Word2Vec, self).__init__()
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.subsample = subsample

        # Seed the random number generator
        random.seed(seed)
        np.random.seed(seed)

        # Build the vocab
        self._id2word, self._word2id, self.word_freqs = self.build_vocab(sentences)
        self.vocab_size = len(self._id2word)

        # Build the distribution for sampling for batches and negative sampling
        self.sampling_probs = self.build_sampling_distribution()
        self.unigram_probs = self.build_unigram_distribution()

        # Build the actual model
        self._model = Word2VecModel(self.vocab_size, self.embedding_size)

        self._optimizer = optim.SGD(self._model.parameters(), lr=self.learning_rate)

        self.global_step = 0

    def nce_loss(self, true_logits, sampled_logits):

        assert len(true_logits) + len(sampled_logits) == self.batch_size

        # cross-entropy(logits, labels)
        true_xent = bce_with_logits(true_logits, J.ones(*true_logits.size()).squeeze(1).long(), size_average=False)
        sampled_xent = bce_with_logits(sampled_logits, J.zeros(*sampled_logits.size()).squeeze(1).long(),
                                       size_average=False)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        return (true_xent + sampled_xent) / self.batch_size

    @staticmethod
    def build_vocab(sentences):
        word_freqs = defaultdict(int)

        # Do first pass through to collect frequencies
        print("Building vocabulary")
        for sentence in tqdm(sentences):
            for word in sentence:
                word_freqs[word] += 1

        # The vocabulary is mapped to id with most frequent being 1
        id2word = list(sorted(word_freqs, key=lambda k: word_freqs[k], reverse=True))
        word2id = {word: i for i, word in enumerate(id2word)}
        return word2id, id2word, word_freqs

    def build_sampling_distribution(self):
        total_words = float(sum(self.word_freqs[w] for w in self.word_freqs))

        word_z = [self.word_freqs[w] / total_words for w in self._id2word]
        # Sampling rate is based on distribution from original word2vec C
        # implementation. See
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        sampling_probs = np.empty(self.vocab_size)
        for i, w in enumerate(self._id2word):
            sampling_probs[i] = (math.sqrt(word_z[i] / float(self.subsample)) + 1) * float(self.subsample) / word_z[i]
        return sampling_probs

    def build_unigram_distribution(self):
        # Sampling rate is based on distribution from original word2vec C
        # implementation. See
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        unigram_probs = np.array([self.word_freqs[w] ** (3/4) for w in self._id2word])
        # Normalize the probabilities
        prob_sum = float(sum(unigram_probs))
        unigram_probs = unigram_probs / prob_sum
        return unigram_probs



    def train(self, sentences, epochs=5):

        for epoch in range(epochs):
            for sentence in sentences:


    def train_sg_pair(self, word, context):
        pos_indicies = [self._word2id[context]]
        # Now sample the negative samples
        neg_indicies = [self.]




