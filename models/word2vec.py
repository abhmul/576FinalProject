import random
from collections import defaultdict
import math

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pyjet.backend as J

from tqdm import tqdm


class Word2VecModel(nn.Module):
    """
    Class that represents the vanilla word2vec module. We'll swap this out with our experimental modules.

    this contains the actual trainable pytorch module and houses the parameters of the model. This model
    should not be trained directly, but rather through the `Word2Vec` class below.
    """

    def __init__(self, vocab_size, embedding_size):
        """
        Initializes a pytorch word2vec module.

        :param vocab_size: The size of the vocabulary of the corpus
        :param embedding_size: The dimension of the embeddings
        """
        super(Word2VecModel, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        # Use sparse for more memory efficient computations
        # Note that only SGD will work with sparse embedding layers on a GPU
        self._encoder = nn.Embedding(self.vocab_size, self.embedding_size, sparse=True)
        self._decoder = nn.Embedding(self.vocab_size, self.embedding_size, sparse=True)

        self._embedding_norms = None

    @staticmethod
    def token2tensor(tokens):
        """Helper to cast a numpy array of tokens to a torch LongTensor"""
        J.LongTensor([token.index for token in tokens]).view(*np.array(tokens).shape)

    def forward(self, input_tokens, ctx_tokens, neg_tokens):
        """
        Computes the forward pass of the pytorch module

        :param input_tokens: The tokens that are input into the model whose embeddings are being trained. Should be of
            shape (Batch Size x 1). This needs to have a shape parameter.
        :param ctx_tokens: The tokens that are in the context of the input tokens. These are the true labels (trying to
            predict 1 for these). Should be of shape (Batch Size x 1). This needs to have a shape parameter.
        :param neg_tokens: The sampled noise tokens. These are the false labels (trying to predict 0 for these). Should
            be of shape (Batch Size x Num Neg Samples). This needs to have a shape parameter.
        :return: The logits for the true and false predictions
        """
        # Quick sanity checks
        assert input_tokens.shape[1:] == (1,)
        assert ctx_tokens.shape[1:] == (1,)
        assert neg_tokens.ndim == 2
        assert neg_tokens.shape[0] == ctx_tokens.shape[0] == input_tokens.shape[0]

        word_embs = self._encoder(self.token2tensor(input_tokens))  # B x 1 x E
        ctx_embs = self._decoder(self.token2tensor(ctx_tokens))  # B x 1 x E
        neg_embs = self._decoder(self.token2tensor(neg_tokens))  # B x N x E

        pos_out = torch.bmm(word_embs, ctx_embs.transpose(1, 2))  # B x 1 x 1
        neg_out = torch.bmm(word_embs, neg_embs.transpose(1, 2))  # B x 1 x N

        return pos_out, neg_out

    @staticmethod
    def loss(true_logits, sampled_logits):
        """
        Computes the loss of the prediction of the network wrt some batch.

        :param true_logits: The predicted logits for the true labels
        :param sampled_logits: The predicted logits for the false labels
        :return: The loss of the predictions.
        """

        # cross-entropy(logits, labels)
        true_xent = F.binary_cross_entropy_with_logits(true_logits, J.ones(*true_logits.size()).long(),
                                                       size_average=False)
        sampled_xent = F.binary_cross_entropy_with_logits(sampled_logits, J.zeros(*sampled_logits.size()).long(),
                                                          size_average=False)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        return (true_xent + sampled_xent) / (len(true_xent) + len(sampled_xent))

    @staticmethod
    def predict(logits):
        """Defines how an actual prediction is computed using the logits."""
        return F.sigmoid(logits)

    def lookup(self, token):
        return self._encoder(token.index)

    def lookup_tokens(self, tokens):
        return self._encoder(self.token2tensor(tokens))

    @property
    def embedding_norms(self):
        if self._embedding_norms is None:
            self._embedding_norms = torch.norm(self._encoder.weight.data, p=2, dim=1)
        return self._embedding_norms

    def similarities(self, tensor):
        norm_tensor = tensor / torch.norm(tensor)
        return torch.matmul(self._encoder.weight.data / self.embedding_norms, norm_tensor)


class Token(object):
    """
    Class for storing token level information
    """

    def __init__(self, index, text, frequency=0):
        """
        Initialize the token for use.

        :param index: The id of the token
        :param text: The text of the token
        :param frequency: The total number of occurrences of the token in the corpus
        """
        self.index = index
        self.text = text
        self.frequency = frequency

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.index == other.index


class Word2Vec(object):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    If you're finished training a model (=no more updates, only querying)
    then switch to the :mod:`gensim.models.KeyedVectors` instance in wv

    # TODO: Need to add save() and load() methods
    The model can be stored/loaded via its `save()` and `load()` methods.

    """
    def __init__(self, sentences, embedding_size=200, learning_rate=0.25, min_learning_rate=0.0001, num_neg_samples=5,
                 batch_size=16, epochs=5, window_size=5, dynamic_window=True, min_count=0, subsample=1e-3, seed=None):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.

        :param sentences: an iterable of `sentences` where each `sentence` is a list of tokens (usually words)
        :param embedding_size: the dimensionality of the feature vectors.
        :param learning_rate: the initial learning rate (will linearly drop to `min_learning_rate` as training progresses)
        :param min_learning_rate: the minimum possible learning rate
        :param num_neg_samples: if > 0, negative sampling will be used, the int for negative specifies how many
            "noise words" should be drawn (usually between 5-20).
            Default is 5. If set to 0, no negative samping is used.
        :param batch_size: The number of sentences to pass through the model before updating the weights
        :param window_size: Window size around a token that defines the context of a token. `window_size` tokens to the
            the left and to the right are included in the context.
        :param dynamic_window: Whether or not to randomly decrease the window size when training like in the original
            word2vec implementation
        :param min_count: ignore all words with total frequency lower than this.
        :param subsample: Subsampling factor to reduce sampling of frequent words in batches
        :param seed: The seed for the random generator. Note that this is only for the python and numpy random
            generators and does not seed pytorch. As of now, pytorch is not being seeded
        """

        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.num_neg_samples = num_neg_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.window_size = window_size
        self.dynamic_window = dynamic_window
        self.min_count = min_count
        self.subsample = subsample

        # Seed the random number generator
        random.seed(seed)
        np.random.seed(random.randint(0, 2**32))

        # Build the vocab
        self.tokens, self.vocab, self.corpus_length = self.build_vocab(sentences, self.min_count)
        self.vocab_size = len(self.tokens)
        # Quick sanity check
        assert all(token.index == i for i, token in enumerate(self.tokens))
        assert len(self.tokens) == len(self.vocab) == self.vocab_size

        # Build the distribution for sampling for batches and negative sampling
        self.sampling_probs = self.build_sampling_distribution()
        self.unigram_probs = self.build_unigram_distribution()

        # Build the actual model
        self._model = Word2VecModel(self.vocab_size, self.embedding_size)

        self._optimizer = optim.SGD(self._model.parameters(), lr=self.learning_rate)

        # Train the model
        self.train(sentences)

    @staticmethod
    def build_vocab(sentences, min_count):
        word_freqs = defaultdict(int)
        corpus_length = 0

        # Do first pass through to collect frequencies
        print("Building vocabulary")
        for sentence in tqdm(sentences):
            corpus_length += 1
            for word in sentence:
                word_freqs[word] += 1

        # The vocabulary is mapped to id with most frequent being 1
        id2word = sorted((w for w in word_freqs if word_freqs[w] >= min_count), key=lambda k: word_freqs[k], reverse=True)
        tokens = np.array([Token(index=i, text=w, frequency=word_freqs[w]) for i, w in enumerate(id2word)])
        # Create a hashable set of the tokens
        vocab = {token.text: token for token in tokens}
        return tokens, vocab, corpus_length

    def build_sampling_distribution(self):
        total_words = float(sum(token.frequency for token in self.tokens))

        word_z = [token.frequency / total_words for token in self.tokens]
        # Sampling rate is based on distribution from original word2vec C
        # implementation. See
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        sampling_probs = np.empty(self.vocab_size)
        for token in self.tokens:
            if self.subsample != 0:
                token.subsample_prob = (math.sqrt(word_z[token.index] / float(self.subsample)) + 1) * float(
                    self.subsample) / word_z[token.index]
            else:
                sampling_probs[token.index] = 1.
            # Just a sanity check
            assert sampling_probs[token.index] <= 1.
        return sampling_probs

    def build_unigram_distribution(self):
        # Sampling rate is based on distribution from original word2vec C
        # implementation. See
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        unigram_probs = np.array([token.frequency ** (3/4) for token in self.tokens])
        # Normalize the probabilities
        unigram_probs = unigram_probs / np.sum(unigram_probs)
        return unigram_probs

    def sample_unigram_distribution(self, true_tokens):
        true_token_indices = np.array([token.index for token in true_tokens])
        # Make the true tokens unsampleable
        true_unigram_probs = self.unigram_probs[true_token_indices]
        self.unigram_probs[true_token_indices] = 0.
        # Renormalize
        masked_sum = self.unigram_probs.sum()
        self.unigram_probs /= masked_sum

        # Sample from the distribution without replacement
        sampled_tokens = np.random.choice(self.tokens, size=self.num_neg_samples, replace=False, p=self.unigram_probs)

        # Undo the masking of the distribution
        self.unigram_probs *= masked_sum
        self.unigram_probs[true_token_indices] = true_unigram_probs

        return sampled_tokens

    def generate_sg_batch(self, sentences):
        token_pairs = []
        sent_count = 0
        for sentence in tqdm(sentences, total=self.corpus_length):
            # Get the tokens of the words in the sentence and prune any words not in the vocabulary
            sent_tokens = [self.vocab[w] for w in sentence if w in self.vocab]
            # Prune the sentence based on subsampling
            sent_tokens = [token for token in sent_tokens if np.random.random() < self.sampling_probs[token.index]]

            # If the sentence is only one word or less, just skip it
            if len(sent_tokens) <= 1:
                continue

            # Turn the sentence into batches for the model
            for pos, token in enumerate(sent_tokens):
                # Modifier to the window size
                window_modifier = np.random.randint(self.window_size) if self.dynamic_window else 0
                # now go over all words from the (reduced) window
                start = max(0, pos - self.window_size + window_modifier)
                end = pos + self.window_size + 1 - window_modifier
                for pos2, ctx_token in enumerate(sent_tokens[start:end], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        token_pairs.append((token, ctx_token))

            # step the sentence count
            sent_count += 1

            if sent_count % self.batch_size == 0:
                # Construct the batch and yield
                yield self.create_sg_batch(token_pairs), self.batch_size
                # Reset the token pairs
                token_pairs = []

        # Sanity check
        assert sent_count == self.corpus_length

        # Yield again if we still have some samples left
        if len(token_pairs) != 0:
            yield self.create_sg_batch(token_pairs), sent_count % self.batch_size

    def create_sg_batch(self, token_pairs):
        input_token_batch = np.empty((len(token_pairs), 1), dtype='O')
        ctx_token_batch = np.empty((len(token_pairs), 1), dtype='O')
        neg_token_batch = np.empty((len(token_pairs), self.num_neg_samples), dtype='O')
        for i, (token, ctx_token) in enumerate(token_pairs):
            # Insert the token and context
            input_token_batch[i] = token
            ctx_token_batch[i] = ctx_token
            # Now sample the negative contrastive tokens
            neg_token_batch[i] = self.sample_unigram_distribution(true_tokens=[token, ctx_token])
        return input_token_batch, ctx_token_batch, neg_token_batch

    def train(self, sentences):
        num_sentences = self.corpus_length * self.epochs
        lr_step = (1 / num_sentences) * (self.learning_rate - self.min_learning_rate)
        # Run each epoch
        num_updates = 0
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch+1, self.epochs))
            # Run through each batch
            for (input_token_batch, ctx_token_batch, neg_token_batch), n_batch_sents in self.generate_sg_batch(
                    sentences):
                # Zero out the current gradient
                self._optimizer.zero_grad()
                # Do the forward pass
                true_logits, neg_logits = self._model(input_token_batch, ctx_token_batch, neg_token_batch)
                loss = self._model.loss(true_logits, neg_logits)
                # Do the backward pass
                loss.backward()
                # Step the optimizer
                self._optimizer.step()
                num_updates += 1

                # Anneal the learning rate
                self.learning_rate = max(self.min_learning_rate, self.learning_rate - n_batch_sents * lr_step)
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

    def most_similar(self, positive=tuple(), negative=tuple(), topn=10, restrict_vocab=None, indexer=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        If topn is False, most_similar returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        # Check that something was provided
        assert len(positive) + len(negative) > 0, "You must provide words to compare to!"

        if isinstance(positive, str):
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        if isinstance(negative, str):
            negative = [negative]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        input_words_weights = [(word, 1.0) if isinstance(word, str) else word for word in positive] + [
            (word, -1.0) if isinstance(word, str) else word for word in negative]
        input_words, input_weights = zip(*input_words_weights)

        # Get each word's respective tokens
        input_tokens = [self.vocab[word] for word in input_words]

        # compute the weighted average of all words
        mean_tensor = torch.matmul(J.Tensor(input_weights), self._model.lookup_tokens(input_tokens))
        similarities = self._model.similarities(mean_tensor)
        # If we don't need the topn, just return the vector
        if not topn:
            return similarities

        # Get the topk
        scores, indices = torch.topk(similarities, k=topn)
        return [(self.tokens[idx], score) for idx, score in zip(indices, scores)]
