import random
from collections import defaultdict
import math
import pickle

import numpy as np

import torch
import torch.optim as optim

import pyjet.backend as J

import time

from tqdm import tqdm

from gensim.models.word2vec import Word2Vec as gensim_Word2Vec


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

    The model can be stored/loaded via its `save()` and `load()` methods.

    """
    def __init__(self, sentences=None, model_func=None, embedding_size=100, learning_rate=0.025,
                 min_learning_rate=0.0001, num_neg_samples=5, batch_size=100, epochs=5, window_size=5,
                 dynamic_window=True, min_count=5, subsample=1e-3, use_adam=False, seed=None, save_fname="",
                 gensim_decoders=""):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.

        :param sentences: an iterable of `sentences` where each `sentence` is a list of tokens (usually words)
        :param embedding_size: the dimensionality of the feature vectors.
        :param learning_rate: the initial learning rate (will linearly drop to `min_learning_rate` as training
            progresses)
        :param min_learning_rate: the minimum possible learning rate
        :param num_neg_samples: if > 0, negative sampling will be used, the int for negative specifies how many
            "noise words" should be drawn (usually between 5-20).
            Default is 5. If set to 0, no negative samping is used.
        :param batch_size: The minimum number of token pairs to pass through the model before updating the weights
        :param window_size: Window size around a token that defines the context of a token. `window_size` tokens to the
            the left and to the right are included in the context.
        :param dynamic_window: Whether or not to randomly decrease the window size when training like in the original
            word2vec implementation
        :param min_count: ignore all words with total frequency lower than this.
        :param subsample: Subsampling factor to reduce sampling of frequent words in batches
        :param seed: The seed for the random generator. Note that this is only for the python and numpy random
            generators and does not seed pytorch. As of now, pytorch is not being seeded
        :param gensim_decoder: Preloads gensim decoders and freezes them in the model.
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
        self.use_adam = use_adam

        # Uninitialized variables
        self.tokens = None
        self.vocab = None
        self.corpus_length = None
        self.vocab_size = None
        self.char2id = None
        self.sampling_probs = None
        self.unigram_probs = None
        self._model_func = model_func
        self._model = None

        # Seed the random number generator
        random.seed(seed)
        np.random.seed(random.randint(0, 2**32))

        # If there are no sentences provided, the model is uninitialized
        if sentences is None:
            return

        # Build the vocab
        self.tokens, self.vocab, self.char2id, self.corpus_length, self.sentences = self.build_vocab(sentences, self.min_count)
        self.vocab_size = len(self.tokens)
        print("Vocabulary Size:", self.vocab_size)
        print("Character Vocabulary Size:", len(self.char2id))
        # Quick sanity check
        assert all(token.index == i for i, token in enumerate(self.tokens))
        assert len(self.tokens) == len(self.vocab) == self.vocab_size

        # Build the distribution for sampling for batches and negative sampling
        self.sampling_probs = self.build_sampling_distribution()
        self.unigram_probs = self.build_unigram_distribution()
        print("Built Sampling Distributions")

        # If no model is provided, don't train the word2vec
        if model_func is None:
            return

        # Build the actual model
        self._model = self.build_model(sparse=(not use_adam))
        print("Built", self._model.__class__.__name__, "with", "sparse" if not use_adam else "dense", "embeddings.")
        if gensim_decoders:
            self.load_decoders_from_gensim(gensim_decoders)

        # Optimizer
        self._optimizer = self.build_optimizer(use_adam)
        print("Built optimizer:", self._optimizer.__class__.__name__)

        # Train the model
        losses = self.train(self.sentences, save_fname=save_fname)
        if save_fname:
            with open(save_fname + "_losses.pkl", 'wb') as save_file:
                pickle.dump(losses, save_file)

    @staticmethod
    def build_vocab(sentences, min_count):
        word_freqs = defaultdict(int)
        corpus_length = 0

        # Do first pass through to collect frequencies
        print("Building vocabulary")
        mem_sentences = []
        for sentence in tqdm(sentences):
            mem_sentences.append(sentence)
            corpus_length += 1
            for word in sentence:
                word_freqs[word] += 1

        char2id = {}
        charids = 0
        charset = set()
        print("WARNING: The character and word id setup is no longer compatible with older models!")
        for word in word_freqs:
            # Get all the characters in the text
            for char in word:
                charset.add(char)
        for char in sorted(charset):
            char2id[char] = charids
            charids += 1

        # The vocabulary is mapped to id with most frequent being 1
        id2word = sorted((w for w in word_freqs if word_freqs[w] >= min_count), key=lambda k: (-word_freqs[k], k))
        print(id2word[:100])
        print(id2word[-100:])
        print(list(sorted([char for char in char2id], key=lambda k: char2id[k])))
        tokens = np.array([Token(index=i, text=w, frequency=word_freqs[w]) for i, w in enumerate(id2word)])
        # Create a hashable set of the tokens
        vocab = {token.text: token for token in tokens}
        return tokens, vocab, char2id, corpus_length, mem_sentences

    def build_optimizer(self, use_adam=False):
        params = [param for param in self._model.parameters() if param.requires_grad]
        if use_adam:
            print("Building Adam")
            return optim.Adam(params)
        print("Building SGD with learning rate", self.learning_rate)
        return optim.SGD(params, lr=self.learning_rate)


    def build_sampling_distribution(self):
        total_words = float(sum(token.frequency for token in self.tokens))

        word_z = [token.frequency / total_words for token in self.tokens]
        # Sampling rate is based on distribution from original word2vec C
        # implementation. See
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        sampling_probs = np.empty(self.vocab_size)
        for token in self.tokens:
            if self.subsample != 0:
                sampling_probs[token.index] = (math.sqrt(word_z[token.index] / float(self.subsample)) + 1) * float(
                    self.subsample) / word_z[token.index]
            else:
                sampling_probs[token.index] = 1.
            # Just a sanity check
            assert 0 < sampling_probs[token.index], sampling_probs[token.index]
        # print(sampling_probs)
        return sampling_probs

    def build_unigram_distribution(self):
        # Sampling rate is based on distribution from original word2vec C
        # implementation. See
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        unigram_probs = np.array([token.frequency ** (3/4) for token in self.tokens])
        # Normalize the probabilities
        unigram_probs = unigram_probs / np.sum(unigram_probs)
        return unigram_probs

    def build_model(self, sparse=True):
        return self._model_func(self.vocab_size, self.embedding_size, char2id=self.char2id, sparse=sparse)

    def generate_sg_batch(self, sentences):
        token_pairs = []
        for sentence in sentences:
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

                if len(token_pairs) >= self.batch_size:
                    # Construct the batch and yield
                    yield self.create_sg_batch(token_pairs)
                    # Reset the token pairs
                    token_pairs = []

        # Yield again if we still have some samples left
        if len(token_pairs) != 0:
            yield self.create_sg_batch(token_pairs)

    def generate_sg_pair(self, sentences):
        # Keep sampling when we need more negative samples
        neg_token_batch = np.random.choice(self.tokens, size=(self.batch_size, self.num_neg_samples),
                                           p=self.unigram_probs)
        neg_i = 0
        for sentence in sentences:
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
                        yield token, ctx_token, neg_token_batch[neg_i]

                        # Change the sampling
                        neg_i += 1
                        if neg_i == self.batch_size:
                            neg_token_batch = np.random.choice(self.tokens,
                                                               size=(self.batch_size, self.num_neg_samples),
                                                               p=self.unigram_probs)
                            neg_i = 0


    def train_sg_pair(self, token, ctx_token, neg_tokens):
        true_logits, neg_logits = self._model([token], [ctx_token], neg_tokens)
        true_xent, sampled_xent = self._model.loss(true_logits, neg_logits)
        # Do the backward pass
        (true_xent + sampled_xent).backward()
        # Step the optimizer
        self._optimizer.step()
        return true_xent, sampled_xent

    def create_sg_batch(self, token_pairs):
        # TODO: Make Numpy placeholders to fill for memory efficiency and speedup.
        input_token_batch, ctx_token_batch = zip(*token_pairs)
        input_token_batch = np.array(input_token_batch).reshape(len(input_token_batch), 1)
        ctx_token_batch = np.array(ctx_token_batch).reshape(len(ctx_token_batch), 1)
        # Now sample the negative contrastive tokens
        neg_token_batch = np.random.choice(self.tokens, size=(len(token_pairs), self.num_neg_samples), p=self.unigram_probs)
        return input_token_batch, ctx_token_batch, neg_token_batch

    def train(self, sentences, save_fname=""):
        # Variables for annealing the learning rate
        lr = self.learning_rate
        last_n = 0
        num_sentences = self.corpus_length * self.epochs
        lr_diff = self.learning_rate - self.min_learning_rate

        # Run each epoch
        num_updates = 0
        losses = []
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch+1, self.epochs))
            true_epoch_losses = []
            sampled_epoch_losses = []
            # Run through each batch
            progbar_sentences = tqdm(sentences, total=self.corpus_length)
            for input_token_batch, ctx_token_batch, neg_token_batch in self.generate_sg_batch(progbar_sentences):
                # Zero out the current gradient
                self._optimizer.zero_grad()
                # Do the forward pass
                true_logits, neg_logits = self._model(input_token_batch, ctx_token_batch, neg_token_batch)
                true_xent, sampled_xent = self._model.loss(true_logits, neg_logits)
                # Do the backward pass
                (true_xent + sampled_xent).backward()
                # Do some logging
                true_epoch_losses.append(true_xent.data[0] / len(input_token_batch))
                sampled_epoch_losses.append(sampled_xent.data[0] / len(input_token_batch))
                progbar_sentences.set_postfix({"true": np.average(true_epoch_losses[-10000:]),
                                               "sampled": np.average(sampled_epoch_losses[-10000:]),
                                               "lr": lr})
                # Step the optimizer
                self._optimizer.step()
                num_updates += 1

                # Anneal the learning rate every sentence if using sgd
                if progbar_sentences.n != last_n and not self.use_adam:
                    sent_passed = epoch * self.corpus_length + progbar_sentences.n
                    lr = max(self.min_learning_rate, self.learning_rate - (sent_passed / num_sentences) * lr_diff)
                    # print(self.learning_rate)
                    for param_group in self._optimizer.param_groups:
                        param_group['lr'] = lr
                    last_n = progbar_sentences.n

            # Log the epoch's loss
            print("True:", np.average(true_epoch_losses))
            print("Sampled:", np.average(sampled_epoch_losses))
            # If we want to save, save it
            if save_fname:
                self.save(save_fname)
                print("Saved file to", save_fname)
            losses.append((true_epoch_losses, sampled_epoch_losses))
        return losses

    def train2(self, sentences, save_fname=""):
        # Variables for annealing the learning rate
        lr = self.learning_rate
        last_n = 0
        num_sentences = self.corpus_length * self.epochs
        lr_diff = self.learning_rate - self.min_learning_rate

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch+1, self.epochs))
            true_epoch_loss = 0
            sampled_epoch_loss = 0
            # Run through each batch
            progbar_sentences = tqdm(sentences, total=self.corpus_length)
            step = 0
            for input_token, ctx_token, neg_tokens in self.generate_sg_pair(progbar_sentences):
                # a = time.time()
                true_xent, sampled_xent = self.train_sg_pair(input_token, ctx_token, neg_tokens)
                # print(time.time() - a)
                true_epoch_loss += true_xent
                sampled_epoch_loss += sampled_xent
                step += 1
                if step % 10000 == 0:
                    progbar_sentences.set_postfix({"true": true_epoch_loss.data[0] / step,
                                                   "sampled": sampled_epoch_loss.data[0] / step,
                                                   "lr": lr})

                # Anneal the learning rate every sentence if using sgd
                if progbar_sentences.n != last_n and not self.use_adam:
                    sent_passed = epoch * self.corpus_length + progbar_sentences.n
                    lr = max(self.min_learning_rate, self.learning_rate - (sent_passed / num_sentences) * lr_diff)
                    # print(self.learning_rate)
                    for param_group in self._optimizer.param_groups:
                        param_group['lr'] = lr
                    last_n = progbar_sentences.n

            # If we want to save, save it
            if save_fname:
                self.save(save_fname)
                print("Saved file to", save_fname)



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

    def export_vectors(self, fname):
        # Lookup all the words
        embeddings = self._model.lookup_tokens(self.tokens[:, np.newaxis], volatile=True).data
        if J.use_cuda:
            embeddings = embeddings.cpu()
        embeddings = embeddings.numpy().reshape(self.vocab_size, self.embedding_size)
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        with open(fname, 'w') as vector_file:
            # First row is "NUM_WORDS EMBEDDING_SIZE"
            vector_file.write("{} {}\n".format(self.vocab_size, self.embedding_size))
            for i, token in enumerate(tqdm(self.tokens)):
                vector_file.write("{} {}\n".format(token.text, " ".join(str(val) for val in embeddings[i])))

    def save(self, fname):
        if self._model is not None:
            torch.save(self._model.state_dict(), fname + ".state_dict")
        # Save the numpy arrays if their there
        if self.tokens is not None:
            np.savez(fname + ".npz", tokens=self.tokens, sampling_probs=self.sampling_probs,
                     unigram_probs=self.unigram_probs)
        # Pickle the params we need to reconstruct the model
        params = {"embedding_size": self.embedding_size, "learning_rate": self.learning_rate,
                  "min_learning_rate": self.min_learning_rate, "num_neg_samples": self.num_neg_samples,
                  "batch_size": self.batch_size, "epochs": self.epochs, "window_size": self.window_size,
                  "dynamic_window": self.dynamic_window, "min_count": self.min_count, "subsample": self.subsample,
                  "seed": random.randint(0, 2 ** 32), "model_func": self._model_func, "use_adam": self.use_adam}
        attributes = {"vocab": self.vocab, "corpus_length": self.corpus_length, "vocab_size": self.vocab_size,
                      "char2id": self.char2id,
                      "model_saved": self._model is not None, "numpy_saved": self.tokens is not None}
        with open(fname + ".pkl", 'wb') as save_file:
            pickle.dump((params, attributes), save_file)

    def load_decoders_from_gensim(self, gensim_w2v_fname):
        gensim_w2v = gensim_Word2Vec.load(gensim_w2v_fname)
        # Get the tokens indexes in gensim's vocabulary
        gensim_inds = [gensim_w2v.wv.vocab[token.text].index for token in self.tokens]
        # Get the decoder numpy array
        decoders = gensim_w2v.syn1neg[gensim_inds]
        assert not np.all(decoders == 0.)
        self._model.set_decoders_from_numpy(decoders)
        self._model.freeze_decoders()


    @staticmethod
    def load(fname, sentences=None, min_count=5):
        print(fname)
        with open(fname + ".pkl", 'rb') as load_file:
            params, attributes = pickle.load(load_file)
        # Create the Word2Vec wrapper
        word2vec = Word2Vec(**params)

        # Load the vocabulary attributes from numpy
        word2vec.vocab = attributes["vocab"]
        word2vec.corpus_length = attributes["corpus_length"]
        word2vec.vocab_size = attributes["vocab_size"]
        if "char2id" in attributes:
            word2vec.char2id = attributes["char2id"]
        # If sentences are provided use them for characters
        if sentences is not None:
            tokens, vocab, char2id, corpus_length, sentences = word2vec.build_vocab(sentences, min_count=min_count)
            word2vec.char2id = char2id
        if attributes["numpy_saved"]:
            vocab_arrays = np.load(fname + ".npz")
            word2vec.tokens = vocab_arrays["tokens"]
            word2vec.sampling_probs = vocab_arrays["sampling_probs"]
            word2vec.unigram_probs = vocab_arrays["unigram_probs"]

        # Load the model if we have one
        if attributes["model_saved"] is not None:
            word2vec._model = word2vec.build_model()
            saved_state = torch.load(fname + ".state_dict", map_location=lambda storage, loc: storage)
            word2vec._model.load_state_dict(saved_state)
            word2vec._optimizer = word2vec.build_optimizer()

        return word2vec



