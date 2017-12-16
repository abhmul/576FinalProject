import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import pyjet.backend as J

from .abstract_word2vec import AWord2Vec
from . import register_model_func

from tqdm import tqdm


class CharWord2Vec(AWord2Vec):

    def lookup_tokens(self, tokens):
        raise NotImplementedError()

    def similarities(self, tensor):
        pass

    def __init__(self, vocab_size, embedding_size, char2id, sparse=False, trainable_char_embeddings=False, **kwargs):
        """
        Initializes a pytorch word2vec module.

        :param vocab_size: The size of the vocabulary of the corpus
        :param embedding_size: The dimension of the embeddings
        :param char2id: A mapping of each character in the vocabulary to its respective id
        :param bidirectional: Whether or not to make the GRU bidirectional
        """
        super(CharWord2Vec, self).__init__(vocab_size, embedding_size, sparse=sparse)
        self.char2id = char2id
        self.num_chars = len(self.char2id)

        # Make a one-hot encoder
        self._char_encoder = nn.Embedding(self.num_chars+1, self.num_chars, padding_idx=self.num_chars)
        if trainable_char_embeddings:
            print("Training character embeddings")
            self._char_encoder.weight.data.uniform_(-0.5 / self.num_chars, 0.5 / self.num_chars)
        else:
            torch.diag(torch.ones(self.num_chars), out=self._char_encoder.weight.data[:-1])
            # Freeze the encodings
            self._char_encoder.weight.requires_grad = False

        self._embedding_norms = None

        if J.use_cuda:
            self.cuda()

    def chartoken2tensor(self, tokens):
        """Helper to cast a list of tokens to a torch LongTensor"""
        assert tokens.ndim == 2, (tokens.ndim, tokens.shape)

        charids = [[self.char2id[char] for char in token.text] for token in tokens.flatten()]
        seq_lens = [len(charid_array) for charid_array in charids]
        # Create the padded one hot encodings
        max_seq_len = max(seq_lens)
        # B*W x L
        charids = Variable(J.LongTensor(
            [charid_array + [self.num_chars] * (max_seq_len - len(charid_array)) for charid_array in charids]))
        # Get the one hot encodings
        onehots = self._char_encoder(charids)  # B*W x L x C
        assert onehots.size(0) == tokens.shape[0] * tokens.shape[1]

        return onehots, seq_lens

    def lookup(self, token):
        return self.lookup_tokens(np.array([[token]]))


class RNNWord2Vec(CharWord2Vec):
    """
    Class that represents the vanilla word2vec module. We'll swap this out with our experimental modules.

    this contains the actual trainable pytorch module and houses the parameters of the model. This model
    should not be trained directly, but rather through the `Word2Vec` class below.
    """

    def __init__(self, vocab_size, embedding_size, char2id, bidirectional=False, sparse=False,
                 trainable_char_embeddings=True, num_encoder_layers=1, linear_size=0, gru=False, **kwargs):
        """
        Initializes a pytorch word2vec module.

        :param vocab_size: The size of the vocabulary of the corpus
        :param embedding_size: The dimension of the embeddings
        :param char2id: A mapping of each character in the vocabulary to its respective id
        :param bidirectional: Whether or not to make the GRU bidirectional
        :param linear_size: The size of the linear layer after the rnn. Should be 0 if no linear layer

        """
        super(RNNWord2Vec, self).__init__(vocab_size, embedding_size, char2id, sparse=sparse,
                                          trainable_char_embeddings=trainable_char_embeddings, **kwargs)
        self.bidirectional = bidirectional
        self.hidden = embedding_size if linear_size == 0 else linear_size
        self.num_recurrent_layers = num_encoder_layers
        rnn_constructor = nn.GRU if gru else nn.LSTM
        print("Using", num_encoder_layers, "layer", self.hidden, "neuron", rnn_constructor.__name__, "as encoder")
        self._rnn = rnn_constructor(self.num_chars, self.hidden // 2 if bidirectional else self.hidden,
                                    num_encoder_layers, batch_first=True, bidirectional=self.bidirectional)
        if linear_size:
            linear = nn.Linear(self.hidden, self.embedding_size)
            print("Using linear layer of size", self.hidden, "on top of encoder")
            self._encoder = torch.nn.Sequential(self._rnn, linear)
        else:
            self._encoder = self._rnn

        if J.use_cuda:
            self.cuda()

    def lookup_tokens(self, tokens):
        # Shape of tokens is B x W
        tokens = np.array(tokens)
        step = 10000
        if tokens.shape[0] > step:
            all_outputs = []

            for i in tqdm(range(0, len(tokens), step)):
                token_slice = tokens[i:i+step]
                # create the one-hot char encodings
                # B*W x L x C
                char_encodings, seq_lens = self.chartoken2tensor(token_slice)
                outputs, _ = self._encoder(char_encodings)
                new_outputs = J.zeros(outputs.size(0), outputs.size(2))
                # Fill it in
                for seq_len in seq_lens:
                    new_outputs = outputs[:, seq_len-1]
                outputs = new_outputs.view(token_slice.shape[0], token_slice.shape[1], self.embedding_size)
                all_outputs.append(outputs)
                # Select the last encoding and reshape into B x W
                # outputs = outputs[:, seq_lens-1, :].view(tokens.shape[0], tokens.shape[1], self.embedding_size)
            return torch.cat(all_outputs, dim=0)

        else:
            char_encodings, seq_lens = self.chartoken2tensor(tokens)
            outputs, _ = self._encoder(char_encodings)
            new_outputs = J.zeros(outputs.size(0), outputs.size(2))
            # Fill it in
            for seq_len in seq_lens:
                new_outputs = outputs[:, seq_len - 1]
            outputs = new_outputs.view(tokens.shape[0], tokens.shape[1], self.embedding_size)
            return outputs


class PoolRNNWord2Vec(RNNWord2Vec):

    def __init__(self, vocab_size, embedding_size, char2id, bidirectional=True, sparse=False,
                 trainable_char_embeddings=False, num_encoder_layers=1, linear_size=0, gru=False, **kwargs):
        super(PoolRNNWord2Vec, self).__init__(vocab_size, embedding_size, char2id, bidirectional, sparse=sparse,
                                              trainable_char_embeddings=trainable_char_embeddings,
                                              num_encoder_layers=num_encoder_layers, linear_size=linear_size,
                                              gru=gru, **kwargs)

    def lookup_tokens(self, tokens):
        # Shape of tokens is B x W
        tokens = np.array(tokens)
        step = 10000
        if tokens.shape[0] > step:
            all_outputs = []
            for i in tqdm(range(0, len(tokens), step)):
                token_slice = tokens[i:i+step]
                # create the one-hot char encodings
                # B*W x L x C
                char_encodings, seq_lens = self.chartoken2tensor(token_slice)
                outputs, _ = self._encoder(char_encodings)  # B*W x L x E
                # print(outputs.size())
                outputs = torch.max(outputs, dim=1)[0].view(token_slice.shape[0], token_slice.shape[1], self.embedding_size)
                all_outputs.append(outputs)
            return torch.cat(all_outputs, dim=0)

        else:
            char_encodings, seq_lens = self.chartoken2tensor(tokens)
            outputs, _ = self._encoder(char_encodings)  # B*W x L x E
            # print(outputs.size())
            outputs = torch.max(outputs, dim=1)[0].view(tokens.shape[0], tokens.shape[1], self.embedding_size)
            return outputs


class CNNWord2Vec(CharWord2Vec):

    def __init__(self, vocab_size, embedding_size, char2id, kernel_size=3, sparse=False,
                 trainable_char_embeddings=False, num_encoder_layers=1, linear_size=0, **kwargs):
        super(CNNWord2Vec, self).__init__(vocab_size, embedding_size, char2id, sparse=sparse,
                                          trainable_char_embeddings=trainable_char_embeddings)
        self.hidden = embedding_size if linear_size == 0 else linear_size
        self.num_convolutional_layers = num_encoder_layers
        padding = (kernel_size - 1) // 2
        self._cnn = nn.Sequential(
            *(nn.Conv1d(self.num_chars, self.hidden, kernel_size=kernel_size, padding=padding) for _ in
              range(self.num_convolutional_layers)))
        print("Using", num_encoder_layers, "layer", self.hidden, "neuron CNN", "as encoder")
        if linear_size:
            linear = nn.Linear(self.hidden, self.embedding_size)
            print("Using linear layer of size", self.hidden, "on top of encoder")
            self._encoder = torch.nn.Sequential(self._cnn, linear)
        else:
            self._encoder = self._cnn

        if J.use_cuda:
            self.cuda()

    def lookup_tokens(self, tokens):
        # Shape of tokens is B x W
        tokens = np.array(tokens)
        # create the one-hot char encodings
        # B*W x L x C
        char_encodings, seq_lens = self.chartoken2tensor(tokens)
        outputs = self._encoder(char_encodings.transpose(1, 2)).transpose(1, 2)  # B*W x L x E
        # print(outputs.size())
        outputs = torch.max(outputs, dim=1)[0].view(tokens.shape[0], tokens.shape[1], self.embedding_size)
        return outputs


register_model_func(RNNWord2Vec)
register_model_func(PoolRNNWord2Vec)
register_model_func(CNNWord2Vec)
