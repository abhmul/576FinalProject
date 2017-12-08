import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J


class VanillaWord2Vec(nn.Module):
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
        super(VanillaWord2Vec, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        # Use sparse for more memory efficient computations
        # Note that only SGD will work with sparse embedding layers on a GPU
        self._encoder = nn.Embedding(self.vocab_size, self.embedding_size, sparse=False)
        self._encoder.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        self._decoder = nn.Embedding(self.vocab_size, self.embedding_size, sparse=False)
        self._decoder.weight.data.zero_()

        self._embedding_norms = None

        if J.use_cuda:
            self.cuda()

    @staticmethod
    def token2tensor(tokens):
        """Helper to cast a numpy array of tokens to a torch LongTensor"""
        tokens = np.array(tokens)
        return Variable(J.LongTensor([token.index for token in tokens.flatten()]).view(*(tokens.shape)))

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
        true_xent = F.binary_cross_entropy_with_logits(true_logits, Variable(J.ones(*true_logits.size())),
                                                       size_average=False)
        sampled_xent = F.binary_cross_entropy_with_logits(sampled_logits, Variable(J.zeros(*sampled_logits.size())),
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