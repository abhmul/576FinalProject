import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J


class AWord2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        """
        Initializes an abstract pytorch word2vec module.

        :param vocab_size: The size of the vocabulary of the corpus
        :param embedding_size: The dimension of the embeddings
        """
        super(AWord2Vec, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

    def token2tensor(self, tokens):
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

        word_embs = self.lookup_tokens(input_tokens)  # B x 1 x E
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

        assert true_logits.size(0) == sampled_logits.size(0)

        # Using pytorch's implementation
        # cross-entropy(logits, labels)
        # true_xent = F.binary_cross_entropy_with_logits(true_logits, Variable(J.ones(*true_logits.size())),
        #                                                size_average=False)
        # sampled_xent = F.binary_cross_entropy_with_logits(sampled_logits, Variable(J.zeros(*sampled_logits.size())),
        #                                                   size_average=False)

        # Using ray1007's implementation
        true_logits = torch.clamp(true_logits, max=10, min=-10)
        sampled_logits = torch.clamp(sampled_logits, max=10, min=-10)
        true_xent = torch.sum(-F.logsigmoid(true_logits))
        sampled_xent = torch.sum(-F.logsigmoid(-sampled_logits))

        # Using tf implementation
        # true_logits = torch.clamp(true_logits, max=10)
        # sampled_logits = torch.clamp(sampled_logits, max=10)
        # def bce_with_logits(x, z):
        #     # x is logit, z is label
        #     return torch.clamp(x, min=0) - (x if z else 0) + torch.log(1 + torch.exp(-torch.abs(x)))
        #
        # true_xent = torch.sum(bce_with_logits(true_logits, True))
        # sampled_xent = torch.sum(bce_with_logits(sampled_logits, False))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        # print(true_xent)
        # print(sampled_xent)
        # Multiply by 1000 to get stable gradients
        return true_xent, sampled_xent

    @staticmethod
    def predict(logits):
        """Defines how an actual prediction is computed using the logits."""
        return F.sigmoid(logits)

    def lookup(self, token):
        raise NotImplementedError()

    def lookup_tokens(self, tokens):
        raise NotImplementedError()

    def similarities(self, tensor):
        raise NotImplementedError()
