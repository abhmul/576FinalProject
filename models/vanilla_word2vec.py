import torch
import torch.nn as nn

import pyjet.backend as J

from .abstract_word2vec import AWord2Vec
from . import register_model_func


class VanillaWord2Vec(AWord2Vec):
    """
    Class that represents the vanilla word2vec module. We'll swap this out with our experimental modules.

    this contains the actual trainable pytorch module and houses the parameters of the model. This model
    should not be trained directly, but rather through the `Word2Vec` class below.
    """

    def __init__(self, vocab_size, embedding_size, sparse=True, **kwargs):
        """
        Initializes a pytorch word2vec module.

        :param vocab_size: The size of the vocabulary of the corpus
        :param embedding_size: The dimension of the embeddings
        """
        super(VanillaWord2Vec, self).__init__(vocab_size, embedding_size)

        # Use sparse for more memory efficient computations
        # Note that only SGD will work with sparse embedding layers on a GPU
        self._encoder = nn.Embedding(self.vocab_size, self.embedding_size, sparse=sparse)
        self._encoder.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        self._decoder = nn.Embedding(self.vocab_size, self.embedding_size, sparse=sparse)
        self._decoder.weight.data.zero_()

        self._embedding_norms = None

        if J.use_cuda:
            self.cuda()

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


register_model_func(VanillaWord2Vec)
