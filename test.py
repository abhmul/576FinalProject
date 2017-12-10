import models
from word2vec import Word2Vec
import numpy as np


def equal_model_params(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_save_load():
    model_func = models.load_model_func("Vani"
                                        "llaWord2Vec")
    sentences = [["1", "2", "3", "4", "5"], ["1", "2"]]
    w2v = Word2Vec(sentences, model_func, embedding_size=2, learning_rate=0.001, min_learning_rate=0.0001,
                   num_neg_samples=1, batch_size=2, epochs=1, window_size=1, dynamic_window=False, min_count=1,
                   subsample=1e-4)
    w2v.save(".tmp/test")
    # Now load
    w2v2 = Word2Vec.load(".tmp/test")
    # Make sure they're all equal
    assert equal_model_params(w2v._model, w2v2._model)
    assert w2v.embedding_size == w2v2.embedding_size
    assert w2v.min_learning_rate == w2v2.min_learning_rate
    assert w2v.learning_rate == w2v2.learning_rate
    assert w2v.num_neg_samples == w2v2.num_neg_samples
    assert w2v.batch_size == w2v2.batch_size
    assert w2v.epochs == w2v2.epochs
    assert w2v.window_size == w2v2.window_size
    assert w2v.dynamic_window == w2v2.dynamic_window
    assert w2v.min_count == w2v2.min_count
    assert w2v.subsample == w2v2.subsample

    assert np.all(w2v.tokens == w2v2.tokens)
    assert np.all(w2v.sampling_probs == w2v2.sampling_probs)
    assert np.all(w2v.unigram_probs == w2v2.unigram_probs)

    assert w2v.vocab_size == w2v2.vocab_size
    assert w2v.vocab == w2v2.vocab
    assert w2v.corpus_length == w2v2.corpus_length

test_save_load()