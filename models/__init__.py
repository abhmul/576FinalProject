from .vanilla_word2vec import VanillaWord2Vec
from .abstract_word2vec import AWord2Vec

# Insert all the different models into here when it is created
MODEL_DICT = {VanillaWord2Vec.__name__: VanillaWord2Vec}


def load_model_func(model_name):
    return MODEL_DICT[model_name]


def get_available_models():
    return list(MODEL_DICT.keys())
