MODEL_DICT = {}


def register_model_func(model_func):
    global MODEL_DICT
    MODEL_DICT[model_func.__name__] = model_func


def load_model_func(model_name):
    return MODEL_DICT[model_name]


def get_available_models():
    return list(MODEL_DICT.keys())


from models.gru_word2vec import *
from models.vanilla_word2vec import *
