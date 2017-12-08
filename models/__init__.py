from .vanilla_word2vec import VanillaWord2Vec

# Insert all the different models into here when it is created
MODEL_DICT = {VanillaWord2Vec.__name__: VanillaWord2Vec}
print(MODEL_DICT)