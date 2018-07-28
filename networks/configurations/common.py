import os
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from core.runtime.embeddings import RusvectoresEmbedding


class CommonSettings:

    test_on_epochs = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=0.1,
        epsilon=10e-6,
        rho=0.95)

    @staticmethod
    def load_embedding():
        w2v_filepath = os.path.join(os.path.dirname(__file__),
                                    "../../data/w2v/news_rusvectores2.bin.gz")

        return RusvectoresEmbedding(Word2Vec.load_word2vec_format(w2v_filepath, binary=True))
