from base import Sample
from core.source.vectors import OpinionVectorCollection
import numpy as np
from numpy.linalg import norm


class NLPSample(Sample):
    """
    NLPSample
    """

    def __init__(self, opinion_vector, position):
        super(NLPSample, self).__init__(position)
        self.opinion_vector = opinion_vector

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def to_network_input(self, news_collection, news_window_size, total_words_count):
        """
        TODO. Add nlp features.
        """
        base_input = super(NLPSample, self).to_network_input(
            news_collection, news_window_size, total_words_count)

        return base_input + [self._normalize(self.opinion_vector.vector)]
