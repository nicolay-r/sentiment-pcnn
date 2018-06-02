
from networks.model import TensorflowModel
from networks.architectures.cnn import VanillaCNN

from networks.processing.batch import BagsCollection

import io_utils


class CNN(TensorflowModel):

    def __init__(
            self,
            word_embedding,
            synonyms_filepath=io_utils.get_synonyms_filepath(),
            train_indices=io_utils.train_indices(),
            test_indices=io_utils.test_indices(),
            words_per_news=25,
            bag_size=1,
            bags_per_minibatch=50,
            callback=None):

        super(CNN, self).__init__(
            io=io_utils.NetworkIOProvider(),
            word_embedding=word_embedding,
            synonyms_filepath=synonyms_filepath,
            train_indices=train_indices,
            test_indices=test_indices,
            bag_size=bag_size,
            words_per_news=words_per_news,
            bags_per_minibatch=bags_per_minibatch,
            callback=callback
        )

    def get_sample_type(self):
        return BagsCollection.ST_BASE

    def set_compiled_network(self, network):
        assert(isinstance(network, VanillaCNN))
        self.network = network

    def create_feed_dict(self, sess, minibatch, news_words_collection, total_words_count, debug=False):
        """
        returns: dict
            Returns dictionary for tf session
        """
        if debug:
            print "CNN feed dictionary"

        X, p1, p2, P1, P2, y = minibatch.to_network_input(
            news_words_collection, total_words_count, self.words_per_news)

        feed_dict = {self.network.x: X,
                     self.network.y: y,
                     self.network.P1: P1,
                     self.network.P2: P2,
                     self.network.p1_ind: p1,
                     self.network.p2_ind: p2,
                     self.network.E: self.E}

        return feed_dict
