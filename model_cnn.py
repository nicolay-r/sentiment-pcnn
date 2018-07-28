
from networks.model import TensorflowModel
from networks.architectures.cnn import VanillaCNN

from networks.processing.batch import MiniBatch
from networks.configurations.cnn import CNNConfig

import io_utils


class CNNModel(TensorflowModel):

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

        super(CNNModel, self).__init__(
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

    def compile_network(self, config):
        assert(isinstance(config, CNNConfig))
        self.network = VanillaCNN(config, embedding_shape=self.E.shape)

    def create_feed_dict(self, sess, minibatch, news_words_collection, total_words_count, is_train, debug=False):
        """
        returns: dict
            Returns dictionary for tf session
        """
        assert (isinstance(minibatch, MiniBatch))

        if debug:
            print "CNN feed dictionary"

        feed = minibatch.to_network_input(news_words_collection,
                                          total_words_count,
                                          self.words_per_news)

        return self.network.get_feed_dict(feed[MiniBatch.I_X_INDS],
                                          feed[MiniBatch.I_LABELS],
                                          dist_from_subj=feed[MiniBatch.I_SUBJ_DISTS],
                                          dist_from_obj=feed[MiniBatch.I_OBJ_DISTS],
                                          embedding=self.E,
                                          pos=feed[MiniBatch.I_POS_INDS])
