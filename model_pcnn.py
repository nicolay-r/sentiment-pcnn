from model_cnn import CNNModel
from networks.architectures.pcnn import PiecewiseCNN
from networks.processing.batch import MiniBatch
from networks.configurations.cnn import CNNConfig


class PCNNModel(CNNModel):

    def set_compiled_network(self, network):
        assert(isinstance(network, PiecewiseCNN))
        self.network = network

    def compile_network(self, config):
        assert(isinstance(config, CNNConfig))
        self.network = PiecewiseCNN(config, self.E.shape)

    def create_feed_dict(self, sess, minibatch, news_words_collection, total_words_count, is_train, debug=False):
        assert(isinstance(minibatch, MiniBatch))

        feed = minibatch.to_network_input(news_words_collection, total_words_count, self.words_per_news)

        return self.network.get_feed_dict(feed[MiniBatch.I_X_INDS],
                                          feed[MiniBatch.I_LABELS],
                                          p_subj_ind=feed[MiniBatch.I_SUBJ_IND],
                                          p_obj_ind=feed[MiniBatch.I_OBJ_IND],
                                          p_subj_dist=feed[MiniBatch.I_SUBJ_DISTS],
                                          p_obj_dist=feed[MiniBatch.I_OBJ_DISTS],
                                          embedding=self.E,
                                          pos=feed[MiniBatch.I_POS_INDS])
