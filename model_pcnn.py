from model_cnn import CNNModel
from networks.architectures.pcnn import PiecewiseCNN
from networks.processing.batch import BagsCollection
from networks.configurations.cnn import CNNConfig


class PCNNModel(CNNModel):

    def get_sample_type(self):
        return BagsCollection.ST_BASE

    def set_compiled_network(self, network):
        assert(isinstance(network, PiecewiseCNN))
        self.network = network

    def compile_network(self, config):
        assert(isinstance(config, CNNConfig))
        self.network = PiecewiseCNN(
            vocabulary_words=self.E.shape[0],
            embedding_size=self.E.shape[1],
            words_per_news=self.words_per_news,
            bags_per_batch=config.bags_per_minibatch,
            bag_size=config.bag_size,
            channels_count=config.filter_size,
            window_size=config.window_size,
            dp=config.position_size,
            dropout=config.dropout)
