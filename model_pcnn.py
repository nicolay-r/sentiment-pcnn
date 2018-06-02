from model_cnn import CNN
from networks.architectures.pcnn import PiecewiseCNN
from networks.processing.batch import BagsCollection


class PCNN(CNN):

    def get_sample_type(self):
        return BagsCollection.ST_BASE

    def set_compiled_network(self, network):
        assert(isinstance(network, PiecewiseCNN))
        self.network = network
