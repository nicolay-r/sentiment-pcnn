from os import makedirs
from os.path import join, exists
from networks.io import NetworkIO, DataType
import io_utils


class ContextLevelNetworkIO(NetworkIO):
    """
    Represents Input interface for NeuralNetwork context
    Now exploited (treated) as input interface only
    """

    def __init__(self, model_name):
        super(ContextLevelNetworkIO, self).__init__(u'ctx_{}'.format(model_name))

    @staticmethod
    def get_entity_filepath(article_index):
        return io_utils.get_entity_filepath(article_index)

    @staticmethod
    def get_news_filepath(article_index):
        return io_utils.get_news_filepath(article_index)

    @staticmethod
    def get_opinion_input_filepath(article_index):
        return io_utils.get_opin_filepath(article_index, is_etalon=True)

    @staticmethod
    def get_neutral_filepath(article_index, data_type):
        if data_type == DataType.Test:
            return io_utils.get_neutral_filepath(article_index, is_train=False)
        if data_type == DataType.Train:
            return io_utils.get_neutral_filepath(article_index, is_train=True)

    def get_model_state_filepath(self, epoch):
        assert(isinstance(epoch, int))
        return join(self._get_model_states_dir(),
                    u'{}__e_{}.state'.format(self.model_name, epoch))

    def _get_model_states_dir(self):
        train_model_root = self.get_model_root(data_type=DataType.Train)
        result_dir = join(train_model_root, u'States/')
        if not exists(result_dir):
            makedirs(result_dir)
        return result_dir

    def get_relations_filepath(self, epoch, data_type):
        assert(isinstance(epoch, int))
        format = None
        if data_type == DataType.Train:
            format = u'{}_rels_train_e_{}.rels'.format(self.model_name, epoch)
        if data_type == DataType.Test:
            format = u'{}_rels_test_e_{}.rels'.format(self.model_name, epoch)

        return join(self._get_model_states_dir(), format)

    def get_relations_prediction_filepath(self, epoch, data_type):
        assert(isinstance(epoch, int))
        format = None
        if data_type == DataType.Test:
            format = u'{}_predict_test_e_{}.preds'.format(self.model_name, epoch)
        if data_type == DataType.Train:
            format = u'{}_predict_train_e_{}.preds'.format(self.model_name, epoch)

        return join(self._get_model_states_dir(), format)


class NetworkCrossValidationIOProvider(ContextLevelNetworkIO):

    def __init__(self, model_name, cv_count):
        assert(isinstance(cv_count, int) and cv_count > 0)

        super(NetworkCrossValidationIOProvider, self).__init__(model_name)
        self.train_indices = None
        self.test_indices = None
        self.cv_iter = io_utils.indices_to_cv_pairs(cv_count)
        self.iterator_index = 0
        self.original_name = self.model_name
        self.next_train_test_pairs()

    def next_train_test_pairs(self):
        self.train_indices, self.test_indices = self.cv_iter.next()
        self.model_name = u"{}_{}".format(self.original_name, self.iterator_index)
        self.iterator_index += 1

    def get_data_indices(self, data_type):
        if data_type == DataType.Test:
            return self.test_indices
        if data_type == DataType.Train:
            return self.train_indices
