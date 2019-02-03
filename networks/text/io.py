from networks.io import NetworkIO
from networks.context.io import ContextLevelNetworkIO
from debug import TextDebugKeys


class TextLevelNetworkIO(NetworkIO):
    """
    IO for Text level classification via Neural Networks
    """

    def __init__(self, model_name, ctx_model_name):
        super(TextLevelNetworkIO, self).__init__(u'text_{}'.format(model_name))
        self.ctx_io_provider = ContextLevelNetworkIO(ctx_model_name)

    def get_relations_predictions_filepath(self, epoch, data_type):
        filepath = self.ctx_io_provider.get_relations_prediction_filepath(epoch, data_type)
        if TextDebugKeys.IOPredictionsFilepath:
            print "Getting relations predictions: {}".format(filepath)
        return filepath

    def get_relations_filepath(self, epoch, data_type):
        filepath = self.ctx_io_provider.get_relations_filepath(epoch, data_type)
        if TextDebugKeys.IORelationsFilepath:
            print "Getting extracted relation collection: {}".format(filepath)
        return filepath
