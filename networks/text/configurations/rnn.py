from networks.text.configurations.base import TextModelSettings


class CellTypes:
    RNN = 'vanilla'
    LSTM = 'lstm'
    GRU = 'gru'


class RNNConfig(TextModelSettings):

    _cell = CellTypes.RNN
    _hidden_size = 128
    _l2_reg_lambda = 0.0

    @property
    def L2RegLambda(self):
        return self._l2_reg_lambda

    @property
    def CellType(self):
        return self._cell

    @property
    def HiddenSize(self):
        return self._hidden_size

    def _internal_get_parameters(self):
        parameters = super(RNNConfig, self)._internal_get_parameters()
        parameters += [
            ("hidden_size", self.HiddenSize),
            ("cell_type", self.CellType),
            ("l2_reg_lambda", self.L2RegLambda)
        ]

        return parameters
