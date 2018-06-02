class NeuralNetwork(object):

    @property
    def ParametersDictionary(self):
        """
        returns: dict
            Dictionary that illustrates a parameter values by it's string keys
        """
        raise Exception("Not implemented")

    @property
    def Cost(self):
        """
        returns: Tensor
            Error result
        """
        raise Exception("Not implemented")

    @property
    def Labels(self):
        """
        returns: Tensor
            Result labels by passed batch through the neural network
        """
        raise Exception("Not implemented")
