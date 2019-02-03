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

    @property
    def Output(self):
        raise Exception("Not implemented")

    @property
    def Log(self):
        """
        return: list, list
            parameter names and perameter values
        """
        return [], []

    @property
    def Accuracy(self):
        raise Exception("Not implemented")

    def create_feed_dict(self, input, data_type):
        raise Exception("Not implemented")
