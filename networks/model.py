
class Model(object):

    def dispose_session(self):
        """
        Tensorflow session dispose method
        """
        raise Exception("Not Implemented")

    def initialize_session(self, gpu_memory_fraction=0.25):
        """
        Tensorflow session initialization
        """
        raise Exception("Not Implemented")

    def predict(self, model_name, test_collection=True):
        raise Exception("Not Implemented")

    def fit(self, epochs, callback=None):
        raise Exception("Not Implemented")
