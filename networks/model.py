import tensorflow as tf

from core.evaluation.eval import Evaluator
from core.processing.lemmatization.base import Stemmer

from networks.callback import Callback
from context.debug import DebugKeys

from io import NetworkIO, DataType


class TensorflowModel(object):
    """
    Base model class, which provides api for
        - tensorflow model compilation
        - fitting
        - training
        - load/save states during fitting/training
        and more.
    """

    def __init__(self, io, callback=None):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(callback, Callback) or callback is None)
        self.sess = None
        self.saver = None
        self.network = None
        self.optimiser = None
        self.io = io
        self.callback = callback

    @property
    def Settings(self):
        """
        Should provide following properties:
        TODO: Create base settings class
        """
        raise Exception("Not Implemented")

    def compile_network(self, config):
        """
        Tensorflow model compilation network.
        """
        raise Exception("Not Implemented")

    def get_gpu_memory_fraction(self):
        raise Exception("Not Implemented")

    def notify_initialized(self):
        if self.callback is not None:
            self.callback.on_initialized(self)

    def load_model(self, filepath):
        """
        Loading tensorflow model into session.
        """
        if DebugKeys.LoadSession:
            print "Loading session: {}".format(filepath)

        tf.reset_default_graph()
        self.saver.restore(self.sess, filepath)

    def save_model(self, filepath):

        if DebugKeys.SaveSession:
            print "Saving session: {}".format(filepath)

        self.saver.save(self.sess, filepath)

    def initialize_session(self):
        """
        Tensorflow session initialization
        """
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.get_gpu_memory_fraction())
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.saver = tf.train.Saver()
        self.sess = sess

    def dispose_session(self):
        """
        Tensorflow session dispose method
        """
        self.sess.close()

    def _compile_network(self):
        self.network = self.compile_network(self.Settings)
        self._set_optimiser()

    @staticmethod
    def _display_log(log_names, log_values):
        assert(len(log_names) == len(log_values))
        print '==========================================='
        for index, log_value in enumerate(log_values):
            print "{}: {}".format(log_names[index], log_value)
        print '==========================================='

    def run(self):
        self._compile_network()
        self.notify_initialized()

        self.initialize_session()
        self.fit()
        self.dispose_session()

    def fit(self):
        raise Exception("Not implemented")

    def predict(self, dest_data_type=DataType.Test):
        raise Exception("Not implemented")

    def _set_optimiser(self):
        raise Exception("Not implemented")

    def _evaluate(self, data_type, stemmer):
        assert(isinstance(stemmer, Stemmer))
        files_to_compare_list = self.io.get_files_to_compare_list(self.io.get_model_root(data_type),
                                                                  self.io.get_data_indices(data_type))

        evaluator = Evaluator(self.io.get_synonyms_collection_filepath(),
                              self.io.get_model_root(data_type),
                              stemmer=self.Settings.Stemmer)
        return evaluator.evaluate(files_to_compare_list, debug=DebugKeys.EvaluateDebug)


